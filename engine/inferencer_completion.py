"""
SAM-Road Completion v2 Inference Script
==========================================
路网补全推理脚本

v2 变更:
  - 支持 4ch 输入 (RGB + traj_heatmap)
  - road_feature_map 2通道 (mask + 节点位置)
  - known_edge_index 最近邻映射到 NMS 关键点
  - 后处理: 已知图边强制 topo_score = 1.0

输入:
  - 遥感影像
  - 已知路网 (pickle 格式邻接表)
  - 可选: 轨迹热力图 (Xian active.png)
输出:
  - 补全后的完整路网
"""

import numpy as np
import os
import sys
# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import cv2
import math

from tools.config_utils import load_config, create_output_dir_and_save_config
from tools.run_info import dump_run_info, mark_run_finished
from data.dataset import cityscale_data_partition, read_rgb_img, get_patch_info_one_img
from data.dataset import spacenet_data_partition
from data.dataset_completion import didi_data_partition, render_graph_feature_map
from models.sam_road_completion import SAMRoadCompletion
from postprocess import graph_extraction
from postprocess import graph_utils
from postprocess import triage
import pickle
import scipy
import rtree
from collections import defaultdict
import time
import json

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--checkpoint", default=None, help="checkpoint of the model to test.")
parser.add_argument("--config", default=None, help="model config.")
parser.add_argument("--output_dir", default=None, help="output dir")
parser.add_argument("--input_graph", default=None,
                    help="path to the known graph pickle file (adj dict format)")
parser.add_argument("--input_graph_dir", default=None,
                    help="directory of known graph pickle files, one per region")
parser.add_argument("--traj_dir", default=None,
                    help="directory of trajectory heatmap images (active.png), for Xian dataset")
parser.add_argument("--device", default="cuda", help="device to use")
args = parser.parse_args()


def crop_img_patch(img, x0, y0, x1, y1):
    return img[y0:y1, x0:x1, :]


def get_batch_img_patches(img, batch_patch_info):
    patches = []
    for _, (x0, y0), (x1, y1) in batch_patch_info:
        patch = crop_img_patch(img, x0, y0, x1, y1)
        patches.append(torch.tensor(patch, dtype=torch.float32))
    batch = torch.stack(patches, 0).contiguous()
    return batch


def load_known_graph(img_id, config):
    """加载已知路网"""
    if args.input_graph:
        graph_path = args.input_graph
    elif args.input_graph_dir:
        if config.DATASET == 'spacenet':
            graph_path = os.path.join(args.input_graph_dir, f'{img_id}__gt_graph.p')
        else:
            graph_path = os.path.join(args.input_graph_dir, f'region_{img_id}_refine_gt_graph.p')
    else:
        return None

    if os.path.exists(graph_path):
        with open(graph_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Warning: known graph not found at {graph_path}")
        return None


def load_traj_heatmap(img_id, config):
    """加载轨迹热力图 (仅 Xian 数据集)"""
    if args.traj_dir is None:
        return None

    if config.DATASET == 'didi':
        traj_path = os.path.join(args.traj_dir, f'region_{img_id}_active.png')
    else:
        return None

    if os.path.exists(traj_path):
        traj_img = cv2.imread(traj_path, cv2.IMREAD_GRAYSCALE)
        return traj_img.astype(np.float32) / 255.0
    else:
        print(f"Warning: traj heatmap not found at {traj_path}")
        return None


def infer_one_img(net, img, config, known_graph_adj=None, traj_heatmap_full=None):
    """
    推理单张影像

    Args:
        net: SAMRoadCompletion 模型
        img: [H, W, 3] RGB 影像
        config: 配置
        known_graph_adj: dict, 已知路网邻接表 {(x,y): [(x1,y1), ...]}
        traj_heatmap_full: [H, W] 轨迹热力图 (None 则用全零)
    """
    image_size = img.shape[0]
    batch_size = config.INFER_BATCH_SIZE

    all_patch_info = get_patch_info_one_img(
        0, image_size, config.SAMPLE_MARGIN, config.PATCH_SIZE, config.INFER_PATCHES_PER_EDGE
    )
    patch_num = len(all_patch_info)
    batch_num = (
        patch_num // batch_size
        if patch_num % batch_size == 0
        else patch_num // batch_size + 1
    )

    fused_keypoint_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device)
    fused_road_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device)
    pixel_counter = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device)

    img_features = list()

    # ---- Pass 1: SAM 编码 + 已知路网特征编码 + 分割 mask ----
    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset: offset + batch_size]
        batch_img_patches = get_batch_img_patches(img, batch_patch_info)

        # 为每个 patch 渲染已知路网特征图 (2通道)
        road_feature_maps = []
        traj_heatmap_patches = []
        for _, (x0, y0), (x1, y1) in batch_patch_info:
            if known_graph_adj is not None:
                rfm = render_graph_feature_map(
                    known_graph_adj, x0, y0, config.PATCH_SIZE
                )
            else:
                rfm = np.zeros((config.PATCH_SIZE, config.PATCH_SIZE, 2), dtype=np.float32)
            road_feature_maps.append(
                torch.tensor(rfm, dtype=torch.float32).permute(2, 0, 1)  # [2, H, W]
            )

            # 轨迹热力图
            if traj_heatmap_full is not None:
                traj_patch = traj_heatmap_full[y0:y1, x0:x1]
                traj_heatmap_patches.append(
                    torch.tensor(traj_patch, dtype=torch.float32).unsqueeze(-1)  # [H, W, 1]
                )
            else:
                traj_heatmap_patches.append(
                    torch.zeros(config.PATCH_SIZE, config.PATCH_SIZE, 1, dtype=torch.float32)
                )

        batch_road_features = torch.stack(road_feature_maps, 0).to(args.device)
        batch_traj_heatmaps = torch.stack(traj_heatmap_patches, 0).to(args.device)

        with torch.no_grad():
            batch_img_patches = batch_img_patches.to(args.device)
            mask_scores, patch_fused_features = net.infer_masks_and_img_features(
                batch_img_patches, batch_traj_heatmaps, batch_road_features
            )
            img_features.append(patch_fused_features)

        for patch_index, patch_info in enumerate(batch_patch_info):
            _, (x0, y0), (x1, y1) = patch_info
            keypoint_patch = mask_scores[patch_index, :, :, 0]
            road_patch = mask_scores[patch_index, :, :, 1]
            fused_keypoint_mask[y0:y1, x0:x1] += keypoint_patch
            fused_road_mask[y0:y1, x0:x1] += road_patch
            pixel_counter[y0:y1, x0:x1] += torch.ones(
                road_patch.shape[0:2], dtype=torch.float32, device=args.device
            )

    fused_keypoint_mask /= pixel_counter
    fused_road_mask /= pixel_counter
    fused_keypoint_mask = (fused_keypoint_mask * 255).to(torch.uint8).cpu().numpy()
    fused_road_mask = (fused_road_mask * 255).to(torch.uint8).cpu().numpy()

    # ---- 提取图节点 ----
    graph_points = graph_extraction.extract_graph_points(
        fused_keypoint_mask, fused_road_mask, config
    )
    print(f'Extracted {graph_points.shape[0]} graph points')
    if graph_points.shape[0] == 0:
        return graph_points, np.zeros((0, 2), dtype=np.int32), fused_keypoint_mask, fused_road_mask

    graph_rtree = rtree.index.Index()
    for i, v in enumerate(graph_points):
        x, y = v
        graph_rtree.insert(i, (x, y, x, y))

    # ---- Pass 2: TopoNet 推理 ----
    edge_scores = defaultdict(float)
    edge_counts = defaultdict(float)

    # 构造已知路网在全局节点中的边索引 (用于 GNN)
    # 使用最近邻映射将已知路网边映射到 NMS 关键点
    known_edge_index_global = _match_known_edges_to_graph_points(
        known_graph_adj, graph_points, config.NEIGHBOR_RADIUS
    )

    # 收集已知路网在全局节点中的所有边 (用于后处理硬覆盖)
    known_edges_set = set()
    if known_graph_adj is not None and known_edge_index_global.shape[1] > 0:
        for e in range(known_edge_index_global.shape[1]):
            s = known_edge_index_global[0, e].item()
            t = known_edge_index_global[1, e].item()
            known_edges_set.add((min(s, t), max(s, t)))

    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset: offset + batch_size]

        topo_data = {'points': [], 'pairs': [], 'valid': []}
        idx_maps = []

        for patch_info in batch_patch_info:
            _, (x0, y0), (x1, y1) = patch_info
            patch_point_indices = list(graph_rtree.intersection((x0, y0, x1, y1)))
            idx_patch2all = {
                patch_idx: all_idx
                for patch_idx, all_idx in enumerate(patch_point_indices)
            }
            patch_point_num = len(patch_point_indices)
            patch_points = graph_points[patch_point_indices, :] - np.array(
                [[x0, y0]], dtype=graph_points.dtype
            )
            patch_kdtree = scipy.spatial.KDTree(patch_points)

            knn_d, knn_idx = patch_kdtree.query(
                patch_points, k=config.MAX_NEIGHBOR_QUERIES + 1,
                distance_upper_bound=config.NEIGHBOR_RADIUS
            )
            knn_idx = knn_idx[:, 1:]
            src_idx = np.tile(
                np.arange(patch_point_num)[:, np.newaxis],
                (1, config.MAX_NEIGHBOR_QUERIES)
            )
            valid = knn_idx < patch_point_num
            tgt_idx = np.where(valid, knn_idx, src_idx)
            pairs = np.stack([src_idx, tgt_idx], axis=-1)

            topo_data['points'].append(patch_points)
            topo_data['pairs'].append(pairs)
            topo_data['valid'].append(valid)
            idx_maps.append(idx_patch2all)

        # collate
        collated = {}
        for key, x_list in topo_data.items():
            length = max([x.shape[0] for x in x_list])
            collated[key] = np.stack([
                np.pad(x, [(0, length - x.shape[0])] + [(0, 0)] * (len(x.shape) - 1))
                for x in x_list
            ], axis=0)

        if collated['points'].shape[1] == 0:
            continue

        batch_features = img_features[batch_index]
        batch_points = torch.tensor(collated['points'], device=args.device)
        batch_pairs = torch.tensor(collated['pairs'], device=args.device)
        batch_valid = torch.tensor(collated['valid'], device=args.device)

        # 为每个 patch 构造 known_edge_index (在 patch 节点空间中)
        batch_known_edge_index = _build_batch_known_edge_index(
            batch_patch_info,
            graph_points=graph_points, graph_rtree=graph_rtree,
            known_edge_index_global=known_edge_index_global
        )

        with torch.no_grad():
            topo_scores = net.infer_toponet(
                batch_features, batch_points, batch_pairs, batch_valid,
                known_edge_index=batch_known_edge_index
            )

        topo_scores = torch.where(
            torch.isnan(topo_scores), -100.0, topo_scores
        ).squeeze(-1).cpu().numpy()

        batch_size_actual, n_samples, n_pairs = topo_scores.shape
        for bi in range(batch_size_actual):
            for si in range(n_samples):
                for pi in range(n_pairs):
                    if not collated['valid'][bi, si, pi]:
                        continue
                    src_idx_patch, tgt_idx_patch = collated['pairs'][bi, si, pi, :]
                    src_idx_all = idx_maps[bi][src_idx_patch]
                    tgt_idx_all = idx_maps[bi][tgt_idx_patch]
                    edge_score = topo_scores[bi, si, pi]
                    assert 0.0 <= edge_score <= 1.0
                    edge_scores[(src_idx_all, tgt_idx_all)] += edge_score
                    edge_counts[(src_idx_all, tgt_idx_all)] += 1.0

    # ---- 聚合边分数并过滤 ----
    pred_new_edges = []
    for edge, score_sum in edge_scores.items():
        score = score_sum / edge_counts[edge]
        # 后处理: 已知图的边强制 topo_score = 1.0 (硬覆盖)
        edge_key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
        if edge_key in known_edges_set:
            score = 1.0
        if score > config.TOPO_THRESHOLD:
            pred_new_edges.append(edge)
    pred_new_edges = np.array(pred_new_edges).reshape(-1, 2)
    pred_nodes = graph_points[:, ::-1]  # to rc

    # ---- 合并已知路网的边 + 预测新增边 ----
    if known_graph_adj is not None:
        known_edges = _get_known_edges_in_graph_points(
            known_graph_adj, graph_points, config.NEIGHBOR_RADIUS
        )
        if len(known_edges) > 0 and len(pred_new_edges) > 0:
            # 去重: 已知边不再添加
            known_set = set(map(tuple, known_edges))
            new_unique = [
                e for e in pred_new_edges
                if tuple(e) not in known_set and (e[1], e[0]) not in known_set
            ]
            if len(new_unique) > 0:
                pred_new_edges_unique = np.array(new_unique)
            else:
                pred_new_edges_unique = np.zeros((0, 2), dtype=np.int32)
            all_edges = np.concatenate([known_edges, pred_new_edges_unique], axis=0)
        elif len(known_edges) > 0:
            all_edges = known_edges
        elif len(pred_new_edges) > 0:
            all_edges = pred_new_edges
        else:
            all_edges = np.zeros((0, 2), dtype=np.int32)
    else:
        all_edges = pred_new_edges

    print(f'Known edges: {len(known_edges) if known_graph_adj is not None and len(known_edges) > 0 else 0}, '
          f'New predicted edges: {len(pred_new_edges)}, '
          f'Final edges: {len(all_edges)}')

    return pred_nodes, all_edges, fused_keypoint_mask, fused_road_mask


def _match_known_edges_to_graph_points(known_graph_adj, graph_points, neighbor_radius):
    """
    将已知路网的边匹配到 NMS 后的全局图节点 (最近邻映射)

    Args:
        known_graph_adj: dict, 已知路网邻接表
        graph_points: [N, 2] NMS 后的关键点坐标
        neighbor_radius: 最近邻匹配的最大距离阈值

    Returns:
        known_edge_index: [2, E] tensor, 已知路网在全局节点索引中的边
    """
    if known_graph_adj is None or len(known_graph_adj) == 0:
        return torch.zeros(2, 0, dtype=torch.long)

    # 收集所有已知路网节点
    known_nodes = set()
    for node, neighbors in known_graph_adj.items():
        known_nodes.add(node)
        for neighbor in neighbors:
            known_nodes.add(neighbor)

    known_nodes_list = list(known_nodes)
    if len(known_nodes_list) == 0:
        return torch.zeros(2, 0, dtype=torch.long)

    known_coords = np.array(known_nodes_list, dtype=np.float32)  # [N_known, 2]

    # 用 KDTree 将已知路网节点匹配到最近的 NMS 节点
    if len(graph_points) == 0:
        return torch.zeros(2, 0, dtype=torch.long)

    kdtree = scipy.spatial.KDTree(graph_points)
    distances, indices = kdtree.query(known_coords, k=1)

    # 建立 known_node → graph_point_index 的映射
    known_to_gp = {}
    for i, (dist, gp_idx) in enumerate(zip(distances, indices)):
        if dist < neighbor_radius:
            known_to_gp[known_nodes_list[i]] = gp_idx

    # 构造已知边在全局节点索引中的表示
    edges_src = []
    edges_tgt = []
    for node, neighbors in known_graph_adj.items():
        if node in known_to_gp:
            for neighbor in neighbors:
                if neighbor in known_to_gp:
                    src_idx = known_to_gp[node]
                    tgt_idx = known_to_gp[neighbor]
                    # 避免自环和重复
                    if src_idx != tgt_idx:
                        edges_src.append(src_idx)
                        edges_tgt.append(tgt_idx)

    if len(edges_src) == 0:
        return torch.zeros(2, 0, dtype=torch.long)

    return torch.tensor([edges_src, edges_tgt], dtype=torch.long)


def _build_batch_known_edge_index(batch_patch_info, graph_points, graph_rtree,
                                   known_edge_index_global):
    """
    为 batch 中的每个 patch 构造 known_edge_index

    将全局节点空间的 known_edge_index 转换为 patch 节点空间的索引
    """
    if known_edge_index_global.shape[1] == 0:
        return torch.zeros(len(batch_patch_info), 2, 0, dtype=torch.long, device=args.device)

    batch_edge_indices = []
    for bi, patch_info in enumerate(batch_patch_info):
        _, (x0, y0), (x1, y1) = patch_info
        patch_point_indices = list(graph_rtree.intersection((x0, y0, x1, y1)))
        idx_all2patch = {all_idx: patch_idx for patch_idx, all_idx in enumerate(patch_point_indices)}

        patch_src = []
        patch_tgt = []
        for e in range(known_edge_index_global.shape[1]):
            s = known_edge_index_global[0, e].item()
            t = known_edge_index_global[1, e].item()
            if s in idx_all2patch and t in idx_all2patch:
                patch_src.append(idx_all2patch[s])
                patch_tgt.append(idx_all2patch[t])

        if len(patch_src) > 0:
            edge_index = torch.tensor([patch_src, patch_tgt], dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        batch_edge_indices.append(edge_index)

    # Padding to same length
    max_edges = max(ei.shape[1] for ei in batch_edge_indices)
    if max_edges == 0:
        return torch.zeros(len(batch_patch_info), 2, 0, dtype=torch.long, device=args.device)

    padded = []
    for ei in batch_edge_indices:
        if ei.shape[1] < max_edges:
            pad = torch.zeros(2, max_edges - ei.shape[1], dtype=torch.long)
            ei = torch.cat([ei, pad], dim=1)
        padded.append(ei)

    return torch.stack(padded, dim=0).to(args.device)  # [B, 2, max_edges]


def _get_known_edges_in_graph_points(known_graph_adj, graph_points, neighbor_radius):
    """获取已知路网在全局节点索引中的边 (numpy)"""
    known_edge_index = _match_known_edges_to_graph_points(
        known_graph_adj, graph_points, neighbor_radius
    )
    if known_edge_index.shape[1] == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return known_edge_index.numpy().T  # [E, 2]


if __name__ == "__main__":
    config = load_config(args.config)

    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    net = SAMRoadCompletion(config)

    # 加载 checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    print(f'##### Loading Trained CKPT {args.checkpoint} #####')
    net.load_state_dict(checkpoint["state_dict"], strict=False)
    net.eval()
    net.to(device)

    # 数据集划分
    if config.DATASET == 'cityscale':
        _, _, test_img_indices = cityscale_data_partition()
        rgb_pattern = './datasets/cityscale/20cities/region_{}_sat.png'
    elif config.DATASET == 'spacenet':
        _, _, test_img_indices = spacenet_data_partition()
        rgb_pattern = './datasets/spacenet/RGB_1.0_meter/{}__rgb.png'
    elif config.DATASET == 'didi':
        _, _, test_img_indices = didi_data_partition()
        rgb_pattern = './xian/2019_400/xian_2019_400/region_{}_sat.png'

    output_dir_prefix = './save/infer_completion_'
    if args.output_dir:
        output_dir = create_output_dir_and_save_config(
            output_dir_prefix, config, specified_dir=f'./save/{args.output_dir}'
        )
    else:
        output_dir = create_output_dir_and_save_config(output_dir_prefix, config)

    # 写运行元信息 (与 config.yaml 分离)
    run_info_path = dump_run_info(
        output_dir=output_dir,
        script=__file__,
        args=args,
        config_source=args.config,
        checkpoint=args.checkpoint,
        extra={'task': 'inference', 'model': 'sam_road_completion'},
    )

    total_inference_seconds = 0.0

    # 启动时一次性提示运行模式 (而不是每张图都打印)
    if args.input_graph or args.input_graph_dir:
        graph_src = args.input_graph or args.input_graph_dir
        print(f'==> Completion mode: using known graph from {graph_src}')
    else:
        print('==> No --input_graph / --input_graph_dir given. '
              'Running in full extraction fallback (model degrades to pure SAM-Road).')

    for img_id in test_img_indices:
        print(f'Processing {img_id}')
        img = read_rgb_img(rgb_pattern.format(img_id))

        # 加载已知路网
        known_graph_adj = load_known_graph(img_id, config)
        if known_graph_adj is not None:
            print(f'  Loaded known graph with {len(known_graph_adj)} nodes')
        # else 分支不再每张图都重复打印 fallback 提示

        # 加载轨迹热力图 (仅 Xian)
        traj_heatmap = load_traj_heatmap(img_id, config)
        if traj_heatmap is not None:
            print(f'  Loaded traj heatmap')
        else:
            print(f'  No traj heatmap, using zeros')

        start_seconds = time.time()
        pred_nodes, pred_edges, itsc_mask, road_mask = infer_one_img(
            net, img, config, known_graph_adj, traj_heatmap
        )
        end_seconds = time.time()
        total_inference_seconds += (end_seconds - start_seconds)

        # 保存可视化
        viz_save_dir = os.path.join(output_dir, 'viz')
        if not os.path.exists(viz_save_dir):
            os.makedirs(viz_save_dir)
        viz_img = np.copy(img)
        img_size = viz_img.shape[0]
        if len(pred_nodes) > 0 and len(pred_edges) > 0:
            viz_img = triage.visualize_image_and_graph(
                viz_img, pred_nodes / img_size, pred_edges, img_size
            )
        cv2.imwrite(os.path.join(viz_save_dir, f'{img_id}.png'), viz_img)

        # 保存 mask
        mask_save_dir = os.path.join(output_dir, 'mask')
        if not os.path.exists(mask_save_dir):
            os.makedirs(mask_save_dir)
        cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_road.png'), road_mask)
        cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_itsc.png'), itsc_mask)

        # 保存图
        if config.DATASET == 'spacenet' or config.DATASET == 'didi':
            pred_nodes = np.stack([400 - pred_nodes[:, 0], pred_nodes[:, 1]], axis=1)

        if len(pred_nodes) > 0 and len(pred_edges) > 0:
            large_map_format = graph_utils.convert_to_sat2graph_format(pred_nodes, pred_edges)
        else:
            large_map_format = {}

        graph_save_dir = os.path.join(output_dir, 'graph')
        if not os.path.exists(graph_save_dir):
            os.makedirs(graph_save_dir)
        with open(os.path.join(graph_save_dir, f'{img_id}.p'), 'wb') as f:
            pickle.dump(large_map_format, f)

        print(f'Done for {img_id}.')

    time_txt = f'Inference completed in {total_inference_seconds} seconds.'
    print(time_txt)
    with open(os.path.join(output_dir, 'inference_time.txt'), 'w') as f:
        f.write(time_txt)
    mark_run_finished(run_info_path)
