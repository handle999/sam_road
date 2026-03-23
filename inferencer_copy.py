import numpy as np
import os
import imageio
import torch
import cv2

from utils import load_config, create_output_dir_and_save_config
from dataset_copy import cityscale_data_partition, read_rgb_img, get_patch_info_one_img
from dataset_copy import spacenet_data_partition
# from model import SAMRoad   # 如果使用其他模型，请取消相应的注释
from model_copy import SAMRoad
import graph_extraction
import graph_utils
import triage
# from triage import visualize_image_and_graph, rasterize_graph
import pickle
import scipy
import rtree
from collections import defaultdict
import time

from argparse import ArgumentParser
from tqdm import tqdm       # infer bar


parser = ArgumentParser()
parser.add_argument(
    "--checkpoint", default=None, help="checkpoint of the model to test."
)
parser.add_argument(
    "--config", default=None, help="model config."
)
parser.add_argument(
    "--output_dir", default=None, help="Name of the output dir, if not specified will use timestamp"
)
parser.add_argument("--device", default="cuda", help="device to use for training")

# 任务界定，分为“提取”和“更新”：
parser.add_argument("--task", type=str, default="extraction", choices=["extraction", "update", "full"], 
                    help="extraction: 0 prior (全黑) | update: partial prior (读 sample) | full: full prior (读 gt 图，极端情况)")

# [=== 核心修改：新增用于全自动消融实验的参数覆写接口 ===]
parser.add_argument("--exp_id", type=str, default=None, help="Unique ID for this experiment run")
parser.add_argument("--dataset", type=str, default=None, help="Override config.DATASET")
parser.add_argument("--edge", type=int, default=None, help="Override config.INFER_PATCHES_PER_EDGE")
parser.add_argument("--ratio", type=float, default=None, help="Override config.SAMPLE_RATIO")
parser.add_argument("--road_thresh", type=float, default=None, help="Override config.ROAD_THRESHOLD")
parser.add_argument("--itsc_thresh", type=float, default=None, help="Override config.ITSC_THRESHOLD")
parser.add_argument("--topo_thresh", type=float, default=None, help="Override config.TOPO_THRESHOLD")
parser.add_argument("--road_nms", type=int, default=None, help="Override config.ROAD_NMS_RADIUS")
parser.add_argument("--itsc_nms", type=int, default=None, help="Override config.ITSC_NMS_RADIUS")
parser.add_argument("--nbr_radius", type=int, default=None, help="Override config.NEIGHBOR_RADIUS")
parser.add_argument("--max_nbr_q", type=int, default=None, help="Override config.MAX_NEIGHBOR_QUERIES")
# ==========================================

args = parser.parse_args()


def get_img_paths(root_dir, image_indices):
    img_paths = []

    for ind in image_indices:
        img_paths.append(os.path.join(root_dir, f"region_{ind}_sat.png"))

    return img_paths


def crop_img_patch(img, x0, y0, x1, y1):
    return img[y0:y1, x0:x1, :]


def get_batch_img_patches(img, batch_patch_info):
    patches = []
    for _, (x0, y0), (x1, y1) in batch_patch_info:
        patch = crop_img_patch(img, x0, y0, x1, y1)
        patches.append(torch.tensor(patch, dtype=torch.float32))
    batch = torch.stack(patches, 0).contiguous()
    return batch


def infer_one_img(net, img, config):
    # TODO(congrui): centralize these configs
    image_size = img.shape[0]

    batch_size = config.INFER_BATCH_SIZE
    # list of (i, (x_begin, y_begin), (x_end, y_end))
    all_patch_info = get_patch_info_one_img(
        0, image_size, config.SAMPLE_MARGIN, config.PATCH_SIZE, config.INFER_PATCHES_PER_EDGE)
    patch_num = len(all_patch_info)
    batch_num = (
        patch_num // batch_size
        if patch_num % batch_size == 0
        else patch_num // batch_size + 1
    )

    # [IMG_H, IMG_W]
    fused_keypoint_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)
    fused_road_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)
    pixel_counter = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)

    # stores img embeddings for toponet
    # list of [B, D, h, w], len=batch_num
    img_features = list()

    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]
        # tensor [B, H, W, C]
        batch_img_patches = get_batch_img_patches(img, batch_patch_info)

        with torch.no_grad():
            batch_img_patches = batch_img_patches.to(args.device, non_blocking=False)
            # [B, H, W, 2]
            mask_scores, patch_img_features = net.infer_masks_and_img_features(batch_img_patches)
            img_features.append(patch_img_features)
        # Aggregate masks
        for patch_index, patch_info in enumerate(batch_patch_info):
            _, (x0, y0), (x1, y1) = patch_info
            keypoint_patch, road_patch = mask_scores[patch_index, :, :, 0], mask_scores[patch_index, :, :, 1]
            fused_keypoint_mask[y0:y1, x0:x1] += keypoint_patch
            fused_road_mask[y0:y1, x0:x1] += road_patch
            pixel_counter[y0:y1, x0:x1] += torch.ones(road_patch.shape[0:2], dtype=torch.float32, device=args.device)
    
    fused_keypoint_mask /= pixel_counter
    fused_road_mask /= pixel_counter
    # range 0-1 -> 0-255
    fused_keypoint_mask = (fused_keypoint_mask * 255).to(torch.uint8).cpu().numpy()
    fused_road_mask = (fused_road_mask * 255).to(torch.uint8).cpu().numpy()

    # ## Astar graph extraction
    # pred_graph = graph_extraction.extract_graph_astar(fused_keypoint_mask, fused_road_mask, config)
    # # Doing this conversion to reuse copied code
    # pred_nodes, pred_edges = graph_utils.convert_from_nx(pred_graph)
    # return pred_nodes, pred_edges, fused_keypoint_mask, fused_road_mask
    # ## Astar graph extraction
    
    ## Extract sample points from masks
    graph_points = graph_extraction.extract_graph_points(fused_keypoint_mask, fused_road_mask, config)
    # print(graph_points.shape[0])    # hhy add 0707
    if graph_points.shape[0] == 0:
        return graph_points, np.zeros((0, 2), dtype=np.int32), fused_keypoint_mask, fused_road_mask

    # for box query
    graph_rtree = rtree.index.Index()
    for i, v in enumerate(graph_points):
        x, y = v
        # hack to insert single points
        graph_rtree.insert(i, (x, y, x, y))
    
    ## Pass 2: infer toponet to predict topology of points from stored img features
    edge_scores = defaultdict(float)
    edge_counts = defaultdict(float)
    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]

        topo_data = {
            'points': [],
            'pairs': [],
            'valid': [],
        }
        idx_maps = []


        # prepares pairs queries
        for patch_info in batch_patch_info:
            _, (x0, y0), (x1, y1) = patch_info
            patch_point_indices = list(graph_rtree.intersection((x0, y0, x1, y1)))
            idx_patch2all = {patch_idx : all_idx for patch_idx, all_idx in enumerate(patch_point_indices)}
            patch_point_num = len(patch_point_indices)
            # normalize into patch
            patch_points = graph_points[patch_point_indices, :] - np.array([[x0, y0]], dtype=graph_points.dtype)
            # for knn and circle query
            patch_kdtree = scipy.spatial.KDTree(patch_points)

            # k+1 because the nearest one is always self
            # idx is to the patch subgraph
            knn_d, knn_idx = patch_kdtree.query(patch_points, k=config.MAX_NEIGHBOR_QUERIES + 1, distance_upper_bound=config.NEIGHBOR_RADIUS)
            # [patch_point_num, n_nbr]
            knn_idx = knn_idx[:, 1:]  # removes self
            # [patch_point_num, n_nbr] idx is to the patch subgraph
            src_idx = np.tile(
                np.arange(patch_point_num)[:, np.newaxis],
                (1, config.MAX_NEIGHBOR_QUERIES)
            )
            valid = knn_idx < patch_point_num
            tgt_idx = np.where(valid, knn_idx, src_idx)
            # [patch_point_num, n_nbr, 2]
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

        # skips this batch if there's no points
        if collated['points'].shape[1] == 0:
            continue
        
        # infer toponet
        # [B, D, h, w]
        batch_features = img_features[batch_index]
        # [B, N_sample, N_pair, 2]
        batch_points = torch.tensor(collated['points'], device=args.device)
        batch_pairs = torch.tensor(collated['pairs'], device=args.device)
        batch_valid = torch.tensor(collated['valid'], device=args.device)


        with torch.no_grad():
            # [B, N_samples, N_pairs, 1]
            topo_scores = net.infer_toponet(batch_features, batch_points, batch_pairs, batch_valid)
                
        # all-invalid (padded, no neighbors) queries returns nan scores
        # [B, N_samples, N_pairs]
        topo_scores = torch.where(torch.isnan(topo_scores), -100.0, topo_scores).squeeze(-1).cpu().numpy()

        # aggregate edge scores
        batch_size, n_samples, n_pairs = topo_scores.shape
        for bi in range(batch_size):
            for si in range(n_samples):
                for pi in range(n_pairs):
                    if not collated['valid'][bi, si, pi]:
                        continue
                    # idx to the full graph
                    src_idx_patch, tgt_idx_patch = collated['pairs'][bi, si, pi, :]
                    src_idx_all, tgt_idx_all = idx_maps[bi][src_idx_patch], idx_maps[bi][tgt_idx_patch]
                    edge_score = topo_scores[bi, si, pi]
                    assert 0.0 <= edge_score <= 1.0
                    edge_scores[(src_idx_all, tgt_idx_all)] += edge_score
                    edge_counts[(src_idx_all, tgt_idx_all)] += 1.0

    # avg edge scores and filter
    pred_edges = []
    for edge, score_sum in edge_scores.items():
        score = score_sum / edge_counts[edge] 
        # print(edge, score)  # hhy add 0707
        if score > config.TOPO_THRESHOLD:
            pred_edges.append(edge)
    pred_edges = np.array(pred_edges).reshape(-1, 2)
    pred_nodes = graph_points[:, ::-1]  # to rc
    
    return pred_nodes, pred_edges, fused_keypoint_mask, fused_road_mask

    
if __name__ == "__main__":
    config = load_config(args.config)
    
    # [=== 修改 1：参数覆盖逻辑 (Override) ===]
    # 按照 命令行参数(args) > 配置文件(config) 的优先级进行覆盖
    if args.dataset is not None:     config.DATASET = args.dataset
    if args.edge is not None:        config.INFER_PATCHES_PER_EDGE = args.edge
    if args.ratio is not None:       config.SAMPLE_RATIO = args.ratio
    if args.road_thresh is not None: config.ROAD_THRESHOLD = args.road_thresh
    if args.itsc_thresh is not None: config.ITSC_THRESHOLD = args.itsc_thresh
    if args.topo_thresh is not None: config.TOPO_THRESHOLD = args.topo_thresh
    if args.road_nms is not None:    config.ROAD_NMS_RADIUS = args.road_nms
    if args.itsc_nms is not None:    config.ITSC_NMS_RADIUS = args.itsc_nms
    if args.nbr_radius is not None:  config.NEIGHBOR_RADIUS = args.nbr_radius
    if args.max_nbr_q is not None:   config.MAX_NEIGHBOR_QUERIES = args.max_nbr_q

    # [=== 修改 2：确定输出目录与初始化日志 ===]
    if args.exp_id:
        output_dir = os.path.join('./save', args.exp_id)
        os.makedirs(output_dir, exist_ok=True)
        # 存一份当前实验实际使用的 config 副本，防止以后忘了参数
        import yaml
        with open(os.path.join(output_dir, 'run_config.yaml'), 'w') as f:
            # 兼容 config 是字典或命名空间对象的情况
            yaml.dump(config if isinstance(config, dict) else config.__dict__, f)
    else:
        output_dir_prefix = './save/infer_'
        if args.output_dir:
            output_dir = create_output_dir_and_save_config(output_dir_prefix, config, specified_dir=f'./save/{args.output_dir}')
        else:
            output_dir = create_output_dir_and_save_config(output_dir_prefix, config)

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "run.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"=== Starting Experiment: {args.exp_id if args.exp_id else 'Default'} ===")
    logger.info(f"Task Mode: {args.task}")
    logger.info(f"Using Device: {args.device}")

    # Builds eval model    
    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")
    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    net = SAMRoad(config)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    logger.info(f'##### Loading Trained CKPT: {args.checkpoint} #####')
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    net.to(device)

    # 数据集路径与 Index 加载
    if config.DATASET == 'cityscale':
        _, _, test_img_indices = cityscale_data_partition()
        rgb_pattern = './cityscale/20cities/region_{}_sat.png'
        gt_graph_pattern = 'cityscale/20cities/region_{}_graph_gt.pickle'
    elif config.DATASET == 'spacenet':
        _, _, test_img_indices = spacenet_data_partition()
        rgb_pattern = './spacenet/RGB_1.0_meter/{}__rgb.png'
        gt_graph_pattern = './spacenet/RGB_1.0_meter/{}__gt_graph.p'
    elif config.DATASET == 'didi':
        with open('./xian/2019_400/data_split.json','r') as jf:
            import json
            data_list = json.load(jf)
            train_img_indices = data_list['train']
            val_img_indices = data_list['validation']
            test_img_indices = data_list['test']

        rgb_pattern = './xian/2019_400/xian_2019_400/region_{}_sat.png'
        gt_graph_pattern = './xian/2019_400/xian_2019_400/region_{}_graph_gt.pickle'
    else:
        raise ValueError(f"Unknown dataset type: {config.DATASET}")
    
    total_inference_seconds = 0.0

    # 动态获取 ratio 的字符串形式 (例如 0.5 或 0.25)，用于拼接路径
    ratio_val = getattr(config, 'SAMPLE_RATIO', 0.5) 
    # {ratio_val:g} 可以去掉末尾多余的0，比如 0.50 变成 0.5
    ratio_folder = f"sample_{ratio_val:g}" 

    pbar = tqdm(test_img_indices, desc="Inference Progress")
    for img_id in pbar:
        # 动态更新进度条前面的文字，这样不会像普通 print 那样刷屏
        pbar.set_description(f'Processing {img_id}')
        # [H, W, C] RGB
        img = read_rgb_img(rgb_pattern.format(img_id))

        # # [=== 修改 3：根据 task 决定先验，动态拼接残缺图路径 ===]
        if args.task == "extraction":
            # 任务：盲提取 -> 使用全黑的先验
            prior = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
            
        elif args.task == "update":
            # 任务：路网更新 -> 读取生成的残缺先验 (partial prior)
            if config.DATASET == 'cityscale':
                partial_prior_path = f'./cityscale/{ratio_folder}/region_{img_id}_refine_gt_graph_partial.png'
            elif config.DATASET == 'spacenet':
                partial_prior_path = f'./spacenet/{ratio_folder}/{img_id}__gt_graph_partial.png'
            elif config.DATASET == 'didi':
                partial_prior_path = f'./xian/2019_400/{ratio_folder}/region_{img_id}_refine_gt_graph_partial.png'
            
            if os.path.exists(partial_prior_path):
                partial_img = cv2.imread(partial_prior_path, cv2.IMREAD_GRAYSCALE)
                prior = (partial_img.astype(np.float32) / 255.0)
                prior = np.expand_dims(prior, axis=-1)
            else:
                logger.warning(f"[WARN] Partial prior not found: {partial_prior_path}. Falling back to empty prior.")
                prior = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
                
        elif args.task == "full":
            if config.DATASET == 'cityscale':
                partial_prior_path = f'./cityscale/20cities/region_{img_id}_gt.png'
            elif config.DATASET == 'spacenet':
                partial_prior_path = f'./spacenet/RGB_1.0_meter/{img_id}__gt.png'
            elif config.DATASET == 'didi':
                partial_prior_path = f'./xian/2019_400/xian_2019_400/region_{img_id}_gt.png'
            
            if os.path.exists(partial_prior_path):
                partial_img = cv2.imread(partial_prior_path, cv2.IMREAD_GRAYSCALE)
                prior = (partial_img.astype(np.float32) / 255.0)
                prior = np.expand_dims(prior, axis=-1)
            else:
                logger.warning(f"[WARN] Full prior not found: {partial_prior_path}. Falling back to empty prior.")
                prior = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
        else:
            raise ValueError(f"Unknown task: {args.task}")
        # ==========================================

        # 拼成 4 通道 [H, W, 4]
        img_4c = np.concatenate([img, prior], axis=-1)

        start_seconds = time.time()
        # coords in (r, c)
        pred_nodes, pred_edges, itsc_mask, road_mask = infer_one_img(net, img_4c, config)
        end_seconds = time.time()
        total_inference_seconds += (end_seconds - start_seconds)

        gt_graph_path = gt_graph_pattern.format(img_id)
        gt_graph = pickle.load(open(gt_graph_path, "rb"))
        gt_nodes, gt_edges = graph_utils.convert_from_sat2graph_format(gt_graph)
        if len(gt_nodes) == 0:
            gt_nodes = np.zeros([0, 2], dtype=np.float32)

        if config.DATASET == 'spacenet':
            # convert ??? -> xy -> rc
            gt_nodes = np.stack([gt_nodes[:, 1], 400 - gt_nodes[:, 0]], axis=1)
            gt_nodes = gt_nodes[:, ::-1]

        # RGB already
        # 保留原版 3 通道 img 用于 OpenCV 绘图
        # 如果用 4 通道的 img_4c，OpenCV 的 cv2.imwrite 和 triage 画图会报错
        viz_img = np.copy(img)
        img_size = viz_img.shape[0]

        # visualizes fused masks
        mask_save_dir = os.path.join(output_dir, 'mask')
        if not os.path.exists(mask_save_dir):
            os.makedirs(mask_save_dir)
        cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_road.png'), road_mask)
        cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_itsc.png'), itsc_mask)

        # # Visualizes the diff between rasterized pred/gt graphs.
        # rast_pred = triage.rasterize_graph(pred_nodes / img_size, pred_edges, img_size, dilation_radius=1)
        # rast_pred_dilate = triage.rasterize_graph(pred_nodes / img_size, pred_edges, img_size, dilation_radius=5)
        # rast_gt = triage.rasterize_graph(gt_nodes / img_size, gt_edges, img_size, dilation_radius=1)
        # rast_gt_dilate = triage.rasterize_graph(gt_nodes / img_size, gt_edges, img_size, dilation_radius=5)

        # fp_pred = (np.less_equal(rast_gt_dilate, 0) * np.greater(rast_pred, 0)).astype(np.uint8)
        # missed_gt = (np.less_equal(rast_pred_dilate, 0) * np.greater(rast_gt, 0)).astype(np.uint8)

        # diff_img = np.array(viz_img)
        # # FP in blue, missed in red (BGR for opencv)
        # diff_img = diff_img * np.less_equal(fp_pred, 0) + fp_pred * np.array([255, 0, 0], dtype=np.uint8)
        # diff_img = diff_img * np.less_equal(missed_gt, 0) + missed_gt * np.array([0, 0, 255], dtype=np.uint8)

        # diff_save_dir = os.path.join(output_dir, 'diff')
        # if not os.path.exists(diff_save_dir):
        #     os.makedirs(diff_save_dir)
        # cv2.imwrite(os.path.join(diff_save_dir, f'{img_id}.png'), diff_img)

        # Visualizes merged large map
        viz_save_dir = os.path.join(output_dir, 'viz')
        if not os.path.exists(viz_save_dir):
            os.makedirs(viz_save_dir)
        viz_img = triage.visualize_image_and_graph(viz_img, pred_nodes / img_size, pred_edges, viz_img.shape[0])
        cv2.imwrite(os.path.join(viz_save_dir, f'{img_id}.png'), viz_img)

        # Saves the large map
        if config.DATASET == 'spacenet':
            # r, c -> ???
            pred_nodes = np.stack([400 - pred_nodes[:, 0], pred_nodes[:, 1]], axis=1)
        large_map_sat2graph_format = graph_utils.convert_to_sat2graph_format(pred_nodes, pred_edges)
        graph_save_dir = os.path.join(output_dir, 'graph')
        if not os.path.exists(graph_save_dir):
            os.makedirs(graph_save_dir)
        graph_save_path = os.path.join(graph_save_dir, f'{img_id}.p')
        with open(graph_save_path, 'wb') as file:
            pickle.dump(large_map_sat2graph_format, file)
        
        # print(f'Done for {img_id}.')
    
    # log inference time
    time_txt = f'Inference completed for {args.config} in {total_inference_seconds:.2f} seconds.'
    logger.info(time_txt)
    with open(os.path.join(output_dir, 'inference_time.txt'), 'w') as f:
        f.write(time_txt)
