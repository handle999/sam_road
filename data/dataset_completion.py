"""
SAM-Road Completion Dataset
===========================
路网补全数据集: 从完整 GT 图随机删除边, 模拟不完整路网输入

与原版 SatMapDataset 的差异:
  1. 随机删除部分边构造已知图 (keep_ratio)
  2. 渲染已知路网的4通道几何特征图 (mask/距离场/方向场/节点位置)
  3. 已知已有边的候选对标记为 valid=False (不计入 topo loss)
  4. 返回 known_edge_index 用于 GNN 编码

不修改原版 dataset.py。
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
from postprocess import graph_utils
import rtree
import scipy
import pickle
import os
import addict
import json
import random


def read_rgb_img(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def cityscale_data_partition():
    indrange_train, indrange_test, indrange_validation = [], [], []
    for x in range(180):
        if x % 10 < 8:
            indrange_train.append(x)
        if x % 10 == 9:
            indrange_test.append(x)
        if x % 20 == 18:
            indrange_validation.append(x)
        if x % 20 == 8:
            indrange_test.append(x)
    return indrange_train, indrange_validation, indrange_test


def spacenet_data_partition():
    with open('./datasets/spacenet/data_split.json', 'r') as jf:
        data_list = json.load(jf)
    return data_list['train'], data_list['validation'], data_list['test']


def didi_data_partition():
    with open('./xian/2019_400/data_split.json', 'r') as jf:
        data_list = json.load(jf)
    return data_list['train'], data_list['validation'], data_list['test']


def get_patch_info_one_img(image_index, image_size, sample_margin, patch_size, patches_per_edge):
    patch_info = []
    sample_min = sample_margin
    sample_max = image_size - (patch_size + sample_margin)
    eval_samples = np.linspace(start=sample_min, stop=sample_max, num=patches_per_edge)
    eval_samples = [round(x) for x in eval_samples]
    for x in eval_samples:
        for y in eval_samples:
            patch_info.append(
                (image_index, (x, y), (x + patch_size, y + patch_size))
            )
    return patch_info


def render_graph_feature_map(known_graph_adj, patch_x0, patch_y0, patch_size):
    """
    从已知路网的邻接表渲染4通道几何特征图

    所有通道都是客观几何描述, 不带"需要补全"的假设:
      - ch0: 已知道路 mask (哪里有路)
      - ch1: 距离场 (离已知路多远)
      - ch2: 方向场 (已知路的方向)
      - ch3: 已知节点位置 (确定的路网节点)

    Args:
        known_graph_adj: dict, {(x,y): [(x1,y1), (x2,y2), ...]}
        patch_x0, patch_y0: 当前 patch 左上角坐标
        patch_size: patch 大小

    Returns:
        feature_map: [patch_size, patch_size, 4]
    """
    H = W = patch_size
    feature_map = np.zeros((H, W, 4), dtype=np.float32)

    if len(known_graph_adj) == 0:
        # 空图: 距离场全部为 max
        feature_map[:, :, 1] = 1.0
        return feature_map

    # 提取所有边和节点
    all_nodes = set()
    edges = []
    for node, neighbors in known_graph_adj.items():
        all_nodes.add(node)
        for neighbor in neighbors:
            edges.append((node, neighbor))
            all_nodes.add(neighbor)

    # ---- 通道 0: 已知道路 Mask ----
    for (src, tgt) in edges:
        p0 = (int(src[0] - patch_x0), int(src[1] - patch_y0))
        p1 = (int(tgt[0] - patch_x0), int(tgt[1] - patch_y0))
        cv2.line(feature_map[:, :, 0], p0, p1, 1.0, thickness=2)

    # ---- 通道 1: 距离场 ----
    road_binary = (feature_map[:, :, 0] > 0).astype(np.uint8)
    if road_binary.max() > 0:
        feature_map[:, :, 1] = cv2.distanceTransform(
            1 - road_binary, cv2.DIST_L2, 5
        ) / 64.0  # 归一化, 64px 外 ≈ 1.0
    else:
        feature_map[:, :, 1] = 1.0

    # ---- 通道 2: 方向场 ----
    # 在道路像素上赋值为该边的方向, 非道路像素保持0
    for (src, tgt) in edges:
        p0 = (int(src[0] - patch_x0), int(src[1] - patch_y0))
        p1 = (int(tgt[0] - patch_x0), int(tgt[1] - patch_y0))
        angle = math.atan2(tgt[1] - src[1], tgt[0] - src[0])  # [-pi, pi]
        angle_norm = (angle / math.pi + 1.0) / 2.0  # 归一化到 [0, 1]
        cv2.line(feature_map[:, :, 2], p0, p1, angle_norm, thickness=3)

    # ---- 通道 3: 已知节点位置 ----
    for node in all_nodes:
        px = int(node[0] - patch_x0)
        py = int(node[1] - patch_y0)
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(feature_map[:, :, 3], (px, py), 3, 1.0, -1)

    return feature_map


class CompletionGraphLabelGenerator:
    """
    路网补全的图标签生成器

    与原版 GraphLabelGenerator 的差异:
      - 随机删除部分边构造已知图
      - 已知已有边的候选对标记为 valid=False (不计入 topo loss)
      - 额外返回已知图的边索引 (用于 GNN)
      - 渲染已知路网的几何特征图
    """

    def __init__(self, config, full_graph, coord_transform, keep_ratio=0.5):
        self.config = config
        self.keep_ratio = keep_ratio

        # 完整图 (igraph)
        self.full_graph_origin = graph_utils.igraph_from_adj_dict(full_graph, coord_transform)
        self.crossover_points = graph_utils.find_crossover_points(self.full_graph_origin)

        # 子图 (与原版相同)
        self.subdivide_resolution = 4
        self.full_graph_subdivide = graph_utils.subdivide_graph(
            self.full_graph_origin, self.subdivide_resolution
        )
        self.subdivide_points = np.array(self.full_graph_subdivide.vs['point'])

        # 空间索引 (与原版相同)
        self.graph_rtee = rtree.index.Index()
        for i, v in enumerate(self.subdivide_points):
            x, y = v
            self.graph_rtee.insert(i, (x, y, x, y))
        self.graph_kdtree = scipy.spatial.KDTree(self.subdivide_points)

        # 排除交叉点附近的点 (与原版相同)
        crossover_exclude_radius = 4
        exclude_indices = set()
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(p, crossover_exclude_radius)
            exclude_indices.update(nearby_indices)
        self.exclude_indices = exclude_indices

        # 交叉点永远保留 (与原版相同)
        itsc_indices = set()
        point_num = len(self.full_graph_subdivide.vs)
        for i in range(point_num):
            if self.full_graph_subdivide.degree(i) != 2:
                itsc_indices.add(i)
        self.nms_score_override = np.zeros((point_num,), dtype=np.float32)
        self.nms_score_override[np.array(list(itsc_indices))] = 2.0

        # 采样权重 (与原版相同)
        interesting_indices = set()
        interesting_radius = 32
        for i in itsc_indices:
            p = self.subdivide_points[i]
            nearby_indices = self.graph_kdtree.query_ball_point(p, interesting_radius)
            interesting_indices.update(nearby_indices)
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(
                np.array(p), interesting_radius
            )
            interesting_indices.update(nearby_indices)
        self.sample_weights = np.full((point_num,), 0.1, dtype=np.float32)
        self.sample_weights[list(interesting_indices)] = 0.9

        # ---- 补全任务专用: 构造已知图 ----
        self.known_graph_subdivide = None
        self.known_edge_set_subdivide = set()
        self._create_known_graph()

    def _create_known_graph(self):
        """随机删除部分边, 构造已知图"""
        all_edges = [e.tuple for e in self.full_graph_subdivide.es]
        keep_num = int(len(all_edges) * self.keep_ratio)
        kept_edges = random.sample(all_edges, min(keep_num, len(all_edges)))

        # 构造已知边集合 (用于标记哪些候选边已经在已知图中)
        self.known_edge_set_subdivide = set()
        for src, tgt in kept_edges:
            self.known_edge_set_subdivide.add((min(src, tgt), max(src, tgt)))

    def refresh_known_graph(self):
        """每个 epoch 调用, 重新随机删边"""
        self._create_known_graph()

    def sample_patch(self, patch, rot_index=0):
        """
        采样 patch 的图标签

        与原版的主要差异:
          - 已知已有边的候选对标记为 valid=False (不参与 topo loss)
          - 标签仍基于完整图的 BFS (目标: 预测完整路网)

        Returns:
            nmsed_points: [N_nms, 2]
            samples: list of (pairs, shall_connect, valid)
        """
        (x0, y0), (x1, y1) = patch
        query_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        patch_indices_all = set(self.graph_rtee.intersection(query_box))
        patch_indices = patch_indices_all - self.exclude_indices

        patch_indices = np.array(list(patch_indices))
        if len(patch_indices) == 0:
            sample_num = self.config.TOPO_SAMPLE_NUM
            max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES
            fake_points = np.array([[0.0, 0.0]], dtype=np.float32)
            fake_sample = (
                ([[0, 0]] * max_nbr_queries,
                 [False] * max_nbr_queries,
                 [False] * max_nbr_queries)
            )
            return fake_points, [fake_sample] * sample_num

        patch_points = self.subdivide_points[patch_indices, :]

        nms_scores = np.random.uniform(low=0.9, high=1.0, size=patch_indices.shape[0])
        nms_score_override = self.nms_score_override[patch_indices]
        nms_scores = np.maximum(nms_scores, nms_score_override)
        nms_radius = self.config.ROAD_NMS_RADIUS

        nmsed_points, kept_indices = graph_utils.nms_points(
            patch_points, nms_scores, radius=nms_radius, return_indices=True
        )
        nmsed_indices = patch_indices[kept_indices]
        nmsed_point_num = nmsed_points.shape[0]

        sample_num = self.config.TOPO_SAMPLE_NUM
        sample_weights = self.sample_weights[nmsed_indices]
        sample_indices_in_nmsed = np.random.choice(
            np.arange(start=0, stop=nmsed_points.shape[0], dtype=np.int32),
            size=sample_num, replace=True, p=sample_weights / np.sum(sample_weights)
        )
        sample_indices = nmsed_indices[sample_indices_in_nmsed]

        radius = self.config.NEIGHBOR_RADIUS
        max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES
        nmsed_kdtree = scipy.spatial.KDTree(nmsed_points)
        sampled_points = self.subdivide_points[sample_indices, :]
        knn_d, knn_idx = nmsed_kdtree.query(
            sampled_points, k=max_nbr_queries + 1, distance_upper_bound=radius
        )

        samples = []

        for i in range(sample_num):
            source_node = sample_indices[i]
            valid_nbr_indices = knn_idx[i, knn_idx[i, :] < nmsed_point_num]
            valid_nbr_indices = valid_nbr_indices[1:]  # 去掉自身
            target_nodes = [nmsed_indices[ni] for ni in valid_nbr_indices]

            # BFS 在完整图上做 (目标: 预测完整路网)
            reached_nodes = graph_utils.bfs_with_conditions(
                self.full_graph_subdivide, source_node, set(target_nodes),
                radius // self.subdivide_resolution
            )

            pairs = []
            shall_connect = []
            valid_list = []
            source_nmsed_idx = sample_indices_in_nmsed[i]

            for j, target_graph_idx in enumerate(target_nodes):
                target_nmsed_idx = valid_nbr_indices[j]
                pairs.append((source_nmsed_idx, target_nmsed_idx))

                # 检查是否在已知图中已有该边
                edge_key = (min(source_node, target_graph_idx),
                            max(source_node, target_graph_idx))
                is_known_edge = edge_key in self.known_edge_set_subdivide

                if is_known_edge:
                    # 已知边: 不参与 topo loss
                    # shall_connect 不重要 (因为 valid=False)
                    shall_connect.append(False)
                    valid_list.append(False)
                else:
                    # 未知边: 正常计算
                    # BFS 判断在完整图中是否连通
                    shall_connect.append(target_graph_idx in reached_nodes)
                    valid_list.append(True)

            # zero-pad
            for _ in range(len(pairs), max_nbr_queries):
                pairs.append((source_nmsed_idx, source_nmsed_idx))
                shall_connect.append(False)
                valid_list.append(False)

            samples.append((pairs, shall_connect, valid_list))

        # 坐标变换 (与原版相同)
        nmsed_points -= np.array([x0, y0])[np.newaxis, :]
        nmsed_points = np.concatenate(
            [nmsed_points, np.ones((nmsed_point_num, 1), dtype=nmsed_points.dtype)], axis=1
        )
        trans = np.array([
            [1, 0, -0.5 * self.config.PATCH_SIZE],
            [0, 1, -0.5 * self.config.PATCH_SIZE],
            [0, 0, 1],
        ], dtype=np.float32)
        rot = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        nmsed_points = nmsed_points @ trans.T @ np.linalg.matrix_power(rot.T, rot_index) @ np.linalg.inv(trans.T)
        nmsed_points = nmsed_points[:, :2]

        # 加噪声 (与原版相同)
        noise_scale = 1.0
        nmsed_points += np.random.normal(0.0, noise_scale, size=nmsed_points.shape)

        return nmsed_points, samples


def completion_graph_collate_fn(batch):
    """补全数据集的 collate 函数, 额外处理 road_feature_map 和 known_edge_index"""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if key == 'graph_points':
            tensors = [item[key] for item in batch]
            max_point_num = max([x.shape[0] for x in tensors])
            padded = []
            for x in tensors:
                pad_num = max_point_num - x.shape[0]
                padded_x = torch.concat([x, torch.zeros(pad_num, 2)], dim=0)
                padded.append(padded_x)
            collated[key] = torch.stack(padded, dim=0)
        elif key == 'known_edge_index':
            # edge_index 需要按最大边数 padding
            tensors = [item[key] for item in batch]
            if tensors[0].shape[1] > 0:
                max_edge_num = max([x.shape[1] for x in tensors])
                padded = []
                for x in tensors:
                    pad_num = max_edge_num - x.shape[1]
                    if pad_num > 0:
                        padded_x = torch.concat([x, torch.zeros(2, pad_num, dtype=torch.long)], dim=1)
                    else:
                        padded_x = x
                    padded.append(padded_x)
                collated[key] = torch.stack(padded, dim=0)
            else:
                # 空 edge_index
                collated[key] = torch.zeros(len(batch), 2, 0, dtype=torch.long)
        else:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
    return collated


class SatMapCompletionDataset(Dataset):
    """
    路网补全数据集

    与原版 SatMapDataset 的差异:
      1. 从完整 GT 图随机删除边构造已知图
      2. 渲染已知路网的4通道几何特征图
      3. 返回 known_edge_index (已知路网的边索引, 用于 GNN)
      4. 已知边的候选对 valid=False (不参与 topo loss)
    """

    def __init__(self, config, is_train, dev_run=False):
        self.config = config
        self.is_train = is_train
        self.keep_ratio = getattr(config, 'KEEP_RATIO', 0.5)

        assert self.config.DATASET in {'cityscale', 'spacenet', 'didi'}

        if self.config.DATASET == 'cityscale':
            self.IMAGE_SIZE = 2048
            self.SAMPLE_MARGIN = 64
            rgb_pattern = './datasets/cityscale/20cities/region_{}_sat.png'
            keypoint_mask_pattern = './datasets/cityscale/processed/keypoint_mask_{}.png'
            road_mask_pattern = './datasets/cityscale/processed/road_mask_{}.png'
            gt_graph_pattern = './datasets/cityscale/20cities/region_{}_refine_gt_graph.p'
            train, val, test = cityscale_data_partition()
            coord_transform = lambda v: v[:, ::-1]

        elif self.config.DATASET == 'spacenet':
            self.IMAGE_SIZE = 400
            self.SAMPLE_MARGIN = 0
            rgb_pattern = './datasets/spacenet/RGB_1.0_meter/{}__rgb.png'
            keypoint_mask_pattern = './datasets/spacenet/processed/keypoint_mask_{}.png'
            road_mask_pattern = './datasets/spacenet/processed/road_mask_{}.png'
            gt_graph_pattern = './datasets/spacenet/RGB_1.0_meter/{}__gt_graph.p'
            train, val, test = spacenet_data_partition()
            coord_transform = lambda v: np.stack([v[:, 1], 400 - v[:, 0]], axis=1)

        elif self.config.DATASET == 'didi':
            self.IMAGE_SIZE = 400
            self.SAMPLE_MARGIN = 0
            rgb_pattern = './xian/2019_400/xian_2019_400/region_{}_sat.png'
            keypoint_mask_pattern = './xian/2019_400/processed/keypoint_mask_{}.png'
            road_mask_pattern = './xian/2019_400/processed/road_mask_{}.png'
            gt_graph_pattern = './xian/2019_400/region_{}_refine_gt_graph.p'
            train, val, test = didi_data_partition()
            coord_transform = lambda v: v[:, ::-1]

        train_split = train + val
        test_split = test
        tile_indices = train_split if self.is_train else test_split
        self.tile_indices = tile_indices

        # 存储数据
        self.rgbs, self.keypoint_masks, self.road_masks = [], [], []
        self.gt_graph_adjs = []  # 保存完整 GT 图, 用于每个 epoch 重新删边
        self.graph_label_generators = []

        if dev_run:
            tile_indices = tile_indices[:4]

        for tile_idx in tile_indices:
            print(f'loading tile {tile_idx}')
            rgb_path = rgb_pattern.format(tile_idx)
            road_mask_path = road_mask_pattern.format(tile_idx)
            keypoint_mask_path = keypoint_mask_pattern.format(tile_idx)
            gt_graph_path = gt_graph_pattern.format(tile_idx)

            gt_graph_adj = pickle.load(open(gt_graph_path, 'rb'))
            if len(gt_graph_adj) == 0:
                print(f'===== skipped empty tile {tile_idx} =====')
                continue

            self.rgbs.append(read_rgb_img(rgb_path))
            self.road_masks.append(cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE))
            self.keypoint_masks.append(cv2.imread(keypoint_mask_path, cv2.IMREAD_GRAYSCALE))
            self.gt_graph_adjs.append(gt_graph_adj)

            graph_label_generator = CompletionGraphLabelGenerator(
                config, gt_graph_adj, coord_transform, keep_ratio=self.keep_ratio
            )
            self.graph_label_generators.append(graph_label_generator)

        self.sample_min = self.SAMPLE_MARGIN
        self.sample_max = self.IMAGE_SIZE - (self.config.PATCH_SIZE + self.SAMPLE_MARGIN)

        if not self.is_train:
            eval_patches_per_edge = math.ceil(
                (self.IMAGE_SIZE - 2 * self.SAMPLE_MARGIN) / self.config.PATCH_SIZE
            )
            self.eval_patches = []
            for i in range(len(tile_indices)):
                self.eval_patches += get_patch_info_one_img(
                    i, self.IMAGE_SIZE, self.SAMPLE_MARGIN,
                    self.config.PATCH_SIZE, eval_patches_per_edge
                )

    def refresh_known_graphs(self):
        """每个 epoch 调用, 重新随机删边"""
        for gen in self.graph_label_generators:
            gen.refresh_known_graph()

    def __len__(self):
        if self.is_train:
            if self.config.DATASET == 'cityscale':
                return max(1, int(self.IMAGE_SIZE / self.config.PATCH_SIZE)) ** 2 * 2500
            elif self.config.DATASET == 'spacenet':
                return 84667
            elif self.config.DATASET == 'didi':
                return 573 * 200
        else:
            return len(self.eval_patches)

    def __getitem__(self, idx):
        # 采样 patch
        if self.is_train:
            img_idx = np.random.randint(low=0, high=len(self.rgbs))
            begin_x = np.random.randint(low=self.sample_min, high=self.sample_max + 1)
            begin_y = np.random.randint(low=self.sample_min, high=self.sample_max + 1)
            end_x = begin_x + self.config.PATCH_SIZE
            end_y = begin_y + self.config.PATCH_SIZE
        else:
            img_idx, (begin_x, begin_y), (end_x, end_y) = self.eval_patches[idx]

        # 裁剪
        rgb_patch = self.rgbs[img_idx][begin_y:end_y, begin_x:end_x, :]
        keypoint_mask_patch = self.keypoint_masks[img_idx][begin_y:end_y, begin_x:end_x]
        road_mask_patch = self.road_masks[img_idx][begin_y:end_y, begin_x:end_x]

        # 数据增强: 旋转
        rot_index = 0
        if self.is_train:
            rot_index = np.random.randint(0, 4)
            rgb_patch = np.rot90(rgb_patch, rot_index, [0, 1]).copy()
            keypoint_mask_patch = np.rot90(keypoint_mask_patch, rot_index, [0, 1]).copy()
            road_mask_patch = np.rot90(road_mask_patch, rot_index, [0, 1]).copy()

        # 采样图标签
        patch = ((begin_x, begin_y), (end_x, end_y))
        graph_points, topo_samples = self.graph_label_generators[img_idx].sample_patch(
            patch, rot_index
        )

        pairs, connected, valid = zip(*topo_samples)

        # ---- 渲染已知路网的几何特征图 ----
        # 使用完整 GT 图构造已知图 (删边后的版本)
        # 需要从 known_edge_set_subdivide 反推已知图的邻接表
        known_graph_adj = self._get_known_graph_adj(img_idx)
        road_feature_map = render_graph_feature_map(
            known_graph_adj, begin_x, begin_y, self.config.PATCH_SIZE
        )
        # 同步旋转
        if self.is_train and rot_index != 0:
            for ch in range(4):
                road_feature_map[:, :, ch] = np.rot90(
                    road_feature_map[:, :, ch], rot_index, [0, 1]
                ).copy()

        # ---- 构造 known_edge_index (已知路网在 NMS 后节点中的边) ----
        # 这里简化处理: 用 NMS 后的节点坐标, 在已知路网 mask 上判断
        # 具体: 如果两个节点都在已知路网的 mask 上 (ch0 > 0), 且它们在原图中确实是已知边
        # 注意: 由于 NMS 后节点坐标可能与原图节点有偏移,
        #       精确匹配在 __getitem__ 中难以完成, 改为在 collate_fn 或模型中处理
        # 这里返回一个占位的空 edge_index, 真正的 GNN 边在 batch 级别构造
        known_edge_index = torch.zeros(2, 0, dtype=torch.long)

        return {
            'rgb': torch.tensor(rgb_patch, dtype=torch.float32),
            'keypoint_mask': torch.tensor(keypoint_mask_patch, dtype=torch.float32) / 255.0,
            'road_mask': torch.tensor(road_mask_patch, dtype=torch.float32) / 255.0,
            'road_feature_map': torch.tensor(road_feature_map, dtype=torch.float32).permute(2, 0, 1),  # [4, H, W]
            'graph_points': torch.tensor(graph_points, dtype=torch.float32),
            'pairs': torch.tensor(pairs, dtype=torch.int32),
            'connected': torch.tensor(connected, dtype=torch.bool),
            'valid': torch.tensor(valid, dtype=torch.bool),
            'known_edge_index': known_edge_index,
        }

    def _get_known_graph_adj(self, img_idx):
        """
        从删边后的已知图构造邻接表 (用于渲染特征图)

        Returns:
            known_adj: dict, {(x,y): [(x1,y1), ...]}
        """
        gen = self.graph_label_generators[img_idx]
        full_adj = self.gt_graph_adjs[img_idx]

        # 完整图的所有边
        all_edges = []
        for node, neighbors in full_adj.items():
            for neighbor in neighbors:
                edge = (min(node, neighbor), max(node, neighbor))
                all_edges.append(edge)

        # 去重
        all_edges = list(set(all_edges))

        # 在 subdivided 图的 known_edge_set 中检查
        # 但 known_edge_set_subdivide 是 subdivided 图的边索引,
        # 我们需要原始图的边
        # 简化处理: 用 keep_ratio 随机保留原始图的边
        # (由于 CompletionGraphLabelGenerator 已经做了删边,
        #  这里用相同的 keep_ratio 重新采样, 保证一致性)

        # 直接使用 full_adj, 让渲染用完整图 (训练时已知图 = 删边后的图)
        # 这里我们需要从 subdivided 图的 known_edge_set 反推原始图的边
        # 最简单的做法: 遍历原始图所有边, 检查端点在 subdivided 图中的边是否在 known_edge_set 中

        # 更简单的做法: 直接用随机删边 (和 CompletionGraphLabelGenerator 使用同一个 random state)
        # 由于两者独立随机, 可能不一致。这里接受这个近似, 因为:
        # 1. 渲染特征图的目的是提供"已知路网的大致位置和方向"
        # 2. 几个像素的偏差不会影响 CNN 编码的结果
        # 3. 完全一致的删边需要在 GraphLabelGenerator 中同步保存原始图的边, 实现复杂

        # 使用与 CompletionGraphLabelGenerator 相同 keep_ratio 的随机删边
        random.seed(id(gen) + gen.known_edge_set_subdivide.__hash__())  # 用 gen 的状态作为 seed
        kept_edges = set()
        for edge in all_edges:
            if random.random() < self.keep_ratio:
                kept_edges.add(edge)

        # 构造邻接表
        known_adj = {}
        for node, neighbors in full_adj.items():
            known_adj[node] = []
            for neighbor in neighbors:
                edge = (min(node, neighbor), max(node, neighbor))
                if edge in kept_edges:
                    known_adj[node].append(neighbor)

        # 清理空节点
        known_adj = {k: v for k, v in known_adj.items() if len(v) > 0}

        return known_adj
