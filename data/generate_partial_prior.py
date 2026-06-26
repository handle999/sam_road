import os
import pickle
import random
import argparse
import numpy as np
import cv2
import glob
from tqdm import tqdm
import copy
import networkx as nx

class PartialGraphSampler:
    def __init__(self, dataset_type, keep_ratio, image_size=None, line_thickness=3,
                 strategy='component'):
        """
        初始化路网采样器
        :param dataset_type: 数据集类型 ('cityscale', 'spacenet', 'didi')
        :param keep_ratio: 保留边的比例 (0.0 ~ 1.0)
        :param image_size: 图像尺寸 (不指定则使用默认值)
        :param line_thickness: 渲染 PNG 时的线宽
        :param strategy: 采样策略
            - 'edge_random': 按边随机删 (旧策略, 破碎)
            - 'component': 按连通块保 (小块按概率整块留/删 + 大块BFS生长, 不破碎, 推荐)
            - 'bfs': BFS生长 (单连通, 过理想)
        """
        self.dataset_type = dataset_type
        self.keep_ratio = keep_ratio
        self.line_thickness = line_thickness
        self.strategy = strategy

        # 1. 设定图像默认尺寸
        if image_size is not None:
            self.image_size = image_size
        else:
            if self.dataset_type == 'cityscale':
                self.image_size = 2048
            elif self.dataset_type in ['spacenet', 'didi']:
                self.image_size = 400
            else:
                raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        # 2. 设定坐标变换逻辑 (将 Pickle 中的坐标统一转为 OpenCV 绘图需要的 (X, Y))
        # 根据 dataset.py 中的 coord_transform 逻辑逆向推导
        # 注意: 此变换仅用于 render_to_png 画图, pickle 保存的是原始坐标
        # (坐标转换由 load_known_graph 入口统一负责, 这里不存变换后坐标)
        if self.dataset_type in ['cityscale', 'didi']:
            # Pickle 中是 (Row, Col) -> 转换为 (X, Y) 即 (Col, Row)
            self.transform_node = lambda v0, v1: (int(v1), int(v0))
        elif self.dataset_type == 'spacenet':
            # Pickle 中是 (raw_y, raw_x) -> 转换为 X=raw_x, Y=400-raw_y
            self.transform_node = lambda v0, v1: (int(v1), int(self.image_size - v0))

    def sample_graph(self, adj_dict, file_seed=None):
        """
        从完整邻接字典采样 partial 路网 (按 self.strategy 策略)
        :param adj_dict: 原始邻接字典 {node: [neighbor1, ...]}
        :param file_seed: 每文件独立种子 (基于全局seed+文件名), 保证可复现且与处理顺序无关.
                          None 时退回全局 random (旧行为, 不推荐).
        :return: 采样后的新邻接字典 (保持原始 pickle 坐标, 不做坐标变换)
        """
        # 每文件用独立 Random 实例, 隔离随机流 (修复: 同seed两次跑结果不一致)
        rng = random.Random(file_seed) if file_seed is not None else random
        if self.strategy == 'edge_random':
            return self._sample_edge_random(adj_dict, rng)
        elif self.strategy == 'component':
            return self._sample_component(adj_dict, rng)
        elif self.strategy == 'bfs':
            return self._sample_bfs(adj_dict, rng)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _adj_to_graph(self, adj_dict):
        """邻接字典 -> networkx 无向图"""
        G = nx.Graph()
        for u, neighbors in adj_dict.items():
            for v in neighbors:
                G.add_edge(u, v)
        return G

    def _graph_to_adj(self, G):
        """networkx 图 -> 邻接字典 (双向)"""
        adj = {}
        for u, v in G.edges():
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        return adj

    def _sample_edge_random(self, adj_dict, rng):
        """旧策略: 按边随机删 (破碎)"""
        edges = set()
        for u, neighbors in adj_dict.items():
            for v in neighbors:
                edges.add(tuple(sorted((u, v))))
        edges = list(edges)
        keep_num = int(len(edges) * self.keep_ratio)
        sampled = rng.sample(edges, min(keep_num, len(edges)))
        new_adj = {}
        for u, v in sampled:
            new_adj.setdefault(u, []).append(v)
            new_adj.setdefault(v, []).append(u)
        return new_adj

    def _sample_component(self, adj_dict, rng):
        """按连通块保: 小块按keep_ratio概率整块留/删 + 大块BFS生长补足.
        形态: 几个完整路段(小块有抹去概率) + 大路段缺口 = 真实地图.
        块内始终连续不破碎, 块数会减少."""
        G = self._adj_to_graph(adj_dict)
        if G.number_of_edges() == 0:
            return {}
        target = int(G.number_of_edges() * self.keep_ratio)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        kept_edges = []
        threshold = max(1, target * 0.3)
        small = [c for c in comps if G.subgraph(c).number_of_edges() <= threshold]
        big = [c for c in comps if G.subgraph(c).number_of_edges() > threshold]
        # 小块按 keep_ratio 概率整块保留 (整块留或整块删, 不拆碎)
        for c in small:
            if rng.random() < self.keep_ratio:
                kept_edges.extend(G.subgraph(c).edges())
        # 大块 BFS 生长补足
        remaining = target - len(kept_edges)
        for c in big:
            if remaining <= 0:
                break
            sub = G.subgraph(c)
            need = min(remaining, int(sub.number_of_edges() * self.keep_ratio))
            nodes = list(c); rng.shuffle(nodes)
            visited = {nodes[0]}; queue = [nodes[0]]; tree = []
            while queue and len(tree) < need:
                node = queue.pop(0)
                for nb in sub.neighbors(node):
                    if nb not in visited:
                        visited.add(nb); tree.append((node, nb)); queue.append(nb)
                        if len(tree) >= need:
                            break
            kept_edges.extend(tree); remaining -= len(tree)
        new_adj = {}
        for u, v in kept_edges:
            new_adj.setdefault(u, []).append(v)
            new_adj.setdefault(v, []).append(u)
        return new_adj

    def _sample_bfs(self, adj_dict, rng):
        """BFS生长: 从最大块随机种子BFS生长到keep_ratio, 单连通."""
        G = self._adj_to_graph(adj_dict)
        if G.number_of_edges() == 0:
            return {}
        target = int(G.number_of_edges() * self.keep_ratio)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        kept_edges = []
        for c in comps:
            if len(kept_edges) >= target:
                break
            need = target - len(kept_edges)
            sub = G.subgraph(c)
            nodes = list(c); rng.shuffle(nodes)
            visited = {nodes[0]}; queue = [nodes[0]]; tree = []
            while queue and len(tree) < need:
                node = queue.pop(0)
                for nb in sub.neighbors(node):
                    if nb not in visited:
                        visited.add(nb); tree.append((node, nb)); queue.append(nb)
                        if len(tree) >= need:
                            break
            kept_edges.extend(tree)
        new_adj = {}
        for u, v in kept_edges:
            new_adj.setdefault(u, []).append(v)
            new_adj.setdefault(v, []).append(u)
        return new_adj

    def render_to_png(self, adj_dict):
        """
        将邻接字典渲染为单通道 2D 掩码图像
        :param adj_dict: 邻接字典
        :return: np.array (H, W) 掩码图像
        """
        img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # 为了避免重复画线，用 set 记录画过的边
        drawn_edges = set()
        for u, neighbors in adj_dict.items():
            for v in neighbors:
                edge_key = tuple(sorted((u, v)))
                if edge_key not in drawn_edges:
                    # 应用数据集特定的坐标转换
                    pt1 = self.transform_node(u[0], u[1])
                    pt2 = self.transform_node(v[0], v[1])
                    
                    cv2.line(img, pt1, pt2, color=255, thickness=self.line_thickness)
                    drawn_edges.add(edge_key)
        
        return img

    def process_file(self, input_p_path, out_p_path, out_png_path=None, file_seed=42):
        """
        处理单个文件：加载 -> 采样 -> 保存 p (-> 可选保存 png)
        :param out_png_path: 若提供则保存 PNG 可视化, None 则跳过
        :param file_seed: 每张图独立用此 seed (默认42), 保证可复现且与处理顺序无关.
                          每张图的采样只依赖 file_seed + 该图本身, 不受其他图/空图/顺序影响.
        """
        with open(input_p_path, 'rb') as f:
            original_adj_dict = pickle.load(f)

        if len(original_adj_dict) == 0:
            print(f"[Warn] Empty graph found: {input_p_path}")
            sampled_adj_dict = {}
        else:
            sampled_adj_dict = self.sample_graph(original_adj_dict, file_seed=file_seed)

        # 保存 Sampled Pickle (原始坐标, 不做变换 — 坐标转换由 load 入口负责)
        with open(out_p_path, 'wb') as f:
            pickle.dump(sampled_adj_dict, f)

        # 可选: 保存 Rendered PNG (用 transform_node 转坐标画图)
        if out_png_path is not None:
            img = self.render_to_png(sampled_adj_dict) if sampled_adj_dict else \
                  np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            cv2.imwrite(out_png_path, img)


def main():
    parser = argparse.ArgumentParser(description="Generate partial road network priors.")
    parser.add_argument("--dataset", type=str, required=True, choices=['spacenet', 'cityscale', 'didi'],
                        help="Type of dataset to handle coordinate systems correctly.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing original GT .p files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save sampled .p (and .png if --viz) files.")
    parser.add_argument("--keep_ratio", type=float, default=0.5,
                        help="Ratio of edges to keep (0.0 to 1.0). Default is 0.5 (50%).")
    parser.add_argument("--thickness", type=int, default=3,
                        help="Line thickness for rendered PNG. Default is 3.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (infer 需固定). Default 42.")
    parser.add_argument("--strategy", type=str, default='edge_random',
                        choices=['edge_random', 'component', 'bfs'],
                        help="Sampling strategy: edge_random(按边随机,默认,训练稳定) / component(按连通块保,不破碎但训练震荡) / bfs(单连通). Default edge_random.")
    parser.add_argument("--viz", action='store_true', default=True,
                        help="Also render PNG visualization (default on). Use --no-viz to disable.")
    parser.add_argument("--no-viz", dest='viz', action='store_false',
                        help="Disable PNG visualization (only save .p).")

    args = parser.parse_args()

    # 每张图独立用 args.seed 采样 (见 process_file file_seed), 不依赖全局随机流.
    # 这样同一 seed 两次运行结果完全一致, 且与文件处理顺序/空图跳过无关.
    # 全局 random 仍 seed 一下 (np 等其他用途), 但采样本身用 per-file Random 实例.
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    sampler = PartialGraphSampler(
        dataset_type=args.dataset,
        keep_ratio=args.keep_ratio,
        line_thickness=args.thickness,
        strategy=args.strategy
    )

    # 查找输入目录下所有的 .p 或 .pickle 文件
    search_pattern1 = os.path.join(args.input_dir, "*.p")
    search_pattern2 = os.path.join(args.input_dir, "*.pickle")
    file_list = glob.glob(search_pattern1) + glob.glob(search_pattern2)

    # 只采样训练/推理实际用的 GT 图, 排除其他 (评测 GT / dense 变体 / active 图 / 已生成的 partial):
    #   didi:      region_{c}_refine_gt_graph.p   (排除 graph_gt.pickle 评测GT, active_graph.pickle)
    #   spacenet:  {id}__gt_graph.p               (排除 _dense*, _dense_spacenet 等变体)
    #   cityscale: region_{c}_refine_gt_graph.p
    # 已有 _partial 后缀的也排除, 防止重复运行时把 partial 当输入再采样.
    def is_target_input(fname):
        if '_partial' in fname:
            return False
        if args.dataset == 'didi':
            return fname.endswith('refine_gt_graph.p')
        elif args.dataset == 'cityscale':
            return fname.endswith('refine_gt_graph.p')
        elif args.dataset == 'spacenet':
            # __gt_graph.p 但排除 __gt_graph_dense*.p
            return fname.endswith('__gt_graph.p')
        return False

    file_list = [f for f in file_list if is_target_input(os.path.basename(f))]

    if len(file_list) == 0:
        print(f"No pickle files found in {args.input_dir}")
        return

    print(f"Found {len(file_list)} files. Starting sampling (Keep Ratio: {args.keep_ratio}, seed: {args.seed})...")

    print(f"Strategy: {args.strategy}, keep_ratio: {args.keep_ratio}, seed: {args.seed} (per-file), viz: {args.viz}")
    print(f"Found {len(file_list)} files. Starting sampling...")

    for file_path in tqdm(file_list):
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]

        out_p_path = os.path.join(args.output_dir, f"{name_without_ext}_partial.p")
        out_png_path = os.path.join(args.output_dir, f"{name_without_ext}_partial.png") if args.viz else None

        # 每张图独立用 args.seed 采样, 可复现且与顺序无关
        sampler.process_file(file_path, out_p_path, out_png_path, file_seed=args.seed)

    print(f"All done! Files saved to {args.output_dir}")
    print(f"  .p files: {len(file_list)}")
    if args.viz:
        print(f"  .png viz: {len(file_list)} (用 transform_node 按 {args.dataset} 坐标系渲染)")


if __name__ == "__main__":
    main()
    