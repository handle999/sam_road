import os
import pickle
import random
import argparse
import numpy as np
import cv2
import glob
from tqdm import tqdm
import copy

class PartialGraphSampler:
    def __init__(self, dataset_type, keep_ratio, image_size=None, line_thickness=3):
        """
        初始化路网采样器
        :param dataset_type: 数据集类型 ('cityscale', 'spacenet', 'didi')
        :param keep_ratio: 保留边的比例 (0.0 ~ 1.0)
        :param image_size: 图像尺寸 (不指定则使用默认值)
        :param line_thickness: 渲染 PNG 时的线宽
        """
        self.dataset_type = dataset_type
        self.keep_ratio = keep_ratio
        self.line_thickness = line_thickness

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
        if self.dataset_type in ['cityscale', 'didi']:
            # Pickle 中是 (Row, Col) -> 转换为 (X, Y) 即 (Col, Row)
            self.transform_node = lambda v0, v1: (int(v1), int(v0))
        elif self.dataset_type == 'spacenet':
            # Pickle 中是 (raw_y, raw_x) -> 转换为 X=raw_x, Y=400-raw_y
            self.transform_node = lambda v0, v1: (int(v1), int(self.image_size - v0))

    def sample_graph(self, adj_dict):
        """
        从完整的邻接字典中随机保留一定比例的边 (无向图逻辑)
        :param adj_dict: 原始邻接字典 {node: [neighbor1, neighbor2, ...]}
        :return: 采样后的新邻接字典
        """
        # 1. 提取所有无向边 (去重)
        edges = set()
        for u, neighbors in adj_dict.items():
            for v in neighbors:
                # 排序保证 (u, v) 和 (v, u) 视为同一条边
                edges.add(tuple(sorted((u, v))))
        
        edges = list(edges)
        
        # 2. 随机采样
        keep_num = int(len(edges) * self.keep_ratio)
        sampled_edges = random.sample(edges, keep_num)

        # 3. 重建邻接字典
        new_adj_dict = {}
        for u, v in sampled_edges:
            if u not in new_adj_dict:
                new_adj_dict[u] = []
            if v not in new_adj_dict:
                new_adj_dict[v] = []
            
            # 保持双向连通性
            new_adj_dict[u].append(v)
            new_adj_dict[v].append(u)

        return new_adj_dict

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

    def process_file(self, input_p_path, out_p_path, out_png_path):
        """
        处理单个文件：加载 -> 采样 -> 保存 p -> 保存 png
        """
        with open(input_p_path, 'rb') as f:
            original_adj_dict = pickle.load(f)

        if len(original_adj_dict) == 0:
            print(f"[Warn] Empty graph found: {input_p_path}")
            sampled_adj_dict = {}
            img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            sampled_adj_dict = self.sample_graph(original_adj_dict)
            img = self.render_to_png(sampled_adj_dict)

        # 保存 Sampled Pickle
        with open(out_p_path, 'wb') as f:
            pickle.dump(sampled_adj_dict, f)

        # 保存 Rendered PNG
        cv2.imwrite(out_png_path, img)


def main():
    parser = argparse.ArgumentParser(description="Generate partial road network priors.")
    parser.add_argument("--dataset", type=str, required=True, choices=['spacenet', 'cityscale', 'didi'],
                        help="Type of dataset to handle coordinate systems correctly.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing original GT .p files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save sampled .p and .png files.")
    parser.add_argument("--keep_ratio", type=float, default=0.5,
                        help="Ratio of edges to keep (0.0 to 1.0). Default is 0.5 (50%).")
    parser.add_argument("--thickness", type=int, default=3,
                        help="Line thickness for rendered PNG. Default is 3.")
    
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    sampler = PartialGraphSampler(
        dataset_type=args.dataset, 
        keep_ratio=args.keep_ratio,
        line_thickness=args.thickness
    )

    # 查找输入目录下所有的 .p 或 .pickle 文件
    search_pattern1 = os.path.join(args.input_dir, "*.p")
    search_pattern2 = os.path.join(args.input_dir, "*.pickle")
    file_list = glob.glob(search_pattern1) + glob.glob(search_pattern2)

    if len(file_list) == 0:
        print(f"No pickle files found in {args.input_dir}")
        return

    print(f"Found {len(file_list)} files. Starting sampling (Keep Ratio: {args.keep_ratio})...")

    for file_path in tqdm(file_list):
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]

        out_p_path = os.path.join(args.output_dir, f"{name_without_ext}_partial.p")
        out_png_path = os.path.join(args.output_dir, f"{name_without_ext}_partial.png")

        sampler.process_file(file_path, out_p_path, out_png_path)

    print(f"All done! Files saved to {args.output_dir}")


if __name__ == "__main__":
    main()
    