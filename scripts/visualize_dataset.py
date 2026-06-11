"""
SAM-Road 数据集可视化验证脚本
================================
验证 CityScale、SpaceNet 和 Xian 数据集的组织方式，特别是坐标系问题。

核心验证点：
1. pickle 文件中图的坐标格式是什么（左下原点？左上原点？）
2. RGB 图片和 GT mask 是否对齐
3. coord_transform 转换是否正确
4. 生成的 keypoint_mask / road_mask 和 RGB 是否视觉一致

使用方法:
  cd /home/hanhaoyu/workspace/research/sam_road
  conda activate samroad
  python scripts/visualize_dataset.py

输出文件说明 (全部保存在 outputs/viz/ 目录下):
================================

【通用命名规则】
  cityscale_region_{编号}_*.png → CityScale 数据集的可视化
  spacenet_{样本ID}_*.png       → SpaceNet 数据集的可视化
  xian_region_{编号}_*.png     → Xian 数据集的可视化

【visualize_cityscale() 产生的图】

  cityscale_region_{编号}_alignment_check.png
    含义: ★ CityScale 核心验证图 ★ swap 变换的 road_mask 与 GT.png/processed mask 对比
    布局: 1×4 子图
      - 第1列: RGB 裁剪区域 + GT.png 叠加 (正确参考)
      - 第2列: RGB + processed/road_mask_{id}.png 叠加 (generate_labels.py 生成的)
      - 第3列: RGB + swap(cityscale) 变换的 road_mask
      - 第4列: RGB + spacenet(flip-y) 变换的 road_mask
    用途: 验证 cityscale 的 swap 变换是否正确
    验证结论: CityScale → swap 正确 (左上原点)

  cityscale_region_{编号}_existing_masks.png
    含义: 查看 generate_labels.py 已生成的 processed/ 目录下的 mask
    布局: 2×2 子图
      - 左上: RGB
      - 右上: 现有 road_mask (灰度)
      - 左下: RGB + road_mask 叠加
      - 右下: 现有 keypoint_mask (灰度)
    用途: 验证 generate_labels.py 生成的 mask 是否与 RGB 对齐

【detailed_coord_analysis() 产生的图】

  spacenet_4transforms.png
    含义: SpaceNet 第一个样本，在 RGB 上叠加 4 种不同坐标变换后的图结构
    布局: 2×2 子图，分别是 raw(无变换) / swap(交换行列) / spacenet(flip-y) / flip-y(另一种翻转)
    用途: 一眼看出哪种变换能让绿色道路线条和 RGB 图片中的真实道路重合
    验证结论: spacenet(flip-y) 变换正确，道路与 RGB 完美对齐

  xian_4transforms.png
    含义: Xian region_0，在 RGB 上叠加 4 种不同坐标变换后的图结构
    布局: 2×2 子图，同上 4 种变换
    用途: 一眼看出哪种变换能让绿色道路线条和 RGB 图片中的真实道路重合
    验证结论: swap(交换行列) 变换正确，道路与 RGB 完美对齐

【visualize_spacenet() 产生的图】

  spacenet_{样本ID}_spacenet_transform.png
    含义: 用 SpaceNet 官方 coord_transform (flip-y) 在 RGB 上画图
    布局: 1×3 子图
      - 左: 原始 pickle 坐标直接画 (红色，不做变换)
      - 中: 应用 spacenet flip-y 变换后画 (绿色)
      - 右: 用 OpenCV 按 generate_labels.py 方式画道路和关键节点 (黄色线 + 红色点)
    用途: 对比无变换和有变换的效果，验证 spacenet 变换正确性

  spacenet_{样本ID}_cityscale_transform.png
    含义: 用 cityscale 式 coord_transform (仅 swap) 在 RGB 上画图
    布局: 同上 1×3
    用途: 对比错误的变换 (swap) 和正确变换的差异
    验证结论: swap 变换下道路线条出现在错误位置 (y 轴翻转)

  spacenet_{样本ID}_gt_overlay.png
    含义: GT.png (数据集自带的 GT 道路 mask) 与 RGB 叠加
    布局: 1×3 子图
      - 左: 原始 RGB
      - 中: GT.png 灰度图
      - 右: RGB + GT.png 半透明叠加
    用途: 查看 GT.png 是否和 RGB 对齐 (GT.png 是正确参考)

  spacenet_{样本ID}_transform_compare.png
    含义: 对比两种坐标变换生成的 keypoint/road mask
    布局: 2×3 子图
      - 上排: spacenet 变换 → RGB / road_mask / keypoint_mask
      - 下排: cityscale 变换 → RGB / road_mask / keypoint_mask
    用途: 对比两种变换生成的 mask 差异 (spacenet 正确时道路 mask 与 RGB 对齐)

  spacenet_{样本ID}_alignment_check.png
    含义: ★ 最核心的验证图 ★ 将两种变换生成的 road_mask 半透明叠加到 RGB 上
    布局: 1×2 子图
      - 左: RGB + spacenet flip-y 变换的 road_mask
      - 右: RGB + cityscale swap 变换的 road_mask
    用途: 直接目视哪种变换让道路 (彩色) 与 RGB 中的真实道路重合
    验证结论: SpaceNet → flip-y 正确 (左图对齐)

【visualize_xian() 产生的图】

  xian_region_{编号}_xian_transform.png
    含义: 用 xian/cityscale 式 coord_transform (仅 swap) 在 RGB 上画图
    布局: 1×3 子图 (同 spacenet_transform 的布局)
    用途: 验证 xian 的 swap 变换是否正确

  xian_region_{编号}_spacenet_transform.png
    含义: 用 spacenet 式 coord_transform (flip-y) 在 RGB 上画图
    布局: 1×3 子图
    用途: 对比错误的 flip-y 变换
    验证结论: flip-y 变换下道路出现在错误位置

  xian_region_{编号}_transform_compare.png
    含义: 对比两种坐标变换生成的 keypoint/road mask
    布局: 2×3 子图 (同 spacenet 的布局)
    用途: 对比两种变换的 mask 差异

  xian_region_{编号}_alignment_check.png
    含义: ★ 最核心的验证图 ★ 将两种变换生成的 road_mask 半透明叠加到 RGB 上
    布局: 1×2 子图
      - 左: RGB + xian swap 变换的 road_mask
      - 右: RGB + spacenet flip-y 变换的 road_mask
    用途: 直接目视哪种变换让道路与 RGB 重合
    验证结论: Xian → swap 正确 (左图对齐)

  xian_region_{编号}_existing_masks.png
    含义: 查看 generate_labels.py 已生成的 processed/ 目录下的 mask
    布局: 2×2 子图
      - 左上: RGB
      - 右上: 现有 road_mask (灰度)
      - 左下: RGB + road_mask 叠加
      - 右下: 现有 keypoint_mask (灰度)
    用途: 验证 generate_labels.py 生成的 mask 是否与 RGB 对齐

  xian_region_{编号}_gt_overlay.png
    含义: GT.png 与 RGB 叠加
    布局: 1×3 子图 (同 spacenet 的 gt_overlay)
    用途: 查看 GT.png 是否和 RGB 对齐

  xian_region_{编号}_active_overlay.png
    含义: Active.png (已有路网先验) 与 RGB 叠加
    布局: 1×3 子图
      - 左: RGB
      - 中: Active mask 灰度图
      - 右: RGB + Active 半透明叠加
    用途: 查看 Active mask (4通道模型输入) 是否和 RGB 中的道路对齐
"""

import os
import sys
import pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import networkx as nx

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = '/home/hanhaoyu/workspace/research/sam_road/outputs/viz'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SpaceNet 样本
SPACENET_DIR = '/home/hanhaoyu/workspace/research/sam_road/datasets/spacenet/RGB_1.0_meter'
SPACENET_SAMPLES = ['AOI_2_Vegas_210', 'AOI_2_Vegas_5']  # 用两个样本

# Xian 样本
XIAN_DIR = '/home/hanhaoyu/workspace/research/sam_road/datasets/didi/xian/2019_400/xian_2019_400'
XIAN_PROCESSED_DIR = '/home/hanhaoyu/workspace/research/sam_road/datasets/didi/xian/2019_400/processed'
XIAN_SAMPLES = ['0', '1', '2']  # region_0, region_1, region_2

CITYSCALE_DIR = '/home/hanhaoyu/workspace/research/sam_road/datasets/cityscale/20cities'
CITYSCALE_PROCESSED_DIR = '/home/hanhaoyu/workspace/research/sam_road/datasets/cityscale/processed'
CITYSCALE_SAMPLES = ['0', '10', '166']  # region_0, region_10, region_166

SPACENET_IMAGE_SIZE = 400
XIAN_IMAGE_SIZE = 400
CITYSCALE_IMAGE_SIZE = 2048


def read_rgb_img(path):
    """读取 RGB 图片"""
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"[ERROR] Cannot read: {path}")
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def load_gt_graph(path):
    """加载 GT graph pickle"""
    with open(path, 'rb') as f:
        gt_graph = pickle.load(f)
    return gt_graph


def analyze_pickle_coords(gt_graph, image_size, dataset_name):
    """
    分析 pickle 中坐标的格式
    
    返回:
      - coord_range: 坐标范围
      - sample_coords: 一些样本坐标
    """
    all_coords = []
    for node, neighbors in gt_graph.items():
        all_coords.append(node)
        for nei in neighbors:
            all_coords.append(nei)
    
    all_coords = np.array(all_coords, dtype=np.float64)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Number of unique nodes: {len(gt_graph)}")
    print(f"Total coord entries (with duplicates): {len(all_coords)}")
    print(f"Coord shape: {all_coords.shape}")
    print(f"Dim 0 range: [{all_coords[:, 0].min():.2f}, {all_coords[:, 0].max():.2f}]")
    print(f"Dim 1 range: [{all_coords[:, 1].min():.2f}, {all_coords[:, 1].max():.2f}]")
    print(f"Image size: {image_size}")
    
    # 检查坐标是否在 [0, image_size] 内
    in_range_0 = np.all((all_coords[:, 0] >= 0) & (all_coords[:, 0] <= image_size))
    in_range_1 = np.all((all_coords[:, 1] >= 0) & (all_coords[:, 1] <= image_size))
    print(f"Dim 0 in [0, {image_size}]: {in_range_0}")
    print(f"Dim 1 in [0, {image_size}]: {in_range_1}")
    
    # 采样一些坐标
    print(f"\nSample coordinates (first 10):")
    for i, coord in enumerate(all_coords[:10]):
        print(f"  {i}: ({coord[0]:.2f}, {coord[1]:.2f})")
    
    return all_coords


def draw_graph_on_image(rgb, gt_graph, coord_transform, image_size, title, save_path):
    """
    在 RGB 图上绘制图结构，验证坐标转换是否正确
    
    Args:
        rgb: RGB 图片 [H, W, 3]
        gt_graph: dict, 邻接表
        coord_transform: lambda [N, 2] -> [N, 2]
        image_size: 图片尺寸
        title: 图标题
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # === 子图1: 原始坐标 (不做任何变换) ===
    ax = axes[0]
    ax.imshow(rgb)
    ax.set_title(f"{title}\nRaw coords (no transform)")
    
    # 直接用原始坐标画图
    for node, neighbors in gt_graph.items():
        n0, n1 = int(node[0]), int(node[1])
        # 检查坐标是否在图片范围内
        if 0 <= n0 < image_size and 0 <= n1 < image_size:
            ax.plot(n1, n0, 'r.', markersize=2)  # 注意: plot(x, y), 所以 plot(dim1, dim0)
        for nei in neighbors:
            ne0, ne1 = int(nei[0]), int(nei[1])
            if 0 <= n0 < image_size and 0 <= n1 < image_size and 0 <= ne0 < image_size and 0 <= ne1 < image_size:
                ax.plot([n1, ne1], [n0, ne0], 'r-', linewidth=0.5, alpha=0.5)
    
    # === 子图2: 应用 coord_transform ===
    ax = axes[1]
    ax.imshow(rgb)
    ax.set_title(f"{title}\nWith coord_transform")
    
    # 构建坐标数组并变换
    nodes_list = list(gt_graph.keys())
    edges_list = []
    for node, neighbors in gt_graph.items():
        for nei in neighbors:
            edges_list.append((node, nei))
    
    if len(nodes_list) > 0:
        nodes_arr = np.array(nodes_list, dtype=np.float64)
        transformed_nodes = coord_transform(nodes_arr)
        
        # 创建从原始坐标到变换坐标的映射
        node_to_transformed = {}
        for i, node in enumerate(nodes_list):
            node_to_transformed[node] = (transformed_nodes[i, 0], transformed_nodes[i, 1])
        
        for src, tgt in edges_list:
            t_src = node_to_transformed[src]
            t_tgt = node_to_transformed[tgt]
            # 在图像坐标系中: x=dim0, y=dim1, plot(y, x)?
            # 不对，这里已经是 (x, y) 了
            ax.plot([t_src[0], t_tgt[0]], [t_src[1], t_tgt[1]], 'g-', linewidth=0.5, alpha=0.5)
        
        for t_node in transformed_nodes:
            if 0 <= t_node[0] < image_size and 0 <= t_node[1] < image_size:
                ax.plot(t_node[0], t_node[1], 'g.', markersize=2)
    
    # === 子图3: GT mask (如果有) ===
    ax = axes[2]
    ax.imshow(rgb)
    ax.set_title(f"{title}\nGT overlay")
    
    # 使用 OpenCV 画出 GT（和 generate_labels.py 一样的方式）
    graph = nx.Graph()
    for n, neis in gt_graph.items():
        for nei in neis:
            # 应用 coord_transform
            n_arr = np.array([[n[0], n[1]]], dtype=np.float64)
            nei_arr = np.array([[nei[0], nei[1]]], dtype=np.float64)
            tn = coord_transform(n_arr)[0]
            tnei = coord_transform(nei_arr)[0]
            graph.add_edge((int(tn[0]), int(tn[1])), (int(tnei[0]), int(tnei[1])))
    
    # 画边
    for (x1, y1), (x2, y2) in graph.edges():
        ax.plot([x1, x2], [y1, y2], 'y-', linewidth=1, alpha=0.7)
    
    # 画关键节点
    key_nodes = []
    for node, degree in graph.degree():
        if degree != 2:
            key_nodes.append(node)
    for (x, y) in key_nodes:
        ax.plot(x, y, 'ro', markersize=4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_spacenet():
    """可视化 SpaceNet 数据集"""
    print("\n" + "="*80)
    print("VISUALIZING SPACENET DATASET")
    print("="*80)
    
    # SpaceNet 的 coord_transform: (r, c) -> (x, y)
    # 原始坐标假设是 (row, col)，即 (y, x) -> 左下原点
    # 变换: x = col, y = IMAGE_SIZE - row  (即翻转 y 轴)
    coord_transform_spacenet = lambda v: np.stack([v[:, 1], SPACENET_IMAGE_SIZE - v[:, 0]], axis=1)
    # cityscale 的 coord_transform: 简单交换 (r, c) -> (c, r) 即 (y, x) -> (x, y)
    coord_transform_cityscale_style = lambda v: v[:, ::-1]
    
    for sample_id in SPACENET_SAMPLES:
        rgb_path = os.path.join(SPACENET_DIR, f"{sample_id}__rgb.png")
        gt_graph_path = os.path.join(SPACENET_DIR, f"{sample_id}__gt_graph.p")
        gt_png_path = os.path.join(SPACENET_DIR, f"{sample_id}__gt.png")
        
        if not os.path.exists(rgb_path) or not os.path.exists(gt_graph_path):
            print(f"[SKIP] {sample_id} - files not found")
            continue
        
        rgb = read_rgb_img(rgb_path)
        gt_graph = load_gt_graph(gt_graph_path)
        
        print(f"\n--- SpaceNet sample: {sample_id} ---")
        print(f"RGB shape: {rgb.shape}")
        print(f"GT graph nodes: {len(gt_graph)}")
        
        # 分析坐标
        all_coords = analyze_pickle_coords(gt_graph, SPACENET_IMAGE_SIZE, f"SpaceNet/{sample_id}")
        
        # 绘制对比图: spacenet transform vs cityscale-style transform
        draw_graph_on_image(rgb, gt_graph, coord_transform_spacenet, SPACENET_IMAGE_SIZE,
                          f"SpaceNet {sample_id} (spacenet_transform)",
                          os.path.join(OUTPUT_DIR, f"spacenet_{sample_id}_spacenet_transform.png"))
        
        draw_graph_on_image(rgb, gt_graph, coord_transform_cityscale_style, SPACENET_IMAGE_SIZE,
                          f"SpaceNet {sample_id} (cityscale_transform)",
                          os.path.join(OUTPUT_DIR, f"spacenet_{sample_id}_cityscale_transform.png"))
        
        # 也加载 GT.png 看看
        if os.path.exists(gt_png_path):
            gt_png = cv2.imread(gt_png_path, cv2.IMREAD_GRAYSCALE)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(rgb)
            axes[0].set_title("RGB")
            axes[1].imshow(gt_png, cmap='gray')
            axes[1].set_title("GT mask (from _gt.png)")
            axes[2].imshow(rgb)
            axes[2].imshow(gt_png, cmap='jet', alpha=0.3)
            axes[2].set_title("RGB + GT overlay")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"spacenet_{sample_id}_gt_overlay.png"), dpi=150)
            plt.close()
        
        # 重新生成 keypoint 和 road mask (和 generate_labels.py 一样)
        # 使用 spacenet 变换
        graph_sn = nx.Graph()
        for n, neis in gt_graph.items():
            for nei in neis:
                graph_sn.add_edge((int(n[1]), SPACENET_IMAGE_SIZE - int(n[0])), 
                                  (int(nei[1]), SPACENET_IMAGE_SIZE - int(nei[0])))
        
        key_nodes_sn = [node for node, degree in graph_sn.degree() if degree != 2]
        
        keypoint_mask_sn = np.zeros((SPACENET_IMAGE_SIZE, SPACENET_IMAGE_SIZE), dtype=np.uint8)
        for point in key_nodes_sn:
            cv2.circle(keypoint_mask_sn, point, 3, 255, -1)
        
        road_mask_sn = np.zeros((SPACENET_IMAGE_SIZE, SPACENET_IMAGE_SIZE), dtype=np.uint8)
        for (x1, y1), (x2, y2) in graph_sn.edges():
            cv2.line(road_mask_sn, (x1, y1), (x2, y2), 255, 3)
        
        # 使用 cityscale 变换
        graph_cs = nx.Graph()
        for n, neis in gt_graph.items():
            for nei in neis:
                graph_cs.add_edge((int(n[1]), int(n[0])), (int(nei[1]), int(nei[0])))
        
        key_nodes_cs = [node for node, degree in graph_cs.degree() if degree != 2]
        
        keypoint_mask_cs = np.zeros((SPACENET_IMAGE_SIZE, SPACENET_IMAGE_SIZE), dtype=np.uint8)
        for point in key_nodes_cs:
            cv2.circle(keypoint_mask_cs, point, 3, 255, -1)
        
        road_mask_cs = np.zeros((SPACENET_IMAGE_SIZE, SPACENET_IMAGE_SIZE), dtype=np.uint8)
        for (x1, y1), (x2, y2) in graph_cs.edges():
            cv2.line(road_mask_cs, (x1, y1), (x2, y2), 255, 3)
        
        # 对比两种变换生成的 mask
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes[0, 0].imshow(rgb); axes[0, 0].set_title("RGB")
        axes[0, 1].imshow(road_mask_sn, cmap='gray'); axes[0, 1].set_title("Road mask (spacenet transform)\nx=col, y=IMAGE_SIZE-row")
        axes[0, 2].imshow(keypoint_mask_sn, cmap='gray'); axes[0, 2].set_title("Keypoint mask (spacenet transform)")
        
        axes[1, 0].imshow(rgb); axes[1, 0].set_title("RGB")
        axes[1, 1].imshow(road_mask_cs, cmap='gray'); axes[1, 1].set_title("Road mask (cityscale transform)\nx=col, y=row")
        axes[1, 2].imshow(keypoint_mask_cs, cmap='gray'); axes[1, 2].set_title("Keypoint mask (cityscale transform)")
        
        plt.suptitle(f"SpaceNet {sample_id}: Comparing coord transforms", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"spacenet_{sample_id}_transform_compare.png"), dpi=150)
        plt.close()
        
        # 最重要的验证: road_mask 和 RGB 是否对齐
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb)
        axes[0].imshow(road_mask_sn, cmap='jet', alpha=0.4)
        axes[0].set_title("RGB + Road (spacenet transform)")
        axes[1].imshow(rgb)
        axes[1].imshow(road_mask_cs, cmap='jet', alpha=0.4)
        axes[1].set_title("RGB + Road (cityscale transform)")
        plt.suptitle(f"SpaceNet {sample_id}: Which transform aligns with RGB?", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"spacenet_{sample_id}_alignment_check.png"), dpi=150)
        plt.close()


def visualize_xian():
    """可视化 Xian (DiDi) 数据集"""
    print("\n" + "="*80)
    print("VISUALIZING XIAN (DiDi) DATASET")
    print("="*80)
    
    # Xian 的 coord_transform (和 cityscale 一样):
    coord_transform_xian = lambda v: v[:, ::-1]
    # 也试试 spacenet 的变换
    coord_transform_spacenet_style = lambda v: np.stack([v[:, 1], XIAN_IMAGE_SIZE - v[:, 0]], axis=1)
    
    for sample_id in XIAN_SAMPLES:
        rgb_path = os.path.join(XIAN_DIR, f"region_{sample_id}_sat.png")
        gt_graph_path = os.path.join(XIAN_DIR, f"region_{sample_id}_refine_gt_graph.p")
        gt_png_path = os.path.join(XIAN_DIR, f"region_{sample_id}_gt.png")
        active_png_path = os.path.join(XIAN_DIR, f"region_{sample_id}_active.png")
        keypoint_mask_path = os.path.join(XIAN_PROCESSED_DIR, f"keypoint_mask_{sample_id}.png")
        road_mask_path = os.path.join(XIAN_PROCESSED_DIR, f"road_mask_{sample_id}.png")
        
        if not os.path.exists(rgb_path) or not os.path.exists(gt_graph_path):
            print(f"[SKIP] region_{sample_id} - files not found")
            continue
        
        rgb = read_rgb_img(rgb_path)
        gt_graph = load_gt_graph(gt_graph_path)
        
        print(f"\n--- Xian sample: region_{sample_id} ---")
        print(f"RGB shape: {rgb.shape}")
        print(f"GT graph nodes: {len(gt_graph)}")
        
        # 分析坐标
        all_coords = analyze_pickle_coords(gt_graph, XIAN_IMAGE_SIZE, f"Xian/region_{sample_id}")
        
        # 绘制: xian transform vs spacenet transform
        draw_graph_on_image(rgb, gt_graph, coord_transform_xian, XIAN_IMAGE_SIZE,
                          f"Xian region_{sample_id} (xian/cityscale transform: swap)",
                          os.path.join(OUTPUT_DIR, f"xian_region_{sample_id}_xian_transform.png"))
        
        draw_graph_on_image(rgb, gt_graph, coord_transform_spacenet_style, XIAN_IMAGE_SIZE,
                          f"Xian region_{sample_id} (spacenet transform: flip-y)",
                          os.path.join(OUTPUT_DIR, f"xian_region_{sample_id}_spacenet_transform.png"))
        
        # 重新生成 mask (和 generate_labels.py 一样)
        # Xian generate_labels.py 使用: graph.add_edge((int(n[1]), int(n[0])), (int(nei[1]), int(nei[0])))
        # 即简单的交换 (row, col) -> (col, row)
        graph_xian = nx.Graph()
        for n, neis in gt_graph.items():
            for nei in neis:
                graph_xian.add_edge((int(n[1]), int(n[0])), (int(nei[1]), int(nei[0])))
        
        key_nodes_xian = [node for node, degree in graph_xian.degree() if degree != 2]
        
        road_mask_xian = np.zeros((XIAN_IMAGE_SIZE, XIAN_IMAGE_SIZE), dtype=np.uint8)
        for (x1, y1), (x2, y2) in graph_xian.edges():
            cv2.line(road_mask_xian, (x1, y1), (x2, y2), 255, 3)
        
        keypoint_mask_xian = np.zeros((XIAN_IMAGE_SIZE, XIAN_IMAGE_SIZE), dtype=np.uint8)
        for point in key_nodes_xian:
            cv2.circle(keypoint_mask_xian, point, 3, 255, -1)
        
        # spacenet 变换方式
        graph_sn = nx.Graph()
        for n, neis in gt_graph.items():
            for nei in neis:
                graph_sn.add_edge((int(n[1]), XIAN_IMAGE_SIZE - int(n[0])), 
                                  (int(nei[1]), XIAN_IMAGE_SIZE - int(nei[0])))
        
        key_nodes_sn = [node for node, degree in graph_sn.degree() if degree != 2]
        
        road_mask_sn = np.zeros((XIAN_IMAGE_SIZE, XIAN_IMAGE_SIZE), dtype=np.uint8)
        for (x1, y1), (x2, y2) in graph_sn.edges():
            cv2.line(road_mask_sn, (x1, y1), (x2, y2), 255, 3)
        
        keypoint_mask_sn = np.zeros((XIAN_IMAGE_SIZE, XIAN_IMAGE_SIZE), dtype=np.uint8)
        for point in key_nodes_sn:
            cv2.circle(keypoint_mask_sn, point, 3, 255, -1)
        
        # 对比两种变换
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes[0, 0].imshow(rgb); axes[0, 0].set_title("RGB")
        axes[0, 1].imshow(road_mask_xian, cmap='gray'); axes[0, 1].set_title("Road mask (xian: swap r,c)\nx=col, y=row")
        axes[0, 2].imshow(keypoint_mask_xian, cmap='gray'); axes[0, 2].set_title("Keypoint mask (xian: swap)")
        
        axes[1, 0].imshow(rgb); axes[1, 0].set_title("RGB")
        axes[1, 1].imshow(road_mask_sn, cmap='gray'); axes[1, 1].set_title("Road mask (spacenet: flip y)\nx=col, y=SIZE-row")
        axes[1, 2].imshow(keypoint_mask_sn, cmap='gray'); axes[1, 2].set_title("Keypoint mask (spacenet: flip y)")
        
        plt.suptitle(f"Xian region_{sample_id}: Comparing coord transforms", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"xian_region_{sample_id}_transform_compare.png"), dpi=150)
        plt.close()
        
        # 最重要的验证: road_mask 和 RGB 是否对齐
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb)
        axes[0].imshow(road_mask_xian, cmap='jet', alpha=0.4)
        axes[0].set_title("RGB + Road (xian: swap)")
        axes[1].imshow(rgb)
        axes[1].imshow(road_mask_sn, cmap='jet', alpha=0.4)
        axes[1].set_title("RGB + Road (spacenet: flip-y)")
        plt.suptitle(f"Xian region_{sample_id}: Which transform aligns with RGB?", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"xian_region_{sample_id}_alignment_check.png"), dpi=150)
        plt.close()
        
        # 也看看已有的 processed mask (如果有)
        if os.path.exists(keypoint_mask_path) and os.path.exists(road_mask_path):
            existing_keypoint = cv2.imread(keypoint_mask_path, cv2.IMREAD_GRAYSCALE)
            existing_road = cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes[0, 0].imshow(rgb); axes[0, 0].set_title("RGB")
            axes[0, 1].imshow(existing_road, cmap='gray'); axes[0, 1].set_title("Existing road_mask (from processed/)")
            axes[1, 0].imshow(rgb)
            axes[1, 0].imshow(existing_road, cmap='jet', alpha=0.4)
            axes[1, 0].set_title("RGB + existing road_mask overlay")
            axes[1, 1].imshow(existing_keypoint, cmap='gray'); axes[1, 1].set_title("Existing keypoint_mask")
            plt.suptitle(f"Xian region_{sample_id}: Existing processed masks", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"xian_region_{sample_id}_existing_masks.png"), dpi=150)
            plt.close()
        
        # 看 GT.png
        if os.path.exists(gt_png_path):
            gt_png = cv2.imread(gt_png_path, cv2.IMREAD_GRAYSCALE)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(rgb); axes[0].set_title("RGB")
            axes[1].imshow(gt_png, cmap='gray'); axes[1].set_title("GT mask (_gt.png)")
            axes[2].imshow(rgb)
            axes[2].imshow(gt_png, cmap='jet', alpha=0.3)
            axes[2].set_title("RGB + GT overlay")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"xian_region_{sample_id}_gt_overlay.png"), dpi=150)
            plt.close()
        
        # 看 Active.png
        if os.path.exists(active_png_path):
            active_png = cv2.imread(active_png_path, cv2.IMREAD_GRAYSCALE)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(rgb); axes[0].set_title("RGB")
            axes[1].imshow(active_png, cmap='gray'); axes[1].set_title("Active mask (_active.png)")
            axes[2].imshow(rgb)
            axes[2].imshow(active_png, cmap='jet', alpha=0.3)
            axes[2].set_title("RGB + Active overlay")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"xian_region_{sample_id}_active_overlay.png"), dpi=150)
            plt.close()


def visualize_cityscale():
    """可视化 CityScale 数据集"""
    print("\n" + "="*80)
    print("VISUALIZING CITYSCALE DATASET")
    print("="*80)
    
    # CityScale 的 coord_transform: 简单交换 (r, c) -> (c, r) 即 (y, x) -> (x, y)
    coord_transform_cityscale = lambda v: v[:, ::-1]
    # 对比 spacenet 的变换
    coord_transform_spacenet_style = lambda v: np.stack([v[:, 1], CITYSCALE_IMAGE_SIZE - v[:, 0]], axis=1)
    
    for sample_id in CITYSCALE_SAMPLES:
        rgb_path = os.path.join(CITYSCALE_DIR, f'region_{sample_id}_sat.png')
        gt_graph_path = os.path.join(CITYSCALE_DIR, f'region_{sample_id}_refine_gt_graph.p')
        gt_png_path = os.path.join(CITYSCALE_DIR, f'region_{sample_id}_gt.png')
        road_mask_path = os.path.join(CITYSCALE_PROCESSED_DIR, f'road_mask_{sample_id}.png')
        keypoint_mask_path = os.path.join(CITYSCALE_PROCESSED_DIR, f'keypoint_mask_{sample_id}.png')
        
        if not os.path.exists(rgb_path) or not os.path.exists(gt_graph_path):
            print(f"[SKIP] region_{sample_id} - files not found")
            continue
        
        rgb = read_rgb_img(rgb_path)
        gt_graph = load_gt_graph(gt_graph_path)
        
        print(f"\n--- CityScale sample: region_{sample_id} ---")
        print(f"RGB shape: {rgb.shape}")
        print(f"GT graph nodes: {len(gt_graph)}")
        
        # 分析坐标
        all_coords = analyze_pickle_coords(gt_graph, CITYSCALE_IMAGE_SIZE, f'CityScale/region_{sample_id}')
        
        # 生成 road mask (两种变换)
        graph_cs = nx.Graph()
        for n, neis in gt_graph.items():
            for nei in neis:
                graph_cs.add_edge((int(n[1]), int(n[0])), (int(nei[1]), int(nei[0])))
        
        road_mask_cs = np.zeros((CITYSCALE_IMAGE_SIZE, CITYSCALE_IMAGE_SIZE), dtype=np.uint8)
        for (x1, y1), (x2, y2) in graph_cs.edges():
            cv2.line(road_mask_cs, (x1, y1), (x2, y2), 255, 3)
        
        graph_sn = nx.Graph()
        for n, neis in gt_graph.items():
            for nei in neis:
                graph_sn.add_edge((int(n[1]), CITYSCALE_IMAGE_SIZE - int(n[0])), (int(nei[1]), CITYSCALE_IMAGE_SIZE - int(nei[0])))
        
        road_mask_sn = np.zeros((CITYSCALE_IMAGE_SIZE, CITYSCALE_IMAGE_SIZE), dtype=np.uint8)
        for (x1, y1), (x2, y2) in graph_sn.edges():
            cv2.line(road_mask_sn, (x1, y1), (x2, y2), 255, 3)
        
        # 对齐检查: 用裁剪区域来避免 2048 全图太慢
        # 取中心 512x512 区域
        margin = 512
        crop = (CITYSCALE_IMAGE_SIZE // 2 - margin, CITYSCALE_IMAGE_SIZE // 2 - margin,
                CITYSCALE_IMAGE_SIZE // 2 + margin, CITYSCALE_IMAGE_SIZE // 2 + margin)
        
        gt_png = cv2.imread(gt_png_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_png_path) else None
        existing_road = cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(road_mask_path) else None
        
        # ★ 核心验证图: 对齐检查 ★
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        # 第1列: RGB + GT.png
        ax = axes[0]
        ax.imshow(rgb[crop[1]:crop[3], crop[0]:crop[2]])
        if gt_png is not None:
            ax.imshow(gt_png[crop[1]:crop[3], crop[0]:crop[2]], cmap='jet', alpha=0.3)
        ax.set_title("RGB + GT.png (reference)")
        
        # 第2列: RGB + existing road_mask
        ax = axes[1]
        ax.imshow(rgb[crop[1]:crop[3], crop[0]:crop[2]])
        if existing_road is not None:
            ax.imshow(existing_road[crop[1]:crop[3], crop[0]:crop[2]], cmap='jet', alpha=0.4)
        ax.set_title("RGB + processed road_mask")
        
        # 第3列: RGB + swap road_mask
        ax = axes[2]
        ax.imshow(rgb[crop[1]:crop[3], crop[0]:crop[2]])
        ax.imshow(road_mask_cs[crop[1]:crop[3], crop[0]:crop[2]], cmap='jet', alpha=0.4)
        ax.set_title("RGB + swap(cityscale) road_mask")
        
        # 第4列: RGB + flip-y road_mask
        ax = axes[3]
        ax.imshow(rgb[crop[1]:crop[3], crop[0]:crop[2]])
        ax.imshow(road_mask_sn[crop[1]:crop[3], crop[0]:crop[2]], cmap='jet', alpha=0.4)
        ax.set_title("RGB + flip-y(spacenet) road_mask")
        
        plt.suptitle(f"CityScale region_{sample_id} (center crop): Which coord_transform aligns?", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'cityscale_region_{sample_id}_alignment_check.png'), dpi=150)
        plt.close()
        print(f"Saved: cityscale_region_{sample_id}_alignment_check.png")
        
        # 已有 processed mask 检查
        if existing_road is not None:
            existing_keypoint = cv2.imread(keypoint_mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(keypoint_mask_path) else None
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes[0, 0].imshow(rgb[crop[1]:crop[3], crop[0]:crop[2]]); axes[0, 0].set_title("RGB (crop)")
            axes[0, 1].imshow(existing_road[crop[1]:crop[3], crop[0]:crop[2]], cmap='gray'); axes[0, 1].set_title("Existing road_mask (crop)")
            axes[1, 0].imshow(rgb[crop[1]:crop[3], crop[0]:crop[2]])
            axes[1, 0].imshow(existing_road[crop[1]:crop[3], crop[0]:crop[2]], cmap='jet', alpha=0.4)
            axes[1, 0].set_title("RGB + existing road_mask overlay (crop)")
            if existing_keypoint is not None:
                axes[1, 1].imshow(existing_keypoint[crop[1]:crop[3], crop[0]:crop[2]], cmap='gray'); axes[1, 1].set_title("Existing keypoint_mask (crop)")
            else:
                axes[1, 1].set_title("keypoint_mask not found")
            plt.suptitle(f"CityScale region_{sample_id}: Existing processed masks", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'cityscale_region_{sample_id}_existing_masks.png'), dpi=150)
            plt.close()
            print(f"Saved: cityscale_region_{sample_id}_existing_masks.png")


def detailed_coord_analysis():
    """
    详细分析坐标系
    
    对每个数据集的一个样本:
    1. 检查 pickle 中坐标的最大最小值
    2. 用不同的 coord_transform 绘图
    3. 和 GT.png 对比
    """
    print("\n" + "="*80)
    print("DETAILED COORDINATE ANALYSIS")
    print("="*80)
    
    # === SpaceNet ===
    sample = SPACENET_SAMPLES[0]
    gt_graph_path = os.path.join(SPACENET_DIR, f"{sample}__gt_graph.p")
    rgb_path = os.path.join(SPACENET_DIR, f"{sample}__rgb.png")
    gt_png_path = os.path.join(SPACENET_DIR, f"{sample}__gt.png")
    
    if os.path.exists(gt_graph_path):
        gt_graph = load_gt_graph(gt_graph_path)
        rgb = read_rgb_img(rgb_path)
        gt_png = cv2.imread(gt_png_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_png_path) else None
        
        # 收集所有坐标
        all_coords = []
        for node, neighbors in gt_graph.items():
            all_coords.append(node)
        all_coords = np.array(all_coords, dtype=np.float64)
        
        print(f"\nSpaceNet {sample}:")
        print(f"  Pickle coord[0] range: [{all_coords[:, 0].min():.1f}, {all_coords[:, 0].max():.1f}]")
        print(f"  Pickle coord[1] range: [{all_coords[:, 1].min():.1f}, {all_coords[:, 1].max():.1f}]")
        print(f"  Image size: {SPACENET_IMAGE_SIZE}")
        
        # 尝试 4 种不同的坐标变换方式
        transforms = {
            'raw (n[0], n[1])': lambda v: v,  # 直接使用原始坐标
            'swap (n[1], n[0])': lambda v: v[:, ::-1],  # 交换行列
            'spacenet (n[1], SIZE-n[0])': lambda v: np.stack([v[:, 1], SPACENET_IMAGE_SIZE - v[:, 0]], axis=1),
            'flip-y (SIZE-n[1], n[0])': lambda v: np.stack([SPACENET_IMAGE_SIZE - v[:, 1], v[:, 0]], axis=1),
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes = axes.flatten()
        
        for idx, (name, transform) in enumerate(transforms.items()):
            ax = axes[idx]
            ax.imshow(rgb)
            ax.set_title(name, fontsize=10)
            
            # 变换坐标并画图
            nodes_arr = np.array(list(gt_graph.keys()), dtype=np.float64)
            transformed = transform(nodes_arr)
            
            # 画边
            for i, (node, neighbors) in enumerate(gt_graph.items()):
                tn = transformed[i]
                for nei in neighbors:
                    # 找 nei 的 index
                    # 简单做法: 直接变换
                    nei_arr = np.array([[nei[0], nei[1]]], dtype=np.float64)
                    tnei = transform(nei_arr)[0]
                    
                    if (0 <= tn[0] < SPACENET_IMAGE_SIZE and 0 <= tn[1] < SPACENET_IMAGE_SIZE and
                        0 <= tnei[0] < SPACENET_IMAGE_SIZE and 0 <= tnei[1] < SPACENET_IMAGE_SIZE):
                        ax.plot([tn[0], tnei[0]], [tn[1], tnei[1]], 'g-', linewidth=0.3, alpha=0.5)
            
            # 画点
            for t in transformed:
                if 0 <= t[0] < SPACENET_IMAGE_SIZE and 0 <= t[1] < SPACENET_IMAGE_SIZE:
                    ax.plot(t[0], t[1], 'g.', markersize=1)
        
        plt.suptitle(f"SpaceNet {sample}: 4 coord transforms comparison\nWhich one aligns roads with the RGB image?", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "spacenet_4transforms.png"), dpi=150)
        plt.close()
    
    # === Xian ===
    sample = XIAN_SAMPLES[0]
    gt_graph_path = os.path.join(XIAN_DIR, f"region_{sample}_refine_gt_graph.p")
    rgb_path = os.path.join(XIAN_DIR, f"region_{sample}_sat.png")
    gt_png_path = os.path.join(XIAN_DIR, f"region_{sample}_gt.png")
    
    if os.path.exists(gt_graph_path):
        gt_graph = load_gt_graph(gt_graph_path)
        rgb = read_rgb_img(rgb_path)
        gt_png = cv2.imread(gt_png_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_png_path) else None
        
        all_coords = []
        for node, neighbors in gt_graph.items():
            all_coords.append(node)
        all_coords = np.array(all_coords, dtype=np.float64)
        
        print(f"\nXian region_{sample}:")
        print(f"  Pickle coord[0] range: [{all_coords[:, 0].min():.1f}, {all_coords[:, 0].max():.1f}]")
        print(f"  Pickle coord[1] range: [{all_coords[:, 1].min():.1f}, {all_coords[:, 1].max():.1f}]")
        print(f"  Image size: {XIAN_IMAGE_SIZE}")
        
        transforms = {
            'raw (n[0], n[1])': lambda v: v,
            'swap (n[1], n[0])': lambda v: v[:, ::-1],
            'spacenet (n[1], SIZE-n[0])': lambda v: np.stack([v[:, 1], XIAN_IMAGE_SIZE - v[:, 0]], axis=1),
            'flip-y (SIZE-n[1], n[0])': lambda v: np.stack([XIAN_IMAGE_SIZE - v[:, 1], v[:, 0]], axis=1),
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes = axes.flatten()
        
        for idx, (name, transform) in enumerate(transforms.items()):
            ax = axes[idx]
            ax.imshow(rgb)
            ax.set_title(name, fontsize=10)
            
            nodes_arr = np.array(list(gt_graph.keys()), dtype=np.float64)
            transformed = transform(nodes_arr)
            
            for i, (node, neighbors) in enumerate(gt_graph.items()):
                tn = transformed[i]
                for nei in neighbors:
                    nei_arr = np.array([[nei[0], nei[1]]], dtype=np.float64)
                    tnei = transform(nei_arr)[0]
                    
                    if (0 <= tn[0] < XIAN_IMAGE_SIZE and 0 <= tn[1] < XIAN_IMAGE_SIZE and
                        0 <= tnei[0] < XIAN_IMAGE_SIZE and 0 <= tnei[1] < XIAN_IMAGE_SIZE):
                        ax.plot([tn[0], tnei[0]], [tn[1], tnei[1]], 'g-', linewidth=0.3, alpha=0.5)
            
            for t in transformed:
                if 0 <= t[0] < XIAN_IMAGE_SIZE and 0 <= t[1] < XIAN_IMAGE_SIZE:
                    ax.plot(t[0], t[1], 'g.', markersize=1)
        
        plt.suptitle(f"Xian region_{sample}: 4 coord transforms comparison\nWhich one aligns roads with the RGB image?", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "xian_4transforms.png"), dpi=150)
        plt.close()


def verify_generate_labels_consistency():
    """
    验证 generate_labels.py 生成 mask 的方式和 dataloader 中的 coord_transform 是否一致
    
    这是最关键的检查：generate_labels.py 中画图的方式和 dataset.py 中的 coord_transform
    是否对同一组坐标做了相同的变换。
    """
    print("\n" + "="*80)
    print("VERIFYING generate_labels.py AND dataset.py CONSISTENCY")
    print("="*80)
    
    # SpaceNet 的 generate_labels.py:
    #   graph.add_edge((int(n[1]), IMAGE_SIZE-int(n[0])), (int(nei[1]), IMAGE_SIZE-int(nei[0])))
    #   即: x = col, y = IMAGE_SIZE - row
    #
    # SpaceNet 的 dataset.py:
    #   coord_transform = lambda v : np.stack([v[:, 1], 400 - v[:, 0]], axis=1)
    #   即: x = v[:, 1] = col, y = 400 - v[:, 0] = 400 - row
    #
    # 这两者是一致的! ✅
    
    print("\nSpaceNet:")
    print("  generate_labels.py: (int(n[1]), IMAGE_SIZE-int(n[0]))")
    print("  dataset.py:         np.stack([v[:, 1], 400 - v[:, 0]], axis=1)")
    print("  → CONSISTENT ✅ (both: x=col, y=SIZE-row)")
    
    # CityScale 的 generate_labels.py:
    #   graph.add_edge((int(n[1]), int(n[0])), (int(nei[1]), int(nei[0])))
    #   即: x = col, y = row (简单交换)
    #
    # CityScale 的 dataset.py:
    #   coord_transform = lambda v : v[:, ::-1]
    #   即: x = v[:, 1] = col, y = v[:, 0] = row
    #
    # 这两者是一致的! ✅
    
    print("\nCityScale:")
    print("  generate_labels.py: (int(n[1]), int(n[0]))")
    print("  dataset.py:         v[:, ::-1]")
    print("  → CONSISTENT ✅ (both: x=col, y=row)")
    
    # Xian 的 generate_labels.py:
    #   graph.add_edge((int(n[1]), int(n[0])), (int(nei[1]), int(nei[0])))
    #   即: x = col, y = row (简单交换, 和 cityscale 一样)
    #
    # 但 Xian 的 dataset.py (原始版):
    #   coord_transform = lambda v : v[:, ::-1]
    #   即: x = v[:, 1] = col, y = v[:, 0] = row
    #
    # 这两者是一致的! ✅
    # 
    # 但问题在于：Xian 数据的 pickle 中坐标系到底是什么？
    # 如果 pickle 中是 (row, col)，即左上原点：
    #   swap 后 → (col, row) = (x, y)，这是正确的图像坐标
    # 如果 pickle 中是 (x, y) 已经是左上原点：
    #   swap 后 → (y, x)，这是错的
    
    print("\nXian:")
    print("  generate_labels.py: (int(n[1]), int(n[0]))")
    print("  dataset.py:         v[:, ::-1]")
    print("  → CONSISTENT (both: x=col, y=row)")
    print("  ⚠️  BUT: need to verify if pickle coords are (row, col) or (x, y)")
    
    # SpaceNet 的 pickle 中坐标是 (row, col)，即左下原点 (数学坐标系)
    # 因为 coord[0] (row) 从 0 到 400, coord[1] (col) 从 0 到 400
    # 但 row=0 对应图片底部，所以需要 IMAGE_SIZE - row 来翻转
    #
    # CityScale 的 pickle 中坐标是 (row, col)，即左上原点 (图像坐标系)
    # row=0 对应图片顶部，所以不需要翻转，只需要 swap 就行
    #
    # Xian 的 pickle 中坐标是什么？
    # 让我们通过可视化来验证！


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SAM-Road Dataset Visualization & Verification')
    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'cityscale', 'spacenet', 'xian'],
                        help='Which dataset to visualize (default: all)')
    args = parser.parse_args()
    
    print("SAM-Road Dataset Visualization & Verification")
    print("=" * 60)
    
    if args.dataset in ('all', 'cityscale'):
        # 0. CityScale 可视化 (新增)
        visualize_cityscale()
    
    if args.dataset in ('all', 'spacenet'):
        # 1. 详细坐标分析（4种变换对比）
        detailed_coord_analysis()
        
        # 2. SpaceNet 可视化
        visualize_spacenet()
    
    if args.dataset in ('all', 'xian'):
        # 3. Xian 可视化
        visualize_xian()
    
    if args.dataset == 'all':
        # 4. 一致性检查
        verify_generate_labels_consistency()
    
    print(f"\n✅ All visualizations saved to: {OUTPUT_DIR}")
    print("Please check the output images to verify coordinate alignment.")
