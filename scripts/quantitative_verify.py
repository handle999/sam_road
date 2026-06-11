"""
SAM-Road 定量坐标系验证脚本
================================
用 IoU (交并比) 和像素准确率定量判断哪种坐标变换与 GT.png 更对齐。
不生成图片, 仅在控制台输出数值结果。

使用方法:
  cd /home/hanhaoyu/workspace/research/sam_road
  conda activate samroad
  python scripts/quantitative_verify.py

输出说明 (仅控制台输出, 不生成图片):
================================================

  1. CITYSCALE - Quantitative Verification
     对每个 CityScale 样本, 计算 3 种变换生成的 road_mask 与 GT.png 的 IoU 和 PixelAcc:
       - swap (cityscale):  交换行列, 适用于左上原点 → CityScale下IoU较高 ✅
       - spacenet (flip-y): 交换+翻转y轴, 适用于左下原点 → IoU极低 ❌
       - raw (no transform): 直接用原始坐标 → IoU极低 ❌

  2. SPACE NET - Quantitative Verification
     对每个 SpaceNet 样本, 计算 3 种变换生成的 road_mask 与 GT.png 的 IoU 和 PixelAcc:
       - swap (cityscale):  交换行列, 适用于左上原点 → SpaceNet下IoU极低(~0.03) ❌
       - spacenet (flip-y): 交换+翻转y轴, 适用于左下原点 → IoU较高(~0.62) ✅
       - raw (no transform): 直接用原始坐标 → IoU极低(~0.03) ❌

  3. XIAN - Quantitative Verification
     对每个 Xian 样本, 同样计算 3 种变换的 IoU:
       - swap (cityscale):  IoU较高(~0.61) ✅
       - spacenet (flip-y): IoU较低(~0.10-0.22) ❌
       - raw (no transform): IoU极低(~0.01) ❌

  4. COORDINATE DISTRIBUTION ANALYSIS
     打印多个样本的 pickle 坐标 dim0/dim1 统计 (min, max, mean):
       - SpaceNet dim0 均值偏大 → 坐标从底部开始 (左下原点)
       - Xian dim0 均值分布不固定 → 无法仅靠均值判断, 需IoU验证

  核心结论:
    SpaceNet → coord_transform = lambda v: np.stack([v[:,1], 400-v[:,0]], axis=1)  (flip-y)
    Xian     → coord_transform = lambda v: v[:, ::-1]                                (swap)
    CityScale→ coord_transform = lambda v: v[:, ::-1]                                (swap)
    SpaceNet → coord_transform = lambda v: np.stack([v[:,1], 400-v[:,0]], axis=1)  (flip-y)
    Xian     → coord_transform = lambda v: v[:, ::-1]                                (swap)
"""
import os, pickle
import numpy as np
import cv2
import networkx as nx

def draw_graph_mask(gt_graph, image_size, coord_transform):
    """生成 road mask"""
    graph = nx.Graph()
    for n, neis in gt_graph.items():
        for nei in neis:
            n_arr = np.array([[n[0], n[1]]], dtype=np.float64)
            nei_arr = np.array([[nei[0], nei[1]]], dtype=np.float64)
            tn = coord_transform(n_arr)[0]
            tnei = coord_transform(nei_arr)[0]
            graph.add_edge((int(tn[0]), int(tn[1])), (int(tnei[0]), int(tnei[1])))
    
    road_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    for (x1, y1), (x2, y2) in graph.edges():
        cv2.line(road_mask, (x1, y1), (x2, y2), 255, 3)
    return road_mask

def compute_iou(mask1, mask2):
    """计算两个 mask 的 IoU"""
    m1 = mask1 > 127
    m2 = mask2 > 127
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def compute_pixel_accuracy(mask1, mask2):
    """计算像素级准确率"""
    m1 = mask1 > 127
    m2 = mask2 > 127
    total = m1.size
    correct = (m1 == m2).sum()
    return correct / total


# ============================================================
# CityScale 验证
# ============================================================
print("="*60)
print("CITYSCALE - Quantitative Verification")
print("="*60)

CS_DIR = '/home/hanhaoyu/workspace/research/sam_road/datasets/cityscale/20cities'
CS_IMAGE_SIZE = 2048

cs_transforms = {
    'swap (cityscale)': lambda v: v[:, ::-1],
    'spacenet (flip-y)': lambda v: np.stack([v[:, 1], CS_IMAGE_SIZE - v[:, 0]], axis=1),
    'raw (no transform)': lambda v: v,
}

for sample_id in ['0', '10', '166']:
    gt_graph_path = os.path.join(CS_DIR, f'region_{sample_id}_refine_gt_graph.p')
    gt_png_path = os.path.join(CS_DIR, f'region_{sample_id}_gt.png')
    
    if not os.path.exists(gt_graph_path):
        continue
    
    with open(gt_graph_path, 'rb') as f:
        gt = pickle.load(f)
    
    gt_png = cv2.imread(gt_png_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_png_path) else None
    
    print(f"\nregion_{sample_id}:")
    for name, tf in cs_transforms.items():
        road_mask = draw_graph_mask(gt, CS_IMAGE_SIZE, tf)
        if gt_png is not None:
            iou = compute_iou(road_mask, gt_png)
            acc = compute_pixel_accuracy(road_mask, gt_png)
            print(f"  {name:30s} IoU={iou:.4f}  PixelAcc={acc:.4f}")
        else:
            nonzero_ratio = (road_mask > 0).mean()
            print(f"  {name:30s} NonZeroRatio={nonzero_ratio:.4f}")


# ============================================================
# SpaceNet 验证
# ============================================================
print("\n" + "="*60)
print("SPACE NET - Quantitative Verification")
print("="*60)

SP_DIR = '/home/hanhaoyu/workspace/research/sam_road/datasets/spacenet/RGB_1.0_meter'
IMAGE_SIZE = 400

transforms = {
    'swap (cityscale)': lambda v: v[:, ::-1],
    'spacenet (flip-y)': lambda v: np.stack([v[:, 1], IMAGE_SIZE - v[:, 0]], axis=1),
    'raw (no transform)': lambda v: v,
}

for sample in ['AOI_2_Vegas_210', 'AOI_2_Vegas_5']:
    gt_graph_path = os.path.join(SP_DIR, f'{sample}__gt_graph.p')
    gt_png_path = os.path.join(SP_DIR, f'{sample}__gt.png')
    
    if not os.path.exists(gt_graph_path) or not os.path.exists(gt_png_path):
        continue
    
    with open(gt_graph_path, 'rb') as f:
        gt = pickle.load(f)
    gt_png = cv2.imread(gt_png_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"\n{sample}:")
    for name, tf in transforms.items():
        road_mask = draw_graph_mask(gt, IMAGE_SIZE, tf)
        iou = compute_iou(road_mask, gt_png)
        acc = compute_pixel_accuracy(road_mask, gt_png)
        print(f"  {name:30s} IoU={iou:.4f}  PixelAcc={acc:.4f}")


# ============================================================
# Xian 验证
# ============================================================
print("\n" + "="*60)
print("XIAN - Quantitative Verification")
print("="*60)

XIAN_DIR = '/home/hanhaoyu/workspace/research/sam_road/datasets/didi/xian/2019_400/xian_2019_400'

for sample_id in ['0', '1', '2']:
    gt_graph_path = os.path.join(XIAN_DIR, f'region_{sample_id}_refine_gt_graph.p')
    gt_png_path = os.path.join(XIAN_DIR, f'region_{sample_id}_gt.png')
    
    if not os.path.exists(gt_graph_path):
        continue
    
    with open(gt_graph_path, 'rb') as f:
        gt = pickle.load(f)
    
    gt_png = cv2.imread(gt_png_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_png_path) else None
    
    print(f"\nregion_{sample_id}:")
    for name, tf in transforms.items():
        road_mask = draw_graph_mask(gt, IMAGE_SIZE, tf)
        if gt_png is not None:
            iou = compute_iou(road_mask, gt_png)
            acc = compute_pixel_accuracy(road_mask, gt_png)
            print(f"  {name:30s} IoU={iou:.4f}  PixelAcc={acc:.4f}")
        else:
            # 没有GT.png, 看看mask中非零像素比例
            nonzero_ratio = (road_mask > 0).mean()
            print(f"  {name:30s} NonZeroRatio={nonzero_ratio:.4f}")

# ============================================================
# 额外: SpaceNet 的坐标分布验证
# ============================================================
print("\n" + "="*60)
print("COORDINATE DISTRIBUTION ANALYSIS")
print("="*60)

# SpaceNet
with open(os.path.join(SP_DIR, 'AOI_2_Vegas_210__gt_graph.p'), 'rb') as f:
    gt_sp = pickle.load(f)
coords_sp = np.array(list(gt_sp.keys()), dtype=np.float64)

# Xian
with open(os.path.join(XIAN_DIR, 'region_0_refine_gt_graph.p'), 'rb') as f:
    gt_xi = pickle.load(f)
coords_xi = np.array(list(gt_xi.keys()), dtype=np.float64)

print("\nSpaceNet AOI_2_Vegas_210:")
print(f"  dim0 (row? y?): min={coords_sp[:,0].min():.1f}, max={coords_sp[:,0].max():.1f}, mean={coords_sp[:,0].mean():.1f}")
print(f"  dim1 (col? x?): min={coords_sp[:,1].min():.1f}, max={coords_sp[:,1].max():.1f}, mean={coords_sp[:,1].mean():.1f}")

print("\nXian region_0:")
print(f"  dim0 (row? y?): min={coords_xi[:,0].min():.1f}, max={coords_xi[:,0].max():.1f}, mean={coords_xi[:,0].mean():.1f}")
print(f"  dim1 (col? x?): min={coords_xi[:,1].min():.1f}, max={coords_xi[:,1].max():.1f}, mean={coords_xi[:,1].mean():.1f}")

# 判断标准: 如果 dim0 的均值偏大(接近 IMAGE_SIZE)，说明坐标系是从底部开始的
# 如果 dim0 的均值偏小(接近 0)，说明坐标系是从顶部开始的
# SpaceNet: dim0 mean = 321，明显偏大 → 左下原点（dim0=y轴，从底部开始）
# Xian: dim0 mean = ?

# 看看更多样本的 dim0 均值
print("\n--- Checking dim0 mean across multiple Xian samples ---")
for sample_id in ['0', '1', '2', '3', '4', '5']:
    gt_graph_path = os.path.join(XIAN_DIR, f'region_{sample_id}_refine_gt_graph.p')
    if os.path.exists(gt_graph_path):
        with open(gt_graph_path, 'rb') as f:
            gt = pickle.load(f)
        coords = np.array(list(gt.keys()), dtype=np.float64)
        if len(coords) > 0:
            print(f"  region_{sample_id}: dim0 mean={coords[:,0].mean():.1f}, dim1 mean={coords[:,1].mean():.1f}, nodes={len(coords)}")

print("\n--- Checking dim0 mean across multiple SpaceNet samples ---")
for sample in ['AOI_2_Vegas_210', 'AOI_2_Vegas_5', 'AOI_3_Paris_200', 'AOI_4_Shanghai_200']:
    gt_graph_path = os.path.join(SP_DIR, f'{sample}__gt_graph.p')
    if os.path.exists(gt_graph_path):
        with open(gt_graph_path, 'rb') as f:
            gt = pickle.load(f)
        coords = np.array(list(gt.keys()), dtype=np.float64)
        if len(coords) > 0:
            print(f"  {sample}: dim0 mean={coords[:,0].mean():.1f}, dim1 mean={coords[:,1].mean():.1f}, nodes={len(coords)}")
