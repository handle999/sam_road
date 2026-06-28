"""
可视化 xian 黑边 region 裁剪前后对比
====================================
把 rn graph (node+edge) 画在 sat img 上, 裁剪前(.bak) vs 裁剪后(.p) 并排对比.
所有黑边 region 输出到一张大图, 方便一次性查看.

用法:
  conda run -n samroad python tools/viz_clip_compare.py
输出:
  docs/imgs/xian_clip_compare.png
"""
import os, sys, pickle, json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

BASE = 'datasets/didi/xian/2019_400'
SPLIT = json.load(open('datasets/didi/xian/data_split.json'))
ALL_IDS = sorted(set(SPLIT['train'] + SPLIT['validation'] + SPLIT['test']))


def is_black_edge(sat, thresh=15):
    """检测是否有黑边"""
    black = sat.max(axis=2) < thresh
    r16 = black[:, -16:].mean()
    b16 = black[-16:, :].mean()
    return r16 > 0.9 or b16 > 0.9


def draw_graph_on_sat(sat, adj, node_color=(0, 0, 255), edge_color=(0, 255, 255), node_radius=3, edge_thick=2):
    """把 rn graph (node+edge) 画在 sat img 上.
    adj 的 key 是 (row, col), 画图用 (x=col, y=row)."""
    img = sat.copy()
    drawn = set()
    # 先画 edge
    for node, neighbors in adj.items():
        for nb in neighbors:
            edge = tuple(sorted((node, nb)))
            if edge in drawn:
                continue
            drawn.add(edge)
            p0 = (int(node[1]), int(node[0]))
            p1 = (int(nb[1]), int(nb[0]))
            cv2.line(img, p0, p1, edge_color, edge_thick)
    # 再画 node (覆盖 edge 上)
    for node in adj.keys():
        p = (int(node[1]), int(node[0]))
        cv2.circle(img, p, node_radius, node_color, -1)
    return img


# 找所有黑边 region
black_edge_ids = []
for img_id in ALL_IDS:
    sat_path = f'{BASE}/region_{img_id}_sat.png'
    if not os.path.exists(sat_path):
        continue
    sat = cv2.imread(sat_path)
    if sat is None:
        continue
    if is_black_edge(sat):
        black_edge_ids.append(img_id)

print(f'找到 {len(black_edge_ids)} 个黑边 region: {black_edge_ids}')

# 可视化: 每个 region 3列 (sat原图 | 裁剪前rn | 裁剪后rn), 太多则分批
COLS = 3  # sat, before, after
ROWS = len(black_edge_ids)
BATCH = 12  # 每张图最多12行, 避免太大

batches = [black_edge_ids[i:i+BATCH] for i in range(0, len(black_edge_ids), BATCH)]

for bi, batch_ids in enumerate(batches):
    rows = len(batch_ids)
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 4, rows * 4))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for r, img_id in enumerate(batch_ids):
        sat = cv2.imread(f'{BASE}/region_{img_id}_sat.png')[:, :, ::-1]  # BGR->RGB

        # 裁剪前 (.bak)
        bak_path = f'{BASE}/region_{img_id}_refine_gt_graph.p.bak'
        adj_before = pickle.load(open(bak_path, 'rb')) if os.path.exists(bak_path) else {}

        # 裁剪后 (.p)
        adj_after = pickle.load(open(f'{BASE}/region_{img_id}_refine_gt_graph.p', 'rb'))

        # 画
        sat_only = sat.copy()
        before_img = draw_graph_on_sat(sat, adj_before) if adj_before else sat.copy()
        after_img = draw_graph_on_sat(sat, adj_after) if adj_after else sat.copy()

        axes[r, 0].imshow(sat_only)
        axes[r, 0].set_title(f'region_{img_id} sat', fontsize=9)
        axes[r, 0].axis('off')

        n_before = len(adj_before)
        n_after = len(adj_after)
        axes[r, 1].imshow(before_img)
        axes[r, 1].set_title(f'裁剪前 ({n_before}节点)', fontsize=9, color='red')
        axes[r, 1].axis('off')

        axes[r, 2].imshow(after_img)
        axes[r, 2].set_title(f'裁剪后 ({n_after}节点)', fontsize=9, color='green')
        axes[r, 2].axis('off')

    plt.suptitle(f'xian 黑边 region 裁剪对比 (batch {bi+1}/{len(batches)})\n黄线=edge, 红点=node', fontsize=12)
    plt.tight_layout()
    out = f'docs/imgs/xian_clip_compare_batch{bi+1}.png'
    os.makedirs('docs/imgs', exist_ok=True)
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'✓ {out} ({rows} region)')

print(f'\n共 {len(batches)} 张图, 涵盖 {len(black_edge_ids)} 个黑边 region')
