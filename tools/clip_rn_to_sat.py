"""
裁剪 rn graph 对齐 sat 黑边 (xian 数据集)
==========================================
问题: xian 部分 region 的 sat 有 no-data 黑边, 但 rn graph 在黑边区仍有路.
做法: 检测 sat 有效区(非黑边), 用 cohen-sutherland 把 rn graph 裁到有效区内,
      裁断处补交点 node (与 graph2RegionCoordinate 同思路, 避免悬空边).

用法:
  python tools/clip_rn_to_sat.py --dataset didi --input_dir datasets/didi/xian/2019_400 --dry-run
  python tools/clip_rn_to_sat.py --dataset didi --input_dir datasets/didi/xian/2019_400
"""
import os, sys, pickle, argparse, shutil
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def detect_valid_bbox(sat, black_thresh=15):
    """检测 sat 的有效(非黑边)区域, 返回 (x_min, y_min, x_max, y_max) 像素边界.
    黑边判定: sat.max(axis=2) < black_thresh (整条边缘全黑)."""
    black = sat.max(axis=2) < black_thresh
    H, W = black.shape
    # 逐行/列检测: 找非黑边的范围
    # 行: 从上往下找第一个非全黑行, 从下往上找第一个非全黑行
    row_has_data = ~black.all(axis=1)  # 该行有非黑像素
    col_has_data = ~black.all(axis=0)
    if not row_has_data.any() or not col_has_data.any():
        return None  # 全黑
    y_min = np.argmax(row_has_data)
    y_max = H - 1 - np.argmax(row_has_data[::-1])
    x_min = np.argmax(col_has_data)
    x_max = W - 1 - np.argmax(col_has_data[::-1])
    return int(x_min), int(y_min), int(x_max), int(y_max)


def cohen_sutherland_clip(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    """Cohen-Sutherland 裁剪线段到矩形 (xmin,ymin,xmax,ymax) 内.
    返回 None(全外) 或 ((nx1,ny1),(nx2,ny2)) 裁后线段."""
    INSIDE = 0; LEFT = 1; RIGHT = 2; BOTTOM = 4; TOP = 8

    def code(x, y):
        c = INSIDE
        if x < xmin: c |= LEFT
        elif x > xmax: c |= RIGHT
        if y < ymin: c |= BOTTOM
        elif y > ymax: c |= TOP
        return c

    c1 = code(x1, y1)
    c2 = code(x2, y2)
    while True:
        if c1 == 0 and c2 == 0:
            return ((x1, y1), (x2, y2))  # 全在内
        if c1 & c2:
            return None  # 全外
        c_out = c1 if c1 else c2
        if c_out & TOP:
            x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
            y = ymax
        elif c_out & BOTTOM:
            x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
            y = ymin
        elif c_out & RIGHT:
            y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
            x = xmax
        elif c_out & LEFT:
            y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
            x = xmin
        if c_out == c1:
            x1, y1, c1 = x, y, code(x, y)
        else:
            x2, y2, c2 = x, y, code(x, y)


def clip_graph_to_bbox(adj_dict, xmin, ymin, xmax, ymax):
    """把 rn graph (邻接表, 坐标 (row,col)) 裁剪到有效 bbox 内.
    裁断处补交点 node (与 graph2RegionCoordinate 同思路).
    注意: 坐标是 (row,col), bbox 的 x=col, y=row."""
    new_adj = {}
    processed = set()
    for node, neighbors in adj_dict.items():
        r0, c0 = node[0], node[1]
        for nb in neighbors:
            edge_id = tuple(sorted((node, nb)))
            if edge_id in processed:
                continue
            processed.add(edge_id)
            r1, c1 = nb[0], nb[1]
            # cohen-sutherland 用 (x=col, y=row)
            clipped = cohen_sutherland_clip(c0, r0, c1, r1, xmin, ymin, xmax, ymax)
            if clipped is None:
                continue  # 边全在黑边区, 删
            (nc0, nr0), (nc1, nr1) = clipped
            # 新 node (row, col) — 裁断处补 node
            n1 = (float(nr0), float(nc0))
            n2 = (float(nr1), float(nc1))
            new_adj.setdefault(n1, []).append(n2)
            new_adj.setdefault(n2, []).append(n1)
    return new_adj


def process_region(input_dir, img_id, sat_name_tmpl, graph_name_tmpl, dry_run=True):
    """处理单个 region: 检测黑边 → 裁 rn graph → 保存."""
    sat_path = os.path.join(input_dir, sat_name_tmpl.format(img_id))
    graph_path = os.path.join(input_dir, graph_name_tmpl.format(img_id))
    if not os.path.exists(sat_path) or not os.path.exists(graph_path):
        return None
    sat = cv2.imread(sat_path)
    if sat is None:
        return None
    adj = pickle.load(open(graph_path, 'rb'))
    if len(adj) == 0:
        return None

    bbox = detect_valid_bbox(sat)
    if bbox is None:
        print(f'  region_{img_id}: sat 全黑, 跳过')
        return None

    xmin, ymin, xmax, ymax = bbox
    H, W = sat.shape[:2]
    # 如果有效区 = 全图, 无黑边, 不用裁
    if xmin == 0 and ymin == 0 and xmax == W - 1 and ymax == H - 1:
        return ('no_black_edge', len(adj))

    # 裁剪
    new_adj = clip_graph_to_bbox(adj, xmin, ymin, xmax, ymax)
    old_edges = sum(len(v) for v in adj.values()) // 2
    new_edges = sum(len(v) for v in new_adj.values()) // 2

    if dry_run:
        return ('clipped', old_edges, new_edges, bbox)

    # 备份原 .p, 保存裁后
    backup = graph_path + '.bak'
    if not os.path.exists(backup):
        shutil.copy(graph_path, backup)
    with open(graph_path, 'wb') as f:
        pickle.dump(new_adj, f)
    return ('clipped', old_edges, new_edges, bbox)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['didi', 'spacenet'])
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--dry-run', action='store_true', help='只预览不写')
    args = ap.parse_args()

    if args.dataset == 'didi':
        sat_tmpl = 'region_{}_sat.png'
        graph_tmpl = 'region_{}_refine_gt_graph.p'
        # 评测GT: graph_gt.pickle (APLS/TOPO 评测用, 也需裁剪对齐)
        eval_gt_tmpl = 'region_{}_graph_gt.pickle'
    else:
        sat_tmpl = '{}__rgb.png'
        graph_tmpl = '{}__gt_graph.p'
        eval_gt_tmpl = None  # spacenet 评测GT和训练GT同一个文件

    # 找所有 region id
    import json
    if args.dataset == 'didi':
        split = json.load(open(os.path.join(args.input_dir, '..', 'data_split.json')))
    else:
        split = json.load(open(os.path.join(args.input_dir, 'data_split.json')))
    all_ids = sorted(set(split['train'] + split['validation'] + split['test']))

    print(f'Processing {len(all_ids)} regions (dry_run={args.dry_run})...')
    clipped = 0
    no_black = 0
    for img_id in all_ids:
        # 1. 裁训练GT (refine_gt_graph.p)
        result = process_region(args.input_dir, img_id, sat_tmpl, graph_tmpl, args.dry_run)
        if result is None:
            continue
        if result[0] == 'no_black_edge':
            no_black += 1
        elif result[0] == 'clipped':
            clipped += 1
            _, old_e, new_e, bbox = result
            if args.dry_run:
                print(f'  region_{img_id}: bbox={bbox} refine edges {old_e}→{new_e} (删{old_e-new_e})')

        # 2. 裁评测GT (graph_gt.pickle, didi_xian 才有, 评测用)
        if eval_gt_tmpl is not None:
            result2 = process_region(args.input_dir, img_id, sat_tmpl, eval_gt_tmpl, args.dry_run)
            if result2 is not None and result2[0] == 'clipped' and args.dry_run:
                _, old_e2, new_e2, _ = result2
                print(f'  region_{img_id}: graph_gt edges {old_e2}→{new_e2} (删{old_e2-new_e2})')

    print(f'\n汇总: {clipped} 个 region 有黑边需裁剪, {no_black} 个无黑边')
    if args.dry_run:
        print('(dry-run, 未写文件. 去掉 --dry-run 执行裁剪)')
    else:
        print('✓ 裁剪完成 (refine_gt_graph.p + graph_gt.pickle), 原文件备份为 .bak')
        print('下一步: 重新生成 road_mask/gt.png/partial (跑 generate_labels + generate_partial_prior)')


if __name__ == '__main__':
    main()
