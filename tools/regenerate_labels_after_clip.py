"""
裁剪后重新生成 xian 的所有 PNG 产物
====================================
从裁剪后的 refine_gt_graph.p 重新生成:
1. road_mask / keypoint_mask (generate_labels.py)
2. gt.png (从 graph 渲染, thickness=2, 对齐 prepare_dataset)
3. partial.p / partial.png (generate_partial_prior.py)
"""
import os, sys, pickle, argparse
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def render_gt_png(adj_dict, image_size=400, thickness=2):
    """从 graph 邻接表渲染 gt.png (和 graphVis2048Segmentation 同思路, thickness=2).
    坐标: adj_dict 的 key 是 (row, col), 画图用 (x=col, y=row)."""
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    drawn = set()
    for node, neighbors in adj_dict.items():
        for nb in neighbors:
            edge = tuple(sorted((node, nb)))
            if edge in drawn:
                continue
            drawn.add(edge)
            # (row, col) -> (x=col, y=row) for cv2
            p0 = (int(node[1]), int(node[0]))
            p1 = (int(nb[1]), int(nb[0]))
            cv2.line(img, p0, p1, 255, thickness)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True, help='xian 2019_400 目录')
    ap.add_argument('--dataset', default='didi', choices=['didi', 'spacenet'])
    args = ap.parse_args()

    input_dir = args.input_dir
    import json
    if args.dataset == 'didi':
        split_path = os.path.join(input_dir, '..', 'data_split.json')
        sat_tmpl = 'region_{}_sat.png'
        graph_tmpl = 'region_{}_refine_gt_graph.p'
        gt_tmpl = 'region_{}_gt.png'
    else:
        split_path = os.path.join(input_dir, 'data_split.json')
        sat_tmpl = '{}__rgb.png'
        graph_tmpl = '{}__gt_graph.p'
        gt_tmpl = '{}__gt.png'

    split = json.load(open(split_path))
    all_ids = sorted(set(split['train'] + split['validation'] + split['test']))

    print(f'重新生成 gt.png ({len(all_ids)} 个 region, thickness=2)...')
    count = 0
    for img_id in all_ids:
        graph_path = os.path.join(input_dir, graph_tmpl.format(img_id))
        gt_path = os.path.join(input_dir, gt_tmpl.format(img_id))
        if not os.path.exists(graph_path):
            continue
        adj = pickle.load(open(graph_path, 'rb'))
        if len(adj) == 0:
            # 空 graph, gt.png 全黑
            cv2.imwrite(gt_path, np.zeros((400, 400), dtype=np.uint8))
            count += 1
            continue
        gt_img = render_gt_png(adj, 400, 2)
        cv2.imwrite(gt_path, gt_img)
        count += 1
    print(f'✓ gt.png 重新生成: {count} 个')


if __name__ == '__main__':
    main()
