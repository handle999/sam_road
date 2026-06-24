"""
Visualize graph_gt.pickle over satellite image.

Usage:
    python visualize_coord.py --dataset [cityscale|didi_xian] --id [region_id]
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def load_img(path):
    return np.array(Image.open(path))


def load_graph(graph_path):
    g = pickle.load(open(graph_path, 'rb'))
    nodes = np.array(list(g.keys()))
    edges = []
    for n, neibs in g.items():
        for nei in neibs:
            edges.append((n, nei))
    return nodes, edges, g


def visualize_cityscale(region_id):
    sat_path = f'datasets/cityscale/20cities/region_{region_id}_sat.png'
    gt_path = f'datasets/cityscale/20cities/region_{region_id}_graph_gt.pickle'
    img = load_img(sat_path)
    nodes, edges, _ = load_graph(gt_path)
    fig, axes = plt.subplots(1,1, figsize=(8,8))
    axes.imshow(img)
    axes.scatter(nodes[:, 1], nodes[:, 0], s=1, c='red', zorder=2)
    for n1, n2 in edges:
        axes.plot([n1[1], n2[1]], [n1[0], n2[0]], c='cyan', lw=0.5)
    axes.set_title(f'cityscale region_{region_id} (row,col)')
    plt.tight_layout()
    plt.savefig(f'viz_cityscale_region_{region_id}.png', dpi=60, bbox_inches='tight')
    print(f'Saved viz_cityscale_region_{region_id}.png')
    plt.close()

def visualize_didi_xian(region_id):
    sat_path = f"datasets/didi/xian/2019_400/region_{region_id}_sat.png"
    gt_path = f"datasets/didi/xian/2019_400/region_{region_id}_graph_gt.pickle"
    img = load_img(sat_path)
    img_size = img.shape[0]
    nodes, edges, _ = load_graph(gt_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sub-1: As-is (0=Y_up, 1=X)
    axes[0].imshow(img)
    axes[0].scatter(nodes[:, 1], nodes[:, 0], s=2, c='red', zorder=2)
    for n1, n2 in edges:
        axes[0].plot([n1[1], n2[1]], [n1[0], n2[0]], c='cyan', lw=0.5)
    axes[0].set_title('As-is (0=Y_up, 1=X) ≤ bottom-left')
    
    # Sub-2: Y-flip (to image row-col)
    axes[1].imshow(img)
    nodes_flip = np.stack([nodes[:, 1], img_size - nodes[:, 0]], axis=1)
    axes[1].scatter(nodes_flip[:, 0], nodes_flip[:, 1], s=2, c='red', zorder=2)
    for n1, n2 in edges:
        n1f = (img_size - n1[0], n1[1])
        n2f = (img_size - n2[0], n2[1])
        axes[1].plot([n1f[0], n2f[0]], [n1f[1], n2f[1]], c='cyan', lw=0.5)
    axes[1].set_title('Y-flip ≤ top-left (row,col)')
    
    # Sub-3: Swap (0=col, 1=row)
    axes[2].imshow(img)
    nodes_swap = np.stack([nodes[:, 1], nodes[:, 0]], axis=1)
    axes[2].scatter(nodes_swap[:, 1], nodes_swap[:, 0], s=2, c='red', zorder=2)
    for n1, n2 in edges:
        n1sw = [n1[1], n1[0]]
        n2sw = [n2[1], n2[0]]
        axes[2].plot([n1sw[1], n2sw[1]], [n1sw[0], n2sw[0]], c='cyan', lw=0.5)
    axes[2].set_title('Swap (0=col, 1=row) ≤ NOT standard')
    
    plt.tight_layout()
    plt.savefig(f'viz_didi_xian_region_{region_id}.png', dpi=80, bbox_inches='tight')
    print(f'Saved viz_didi_xian_region_{region_id}.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize graph_gt coordinate system')
    parser.add_argument('--dataset', type=str, required=True, choices=['cityscale', 'didi_xian'])
    parser.add_argument('--id', type=int, default=0, help='Region ID')
    args = parser.parse_args()
    if args.dataset == 'cityscale':
        visualize_cityscale(args.id)
    elif args.dataset == 'didi_xian':
        visualize_didi_xian(args.id)
