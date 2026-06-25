"""
Partial 采样策略可视化 (多图 × 多 seed)
=========================================
把完整GT + 3种采样方案(按边随机/按连通块保/BFS生长)画成网格:
  - 每个数据集选 3 张不同密度的图
  - 每个策略测 3 个 seed (42/123/777)
人工判断哪种 partial 形态最接近真实地图, 以及策略对 seed 的稳定性.

用法:
  conda run -n samroad python tools/viz_partial_sampling.py
输出:
  docs/imgs/partial_sampling_didi_xian.png
  docs/imgs/partial_sampling_spacenet.png
"""
import os, sys, pickle, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- 采样策略 ----------
def sample_by_edge_random(G, keep_ratio, seed=42):
    """当前策略: 按边随机删"""
    rng = random.Random(seed)
    edges = list(G.edges())
    keep_num = int(len(edges) * keep_ratio)
    kept = rng.sample(edges, min(keep_num, len(edges)))
    H = nx.Graph()
    H.add_edges_from(kept)  # 只加保留的边涉及的节点, 不加孤立节点
    return H

def sample_by_road_segment(G, keep_ratio, seed=42):
    """按路段删: 路段=边(路口间), 随机删整条路段. 与按边随机在稠密图上等价,
    但语义上是'删整条路段而非拆碎路段'."""
    return sample_by_edge_random(G, keep_ratio, seed)  # 路段级=边级, 同实现

def sample_by_component(G, keep_ratio, seed=42):
    """按连通块保: 小块按概率整块留/删 + 大块BFS生长补足.
    形态: 几个完整路段(小块按keep_ratio概率保留) + 大路段缺一部分 = 真实地图.
    小块也有概率被整块抹去(整块留或整块删, 不拆碎), 块数会减少."""
    rng = random.Random(seed)
    target = int(G.number_of_edges() * keep_ratio)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    kept_edges = []
    threshold = max(1, target * 0.3)
    small = [c for c in comps if G.subgraph(c).number_of_edges() <= threshold]
    big = [c for c in comps if G.subgraph(c).number_of_edges() > threshold]
    # 小块按 keep_ratio 概率整块保留 (整块留或整块删, 不拆碎)
    for c in small:
        if rng.random() < keep_ratio:
            kept_edges.extend(G.subgraph(c).edges())
    remaining = target - len(kept_edges)
    for c in big:
        if remaining <= 0:
            break
        sub = G.subgraph(c)
        need = min(remaining, int(sub.number_of_edges() * keep_ratio))
        nodes = list(c); rng.shuffle(nodes)
        visited = {nodes[0]}; queue = [nodes[0]]; tree = []
        while queue and len(tree) < need:
            node = queue.pop(0)
            for nb in sub.neighbors(node):
                if nb not in visited:
                    visited.add(nb); tree.append((node, nb)); queue.append(nb)
                    if len(tree) >= need: break
        kept_edges.extend(tree); remaining -= len(tree)
    H = nx.Graph()
    H.add_edges_from(kept_edges)
    return H

def sample_by_bfs_grow(G, keep_ratio, seed=42):
    """BFS生长: 从最大块随机种子BFS生长到keep_ratio, 单连通."""
    rng = random.Random(seed)
    target = int(G.number_of_edges() * keep_ratio)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    kept_edges = []
    for comp in comps:
        if len(kept_edges) >= target:
            break
        need = target - len(kept_edges)
        sub = G.subgraph(comp)
        nodes = list(comp); rng.shuffle(nodes)
        visited = {nodes[0]}; queue = [nodes[0]]; tree = []
        while queue and len(tree) < need:
            node = queue.pop(0)
            for nb in sub.neighbors(node):
                if nb not in visited:
                    visited.add(nb); tree.append((node, nb)); queue.append(nb)
                    if len(tree) >= need: break
        kept_edges.extend(tree)
    H = nx.Graph()
    H.add_edges_from(kept_edges)
    return H

# ---------- 可视化 ----------
def draw_graph(ax, G, title, color='blue'):
    """画路网图, 节点坐标=node本身的坐标(tuple)."""
    pos = {n: (n[0], n[1]) for n in G.nodes()}
    # 注意: pickle坐标是(row,col), 画图时x=col=n[1], y=row=n[0], 但为直观用 (n[0],n[1]) 也行
    # 这里统一用 (x=col, y=row) 即 (n[1], -n[0]) 让y轴向下符合图像习惯
    pos = {n: (n[1], -n[0]) for n in G.nodes()}
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=color, width=1.2, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color, node_size=8, alpha=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.axis('off')

def viz_dataset(name, split_path, full_tmpl, partial_tmpl, out_path):
    with open(split_path) as f: ds = json.load(f)
    test = ds['test']

    # 选 3 张图: 优先多块(展示连通块保持), 兼顾密度差异
    candidates = []
    for img in test[:80]:
        try:
            full_adj = pickle.load(open(full_tmpl.format(img), 'rb'))
        except:
            continue
        G = nx.Graph()
        for u, nb in full_adj.items():
            for v in nb: G.add_edge(u, v)
        if G.number_of_edges() > 0:
            candidates.append((img, G, G.number_of_edges(), nx.number_connected_components(G)))
    # 优先选多块(>=3)的图, 按块数降序; 不足则补单块
    multi = [c for c in candidates if c[3] >= 3]
    single = [c for c in candidates if c[3] < 3]
    multi.sort(key=lambda x: -x[3])  # 块多的优先
    # 取最多块的2张 + 1张中等密度单块
    picks = []
    for c in multi[:2]:
        picks.append(c)
    if len(picks) < 3:
        # 补一张中等密度的
        single.sort(key=lambda x: x[2])
        if single:
            picks.append(single[len(single)//2])
    picks = picks[:3]
    # 转成 (img, G, ne) 格式
    picks = [(p[0], p[1], p[2]) for p in picks]

    seeds = [42, 123, 777]
    keep = 0.5
    strategies = [
        ('按边随机(当前)', sample_by_edge_random, 'red'),
        ('按连通块保', sample_by_component, 'green'),
        ('BFS生长', sample_by_bfs_grow, 'blue'),
    ]

    # 网格: 行=3张图, 列=完整GT + 3策略×3seed = 10列
    n_rows = len(picks)
    n_cols = 1 + len(strategies) * len(seeds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, (img, G, ne) in enumerate(picks):
        # 第一列: 完整GT
        draw_graph(axes[r, 0], G, f'完整GT\n{ne}边{nx.number_connected_components(G)}块', 'black')
        axes[r, 0].set_ylabel(f'region_{img}', fontsize=9)
        # 后续列: 每策略×每seed
        c = 1
        for sname, sfn, color in strategies:
            for seed in seeds:
                H = sfn(G, keep, seed)
                title = f'{sname}\nseed{seed}\n{H.number_of_edges()}边{nx.number_connected_components(H)}块'
                draw_graph(axes[r, c], H, title, color)
                c += 1

    fig.suptitle(f'{name}  partial采样策略对比  (keep_ratio={keep}, 行=不同图, 列=策略×seed)', fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'✓ {name} → {out_path}  ({n_rows}图 × {n_cols}列)')
    # 打印块数汇总
    print(f'  3张图边数: {[p[2] for p in picks]}')
    for img, G, ne in picks:
        print(f'  region_{img} (GT {ne}边{nx.number_connected_components(G)}块):')
        for sname, sfn, _ in strategies:
            blocks = [nx.number_connected_components(sfn(G, keep, s)) for s in seeds]
            print(f'    {sname}: seed42/123/777 块数 = {blocks}')

if __name__ == '__main__':
    viz_dataset('didi_xian',
        'datasets/didi/xian/data_split.json',
        'datasets/didi/xian/2019_400/region_{}_refine_gt_graph.p',
        'datasets/didi/xian/2019_400/region_{}_refine_gt_graph_partial.p',
        'docs/imgs/partial_sampling_didi_xian.png')
    viz_dataset('spacenet',
        'datasets/spacenet/data_split.json',
        'datasets/spacenet/RGB_1.0_meter/{}__gt_graph.p',
        'datasets/spacenet/RGB_1.0_meter/{}__gt_graph_partial.p',
        'docs/imgs/partial_sampling_spacenet.png')
