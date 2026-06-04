# SAM-Road 路网补全方案B设计文档

## 一、背景与目标

### 1.1 原始 SAM-Road 模型

原始 SAM-Road 是一个两阶段的端到端路网提取模型：

```
输入: 卫星影像 [H, W, 3]

Stage 1 (SAM):
  - ImageEncoderViT: 视觉编码
  - MapDecoder: 分割头
  - 输出: 道路 mask + 交叉点 mask

Stage 2 (TopoNet):
  - 从 mask 提取图节点
  - 对候选边进行连接预测
  - 输出: 完整路网图 (节点 + 边)
```

### 1.2 当前方案 (model_copy) 的局限

- **4通道输入**：RGB + 二值 Prior (道路 mask)
- **问题**：Prior 丢失了图的拓扑结构信息
  - 只能表达"哪些像素是道路"
  - 无法表达"哪些点相连"、"图的连通性"
- **实际需求**：路网补全任务
  - 输入：卫星影像 + 不完整图 (部分边已存在)
  - 输出：预测需要添加的边

### 1.3 方案B目标

保留两阶段设计，但让 TopoNet 能够感知**已有图的拓扑结构**：

```
输入: 卫星影像 [H, W, 3] + 已有图 G = (V, E)

Stage 1 (SAM): 不变
  - 输出: 候选节点 V_new (从 mask 提取)

Stage 2 (TopoNet 改进):
  - 输入: 图像特征 + 图结构信息
  - 任务: 预测 V_known ∪ V_new 之间的边
  - 输出: 需要添加的边集合
```

---

## 二、数据集需求与构造

### 2.1 原始数据集格式

每个数据集需要以下文件：

| 文件类型 | 说明 | 格式 |
|---------|------|------|
| 卫星影像 | RGB 图像 | PNG, 400×400 或 2048×2048 |
| GT 道路图 | 二值道路 mask | PNG |
| GT 图 | 邻接字典 | Pickle: `{node: [neighbors]}` |

### 2.2 路网补全数据集构造

从完整 GT 图生成训练样本：

```python
def generate_completion_samples(full_graph, keep_ratio=0.5):
    """
    从完整图生成补全训练样本

    Args:
        full_graph: 完整 GT 图 (邻接字典)
        keep_ratio: 保留边的比例 (0.3~0.7 常用)

    Returns:
        known_graph: 保留的部分边 (不完整图，作为输入)
        positive_edges: 被删除的边 (正样本)
        negative_edges: 不应该相连的点对 (负样本)
    """
    # 1. 获取所有边
    all_edges = set()
    for node, neighbors in full_graph.items():
        for nei in neighbors:
            # 无向图，去重
            edge = tuple(sorted((node, nei)))
            all_edges.add(edge)

    all_edges = list(all_edges)

    # 2. 随机保留部分边
    keep_num = int(len(all_edges) * keep_ratio)
    kept_edges = random.sample(all_edges, keep_num)
    kept_edge_set = set(kept_edges)

    # 3. 正样本: 被删除的边
    positive_edges = all_edges - kept_edge_set

    # 4. 负样本: 空间邻近但图中不相连的点对
    #    策略: 对每个节点，找空间邻近的点，排除已有的和被删除的
    negative_edges = set()
    all_nodes = list(full_graph.keys())

    for node in all_nodes:
        # 找空间邻近的点 (KDTree)
        nearby = find_nearby_nodes(node, all_nodes, radius=64)
        for neighbor in nearby:
            edge = tuple(sorted((node, neighbor)))
            if edge not in kept_edge_set and edge not in positive_edges:
                negative_edges.add(edge)

        # 限制负样本数量 (保持正负比 1:3 ~ 1:5)
        if len(negative_edges) > len(positive_edges) * 5:
            break

    return kept_edges, list(positive_edges), list(negative_edges)
```

### 2.3 数据集目录结构

```
dataset/
├── images/                    # 卫星影像
│   ├── region_0_sat.png
│   └── ...
├── gt_graph/                  # 完整 GT 图
│   ├── region_0_refine_gt_graph.p
│   └── ...
├── processed/                 # 训练标签 (keypoint/road mask)
│   ├── keypoint_mask_0.png
│   └── road_mask_0.png
├── split.json                 # 数据划分
└── completion/                # 补全任务专用
    ├── train_keep_0.5/        # 保留50%边的训练样本
    │   ├── region_0/
    │   │   ├── known_graph.p  # 已知边
    │   │   ├── candidates.p   # 候选边
    │   │   └── labels.p       # 标签
    └── ...
```

---

## 三、模型架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         路网补全模型 (两阶段)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                    卫星影像 [H, W, 3]
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 1: SAM (视觉编码 + 分割) - 保持不变                                    │
│                                                                             │
│  ImageEncoderViT (ViT-B)                                                   │
│    输入: [B, 3, 256, 256]                                                  │
│    输出: [B, 256, 16, 16]  (image_features)                                │
│                                                                             │
│  MapDecoder                                                                │
│    输入: [B, 256, 16, 16]                                                  │
│    输出: [B, 2, 256, 256]  (keypoint_mask, road_mask)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
  mask_scores        image_features      graph_points
  (道路/交叉点)       (用于拓扑预测)       (从mask提取的节点)
        │                 │                 │
        │    ┌────────────┴────────────┐    │
        │    │     图结构信息          │    │
        │    │  - 已知节点 V_known     │    │
        │    │  - 已知边 E_known       │    │
        │    │  - 节点度数             │    │
        │    │  - 邻接表               │    │
        │    └─────────────────────────┘    │
        │                 │                 │
        ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 2: TopoNet (改进 - 融合图结构)                                       │
│                                                                             │
│  输入:                                                                     │
│    1. image_features: [B, 256, 16, 16]  - SAM 提取的图像特征               │
│    2. graph_struct: 图结构信息 (已知图)                                    │
│    3. candidate_edges: [B, N_candidates, 2] - 待预测的候选边               │
│                                                                             │
│  处理流程:                                                                 │
│    1. 采样特征: 从 image_features 采样候选边端点的视觉特征                 │
│    2. 融合图结构: 为每个端点融入图结构特征 (邻居度、路径信息)              │
│    3. Transformer: 学习图结构 + 视觉特征的交互                            │
│    4. 输出: edge_scores [B, N_candidates, 1] - 连接概率                    │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
              预测的边集合 E_pred
              输出: 完整图 G = (V_known ∪ V_new, E_known ∪ E_pred)
```

### 3.2 TopoNet 改进细节

#### 3.2.1 输入设计

```python
class TopoNetForCompletion(nn.Module):
    """
    用于路网补全的 TopoNet
    """

    def forward(self, image_features, graph_struct, candidate_edges):
        """
        Args:
            image_features: [B, D, h, w] - SAM 图像特征
            graph_struct: dict 包含:
                - known_node_indices: 已知节点的索引
                - known_adjacency: 已知图的邻接表
                - node_degrees: 节点度数
            candidate_edges: [B, N_candidates, 2] - 候选边 (端点索引)

        Returns:
            edge_scores: [B, N_candidates, 1] - 每条边的连接概率
        """
        # Step 1: 采样端点视觉特征
        #   - 对每条候选边，采样 src 和 tgt 的特征
        #   - point_features: [B, N_candidates*2, D]
        point_features = self.bilinear_sample(image_features, candidate_edges)

        # Step 2: 融入图结构特征
        #   - 为每个端点计算图结构特征
        #   - 特征: 节点度数、是否在已知图中、与已知节点的关系
        graph_aware_features = self.enhance_with_graph_structure(
            point_features, graph_struct, candidate_edges
        )

        # Step 3: 构造边特征
        #   - 融合: 端点特征 + 相对位置 + 图结构差异
        edge_features = self.construct_edge_features(graph_aware_features)

        # Step 4: Transformer 编码
        #   - 学习局部图结构
        #   - 输入: [B*N_patches, N_edges, D]
        edge_features = self.transformer_encoder(edge_features)

        # Step 5: 输出连接概率
        edge_scores = self.output_proj(edge_features)

        return torch.sigmoid(edge_scores)
```

#### 3.2.2 图结构特征设计

```python
def enhance_with_graph_structure(point_features, graph_struct, candidate_edges):
    """
    为端点特征融入图结构信息

    图结构特征包括:
    1. known_flag: 是否在已知图中 (0/1)
    2. degree: 节点度数 (已知的邻居数)
    3. is_endpoint: 是否是断头路的端点 (关键的补全目标)
    4. neighbor_overlap: 与候选边另一端的邻居重叠度
    """

    # 1. 已知节点标记
    known_mask = torch.zeros_like(point_features[:, :, 0])
    known_mask[:, graph_struct['known_node_indices']] = 1.0

    # 2. 节点度数特征
    degree_features = torch.zeros_like(point_features[:, :, 0])
    for idx, node_idx in enumerate(graph_struct['known_node_indices']):
        degree_features[:, idx] = graph_struct['node_degrees'][node_idx]

    # 3. 断头路端点特征 (度数=1的节点，更可能需要补全)
    dead_end_features = (degree_features == 1.0).float()

    # 4. 融合
    enhanced = torch.cat([
        point_features,
        known_mask.unsqueeze(-1),
        degree_features.unsqueeze(-1) / 10.0,  # 归一化
        dead_end_features.unsqueeze(-1)
    ], dim=-1)

    return enhanced
```

---

## 四、训练数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         训练阶段数据流                                       │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: 数据准备 (CompletionDataset)
─────────────────────────────────────────────────────────────────────────────
  输入: 完整 GT 图

  处理:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 1. 随机删除部分边 (keep_ratio ~ 0.5)                                    │
  │    → 已知图: E_known                                                    │
  │    → 缺失边: E_positive (正样本)                                        │
  │                                                                         │
  │ 2. 采样负样本 (空间邻近但不相连的点对)                                   │
  │    → E_negative                                                         │
  │                                                                         │
  │ 3. 从完整图提取所有节点 V = V_known ∪ V_missing                         │
  │    → NMS + 连通域分析 → V_new (从影像可提取的新节点)                    │
  │                                                                         │
  │ 4. 构造候选边:                                                          │
  │    - E_known 内部的边 (可能被遗漏的)                                    │
  │    - V_known ↔ V_missing (从已知到缺失)                                 │
  │    - V_new 内部的边 (原版任务)                                          │
  │                                                                         │
  │ 5. 标签:                                                                │
  │    - 在 E_positive 中 → 1 (应该添加)                                    │
  │    - 在 E_negative 中 → 0 (不应添加)                                    │
  └─────────────────────────────────────────────────────────────────────────┘

Step 2: 模型前向
─────────────────────────────────────────────────────────────────────────────
  输入:
    rgb:              [B, 256, 256, 3]        影像 patch
    known_nodes:     [B, N_known, 2]          已知图节点
    known_edges:     [B, N_known_edges, 2]    已知图边
    candidate_edges: [B, N_candidates, 2]     候选边
    edge_labels:     [B, N_candidates]        标签 (1=应添加, 0=不添加)

  Stage 1: SAM (不变)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ rgb → normalize → image_encoder → map_decoder                         │
  │                                                                     │
  │ 输出: mask_scores [B, 256, 256, 2] + image_features [B, 256, 16, 16] │
  └─────────────────────────────────────────────────────────────────────────┘

  Stage 2: TopoNet 改进
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 1. 从 mask 提取图节点: graph_points                                    │
  │    → 融合已知节点 + 新提取的节点                                        │
  │                                                                         │
  │ 2. 采样点特征: bilinear_sample(image_features, graph_points)          │
  │                                                                         │
  │ 3. 融入图结构: 构造 enhanced_point_features                            │
  │    → 加入 known_flag, degree, dead_end 等特征                         │
  │                                                                         │
  │ 4. 构造边特征: 对每条候选边融合两端点特征                               │
  │                                                                         │
  │ 5. Transformer: 学习图结构交互                                         │
  │                                                                         │
  │ 6. 输出: edge_scores [B, N_candidates, 1]                              │
  └─────────────────────────────────────────────────────────────────────────┘

Step 3: Loss 计算
─────────────────────────────────────────────────────────────────────────────
  # 分割 Loss (保持不变)
  mask_loss = BCE(keypoint_pred, keypoint_gt) + BCE(road_pred, road_gt)

  # 边预测 Loss (改进)
  #   - 正样本: 缺失的边应该被预测为 1
  #   - 负样本: 不该连的边应该被预测为 0
  edge_loss = BCE(edge_pred, edge_label)

  total_loss = mask_loss + edge_loss
```

---

## 五、推理数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         推理阶段数据流                                       │
└─────────────────────────────────────────────────────────────────────────────┘

输入:
  - 卫星影像 [H, W, 3]
  - 已有图 G_known = (V_known, E_known)  ← 用户提供

Stage 1: SAM (不变)
─────────────────────────────────────────────────────────────────────────────
  1. 滑窗切分: 影像 → patches
  2. 批量推理: SAM 编码 + 解码
  3. 聚合: 像素级平均 → fused_masks
  4. 提取节点: V_new = extract_points(fused_masks)

Stage 2: 边预测 (改进)
─────────────────────────────────────────────────────────────────────────────
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 1. 构造候选边集合:                                                      │
  │    - V_known 内部的边 (可能遗漏的)                                      │
  │    - V_known ↔ V_new (从已知到新节点)                                   │
  │    - V_new 内部的边                                                     │
  │                                                                         │
  │ 2. 采样特征:                                                            │
  │    - 从 image_features 采样端点视觉特征                                 │
  │    - 融入图结构特征                                                     │
  │                                                                         │
  │ 3. TopoNet 预测: edge_scores                                           │
  │                                                                         │
  │ 4. 后处理:                                                              │
  │    - 阈值过滤: score > threshold → 添加边                              │
  │    - 连通性检查: 确保不形成环或断裂                                      │
  │                                                                         │
  │ 5. 输出完整图: G_complete = (V_known ∪ V_new, E_known ∪ E_pred)        │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## 六、与原版的对比

| 对比项 | 原版 SAM-Road | 方案B (路网补全) |
|--------|--------------|-----------------|
| **输入** | 卫星影像 | 卫星影像 + 已有图 |
| **任务** | 从零提取路网 | 补全不完整路网 |
| **Stage 1** | 不变 | 不变 |
| **Stage 2** | 只处理新节点之间 | 处理已知节点 + 新节点 |
| **TopoNet 输入** | 只有图像特征 | 图像特征 + 图结构 |
| **训练数据** | 完整 GT 图 | 从完整图删除部分边 |
| **推理输出** | 全新图 | 在输入图基础上添加边 |

---

## 七、需要修改/新建的文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `dataset_completion.py` | 新建 | 路网补全数据集类 |
| `model_completion.py` | 新建 | 修改 TopoNet 支持图结构 |
| `inferencer_completion.py` | 新建 | 推理时加载已有图 |
| `train_completion.py` | 新建 | 训练脚本 |
| `graph_utils.py` | 修改 | 添加图结构特征提取函数 |

---

## 八、数据集构造脚本示例

```python
# generate_completion_data.py
import os
import pickle
import json
import random
import numpy as np
import networkx as nx

def generate_completion_dataset(
    image_dir,
    gt_graph_dir,
    output_dir,
    keep_ratios=[0.3, 0.5, 0.7]
):
    """
    生成路网补全训练数据

    Args:
        image_dir: 原始卫星影像目录
        gt_graph_dir: GT 图目录 (pickle)
        output_dir: 输出目录
        keep_ratios: 保留边的比例列表
    """

    os.makedirs(output_dir, exist_ok=True)

    # 加载 GT 图列表
    gt_files = sorted([f for f in os.listdir(gt_graph_dir) if f.endswith('.p')])

    for keep_ratio in keep_ratios:
        ratio_dir = os.path.join(output_dir, f'keep_{int(keep_ratio*100)}')
        os.makedirs(ratio_dir, exist_ok=True)

        for gt_file in gt_files:
            region_id = gt_file.replace('_refine_gt_graph.p', '')

            # 加载完整图
            full_graph_path = os.path.join(gt_graph_dir, gt_file)
            with open(full_graph_path, 'rb') as f:
                full_adj = pickle.load(f)

            # 转换为 networkx 图
            G_full = nx.Graph(full_adj)

            # 随机删除边
            all_edges = list(G_full.edges())
            keep_num = int(len(all_edges) * keep_ratio)
            kept_edges = random.sample(all_edges, keep_num)

            G_known = nx.Graph()
            G_known.add_edges_from(kept_edges)

            # 正样本: 被删除的边
            missing_edges = set(all_edges) - set(kept_edges)

            # 负样本: 空间邻近但不相连的点对
            negative_edges = set()
            nodes = list(G_known.nodes())
            for node in nodes:
                # 找邻近点
                neighbors = [n for n in nodes
                            if abs(n[0]-node[0]) + abs(n[1]-node[1]) < 64]
                for neighbor in neighbors:
                    if not G_known.has_edge(node, neighbor):
                        edge = tuple(sorted((node, neighbor)))
                        if edge not in missing_edges:
                            negative_edges.add(edge)

                    if len(negative_edges) > len(missing_edges) * 3:
                        break
                if len(negative_edges) > len(missing_edges) * 3:
                    break

            negative_edges = list(negative_edges)[:len(missing_edges) * 3]

            # 保存
            sample = {
                'known_edges': list(G_known.edges()),
                'positive_edges': list(missing_edges),
                'negative_edges': negative_edges,
            }

            output_path = os.path.join(ratio_dir, f'{region_id}.p')
            with open(output_path, 'wb') as f:
                pickle.dump(sample, f)

    print(f"Generated completion data in {output_dir}")

if __name__ == '__main__':
    # 示例: 为 SpaceNet 生成补全数据
    generate_completion_dataset(
        image_dir='datasets/spacenet/RGB_1.0_meter',
        gt_graph_dir='datasets/spacenet/RGB_1.0_meter',
        output_dir='spacenet/completion',
        keep_ratios=[0.3, 0.5, 0.7]
    )
```

---

## 九、训练命令示例

```bash
# 训练路网补全模型
python train_completion.py \
    --config config/toponet_vitb_256_spacenet_completion.yaml \
    --ckpt_path ./checkpoints/samroad_completion

# 推理 (补全任务)
python inferencer_completion.py \
    --config config/toponet_vitb_256_spacenet_completion.yaml \
    --checkpoint ./checkpoints/samroad_completion/epoch=10.ckpt \
    --input_image ./test/sat.png \
    --input_graph ./test/partial_graph.p \
    --output_dir ./test/completion_result
```

---

## 十、配置文件示例

```yaml
# config/toponet_vitb_256_spacenet_completion.yaml

DATASET: 'spacenet_completion'

# SAM 配置
NO_SAM: False
SAM_VERSION: 'vit_b'
SAM_CKPT_PATH: 'sam_ckpts/sam_vit_b_01ec64.pth'
PATCH_SIZE: 256
BATCH_SIZE: 64
TRAIN_EPOCHS: 30
BASE_LR: 0.001

# TOPONET 配置 (改进)
TOPO_SAMPLE_NUM: 128
TOPONET_VERSION: 'completion'  # 新增: 标识使用补全版本
GRAPH_FEATURE_DIM: 16          # 新增: 图结构特征维度

# 补全任务专用
KEEP_RATIO: 0.5                # 训练时保留边的比例
MAX_NEGATIVE_RATIO: 3.0        # 正负样本比例上限

# 推理配置
INFER_BATCH_SIZE: 64
INFER_PATCHES_PER_EDGE: 16

# 阈值 (可能需要调优)
ITSC_THRESHOLD: 0.2
ROAD_THRESHOLD: 0.34
TOPO_THRESHOLD: 0.5            # 补全任务的阈值可能需要更低
COMPLETION_THRESHOLD: 0.3     # 新增: 边预测阈值
```

---

## 十一、总结

方案B保持了 SAM-Road 的两阶段设计，但**将图结构信息融入 TopoNet**：

1. **Stage 1 (SAM)**：保持不变，提取视觉特征和初始节点
2. **Stage 2 (TopoNet 改进)**：
   - 输入增加已知图的拓扑结构
   - 预测目标从"新节点之间"扩展到"已知↔新 + 已知↔已知"
3. **训练数据**：从完整图随机删除边来模拟不完整输入
4. **推理**：接受用户提供的已有图，输出补全后的图

这样既利用了 SAM-Road 成熟的视觉编码能力，又赋予了模型图结构感知能力，实现路网补全任务。