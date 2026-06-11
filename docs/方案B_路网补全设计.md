# SAM-Road Completion v2：路网补全模型设计文档

> 本文档基于对现有 `sam_road_completion.py` 的深度分析，结合任务设定和数据集实际情况，重新梳理的完整设计方案。

---

## 一、任务定义

### 1.1 核心目标

提出一个路网补全（Road Network Completion）模型，实现三个目标：

| # | 目标 | 场景 | 数据集 | 难度 |
|---|------|------|--------|------|
| (1) | 提出 Completion 模型，兼容有/无轨迹 | 通用 | 全部 | 框架设计 |
| (2) | 无 traj 时超越 SOTA extraction | img + 已知路网 | SpaceNet, CityScale | 对标 SAM-Road |
| (3) | 有 traj 时效果更好 | img + traj + 已知路网 | Xian | 验证 traj 价值 |

### 1.2 Extraction vs Completion

| | Extraction (SAM-Road 原始任务) | Completion (本方案) |
|---|---|---|
| **输入** | 仅卫星影像 | 卫星影像 + 部分已知路网 + (可选)轨迹 |
| **输出** | 完整路网图 | 在已知路网基础上补全缺失部分 |
| **搜索空间** | 全图所有可能的边 | 仅缺失区域的候选边 |
| **先验信息** | 无 | 已知路网的几何与拓扑 |

**关键洞察**：Completion 模型在"已知路网=空"时应退化为 Extraction，保证不劣于 SAM-Road 基线。

### 1.3 已知路网的统一来源

**所有数据集的训练时已知路网，统一用随机采样 GT 图。**

| 数据集 | 已知路网来源 | 说明 |
|--------|-------------|------|
| SpaceNet | GT 随机删边 | 无其他选择，只有 GT |
| CityScale | GT 随机删边 | 同上 |
| Xian | GT 随机删边 | **不用 active_graph**，traj 仅通过热力图注入 |

**为什么 Xian 不用 active_graph**：
1. `active_graph` 的节点坐标和 GT 不精确匹配（同一空间位置的坐标有小数差异，精确匹配率仅 ~21%）
2. `active_graph` 是 traj 做 map matching 的中间产物，其图结构和 GT 不同（节点位置、边划分都不同）
3. 统一用随机删 GT 保证三个数据集的训练逻辑完全一致，便于公平对比

---

## 二、数据集分析

### 2.1 三个数据集的统一视角

| | SpaceNet | CityScale | Xian (DiDi) |
|---|---|---|---|
| **img** | ✅ 400×400 sat.png | ✅ 2048×2048 sat.png | ✅ 400×400 sat.png |
| **GT路网** | ✅ refine_gt_graph.p | ✅ refine_gt_graph.p | ✅ refine_gt_graph.p |
| **GT来源** | OSM | OSM | OSM |
| **traj原始数据** | ❌ 无 | ❌ 无 | ✅ .shp (map matched linestring) |
| **traj渲染产物** | - | - | active.png (二值mask) |
| **pickle坐标原点** | 左下（数学坐标） | 左上（图像坐标） | 左上（图像坐标） |
| **coord_transform** | swap + flip-y | swap | swap |
| **样本数** | 2549 | 180 | ~562 |
| **适用模型** | Completion (无traj) | Completion (无traj) | Completion (有traj) |

### 2.2 Xian 数据集覆盖情况

对 Xian 562 个 region 的 `active_graph` vs GT 覆盖率统计：

| 类别 | 数量 | 说明 |
|------|------|------|
| 完全覆盖 (ratio=1.0) | 171 | active ≈ GT，traj 覆盖了几乎所有路 |
| 部分覆盖 (0 < ratio < 1) | 281 | 典型的补全场景，traj 只覆盖了部分路 |
| 完全为空 (ratio=0) | 61 | 无轨迹数据，需要纯视觉推断 |

**均值覆盖率**：76%（edge_ratio），说明大部分区域已有相当多的路网信息。

**注意**：虽然 `active_graph` 不作为模型输入，但 `active.png`（traj 渲染的热力图 mask）是可以作为轨迹模态使用的。

### 2.3 traj 数据的生成链路

```
原始出租车 GPS 轨迹
  → map matching (匹配到 OSM 路网)
  → 生成 .shp 文件 (linestring, 每条线=一次匹配结果)
  → download_use_osm.py 处理:
      ├── 渲染为 active.png (二值mask, 可用作热力图)
      └── 组织为 active_graph.pickle (邻接表, 本方案不使用)
```

---

## 三、现有 Completion 模型的问题分析

### 3.1 代码实现回顾

当前 `models/sam_road_completion.py` 包含以下新增组件：

| 组件 | 作用 | 问题 |
|------|------|------|
| `RoadGraphEncoder` (CNN) | 编码已知路网的4通道几何特征图 | 4通道中2个大概率无效 |
| `FeatureFusion` (1×1 Conv) | 融合视觉特征+路网特征 | 设计合理 |
| `RoadGraphGNN` (GAT-like) | 编码已知路网拓扑结构 | **edge_index 是空占位符，GNN 没有工作** |
| `TopoNetCompletion` | 扩展 pair_proj 输入维度 | 设计合理，但依赖 GNN 输出 |

### 3.2 关键 Bug：known_edge_index 为空

```python
# data/dataset_completion.py, __getitem__ 中:
known_edge_index = torch.zeros(2, 0, dtype=torch.long)  # ← 永远是空的！
```

这意味着 `RoadGraphGNN` 在训练时**从未接收到真实的已知边信息**，要么：
- `edge_index.shape[2] > 0` 但全是零索引 → GNN 在错误的位置做消息传递
- `edge_index.shape[2] == 0` → GNN 的 `attn_mask=None` → 退化为全连接注意力

无论哪种情况，GNN 都没有学到"已知路网中谁和谁已连通"这个最重要的信息。

### 3.3 road_feature_map 4通道问题

| 通道 | 内容 | 有效性分析 |
|------|------|-----------|
| ch0: 已知道路 mask | ✅ 有效 | "哪里有路"的强先验，CNN 无法从影像 100% 确定 |
| ch1: 距离场 | ❌ 大概率无效 | (1) 图节点在道路上，采样处值≈0；(2) CNN 可从 ch0 自行推断距离；(3) 隐含"需要补全"假设 |
| ch2: 方向场 | ❌ 大概率无效 | (1) 角度循环性未处理(0°≈360°但数值差大)；(2) 多方向叠加污染；(3) CNN 可从 ch0 推断梯度方向 |
| ch3: 已知节点位置 | ✅ 有效 | 区分已知/未知节点，指导 TopoNet 注意力分配 |

**结论**：精简为 2 通道（ch0=道路mask, ch1=节点位置），减少参数和误导信号。

### 3.4 已知部分不改的保证问题

**当前模型不能保证已知图部分不被修改。**

- 训练时：已知边 `valid=False` 不计入 topo_loss，但模型从未被教过"已知边 score=1"
- 推理时：候选边来自 mask 提取的关键点，和训练时的 GT 节点体系完全不同
- GNN 的 edge_index 对不上推理时的节点索引

**正确做法**：推理后处理中，对已知图的边**强制 `topo_score = 1.0`**（硬覆盖）。模型的职责是"发现缺失的边"，不是"确认已有的边"。

### 3.5 keep_ratio=0.5 固定的问题

- 过高(0.9)：补全太简单，退化为"连断头路"
- 过低(0.1)：已知信息太少，等价于从零建图
- 固定0.5：缺乏对真实场景分布的模拟

**改进**：动态采样 `keep_ratio ~ U[0.2, 0.8]`，每个 epoch/每个 sample 随机取值。

### 3.6 _get_known_graph_adj 的一致性 Bug

`dataset_completion.py` 中渲染 road_feature_map 时的删边和 `CompletionGraphLabelGenerator` 中的删边是**独立随机**的。代码注释承认了这一点：

```python
# 简化处理: 用 keep_ratio 随机保留原始图的边
# 由于 CompletionGraphLabelGenerator 已经做了删边,
# 这里用相同的 keep_ratio 重新采样, 保证一致性
# ...
# 使用与 CompletionGraphLabelGenerator 相同 keep_ratio 的随机删边
# 由于两者独立随机, 可能不一致。这里接受这个近似
```

这意味着**渲染特征图用的路网和标签中的已知边可能不一致**，模型收到了矛盾的信号。

---

## 四、Completion v2 模型设计

### 4.1 整体架构

```
输入层 (3种模态，traj可选):
  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐
  │  RGB [3]    │   │ traj_heatmap │   │ known_graph       │
  │  (必须)      │   │ [1] (可选)   │   │ (GT随机删边)       │
  └──────┬──────┘   └──────┬───────┘   └──────┬────────────┘
         │                 │                   │
         ▼                 ▼                   │
  ┌──────────────────────────────┐             │
  │ concat → [4ch]               │             │
  │ (无traj时ch4=zeros)          │             │
  └──────────────┬───────────────┘             │
                 │                             │
                 ▼                             │
  ┌──────────────────────────────┐             │
  │ SAM ImageEncoder (ViT-B)     │             │
  │ + LoRA on qkv + patch_embed  │             │
  └──────────────┬───────────────┘             │
                 │                             │
         image_embeddings                       │
         [B, 256, h, w]                        │
                 │                             │
                 │                   ┌─────────┴──────────┐
                 │                   │ 渲染 road_feat_map  │
                 │                   │ [2, H, W]:          │
                 │                   │  ch0=已知路mask      │
                 │                   │  ch1=已知节点位置    │
                 │                   └─────────┬──────────┘
                 │                             │
                 │                   ┌─────────┴──────────┐
                 │                   │ RoadGraphEnc (CNN)  │
                 │                   │ 2→32→64→128→256    │
                 │                   └─────────┬──────────┘
                 │                             │
                 │                     road_embeddings      │
                 │                     [B, 256, h, w]       │
                 │                             │
                 ▼                             ▼
           ┌─────────────────────────────────────┐
           │ FeatureFusion                        │
           │ cat([img_emb, road_emb]) → Conv1x1  │
           │ → fused_features [B, 256, h, w]      │
           └──────────────┬──────────────────────┘
                          │
              ┌───────────┼───────────────┐
              ▼                           ▼
     ┌────────────────┐          ┌───────────────────┐
     │ MapDecoder      │          │ BilinearSampler    │
     │ (只用img_emb!)  │          │ (从fused_features  │
     │                │          │  采样节点特征)      │
     └───────┬────────┘          └─────────┬─────────┘
             │                             │
      mask_logits/scores          point_features
      [B,H,W,2]                   [B, N, 256]
             │                             │
             │                   ┌─────────┴───────────┐
             │                   │ 随机删边GT的边索引    │
             │                   │ → known_edge_index   │
             │                   │ [B, 2, E] (修复!)    │
             │                   └─────────┬───────────┘
             │                             │
             │                   ┌─────────┴───────────┐
             │                   │ RoadGraphGNN          │
             │                   │ (只让已知边两端互注意) │
             │                   └─────────┬───────────┘
             │                             │
             │                      graph_embeddings
             │                      [B, N, graph_dim]
             │                             │
             │                   ┌─────────┴───────────┐
             │                   │ TopoNetCompletion     │
             │                   │ pair_proj输入:        │
             │                   │ [src_feat; tgt_feat;  │
             │                   │  offset; src_gnn;     │
             │                   │  tgt_gnn]             │
             │                   └─────────┬───────────┘
             │                             │
             ▼                             ▼
      mask输出                      topo_scores
      [B,H,W,2]                    [B,S,P,1]
```

### 4.2 traj 的利用方式：A + B 双路径

#### 路径 A：traj → 热力图 → SAM 第4通道

```
traj原始数据 (.shp linestring / active.png)
  → 渲染到 [H, W] 二值热力图 (轨迹经过=1, 否则=0)
  → concat with RGB → [H, W, 4]
  → SAM 4ch Encoder (patch_embed Conv2d(4,...), 第4通道零初始化)
```

**无traj时**：第4通道全零，等价于3ch SAM。4ch模型的设计已保证这种退化——第4通道零初始化意味着初始时和3ch完全等价，训练后也不会对全零输入产生偏见。

**A路径的价值**：让 SAM 编码器在提取视觉特征时就知道"哪里有轨迹"，相当于告诉视觉系统"这些地方一定有路，请重点关注周边区域"。

**捷径风险分析与缓解**：

理论上，模型可能学到"第4通道有值 ≈ 道路"，直接将 traj_heatmap 上采样作为输出，跳过 RGB 视觉理解。但实际上这种风险可控：

1. **轨迹质量天然不完美**：traj 来自 map matching，存在 GPS 偏移、匹配偏差，渲染的 active.png 和 GT road_mask 不完全对齐（像素级有偏移、有缺失），模型无法简单地 `output = traj_heatmap`
2. **轨迹覆盖率不完整**：Xian 的 active.png 平均只覆盖 GT 的 76%，61 个区域覆盖率为 0，模型必须从 RGB 补全剩余部分
3. **先验增强策略**：沿用 4ch 模型的 dropout 策略作为额外保险：
   - 20% 概率：traj_heatmap 全黑（保持纯视觉能力）
   - 60% 概率：traj_heatmap 腐蚀（随机擦除块，打断捷径）
   - 20% 概率：完整 traj（学习信任先验）
4. **第4通道零初始化**：训练初始时 traj 通道无贡献，视觉通道必须先建立能力，traj 只在后期做增强

#### 路径 B：traj → 改善节点特征 → GNN

traj 不直接给 GNN，而是通过 A 路径间接改善节点特征质量：

```
traj → 热力图 → SAM编码器看到 → 更好的image_embeddings
  → BilinearSampler采样 → 更好的point_features
  → RoadGraphGNN → 更好的graph_embeddings
  → TopoNetCompletion → 更好的拓扑预测
```

**GNN 的 edge_index 始终来自随机删边的 GT**，和 traj 无关。这保证了有无 traj 时模型结构完全一致。

**Train/Infer 对齐分析**：

训练时的 known_graph（随机删边GT）和推理时的 partial-roadmap（用户提供）**概念上等价，但节点体系不同**：

| | 训练时 | 推理时 |
|---|---|---|
| 图节点来源 | GT 图的 subdivided 节点 | mask 提取的 NMS 关键点 |
| 节点坐标 | 精确 GT 坐标 | 有预测噪声 |
| known_edge_index | subdivided 图上的边索引，和 graph_points 在同一套索引体系 | 需要将已知路网边映射到 NMS 关键点索引 |
| road_feature_map | 像素级渲染，天然对齐 | 像素级渲染，天然对齐 |

**关键对齐手段**：

1. **road_feature_map 是跨节点体系的桥梁**：无论节点体系怎么变，road_feature_map 是像素级的，不受索引体系影响。CNN 编码后得到的 road_embeddings 在空间上对齐，BilinearSampler 按坐标采样自动对齐
2. **known_edge_index 需要最近邻映射**：
   - 训练时：在 subdivided GT 图上，已知边的 (src, tgt) 可以直接用 NMS 后节点索引表示
   - 推理时：已知路网的每条边 (A, B)，需要找到 A/B 在 NMS 关键点中最近邻的节点 idx_a/idx_b，如果距离 < 阈值则加入 known_edge_index
3. **图结构粒度差异**：训练时 subdivided 图把每条边细分为 4px 一段，推理时用户的 partial-roadmap 没有这种细分。但 GNN 只关注"谁和谁连"，不关注边的细分粒度，所以不影响

#### A + B 的分工

| 路径 | 作用 | traj的贡献 | 无traj时 |
|------|------|-----------|---------|
| A (4ch) | 让SAM看到"哪里有轨迹" → 更好的视觉特征 | 热力图作为先验 | ch4=zeros，退化为3ch |
| B (GNN) | 让TopoNet知道"哪些边已确认" → 更好的拓扑预测 | 间接（通过更好的point_features） | 同结构，特征质量略低 |
| road_feature_map | 让融合特征包含已知路网几何 | 无直接关系 | 同结构 |

### 4.3 退化逻辑（兼容性设计）

| 输入 | traj_heatmap | road_feature_map | known_edge_index | 等效于 |
|------|-------------|-----------------|-----------------|--------|
| 全空 | zeros | zeros | None | 原始 SAM-Road (extraction) |
| 随机删边 | zeros | GT子集mask+节点 | GT子集edges | 目标(2): Completion无traj |
| Xian active | active.png裁剪 | GT子集mask+节点 | GT子集edges | 目标(3): Completion有traj |

**核心设计原则：traj 是可选的增强信号，不是必需输入。**

### 4.4 模型组件详解

#### 4.4.1 SAM 4ch Encoder

与现有 `sam_road_4ch.py` 的设计一致：

| 改动 | 原始 (3ch) | 4ch |
|------|-----------|-----|
| 输入通道 | 3 (RGB) | 4 (RGB + traj_heatmap) |
| `pixel_mean/std` | `[3]` 维 | `[4]` 维 (第4通道 mean=0, std=1) |
| `patch_embed` | `Conv2d(3,...)` | `Conv2d(4,...)`, 第4通道零初始化 |
| LoRA 解冻 | 仅 qkv | qkv + `patch_embed` |

**零初始化保证**：第4通道权重初始化为0，初始时 `4ch_output = 3ch_output + 0 * traj`，和3ch完全等价。

#### 4.4.2 RoadGraphEncoder (CNN) — 2通道版

```
输入: road_feature_map [B, 2, H, W]  (H=W=PATCH_SIZE=256)

Conv2d(2→32, k=3, p=1) + BN + GELU     → [B, 32, 256, 256]
Conv2d(32→64, k=3, s=4, p=1) + BN + GELU → [B, 64, 64, 64]
Conv2d(64→128, k=3, s=4, p=1) + BN + GELU → [B, 128, 16, 16]
Conv2d(128→256, k=3, p=1)                → [B, 256, 16, 16]

输出: road_embeddings [B, 256, h, w]  与 image_embeddings 维度对齐
```

两步 stride=4 下采样，总下采样 16×，与 SAM ViT 的 patch_size=16 一致。

#### 4.4.3 FeatureFusion

```
输入: cat([image_embeddings, road_embeddings], dim=1) → [B, 512, h, w]
Conv2d(512→256, k=1) + GELU
Conv2d(256→256, k=1)
输出: fused_features [B, 256, h, w]
```

1×1 卷积做通道融合，本质是对每个空间位置的 256 维视觉特征 + 256 维路网特征做加权组合。

#### 4.4.4 MapDecoder — 与原版相同

```
输入: image_embeddings [B, 256, h, w]  ← 注意：不是 fused_features！
4层 ConvTranspose2d (stride=2) 逐步上采样:
  [B, 256, 16, 16] → [B, 128, 32, 32] → [B, 64, 64, 64] → [B, 32, 128, 128] → [B, 2, 256, 256]
输出: mask_logits [B, 2, H, W] → permute → [B, H, W, 2]
      ch0 = keypoint, ch1 = road
```

**设计要点**：MapDecoder 用纯 `image_embeddings` 而非 `fused_features`，防止已知路网信息泄露到分割分支导致过拟合。分割 mask 的目的是"从影像中发现路"，不应受已知路网影响。

#### 4.4.5 BilinearSampler

从 2D 特征图上按节点坐标做双线性插值采样：

```
输入: feature_maps [B, 256, 16, 16]  (特征图)
      sample_points [B, N, 2]          (节点坐标, 像素级)

坐标归一化: pixel / PATCH_SIZE * 2 - 1  → [-1, 1] (grid_sample要求)
grid_sample: 对每个坐标做双线性插值, 取出 256 维特征

输出: sampled_features [B, N, 256]  (每个节点的特征向量)
```

**为什么用双线性插值**：节点坐标有训练噪声（N(0,1)像素），双线性插值让特征连续可微，梯度能传播回坐标。最近邻采样不可微。

**和原始模型的区别**：原始 SAM-Road 从 `image_embeddings` 采样；Completion v2 从 `fused_features`（视觉+路网特征融合后）采样。

#### 4.4.6 RoadGraphGNN

```
输入:
  node_visual_features: [B, N, 256]  从 fused_features 采样的节点特征
  node_coords: [B, N, 2]             节点坐标
  edge_index: [B, 2, E]              已知路网边 (src, tgt) 索引

Step 1: node_proj [B, N, 256] → [B, N, graph_dim=32]
Step 2: coord_proj [B, N, 2] → [B, N, 32], 加到 x 上 (让GNN感知空间位置)
Step 3: 构造邻接注意力掩码 adj_mask [B, N, N]
        - 对角线 True (自环)
        - edge_index 中的边 True (双向, 无向图)
        - 其余 False → -inf 屏蔽
Step 4: 2层 MultiheadAttention (4 heads, graph_dim=32) + LayerNorm + 残差

输出: graph_embeddings [B, N, graph_dim=32]
```

**GNN 的核心价值**：直接编码"谁和谁已连通"。这种拓扑约束是距离场和方向场试图间接表达但表达不好的信息。

**退化设计**：`graph_embeddings=None` 时 TopoNet 退化为原版（补零），方便消融实验。

#### 4.4.7 TopoNetCompletion

与原版 TopoNet 的唯一区别在 `pair_proj` 输入维度：

| 版本 | pair_proj 输入 | 维度 |
|------|---------------|------|
| 原版 TopoNet | `[src_feat; tgt_feat; offset]` | 2×128 + 2 = 258 |
| TopoNetCompletion | `[src_feat; tgt_feat; offset; src_graph; tgt_graph]` | 2×128 + 2 + 2×32 = 322 |

处理流程：

```
Step 1: feature_proj [B, N, 256] → [B, N, 128]  降维
Step 2: 从 pairs 索引提取 src/tgt 特征
        src_features [B, S*P, 128], tgt_features [B, S*P, 128]
        offset = tgt_points - src_points [B, S*P, 2]
Step 3: 融入图拓扑嵌入
        if graph_embeddings is not None:
            src_graph, tgt_graph = 从 graph_embeddings 按 pairs 索引
            pair_features = cat([src_feat, tgt_feat, offset, src_graph, tgt_graph])
        else:
            pair_features = cat([src_feat, tgt_feat, offset, zeros])  # 退化
Step 4: pair_proj [B, S*P, 322] → [B, S*P, 128]
Step 5: Transformer (3层, 4头) 在每个 sample 内做注意力
        (padding_mask 屏蔽无效边)
Step 6: output_proj [B, S, P, 128] → [B, S, P, 1]
        logits → sigmoid → scores
```

---

## 五、训练设计

### 5.1 数据构造

#### SpaceNet / CityScale（无traj）

```
输入:
  rgb_patch [H,W,3]
  traj_heatmap [H,W] = zeros             ← 无traj
  known_graph = GT随机删边(keep_ratio ~ U[0.2, 0.8])
  road_feature_map [2,H,W] = render(known_graph)
    ch0: 已知道路 mask
    ch1: 已知节点位置
  known_edge_index [2,E] = known_graph的边映射到NMS节点索引

标签:
  mask: keypoint_mask + road_mask (完整GT)
  topo: 基于完整GT的BFS连通性
  valid: 已知边valid=False, 未知边valid=True
```

#### Xian（有traj）

```
输入:
  rgb_patch [H,W,3]
  traj_heatmap [H,W] = active.png裁剪    ← 有traj！唯一区别
  known_graph = GT随机删边(keep_ratio ~ U[0.2, 0.8])  ← 不用active_graph
  road_feature_map [2,H,W] = render(known_graph)
  known_edge_index [2,E] = known_graph的边映射到NMS节点索引

标签:
  同上 (完整GT)
```

### 5.2 损失函数

```python
# Mask 分割损失 (与原版相同)
gt_masks = stack([keypoint_mask, road_mask], dim=3)  # [B,H,W,2]
mask_loss = BCEWithLogitsLoss(mask_logits, gt_masks)  # 或 Focal Loss

# 拓扑损失 (关键: 已知边不计入)
topo_gt = batch['connected']          # [B,S,P]  候选边是否应该连通
valid = batch['valid']                # [B,S,P]  已知边为 False → 不参与
topo_loss = BCEWithLogitsLoss(logits, gt, reduction='none')
topo_loss *= valid.unsqueeze(-1)      # 屏蔽已知边
topo_loss = sum / valid.sum()         # 只对未知边求平均

total_loss = mask_loss + topo_loss
```

**已知边为什么不算 loss**：已知边已经确定存在，模型不需要重新预测。如果已知边也参与 loss，模型会学到"把已知边标为连通"这种平凡解，而不是学会"补全缺失的边"。

### 5.3 训练策略

#### 模态 Dropout

以一定概率（如 20%）将已知路网信息和 traj 热力图**全部清零**，强制模型保持纯视觉建图能力：

```python
if self.is_train and random.random() < 0.2:
    road_feature_map = zeros  # 清空已知路网
    traj_heatmap = zeros      # 清空轨迹
    known_edge_index = None   # 清空GNN边
```

这保证了 Completion 模型在无已知信息时不劣于 Extraction。

#### 动态 keep_ratio

```python
keep_ratio = random.uniform(0.2, 0.8)  # 每个sample随机
```

- 不固定在 0.5，增强对各种不完整程度的鲁棒性
- 低 keep_ratio ≈ extraction 场景
- 高 keep_ratio ≈ 小幅补全场景

#### 特征图与标签一致性

**必须修复的 Bug**：渲染 `road_feature_map` 时的删边和标签生成时的删边**必须一致**。解决方案：

```python
# CompletionGraphLabelGenerator 中:
# 保存 known_edge_set_subdivide (已有)
# 新增: 保存原始图级别的已知边集合
self.known_edges_original = kept_edges_original  # 原始GT图的已知边

# __getitem__ 中:
# 直接用 gen.known_edges_original 渲染 road_feature_map
# 而不是重新随机删边
```

---

## 六、推理流程

### 6.1 SpaceNet / CityScale（无traj）

```
Step 1: mask_scores, fused_features = infer_masks_and_img_features(rgb, zeros_heatmap, road_feat_map)
Step 2: 从mask提取关键点 → 构造graph_points, pairs
Step 3: 构造 known_edge_index (将已知路网边映射到NMS关键点, 详见6.4)
Step 4: topo_scores = infer_toponet(fused_features, graph_points, pairs, valid, known_edge_index)
Step 5: 后处理: 已知图边强制topo_score=1.0
```

### 6.2 Xian（有traj）

```
Step 1: mask_scores, fused_features = infer_masks_and_img_features(rgb, traj_heatmap, road_feat_map)
         ↑ 唯一区别：traj_heatmap 不是zeros
Step 2-5: 同上
```

### 6.3 已知图保护（后处理）

推理后处理中，对已知图的边**强制 topo_score = 1.0**（硬覆盖）：

```python
# 对于每条已知边 (src_idx, tgt_idx):
# 在候选边中找到对应的 pair
# 强制设置 topo_score = 1.0
```

这是**后处理层面的保证**，模型本身不负责确认已知边。

### 6.4 known_edge_index 的推理时构造

推理时，已知路网（用户提供或从 OSM 获取）的节点坐标和 mask 提取的 NMS 关键点不在同一套索引体系中。需要做映射：

```python
def map_known_edges_to_nms(known_graph, nms_keypoints, distance_threshold=8.0):
    """
    将已知路网的边映射到 NMS 关键点索引体系
    
    Args:
        known_graph: 已知路网的邻接表 {(x,y): [(x1,y1), ...]}
        nms_keypoints: NMS后的关键点坐标 [N, 2]
        distance_threshold: 最近邻匹配的最大距离阈值(像素)
    
    Returns:
        known_edge_index: [2, E] 映射后的边索引
    """
    # 1. 收集已知路网的所有节点
    known_nodes = set(known_graph.keys())
    for neighbors in known_graph.values():
        known_nodes.update(neighbors)
    known_nodes = list(known_nodes)
    
    # 2. 为每个已知节点找最近的 NMS 关键点
    nms_kdtree = KDTree(nms_keypoints)
    node_to_nms = {}
    for node in known_nodes:
        dist, idx = nms_kdtree.query(node)
        if dist < distance_threshold:
            node_to_nms[node] = idx
    
    # 3. 将已知边映射到 NMS 索引
    edges = []
    for src, neighbors in known_graph.items():
        if src not in node_to_nms:
            continue
        for tgt in neighbors:
            if tgt not in node_to_nms:
                continue
            src_idx = node_to_nms[src]
            tgt_idx = node_to_nms[tgt]
            if src_idx != tgt_idx:  # 避免自环
                edges.append((src_idx, tgt_idx))
    
    if len(edges) == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    
    return torch.tensor(edges, dtype=torch.long).t()  # [2, E]
```

**和训练时的对齐**：训练时也应在 NMS 后节点上做同样的映射，而不是直接使用 subdivided GT 图的边索引。这保证训练和推理使用同一套 edge_index 构造逻辑。

---

## 七、实验设计

### 7.1 实验1：Completion > Extraction（无traj，目标2）

**在 SpaceNet / CityScale 上**：

| 方法 | 输入 | 说明 |
|------|------|------|
| SAM-Road (baseline) | img | 原始 extraction |
| Ours (keep=0%) | img | ≈ SAM-Road，验证退化一致性 |
| Ours (keep=50%) | img + 随机删边GT | Completion 核心实验 |
| Ours (keep=80%) | img + 大部分已知GT | 高覆盖率场景 |

**指标**: APLS, TOPO, road IoU, keypoint IoU

**预期**: Completion(keep=50%) > Extraction，因为已知路网减少了搜索空间。

### 7.2 实验2：traj 有效（Xian 上，目标3）

**在 Xian 上**：

| 方法 | traj_heatmap | known_graph | 说明 |
|------|-------------|-------------|------|
| SAM-Road | - | - | 无先验extraction基线 |
| Completion (no traj) | zeros | GT删边 | 验证Completion在Xian上有效 |
| Completion (+ traj) | active.png | GT删边 | 验证traj有增量贡献 |

**预期**: Completion(+traj) > Completion(no traj) > SAM-Road

### 7.3 消融实验

| 编号 | A (4ch热力图) | B (GNN) | road_feat_map | 说明 |
|------|:---:|:---:|:---:|------|
| Full | ✅ | ✅ | 2ch | 完整模型 |
| w/o A | ❌ | ✅ | 2ch | 去掉traj热力图 |
| w/o B | ✅ | ❌ | 2ch | 去掉GNN |
| w/o A&B | ❌ | ❌ | 2ch | 仅road_feat_map |
| w/o road_feat | ✅ | ✅ | 0ch | 去掉路网几何特征 |
| w/o all | ❌ | ❌ | 0ch | ≈ SAM-Road原始 |

### 7.4 keep_ratio 敏感性分析

- keep_ratio = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
- 验证不同已知路网覆盖率下的补全效果
- keep_ratio=0 时应等价于 SAM-Road extraction
- keep_ratio=1 时应完美（已知完整GT，无需补全）

---

## 八、模型改动清单

基于现有 `models/sam_road_completion.py` 和 `data/dataset_completion.py`：

| 优先级 | 改动 | 文件 | 说明 |
|--------|------|------|------|
| **P0** | 修复 known_edge_index | `data/dataset_completion.py` | 构造真实 edge_index（从随机删边GT映射到NMS节点索引），collate_fn 正确 padding |
| **P0** | 修复特征图一致性 | `data/dataset_completion.py` | 渲染 road_feature_map 时使用和标签生成相同的删边结果 |
| **P1** | road_feature_map 4ch→2ch | `models/sam_road_completion.py`, `data/dataset_completion.py` | 去掉距离场和方向场，RoadGraphEncoder 输入 4→2 |
| **P1** | 引入 traj_heatmap | `models/sam_road_completion.py` | 改为4ch输入（RGB+heatmap），patch_embed扩展，无traj时ch4=zeros |
| **P1** | Xian traj_heatmap 支持 | `data/dataset_completion.py` | Xian 使用 active.png 作为 traj_heatmap，known_graph 仍用随机删GT |
| **P1** | 推理后处理 | `engine/inferencer_completion.py` | 已知图边强制 topo_score=1.0 |
| **P2** | 动态 keep_ratio | `data/dataset_completion.py`, config | 支持 `KEEP_RATIO_MIN/MAX`，训练时 Uniform 采样 |
| **P2** | 模态 dropout | `data/dataset_completion.py` | 20%概率清零 traj_heatmap + road_feature_map + known_edge_index |

---

## 九、论文叙事线

```
1. 现有SOTA (SAM-Road, RNGDet++) 做的是 extraction (从零建图)
2. 实际场景中往往已有部分路网 (OSM/已有图) 甚至有轨迹数据 (出租车/GPS)
3. 我们提出 Completion 模型，能利用已知路网信息补全缺失部分
4. 关键设计:
   a. 双通道已知信息注入: CNN编码几何 + GNN编码拓扑
   b. 兼容有/无轨迹: traj作为可选热力图输入，无traj时退化仍有效
   c. 已知图保护: 后处理强制保留已知边，模型只负责发现缺失边
5. 在SpaceNet/CityScale上: Completion > Extraction (利用了已知路网先验)
6. 在Xian上: +traj 进一步提升 (轨迹提供额外的空间先验)
```

**创新点**：
- 从 extraction 到 completion 的范式转换
- traj 模态的引入（A+B 双路径）
- 模型设计兼容有/无 traj，统一框架

---

## 十、数据流完整追踪

### 10.1 训练时单步数据流（以 Xian, keep_ratio=0.5 为例）

```
1. 数据加载:
   - rgb: region_102_sat.png 裁剪 [256,256,3]
   - keypoint_mask, road_mask: 完整GT
   - gt_graph: region_102_refine_gt_graph.p → 完整GT图
   - active.png: region_102_active.png 裁剪 [256,256] → traj_heatmap

2. 随机删边:
   - GT图有 133 节点, 288 有向边
   - keep_ratio=0.5 → 保留 ~144 条边
   - known_edge_set_subdivide: 保留的边集合 (subdivided图上)
   - known_edges_original: 保留的边集合 (原始GT图上)

3. 渲染已知路网特征:
   - ch0: 已知路mask (cv2.line绘制保留的边, thickness=2) [256,256]
   - ch1: 已知节点位置 (cv2.circle绘制保留节点, radius=3) [256,256]
   - road_feature_map: [2, 256, 256] (permute后)

4. 图标签采样:
   - NMS采样节点 → graph_points [N, 2]
   - BFS判断候选边连通性 → pairs [S,P,2], connected [S,P], valid [S,P]
   - 已知边 valid=False, 未知边正常计算

5. 构造 known_edge_index:
   - 对已知边的每个端点，在 NMS 后节点中找最近邻（距离阈值 8px）
   - 映射为 NMS 节点索引之间的边（和推理时使用同一套映射逻辑）
   - known_edge_index [2, E]

6. 模型前向:
   - RGB + traj_heatmap → concat [4ch] → SAM Encoder → image_embeddings [B,256,16,16]
   - road_feature_map [2,256,256] → RoadGraphEncoder → road_embeddings [B,256,16,16]
   - cat + FeatureFusion → fused_features [B,256,16,16]
   - MapDecoder(image_embeddings) → mask [B,2,256,256]
   - BilinearSampler(fused_features, graph_points) → point_features [B,N,256]
   - RoadGraphGNN(point_features, graph_points, known_edge_index) → graph_embeddings [B,N,32]
   - TopoNetCompletion(points, point_features, pairs, valid, graph_embeddings) → topo_scores [B,S,P,1]

7. Loss计算:
   - mask_loss: BCE/Focal
   - topo_loss: BCE(masked, 已知边不计入)
   - total = mask_loss + topo_loss
```

### 10.2 推理时数据流

```
1. 输入:
   - 卫星影像 (整张, 如 400×400)
   - 已知路网 (pickle或用户提供的图)
   - (可选) 轨迹热力图

2. 滑窗切分为 patches

3. 每个 patch:
   - Step 1: infer_masks_and_img_features(rgb, traj_heatmap, road_feat_map)
     → mask_scores [B,H,W,2]
     → fused_features [B,256,h,w]
   
   - Step 2: 从 mask 提取关键点 → 构造 graph_points, pairs
   - Step 3: 将已知路网边映射到 NMS 关键点 → known_edge_index (详见6.4)
   
   - Step 3: infer_toponet(fused_features, graph_points, pairs, valid, known_edge_index)
     → topo_scores [B,S,P,1]
   
   - Step 4: 后处理
     → 阈值过滤
     → 已知图边强制 topo_score=1.0
     → 输出: 补全后的完整图
```

---

## 十一、与现有代码的对应关系

| 设计要素 | 现有代码 | v2改动 |
|---------|---------|--------|
| SAM Encoder | 3ch输入 | 4ch输入（+traj_heatmap） |
| RoadGraphEncoder | 输入4ch | 输入2ch（去掉距离场+方向场） |
| FeatureFusion | 不变 | 不变 |
| MapDecoder | 用image_embeddings | 不变 |
| RoadGraphGNN | edge_index为空 | **修复：填入真实已知边** |
| TopoNetCompletion | pair_proj 322维 | 不变 |
| BilinearSampler | 从fused_features采样 | 不变 |
| Dataset | 随机删边, fixed 0.5 | 动态keep_ratio, Xian用active.png |
| Loss | 已知边valid=False | 不变, 加模态dropout |
| 推理 | 无已知图保护 | **新增：强制已知边score=1.0** |
