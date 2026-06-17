# SAM-Road Completion v2 改进计划

> 基于对 `sam_road_completion.py` / `dataset_completion.py` / `train_completion.py` / `inferencer_completion.py` 的完整代码审查，梳理当前实现的问题、分析改进方向。

---

## 一、当前实现现状总结

### 1.1 模型架构（三条注入路径）

```
输入模态:
  RGB [B,H,W,3]  ──┐
  traj [B,H,W,1] ──→├→ concat [B,4,H,W] → SAM Encoder → image_embeddings [B,256,h,w]
                    ┘     ↑ 路径A: 第4通道

  road_feat [B,2,H,W] → RoadGraphEncoder(CNN) → road_embeddings [B,256,h,w]
                           ↓
  image_embeddings + road_embeddings → FeatureFusion(1×1 Conv) → fused_features [B,256,h,w]

下游:
  MapDecoder(image_embeddings) → mask_logits          ← 分割头: 纯视觉
  BilinearSampler(fused_features) → point_features    ← 拓扑头: 视觉+路网

  known_edge_index [B,2,E] → RoadGraphGNN(point_features, coords, edges)
                               → graph_embeddings [B,N,32]  ← 路径B: GNN拓扑

  TopoNetCompletion(point_features, pairs, graph_embeddings) → topo_scores
```

### 1.2 训练行为

| 数据集 | traj_heatmap | known_graph | keep_ratio | 模态Dropout |
|--------|-------------|-------------|------------|------------|
| SpaceNet | 全零（退化3ch） | GT随机删边 | U[0.2, 0.8] | 20%清零 |
| CityScale | 全零（退化3ch） | GT随机删边 | U[0.2, 0.8] | 20%清零 |
| Xian | active.png裁剪 | GT随机删边 | U[0.2, 0.8] | 20%清零 |

**关键事实**：所有数据集训练时始终自动从 GT 采样 known_graph，不存在"不给 known_graph"的选项。

### 1.3 已修复的问题（v2 → 当前代码）

- [x] known_edge_index：从空占位符修复为真实边映射
- [x] road_feature_map：4ch → 2ch（去掉距离场+方向场）
- [x] 特征图一致性：渲染和标签使用同一组删边
- [x] 动态 keep_ratio：U[0.2, 0.8]
- [x] 推理后处理：已知边硬覆盖 score=1.0
- [x] fp16 NaN 修复：六层防御

---

## 二、核心问题分析

### 2.1 问题一：分割头与路网先验割裂

**现象**：

```python
# forward() 中:
mask_logits = self.map_decoder(image_embeddings)                    # ← 纯视觉
point_features = self.bilinear_sampler(fused_features, graph_points) # ← 视觉+路网
```

分割头（MapDecoder）使用 `image_embeddings`（纯视觉），TopoNet 使用 `fused_features`（视觉+路网融合）。

**为什么这样设计**：代码注释说"避免过拟合"——如果分割头看到已知路网 mask，可能学到"直接复制 road_feature_map 的 ch0 到输出"的捷径，而非真正从卫星图判断。

**为什么 image_embeddings 不需要担心视觉遮挡问题**：如果使用了完整的 4ch 输入（有 traj），traj 的图像特征会补齐遥感影像中被遮挡的部分——traj 告诉 SAM encoder "这些地方一定有路"，SAM encoder 在提取视觉特征时已经把 traj 信号编码进去了。所以 image_embeddings 在有 traj 时实际上已经隐含了"哪里有路"的先验，不需要 fused_features 来补。但对于 SpaceNet/CityScale 这种无 traj 的场景，image_embeddings 确实无法利用路网先验。

**问题本质**：

```
已知路段视觉不显著（遮挡/低对比度）
  → 分割头（纯视觉）漏检这些路段
  → mask 提取不出对应节点
  → TopoNet 缺少关键节点
  → GNN 无法在这些节点上传播已知拓扑
  → 补全失败
```

这是一个**级联失效链**：分割头漏检 → 节点缺失 → GNN 失效 → TopoNet 补全失败。

**消融点**：是否应该让分割头也使用 `fused_features`？这是一个需要实验验证的设计取舍：

| 方案 | 优点 | 缺点 |
|------|------|------|
| 当前：分割头用 image_embeddings | 避免捷径依赖；纯视觉 mask 更鲁棒 | 已知路段可能漏检；级联失效风险 |
| 改进：分割头用 fused_features | 已知路段不漏检；mask 质量可能整体提升 | 可能过拟合已知路网；先验错误时 mask 跟着错 |

**建议改进方案**（折中）：

```python
# 方案: 分割头用 fused_features, 但增加 mask 捷径缓解
mask_logits = self.map_decoder(fused_features)   # ← 改用 fused_features

# 配合更强的 mask 捷径缓解:
# 1. 分割头的 mask loss 仍然用完整 GT (不区分已知/未知路段)
# 2. road_feature_map 加入随机遮挡 (类似 traj 的腐蚀增强)
#    模拟先验质量差的情况, 迫使模型不能只抄 road_feature_map
```

### 2.2 问题二：无先验时 Extraction 能力更弱

**根本原因**：loss 信号稀疏 + 训练偏置。

#### loss 信号稀疏

| | 原版 SAM-Road | Completion (keep=0.5) |
|---|---|---|
| 参与loss的候选对数 | 全部（如16对） | ~8对（已知边被 mask） |
| 正样本数 | ~8 (BFS可达) | ~4 (未知边中可达) |
| 负样本数 | ~8 | ~4 |

每个 batch 的 topo loss 信号量减半。模型在"判断候选边是否该连"上收到的梯度信号更少，学习效率更低。

#### 训练偏置

模型在 80% 的训练时间中有已知路网先验（模态 Dropout 20% 清零），会形成对先验的依赖：
- GNN 学到了"利用已知拓扑推断未知连接"的能力，但没有充分发展"纯粹从视觉判断"的能力
- FeatureFusion 在 80% 时间看到有意义的 road_embeddings，可能学会了"路网特征存在时和不存在时用不同策略"

#### 推理时的退化

推理时如果不给 known_graph：
- road_feature_map = 全零 → road_embeddings = 全零 → FeatureFusion 退化为 `Conv(image_embeddings, 0)` ≈ 降维
- known_edge_index = None → GNN 退化为全连接注意力（等价于无图先验）
- 但模型**没有在"全零先验"的场景下充分训练过**（只有 20% 的模态 Dropout），所以退化的性能大概率不如原版 SAM-Road

**结论**：

| 场景 | Completion 模型表现 | 原版 SAM-Road 表现 | 预期优劣 |
|------|-------------------|-------------------|---------|
| 有 known_graph (高 keep) | ✅ 强 | ❌ 不适用 | Completion >> Extraction |
| 有 known_graph (低 keep) | ⚠️ 中 | ❌ 不适用 | Completion > Extraction |
| 无 known_graph | ❌ 弱（退化） | ✅ 强 | Completion < Extraction |

**这不是 bug，是设计取舍**。Completion 模型的价值在于"有先验时比 Extraction 好"，而不是"无先验时也比 Extraction 好"。

#### 改进方向：增强无先验退化能力

```python
# 1. 提高模态 Dropout 比例 (20% → 30-40%)
MODALITY_DROPOUT_PROB: 0.3   # 强制模型更频繁地面对无先验场景

# 2. 对 road_feature_map 做随机遮挡 (类似 traj 的腐蚀增强)
# 模拟先验质量差/部分缺失的情况
if self.is_train and not drop_all:
    rand_val = random.random()
    if rand_val < 0.15:
        # 15%: 随机擦除 road_feature_map 的部分区域
        for ch in range(2):
            for _ in range(random.randint(1, 3)):
                erase_w = random.randint(32, 128)
                erase_h = random.randint(32, 128)
                ex = random.randint(0, max(1, patch_size - erase_w))
                ey = random.randint(0, max(1, patch_size - erase_h))
                road_feature_map[ey:ey+erase_h, ex:ex+erase_w, ch] = 0

# 3. 在 val 阶段同时评估"有先验"和"无先验"两种模式
# 确保无先验退化不过度
```

### 2.3 问题三：已知边映射到 NMS 节点的信息损失

#### 训练时的映射（`map_known_edges_to_nms`）

**硬性要求**：已知边的两个端点**必须都在 NMS 后的节点集合中**，这条边才被保留。

```
示例:
  已知路网: A ——B—— C
                 ^
                 这条边已知

  NMS 后: A ———————— C    (B 被 NMS 删掉了)

  结果: 边(A,B) 和 (B,C) 的端点 B 不在 NMS 集合中 → 两条已知边都丢失
```

**为什么这样可行**：subdivide 图的节点密度远高于 NMS 节点密度。subdivide_resolution=4 意味着每 4 像素一个节点，NMS radius=16 意味着每 ~16 像素保留一个节点。所以大多数 NMS 节点附近都有 subdivide 节点，映射成功率较高。

**但在稀疏路段的风险**：如果已知路段的端点恰好落在 NMS 节点的间隙中（比如被 crossover exclude 排除了），那条已知边就会丢失。

#### 推理时的映射（`_match_known_edges_to_graph_points`）

用 KDTree 最近邻，距离 < `neighbor_radius`（64像素）算匹配成功。比训练时宽松，但有距离误差——已知路网节点可能被匹配到不太精确的 NMS 关键点。

**改进方向**：

```python
# 方案: 训练时也用最近邻映射, 而非硬性要求端点精确在 NMS 集合中
# 这和推理时的逻辑对齐, 减少训练/推理的 gap

def map_known_edges_to_nms_v2(known_edge_set_subdivide, nmsed_indices,
                               subdivide_points, nmsed_points,
                               distance_threshold=8.0):
    """用 KDTree 做最近邻映射 (与推理时对齐)"""
    if len(nmsed_indices) == 0 or len(known_edge_set_subdivide) == 0:
        return torch.zeros(2, 0, dtype=torch.long)

    # 建立KDTree
    nms_kdtree = scipy.spatial.KDTree(nmsed_points)

    # 建立旧索引到NMS索引的映射 (精确匹配)
    sub_to_nms_exact = {}
    for nms_idx, sub_idx in enumerate(nmsed_indices):
        sub_to_nms_exact[sub_idx] = nms_idx

    edges_src, edges_tgt = [], []
    for (s, t) in known_edge_set_subdivide:
        s_mapped = _map_one_endpoint(s, sub_to_nms_exact, nms_kdtree,
                                      subdivide_points, distance_threshold)
        t_mapped = _map_one_endpoint(t, sub_to_nms_exact, nms_kdtree,
                                      subdivide_points, distance_threshold)
        if s_mapped is not None and t_mapped is not None and s_mapped != t_mapped:
            edges_src.append(s_mapped)
            edges_tgt.append(t_mapped)

    if len(edges_src) == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([edges_src, edges_tgt], dtype=torch.long)
```

### 2.4 问题四：FeatureFusion 的简单性

当前融合方式：

```python
self.feature_fusion = nn.Sequential(
    nn.Conv2d(512, 256, 1),  # concat 后降维
    nn.GELU(),
    nn.Conv2d(256, 256, 1),  # 输出
)
```

这是逐像素的 1×1 Conv 融合，无法建模"已知路网的某条边如何影响远处未知区域"这种长程依赖。GNN 路径B 部分弥补了这一点（通过消息传递），但 GNN 只作用于 TopoNet 的节点级特征，不影响分割 mask 的质量。

**改进方向（可选，视实验结果决定）**：

| 方案 | 复杂度 | 效果预期 | 风险 |
|------|--------|---------|------|
| 当前 1×1 Conv | 低 | 基线 | 无法建模长程依赖 |
| 3×3 Conv 堆叠 | 中 | 局部空间交互 | 可能足够 |
| Cross-Attention | 高 | 长程依赖 | 训练不稳定、计算量大 |
| 空间金字塔池化 | 中 | 多尺度融合 | 需要调参 |

**建议**：先用当前 1×1 Conv 做实验基线，如果发现"远距离已知路网对补全影响不大"则不需要改。如果发现需要，先尝试 3×3 Conv 堆叠，最后才考虑 Cross-Attention。

---

## 三、问题优先级与改进方案

### 3.1 P0：分割头特征源

**问题**：分割头用 `image_embeddings`（纯视觉），已知路段可能漏检 → 级联失效。

**方案**：分割头改用 `fused_features`。

```python
# 改动: models/sam_road_completion.py

# forward() 中:
# 旧: mask_logits = self.map_decoder(image_embeddings)
# 新:
mask_logits = self.map_decoder(fused_features)  # ← 改用 fused_features

# infer_masks_and_img_features() 中同理
# 旧: mask_logits = self.map_decoder(image_embeddings)
# 新:
mask_logits = self.map_decoder(fused_features)
```

**配合防捷径措施**：

1. mask loss 仍用完整 GT（不区分已知/未知），模型必须对**所有**路段做准确预测
2. 对 `road_feature_map` 增加随机遮挡增强（模拟先验缺失/错误）
3. 保持模态 Dropout

**消融实验**：

| 实验 | 分割头特征源 | 预期 |
|------|------------|------|
| A (当前) | image_embeddings | 已知路段可能漏检 |
| B (改进) | fused_features | 已知路段不漏检，但可能过拟合 |
| C (折中) | 0.5×image_embeddings + 0.5×fused_features | 介于 A/B 之间 |

### 3.2 P1：增强无先验退化能力

**问题**：模态 Dropout 20% 不够，模型对先验过度依赖。

**方案**：

```yaml
# config 改动:
MODALITY_DROPOUT_PROB: 0.3        # 20% → 30%, 增加无先验训练比例
ROAD_FEAT_ERASE_PROB: 0.15       # 新增: road_feature_map 随机遮挡概率
ROAD_FEAT_ERASE_REGIONS: [1, 3]   # 新增: 每次遮挡 1-3 个矩形区域
```

```python
# dataset_completion.py __getitem__ 中:
if self.is_train and not drop_all:
    rand_val = random.random()
    if rand_val < self.road_feat_erase_prob:
        for ch in range(2):
            for _ in range(random.randint(*self.road_feat_erase_regions)):
                erase_w = random.randint(32, 128)
                erase_h = random.randint(32, 128)
                ex = random.randint(0, max(1, patch_size - erase_w))
                ey = random.randint(0, max(1, patch_size - erase_h))
                road_feature_map[ey:ey+erase_h, ex:ex+erase_w, ch] = 0
```

### 3.3 P1：训练/推理边映射对齐

**问题**：训练时硬性要求端点精确在 NMS 集合中（可能丢边），推理时用 KDTree 最近邻（有距离误差）。两者逻辑不一致。

**方案**：训练时也改用 KDTree 最近邻映射，与推理对齐。见 2.3 中的 `map_known_edges_to_nms_v2`。

### 3.4 P2：推理时已知路段节点保障

**问题**：如果已知路段在视觉上不显著，mask 阶段可能提取不出这些节点，导致 GNN 无法传播拓扑信息。

**方案**：推理时，将已知路网的节点**直接注入**图节点集合，而非仅依赖 mask 提取。

```python
# inferencer_completion.py 中:

# 1. 正常从 mask 提取节点
graph_points = extract_graph_points(fused_keypoint_mask, fused_road_mask, config)

# 2. 将已知路网节点也加入 graph_points (去重)
known_nodes = set()
for node, neighbors in known_graph_adj.items():
    known_nodes.add(node)
    for neighbor in neighbors:
        known_nodes.add(neighbor)

known_coords = np.array(list(known_nodes), dtype=np.float32)
# 用 KDTree 检查已知节点是否已在 graph_points 中
existing_kdtree = scipy.spatial.KDTree(graph_points)
dists, _ = existing_kdtree.query(known_coords, k=1)
# 距离 > NMS_RADIUS 的节点才需要注入 (避免重复)
inject_mask = dists > config.ROAD_NMS_RADIUS
inject_coords = known_coords[inject_mask]
if len(inject_coords) > 0:
    graph_points = np.concatenate([graph_points, inject_coords], axis=0)
```

**注意**：注入的节点没有 mask 提取的置信度，在 TopoNet 中可能引入噪声。需要实验验证是否有效。

### 3.5 P2：val 阶段双模式评估

**问题**：当前 val 阶段保留 known_graph 但清空 traj，只评估"有先验"模式。无法衡量无先验退化程度。

**方案**：val 阶段同时跑两种模式，分别记录指标。

```python
# 可在 validation_step 中根据 batch_idx 选择模式:
# batch_idx % 2 == 0: 有 known_graph (当前行为)
# batch_idx % 2 == 1: 清空 known_graph (退化模式)
# 分别记录 val_loss_with_prior 和 val_loss_without_prior
```

---

## 四、实验计划

### 4.1 消融实验矩阵

| # | 分割头特征源 | 模态Dropout | 边映射方式 | 节点注入 | 说明 |
|---|------------|------------|-----------|---------|------|
| 1 | image_emb (当前) | 20% | 精确匹配 (当前) | 无 | 基线 |
| 2 | fused_features | 20% | 精确匹配 | 无 | P0: 分割头改特征源 |
| 3 | fused_features | 30% | 精确匹配 | 无 | P1: 增强退化 |
| 4 | fused_features | 30% | KDTree匹配 | 无 | P1: 边映射对齐 |
| 5 | fused_features | 30% | KDTree匹配 | 有 | P2: 节点注入 |

### 4.2 与原版 SAM-Road 的对比

| 方法 | 输入 | SpaceNet APLS | SpaceNet TOPO | Xian APLS | Xian TOPO |
|------|------|:---:|:---:|:---:|:---:|
| SAM-Road (原版) | img | ? | ? | - | - |
| Completion (无先验退化) | img | ? | ? | - | - |
| Completion (keep=50%) | img + GT删边 | ? | ? | ? | ? |
| Completion (keep=50% + traj) | img + GT删边 + traj | - | - | ? | ? |

**关键对比**：Completion(无先验退化) vs SAM-Road(原版)。如果退化模式性能差距 > 5%，说明需要加强 P1 改进。

---

## 五、4ch 第4通道的消融分析

### 5.1 当前实现

```python
# SAM Encoder 输入:
x = torch.cat([rgb, traj_heatmap], dim=3)  # [B, H, W, 4]
x = (x - pixel_mean) / pixel_std           # 第4通道: mean=0, std=1
image_embeddings = self.image_encoder(x)    # SAM ViT

# pixel_mean/std: [123.675, 116.28, 103.53, 0.0] / [58.395, 57.12, 57.375, 1.0]
```

### 5.2 无 traj 时 (SpaceNet/CityScale)

- `traj_heatmap` = 全零 [B,H,W,1]
- 归一化后: `(0 - 0) / 1.0 = 0`
- 第4通道 patch_embed 权重零初始化 → `0 × weight = 0`
- **数学上等价于 3ch SAM**，无性能损失，但有额外计算开销（4ch 的 Conv2d 比 3ch 多 1/3 通道的计算量）

### 5.3 有 traj 时 (Xian)

- `traj_heatmap` = active.png 归一化到 [0, 1]
- 归一化后: `(1 - 0) / 1.0 = 1.0`（最大值），`(0 - 0) / 1.0 = 0`（最小值）
- SAM encoder 在提取视觉特征时**已经把 traj 信号编码进去了**

### 5.4 对问题 2.1 的影响

当有 traj 时，`image_embeddings` 已经隐含了"哪里有路"的先验信息（通过第4通道）。所以分割头用 `image_embeddings` 时，有 traj 场景下 image_embeddings 本身就比纯 3ch 的 image_embeddings 更好——traj 通道相当于给分割头也注入了路网先验，只是注入方式更浅（通过 SAM encoder 的隐式编码，而非直接拼接 road_feature_map）。

**但对于无 traj 场景（SpaceNet/CityScale）**，image_embeddings 确实完全不含路网先验，分割头的级联失效风险仍然存在。

### 5.5 消融点

| 实验 | traj | 分割头特征源 | 说明 |
|------|------|------------|------|
| A1 | 无 | image_embeddings | 当前基线 |
| A2 | 无 | fused_features | 改进：无traj时分割头也能用路网先验 |
| B1 | 有 | image_embeddings | 有traj时，image_embeddings已含先验 |
| B2 | 有 | fused_features | 有traj时，双重先验注入 |

**预期**：B1 vs B2 差异较小（traj 已经给 image_embeddings 注入了先验），A1 vs A2 差异较大（无 traj 时 fused_features 是唯一先验来源）。

---

## 六、改动文件清单

| 优先级 | 改动 | 文件 | 改动量 |
|--------|------|------|--------|
| P0 | 分割头改用 fused_features | `models/sam_road_completion.py` | ~2行 |
| P1 | 增加模态 Dropout 比例 + road_feature_map 遮挡增强 | `data/dataset_completion.py`, config yaml | ~20行 |
| P1 | 训练时边映射改为 KDTree 最近邻 | `data/dataset_completion.py` | ~30行 |
| P2 | 推理时已知路网节点注入 | `engine/inferencer_completion.py` | ~30行 |
| P2 | val 双模式评估 | `models/sam_road_completion.py` | ~20行 |
