# 补全改进计划 — 基于 DeepSeek-v4-pro 分析

> 本文档记录对 SAM-Road Completion v2 模型的深度分析，聚焦三个先验的数据流、Dropout 的设计意图与实现缺陷、以及理论修正方案。

---

## 一、三个先验的消费点与数据流

Completion 模型在原始 SAM-Road 基础上引入三个额外先验信号。以下从 `SAMRoadCompletion.forward` 精确追踪其消费路径。

### 先验 1：`traj_heatmap` [B, H, W, 1]

**消费点（唯一）**：SAM Encoder 的 patch_embed 层。

```python
# sam_road_completion.py line 564-567
x = torch.cat([rgb, traj_heatmap], dim=3)  # [B, H, W, 4]
x = (x - self.pixel_mean) / self.pixel_std
image_embeddings = self.image_encoder(x)   # ← 唯一消费点
```

**注入层级**：SAM Encoder 浅层（patch_embed Conv2d(4→768)）。

**携带信息**：「哪里有轨迹」——空间先验，告诉 ViT backbone 这些区域确定有路。

**影响范围**：全局——`image_embeddings` 是分割分支（MapDecoder）和拓扑分支（TopoNet）的共同上游。

**数据来源**：
- 训练时：Xian 数据集加载 `active.png` 裁剪，归一化到 [0,1]；SpaceNet/CityScale 为全零张量
- 推理时：用户提供的轨迹热力图，或全零（无轨迹场景）

**退化**：第4通道零初始化 → 初始时 4ch 输出 ≡ 3ch 输出；全零输入等价于原始 SAM-Road。

### 先验 2：`road_feature_map` [B, 2, H, W]

**消费点（唯一）**：RoadGraphEncoder (CNN) → FeatureFusion (1×1 Conv)。

```python
# sam_road_completion.py line 570-575
road_embeddings = self.road_feat_encoder(road_feature_map)  # CNN: [2,H,W]→[256,h,w]
fused_features = self.feature_fusion(
    torch.cat([image_embeddings, road_embeddings], dim=1)
)
```

**注入层级**：FeatureFusion 层——image_embeddings 和 road_embeddings 在 ViT 输出层级拼接后融合。

**携带信息**：「已知路网的几何位置」：
- ch0：已知道路 mask（cv2.line 绘制保留的边，thickness=2）
- ch1：已知节点位置（cv2.circle 绘制保留节点，radius=3）

**影响范围**：**仅拓扑分支**。`fused_features` 被 BilinearSampler 采样进入 TopoNet，但 MapDecoder（分割头）始终使用原始 `image_embeddings`，不受 road_feature_map 影响。这是有意设计——分割任务应该从影像发现路，不应受已知路网污染。

**数据来源**：
- 训练时：`render_graph_feature_map()` 从 `known_edges_original` 渲染（与标签使用同一组删边，保证一致性）
- 推理时：从用户提供的部分路网渲染

### 先验 3：`known_edge_index` [B, 2, E]

**消费点（唯一）**：RoadGraphGNN。

```python
# sam_road_completion.py line 604-612
if known_edge_index is not None and known_edge_index.shape[2] > 0:
    graph_embeddings = self.road_graph_gnn(
        point_features, graph_points, known_edge_index
    )
```

**注入层级**：GNN → TopoNetCompletion 的 pair_proj 输入。

**携带信息**：「已知路网中谁连谁」——拓扑先验。GNN 内部将 edge_index 转为稀疏邻接掩码，只允许已知边两端节点互相注意，其余全部屏蔽（-1e4）。

**影响范围**：**仅拓扑分支的最深层**——graph_embeddings [B, N, 32] 作为额外特征拼入 TopoNet 的 pair_proj。

**数据来源**：
- 训练时：GT 随机删边 → subdivided 图上的已知边 → 映射到 NMS 节点索引
- 推理时：用户路网节点 → KDTree 最近邻映射到 NMS 关键点 → 边索引

### 数据流总图

```
traj_heatmap ──→ SAM Encoder (4ch patch_embed) ──→ image_embeddings ──→ MapDecoder (分割)
                                       │                                      ↑
road_feature_map ──→ RoadGraphEncoder ──┤                                      │
                                       ├──→ FeatureFusion ──→ fused_features   │
                                       │                         │             │
known_edge_index ──→ RoadGraphGNN ◄────┘                    BilinearSampler   │
                            │                                     │             │
                            └──→ graph_embeddings ──→ TopoNetCompletion ←──────┘
                                               (pair_proj: 322→128→1)
```

### 三先验的分工总结

| 先验 | 注入层级 | 影响范围 | 携带信息 | 训练来源 | 推理来源 |
|------|---------|---------|---------|---------|---------|
| traj_heatmap | SAM Encoder 浅层 | 全局（分割+拓扑） | 哪里有轨迹 | active.png 或 zeros | 用户轨迹或 zeros |
| road_feature_map | FeatureFusion | 仅拓扑分支 | 已知路网几何位置 | GT删边渲染 | 用户路网渲染 |
| known_edge_index | GNN (最深层) | 仅拓扑分支 | 已知路网拓扑连接 | GT删边→NMS映射 | 用户路网→KDTree映射 |

---

## 二、Dropout 的设计意图与实现分析

### 2.1 为什么需要 Dropout

三个先验在训练和推理时存在**来源 gap**：

| 先验 | 训练时的质量 | 推理时可能的质量 | gap |
|------|------------|----------------|-----|
| traj_heatmap | 从 active.png 裁剪，质量较高 | 用户轨迹可能有偏移/缺失/噪声 | 精度和覆盖率 |
| road_feature_map | 从精确 GT 坐标渲染 | 从用户部分路网渲染，坐标精度不确定 | 节点位置精度 |
| known_edge_index | 精确 GT subdivided 图边索引 | KDTree 最近邻映射，可能误匹配 | 节点体系差异 |

**如果模型永远只在高品质先验下训练**，推理时遇到低品质先验就是分布外输入，可能崩溃。

### 2.2 三种 Dropout 策略

从 `dataset_completion.py:665-689` 提取：

**策略 A：模态 Dropout（20%）——模拟"用户无任何先验"**

```python
if random.random() < 0.2:
    road_feature_map = zeros      # 清空已知路网几何
    known_edge_index = empty      # 清空 GNN 边
    traj_heatmap = zeros          # 清空轨迹
```

**策略 B：traj 热力图增强（仅 Xian 有 traj 时）**

```python
if random.random() < 0.2:
    traj_heatmap = zeros              # 20%: 全黑（保持纯视觉能力）
elif random.random() < 0.6:
    traj_heatmap = 随机腐蚀(3~10块)     # 40%: 模拟低质量轨迹（打断捷径）
# else: 40%: 完整 traj                 # 学习信任可靠先验
```

### 2.3 Dropout 的意图：构建完整能力谱系

Dropout 的目标不是让模型更强，而是**防止模型退化**——让它在「完美先验 / 破损先验 / 无先验」三种输入分布下都被训练过，推理时无论先验质量如何都能应对。

期望的退化谱系：

```
纯 Extraction ←── 模态 dropout 20% ──→ Completion (有先验)
   ↑                                        ↑
  无任何先验                              80% 训练步有先验
```

---

## 三、关键缺陷：Dropout 步的有效监督信号损失

### 3.1 问题定位

已知边的排除逻辑在 `CompletionGraphLabelGenerator.sample_patch` 中（`dataset_completion.py:386-397`）：

```python
if is_known_edge:
    shall_connect.append(False)   # 不参与连通性标签
    valid_list.append(False)      # ← 不参与 topo loss
```

已知边从 `valid` 掩码中被排除，不贡献 topo 训练的监督信号。

而模态 dropout 在 `__getitem__` 中**晚于** `sample_patch` 执行（`dataset_completion.py:665-671`）：

```python
# sample_patch 已经算好了 valid（已知边=False）
graph_points, topo_samples, known_edge_index = ...sample_patch(...)
pairs, connected, valid = zip(*topo_samples)

# dropout 只清零了先验，没有恢复 valid
if drop_all:
    road_feature_map = zeros     # ✅ 先验清零
    known_edge_index = empty     # ✅ 先验清零
    traj_heatmap = zeros         # ✅ 先验清零
    # ❌ valid 掩码未恢复 → 已知边仍然是 False → 不被监督
```

### 3.2 影响量化

假设 `keep_ratio ~ U[0.2, 0.8]`，期望值 0.5：

| 训练步类型 | 先验信息 | 被监督边比例 | 步占比 |
|-----------|---------|-------------|--------|
| Completion 正常步 | road_feat_map + GNN | ~50%（未知边） | 80% |
| Completion dropout 步 | **全空** | **~50%** | 20% |
| 原版 SAM-Road | 全空 | **100%** | 100% |

**Dropout 步严格劣于原版 SAM-Road 的同类型训练步**——同样是全空先验，Completion 的 dropout 步只有一半的边被监督。

### 3.3 根本原因

`valid` 掩码的语义在 Completion 模型中是**矛盾的**：

- **正常语义**：`valid=False` 表示「这条边是已知的，不需要学习预测」
- **Dropout 时语义**：已知路网已被清零，不存在「已知边」的概念，但 `valid` 掩码仍沿用正常步的值

这导致模型在 20% 的步中处于双重劣势：既没有先验辅助，又只被训练了一半的边。

---

## 四、修正方案

核心思路：**在 dropout 步恢复 valid 为全 True，让模型做纯 Extraction 训练。**

### 方案 A：在 `__getitem__` 的 dropout 分支中覆盖 valid

```python
if drop_all:
    road_feature_map = np.zeros_like(road_feature_map)
    known_edge_index = torch.zeros(2, 0, dtype=torch.long)
    traj_heatmap = np.zeros_like(traj_heatmap)
    # ✅ 恢复 valid：让所有候选边都参与 loss（纯 extraction 模式）
    valid = tuple(
        tuple(True for _ in v) for v in valid
    )
```

### 方案 B：在 `sample_patch` 中增加 `force_all_valid` 参数

```python
def sample_patch(self, patch, rot_index=0, force_all_valid=False):
    ...
    if force_all_valid:
        # dropout 步：所有边都参与 loss
        valid_list = [True] * len(target_nodes)
    else:
        # 正常步：已知边不参与
        valid_list = [...]  # 原有逻辑

# 调用处：
if drop_all:
    ...sample_patch(patch, rot_index, force_all_valid=True)
```

### 推荐方案 A

改动最小，逻辑内聚在 `__getitem__` 中，不影响 `sample_patch` 的接口语义。

### 修正后的监督信号对比

| 训练步类型 | 先验信息 | 被监督边比例 | 步占比 |
|-----------|---------|-------------|--------|
| Completion 正常步 | road_feat_map + GNN | ~50%（未知边） | 80% |
| Completion dropout 步 | 全空 | **100%**（修复后） | 20% |
| 原版 SAM-Road | 全空 | 100% | 100% |

修复后 dropout 步等价于纯 Extraction 训练，模型的 Extraction 能力得到充分保障。

---

## 五、与原始 Extraction 的理论对比

### 5.1 Completion 的优势

- GNN + road_feature_map 提供额外的约束信息（已知路网的几何和拓扑），缩小搜索空间
- 不需要预测已知边，任务是「在约束下补全缺失边」，比全图建图简单

### 5.2 Completion 的劣势

- 正常训练步只被监督 ~50% 的边，监督信号量减半
- GNN 存在 training/inference gap（训练时精确 GT 节点 ↔ 推理时 mask 预测节点）
- 4ch encoder 的 patch_embed 第4通道权重在多数据集间共享，无 traj 数据集上可能学到偏见

### 5.3 修正后的期望

修正 dropout 步后，Completion 模型的能力谱系变为：

| 输入条件 | 行为 | 等价于 |
|---------|------|--------|
| 无任何先验（dropout步） | 纯 Extraction，100%边监督 | ≈ SAM-Road |
| 无 traj + 有已知路网（SpaceNet/CityScale） | Completion 无 traj | 目标 (2) |
| 有 traj + 有已知路网（Xian） | Completion 有 traj | 目标 (3) |

使模型在 Extraction 退化场景下的能力与原始 SAM-Road 在理论对齐，消除了 "dropout 步严格劣于基线" 的结构性缺陷。

---

## 六、分析总结

| 维度 | 评价 |
|------|------|
| 三先验的消费路径设计 | **合理**——注入层级有梯度（浅层→中层→深层），影响范围有分工（全局→仅拓扑） |
| 退化逻辑 | **正确**——第4通道零初始化、GNN 空输入退化、模态 dropout 覆盖全空场景 |
| Dropout 意图 | **正确**——防止过拟合训练时高质量先验 |
| Dropout 实现 | **有缺陷**——清零了先验但未恢复 valid，导致 20% 训练步信息量不足 |
| 修正方案 | **简单**——在 dropout 分支恢复 valid 为全 True |
| 超越 Extraction 的理论依据 | **有但需验证**——信息量的 tradeoff（更少边 vs 更多约束）需要消融实验定量 |
