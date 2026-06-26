# Completion Loss 改法分析

> **目标**:改 loss 让 component(不破碎)partial 也能训好,兼得"真实不破碎 + 训练有效"。

---

## 一、当前 loss 机制(问题根源)

### 1.1 数据流
```
sample_patch 对每条候选边 (src, tgt):
  if 是 known edge:
    shall_connect = False   ← 故意设 False (其实真实标签是 True, 该边存在)
    valid = False           ← 排除出 loss
  else (未知边):
    shall_connect = BFS可达性  ← 真实标签
    valid = True            ← 参与 loss

training_step:
  topo_loss = BCE(topo_logits[valid], connected[valid])   ← 只算未知边
```

### 1.2 问题
- **known edge 完全不参与 loss**(valid=False)
- component 采样让 known edge 成块聚集 → 某些 patch 全是 known edge → **无 topo 监督**
- 按边随机时 known edge 散布,每 patch 约 50% 未知边 → 都有监督,训练稳

### 1.3 为什么 originally 这么设计
"模型不需要学已知边,只学补全未知边"——任务定义上合理。但**忽略了 known edge 也能提供监督信号**(模型应学会"已知边确实该连",强化拓扑理解)。

---

## 二、五种改法对比

### 方案1:known edge 也参与 loss,标签=True(最简单)

```python
if is_known_edge:
    shall_connect = True    ← 改: 真实标签 (该边存在=应连)
    valid = True            ← 改: 参与 loss
```

**逻辑**:known edge 在完整图里确实存在(连通),标签应是 True。让它参与 loss,模型学到"已知边该连"。

**优点**:
- 改动最小(2 行)
- component patch 全是 known edge 时,也有监督(标签全 True)
- 监督信号均匀(每 patch 都有 known+unknown 混合监督)

**缺点**:
- known edge 标签恒为 True,模型可能学到"看到 known 就输出 True"的捷径
- 但 known edge 在推理时被后处理硬覆盖(score=1.0),训练时学它是否多余?
  - 不多余:训练时学 known edge 帮助 GNN 理解拓扑结构,间接提升未知边预测

**风险**:known edge 占比大时(component 高 keep_ratio),topo_loss 被 True 标签主导,模型偏向输出 True → precision 下降。需看占比。

---

### 方案2:known edge 参与 loss,但降权

```python
if is_known_edge:
    shall_connect = True
    valid = True
    weight = 0.3   ← 降权 (未知边 weight=1.0)
```

**逻辑**:known edge 参与但权重低,主要监督来自未知边,known edge 起辅助作用。

**优点**:避免 known edge 主导 loss,同时补充监督。

**缺点**:需调 weight;实现稍复杂(BCE 需带 weight)。

---

### 方案3:known edge 作为"正样本锚点",未知边正常监督

```python
# 未知边: 正常 BFS 标签, weight=1.0
# known edge: 标签=True, weight=0.5, 作为正样本锚点
# 还可加: 已知不存在的边(负采样) weight=0.5
```

**逻辑**:known edge 是"确定该连"的正样本,帮模型学到正确的连接模式;未知边是"需判断"的样本。

**优点**:监督信号最完整(正负样本都有锚点)。

**缺点**:实现复杂;负采样需额外逻辑。

---

### 方案4:不排除 known edge,用 connected 真实标签(BFS 对 known edge 也是 True)

```python
if is_known_edge:
    shall_connect = target_graph_idx in reached_nodes  ← 和未知边一样用 BFS
    valid = True
```

**逻辑**:known edge 在完整图 BFS 里本就可达(reached_nodes 包含它),所以 shall_connect 自然是 True。统一用 BFS 标签,不区分 known/unknown。

**优点**:最干净,统一标签来源。

**缺点**:和方案1 类似(known edge 标签恒 True),但语义更一致(都是 BFS)。

---

### 方案5:保持 valid=False,但补"未知边负样本"到全已知 patch

```python
# 当前: known edge valid=False
# 改: 当某 patch 全是 known edge (无未知边) 时, 额外采样负样本边 (不存在的边) 参与 loss
```

**逻辑**:不改变 known edge 处理,但保证每 patch 都有监督(全已知时补负样本)。

**优点**:不改 known edge 语义,只补监督。

**缺点**:实现复杂;负采样边定义模糊;治标不治本。

---

## 三、推荐:方案1(known edge 参与 loss,标签 True)

### 3.1 理由
1. **改动最小**(2 行),风险可控
2. **直击根因**:component 退化是因为 known edge 无监督,方案1 让 known edge 有监督(标签 True)
3. **语义正确**:known edge 确实存在(应连),标签 True 是真实的
4. **监督均匀**:每 patch 的 known+unknown 边都参与,信号稳定

### 3.2 潜在问题与缓解
- **捷径学习**(模型见 known 就输出 True):
  - 缓解:known edge 在训练时通过 road_feature_map + GNN 注入,模型需"看到"先验才输出 True,不是无脑输出
  - 推理时 known edge 硬覆盖 score=1.0,捷径不影响推理结果
  - 真正受益的是:GNN 通过 known edge 的监督,更好理解拓扑,提升未知边预测

- **known edge 占比主导**(高 keep_ratio):
  - 可先用方案1 验证,若 precision 下降明显,再升级方案2(降权)

### 3.3 验证计划
1. 改 `sample_patch`(2 行):known edge 的 shall_connect=True, valid=True
2. 用 component 采样 + 方案1 loss 重训 didi_xian completion
3. 对比:
   - vs 0626(component + 旧loss,0.4571):应回升
   - vs 0625(按边随机 + 旧loss,0.5878):目标追平或超过
4. 看 val_loss 是否还震荡(应稳定)
5. 看 precision/recall 分布(known edge 监督是否导致 precision 下降)

---

## 四、实现细节(方案1)

### 4.1 sample_patch 改动
```python
# data/dataset_completion.py, sample_patch 里
if is_known_edge:
    shall_connect.append(True)   # 改: known edge 存在, 标签 True (原 False)
    valid_list.append(True)      # 改: 参与 loss (原 False)
else:
    shall_connect.append(target_graph_idx in reached_nodes)
    valid_list.append(True)
```

### 4.2 P0-1 修复的兼容
P0-1(Dropout 恢复 valid)在模态 dropout 时把所有真实候选对 valid 恢复 True。方案1 后,known edge 本就 valid=True,P0-1 逻辑不受影响(它只处理 dropout 步)。

### 4.3 不需要改的
- training_step 的 loss 计算:不变(它只看 valid,known edge 现在 valid=True 自动参与)
- validation_step:同上
- 推理 inferencer:不变(known edge 后处理硬覆盖 score=1.0,与训练 loss 无关)

### 4.4 测试侧(test_completion)
test_completion 的 PR 曲线也用 valid 算。方案1 后 known edge 参与,PR 曲线会包含 known edge(标签 True)。这改变了阈值标定的样本分布,需重新标定阈值。但这更接近真实评估(known edge 也算)。

---

## 五、风险与备选

### 5.1 主要风险
方案1 可能让模型对 known edge 过度自信,影响未知边的 precision。如果实验发现 precision 大降:
- 升级方案2(known edge 降权 0.3)
- 或方案4(统一 BFS 标签,语义更一致)

### 5.2 最坏情况
若方案1/2/4 都不能让 component 追上按边随机,说明 loss 不是唯一原因,可能 component 采样本身让 GNN 拓扑编码变难(成块 vs 散布的拓扑结构差异)。那时回退方案A(按边随机)保效果。

---

## 六、决策建议

**先实施方案1**(最小改动,直击根因),用 component 重训 didi_xian,看:
1. val_loss 是否稳定(不再震荡)
2. APLS 是否回升(目标 ≥ 0.5878)
3. precision 是否大降(若降则升级方案2)

这个实验成本低(改 2 行 + 重训一次),能验证"loss 是不是根因"。
