# Completion 推理坐标系 Bug 分析

> **症状**:completion × didi_xian 的 APLS=0.337、TOPO F1=0.509,**比 extraction×didi_xian (APLS=0.375) 还差**。给了已知路网先验反而更糟。
>
> **根因**:推理时 known graph(已知路网)的坐标系与 graph_points 不一致,导致 known edges 几乎 100% 被连到错误节点上,后处理还强制保留这些错误边。

---

## 一、Bug 定位(有定量证据)

### 1.1 坐标系不一致

| 对象 | 坐标系 | 来源 |
|------|--------|------|
| `graph_points`(mask 提取的节点) | **(x, y)** | `extract_graph_points` 内部 `rcs[:, ::-1]` |
| `known_graph_adj`(partial prior pickle) | **(row, col)** | `load_known_graph` 直接 `pickle.load`,**未做 coord_transform** |
| 训练时 known graph | **(x, y)** | `igraph_from_adj_dict(full_graph, coord_transform)` 已转换 |

`inferencer_completion.py` 的 `_match_known_edges_to_graph_points`(L411)和 `_inject_known_nodes`(L535)直接把 pickle 的 `(row,col)` 坐标当 `(x,y)` 与 graph_points 做 KDTree 匹配——**对 didi_xian/cityscale 是 x/y 互换,对 spacenet 是 x/y 互换 + y 未翻转**。

### 1.2 定量证据(didi_xian test[0]=region_92,真实数据)

用真实 partial prior 和 GT 图测:

| 指标 | 错误坐标系(当前代码) | 正确坐标系(加 transform) |
|------|:---:|:---:|
| 节点匹配率(距离<64px) | 96.1% | 100% |
| **边端点全对率** | **0/72 (0%)** | **72/72 (100%)** |
| 端点错位边数 | 67/72 | 0 |
| 平均匹配距离 | 23.1px | 0.0px |

**节点匹配率 96% 是假象**——节点碰巧匹配到邻近的某个节点,但是错的那个。看边端点正确率才暴露问题:**72 条 known edge 全部端点错位**。

### 1.3 为什么 mask 对、结果差

- mask 来自 `MapDecoder(image_embeddings)`,纯视觉,与 known graph 坐标无关 → road/keypoint png 正确
- completion 最终图 = `known_edges` + `pred_new_edges`。known_edges 经 `_match_known_edges_to_graph_points` 获取,坐标系错 → 全部连错
- 后处理(L343-346)把 known edge `score` 硬覆盖为 1.0 强制保留 → **错误边被强制注入最终路网**

这完全解释了"completion 比 extraction 差":extraction 没有 known graph 注入,反而没有这批错误边。

### 1.4 Bug 历史

- 首次出现:`2d5dc74`(completion 模型诞生)起就存在
- `a8c8fbb`("fix(coord)")只修了保存图时的坐标分支,**未碰 `_match_known_edges_to_graph_points`**
- 我的 P0-1/P1-3/P2-1/P2-2 改动里,P2-1 的 `_inject_known_nodes` 复用了同一错误坐标系,**继承了 bug 但非根源**
- **结论:这是 completion 模型一直存在的预存 bug,不是我引入的。但 completion 此前所有 infer 结果都受影响。**

---

## 二、影响范围

| 受影响 | 不受影响 |
|--------|---------|
| completion 推理的 known edges 匹配与注入(`_match_known_edges_to_graph_points`、`_inject_known_nodes`、`_get_known_edges_in_graph_points`) | mask 分割(MapDecoder) |
| completion 最终路网图(含错误 known edges) | TopoNet 预测的 pred_new_edges(它用 graph_points,坐标系自洽) |
| completion 的 APLS/TOPO 指标 | extraction(无 known graph) |
| **didi_xian、cityscale、spacenet 三个数据集都受影响**(spacenet 错得更狠:x/y 互换+y未翻转) | 训练(训练侧 known graph 经 coord_transform,坐标系正确) |

注意:**训练是正确的**,bug 只在推理。所以 ckpt 本身没问题,修了推理代码后可直接重跑 infer+eval,无需重训。

---

## 三、修复方案

### 3.1 核心思路

在推理侧,从 `known_graph_adj` 提取坐标后、与 graph_points 做 KDTree 前,对 known 坐标做与训练相同的 `coord_transform`。最干净的做法:**在 `load_known_graph` 里一次性把 pickle 的 `(row,col)` 邻接表转成 `(x,y)` 格式**,与训练 `igraph_from_adj_dict` 对齐。

### 3.2 具体改动点

**`engine/inferencer_completion.py`**:

1. 新增一个 `transform_known_graph_coords(known_graph_adj, dataset)` 函数,按 dataset 做坐标变换:
   - didi_xian / cityscale:`(r,c) → (c,r)` 即 key `(r,c)` 变 `(c,r)`
   - spacenet:`(r,c) → (c, 400-r)`

2. 在 `infer_one_img` 开头,`load_known_graph` 返回后调用此函数,把邻接表统一转成 `(x,y)`。之后所有函数(`_match_known_edges_to_graph_points`、`_inject_known_nodes`、`render_graph_feature_map`)拿到的 known 坐标都是 `(x,y)`,与 graph_points 一致。

**注意 `render_graph_feature_map`**:它现在训练/推理都把 `(row,col)` 当 `(x,y)` 画线(两边一致但语义错)。修复后推理侧传入 `(x,y)` 反而和训练不一致了。需要同步检查训练侧 `render_graph_feature_map` 的坐标——如果训练侧也该用 `(x,y)`,则两边一起修;如果训练侧故意用 `(row,col)` 渲染(因为 road_feature_map 只是几何先验,坐标翻转不影响 CNN 学到"哪里有线"),则推理侧 render 也应保持 `(row,col)`。

**这是个需要仔细验证的点**:render 的坐标一致性 vs 匹配的坐标正确性,两者要求不同。建议:
- 匹配/注入路径:必须用 `(x,y)`(修)
- render 路径:训练推理一致即可,当前都 `(row,col)` 不动,避免引入新的 train/infer gap

### 3.3 验证方法

修复后,重跑 region_92 的匹配实验,确认边端点正确率从 0% → 100%。然后重跑 completion infer+eval,对比 APLS/TOPO 是否回升到 ≥ extraction 水平。

### 3.4 待确认问题(已验证,见下文 §五)

1. **训练侧 `render_graph_feature_map` 坐标**:`get_known_adj_for_render` 返回 `(row,col)`,render 当 `(x,y)` 画。训练时 road_feature_map 的 ch0(路mask)画在 `(row,col)` 当 `(x,y)` 的位置——这对 didi_xian 等于把路网画在转置位置。CNN 能学到(因为 train/infer 一致),但语义上 road_feature_map 和 image_embeddings 在空间上没对齐。**这可能是一个独立的、更深的 bug**,需要单独验证 road_feature_map 渲染后与真实路网 mask 的 IoU。

2. **spacenet 的 y 翻转**:spacenet pickle 是 `(raw_y, raw_x)`,coord_transform 是 `(raw_x, 400-raw_y)`。推理修复时必须严格对齐这个公式,不能只做 `[::-1]`。

---

## 五、第二个 Bug:`render_graph_feature_map` 坐标错位(已验证)

### 5.1 验证结果(didi_xian region_92,真实数据)

对比 `render_graph_feature_map` 渲染出的路网 mask 与正确 `(x,y)` 渲染的路网:

| 对比 | IoU |
|------|:---:|
| 当前 render(`(r,c)` 当 `(x,y)`)vs 正确 `(x,y)` | **0.040**(几乎不重叠) |
| 当前 render **转置后** vs 正确 `(x,y)` | **0.315**(高 8 倍) |

**结论**:`render_graph_feature_map` 把路网画在了转置位置。`road_feature_map` 与 `image_embeddings` 在空间上不对齐。

### 5.2 这是训练侧 bug 吗?

**严格说不是 train/infer gap**(训练推理都用同样的错误坐标,一致),但它是**更深的语义 bug**:

- `road_feature_map`(ch0 路mask)画在 `(r,c)` 当 `(x,y)` 的转置位置
- `image_embeddings` 是 SAM encoder 在 `(x,y)` 空间提取的视觉特征
- `FeatureFusion` 把两者 concat 融合 → **几何先验和视觉特征空间错位**,先验信号被削弱甚至变成噪声

这解释了为什么 completion 即使修了推理匹配 bug,效果也可能上不去——**训练时 fusion 就在错位特征上学的**。

### 5.3 修复影响

修 `render_graph_feature_map` 坐标会**改变训练输入分布**,需要重训。而修推理匹配坐标系(§三)不改变训练,可复用 ckpt。

**因此分两步**:
1. 先修推理匹配(§三),复用 ckpt 重跑 infer+eval,看 completion 能否回到 ≥ extraction
2. 再修 render 坐标(训练侧),重训 completion,看能否进一步提升

---

## 六、完整修复+验证计划

### 阶段 1:修推理匹配坐标系(无需重训)

**改动**:`engine/inferencer_completion.py`
- 新增 `transform_known_graph_coords(known_graph_adj, dataset)`:按 dataset 把 `(r,c)` 转 `(x,y)`
  - didi_xian/cityscale:`(r,c)→(c,r)`
  - spacenet:`(r,c)→(c, 400-r)`
- `infer_one_img` 开头调用,把 known_graph_adj 统一转 `(x,y)`
- 此后 `_match_known_edges_to_graph_points`、`_inject_known_nodes` 拿到的都是 `(x,y)`,与 graph_points 一致

**验证**:
- 重跑 region_92 匹配实验,边端点正确率应 0%→100%
- `run.py --task completion --dataset didi_xian --steps infer,eval --resume-run`
- 对比 APLS/TOPO 是否回升到 ≥ extraction(0.375/0.42)

**注意**:`render_graph_feature_map` 在推理时也调用。阶段 1 **先不修 render**(保持 train/infer 一致),只修匹配路径。render 仍是转置的,但 train/infer 一致不影响。render 的修复留到阶段 2(需重训)。

### 阶段 2:修 render 坐标系(需重训)

**改动**:`data/dataset_completion.py` 的 `render_graph_feature_map` + `get_known_adj_for_render`
- `get_known_adj_for_render` 返回 `(x,y)` 而非 `(r,c)`(对 known_edges_original 的端点做 coord_transform)
- 或在 `render_graph_feature_map` 内部对 node 坐标做 transform
- 推理侧 `inferencer_completion.py` 的 render 调用同步改

**验证**:
- 画 road_feature_map 与 GT road_mask 叠加图,IoU 应 >0.5
- 重训 completion,重跑 infer+eval,对比阶段 1 是否进一步提升

### 阶段 3:人眼最终确认

修之前,先可视化几张服务器 completion infer 结果,对照 partial prior png,人眼确认 known edges 确实连错。修复后再可视化同样几张,确认连对了。

---

## 七、总结

completion 比 extraction 差,是**两个坐标系 bug 叠加**:

| Bug | 位置 | 影响 | 修复成本 |
|-----|------|------|---------|
| Bug 1:推理匹配坐标系 | `_match_known_edges_to_graph_points`、`_inject_known_nodes` | known edges 100% 连错,强制注入错误边 | 低(改推理,复用 ckpt) |
| Bug 2:render 坐标系 | `render_graph_feature_map`(训练+推理) | road_feature_map 与 image_embeddings 空间错位,先验信号削弱 | 高(改训练,需重训) |

**建议先修 Bug 1**(立竿见影,无需重训),验证 completion 回升后再决定是否修 Bug 2。两个 bug 都是 completion 模型诞生起就存在的预存问题,非本次编排重构引入。

---

## 八、修复实施(2026-06-25,已落地)

### 8.1 修法:入口统一转

新增共享函数 `transform_known_graph_coords(known_graph_adj, dataset)`(在 `data/dataset_completion.py`),按数据集把 pickle `(row,col)` 转 `(x,y)`:
- didi_xian/cityscale:`(r,c)→(c,r)`
- spacenet:`(r,c)→(c, 400-r)`

在**两个入口**调用,下游所有消费点不动:

| 入口 | 文件 | 修复的 Bug |
|------|------|-----------|
| `get_known_adj_for_render()` | `data/dataset_completion.py` | Bug 2(训练侧 render) |
| `load_known_graph()` | `engine/inferencer_completion.py` | Bug 1(推理匹配)+ Bug 2(推理 render) |

转完后 `render_graph_feature_map`、`_match_known_edges_to_graph_points`、`_inject_known_nodes` 拿到的都是 `(x,y)`,与 graph_points / image_embeddings 对齐。这些函数内部不用改(它们的 docstring 本就期待 `(x,y)`,是调用方之前传错了)。

### 8.2 验证结果(真实数据)

**Bug 1(边端点正确率)**:

| 数据集 | 修复前 | 修复后 |
|------|:---:|:---:|
| didi_xian (72 边) | 0% | **100%** |
| spacenet (48 边) | 0% | **100%** |

**Bug 2(render IoU)**:

| 数据集 | 修复前 | 修复后 |
|------|:---:|:---:|
| didi_xian | 0.040 | **0.593** |
| spacenet | 0.019 | **0.609** |

(修复后 IoU 非 1.0 是因为 partial prior 稀疏路网画粗线的 thickness 差异,坐标已对齐)

### 8.3 ⚠️ 关键影响:现有 ckpt 的复用问题

Bug 2 修了训练侧 `get_known_adj_for_render`(现在返回 `(x,y)`),**改变了训练输入分布**。这意味着:

- **现有 completion ckpt**(在 render 转置错位下训练)与新代码(render 对齐)**train/infer 不一致**。用旧 ckpt + 新代码跑 infer,road_feature_map 从"转置"变成"对齐",模型没见过这种输入,效果可能变差。
- **Bug 1 是纯推理修复**,不改训练,但因为它和 Bug 2 同在 `load_known_graph` 一起修了,无法单独验证 Bug 1 对旧 ckpt 的效果。

**因此 completion 必须重训**才能享受两个 bug 修复的完整收益。重训后:
- 训练时 road_feature_map 对齐 → FeatureFusion 真正用上几何先验
- 推理时 known edges 连对 → 不再注入错误边

**重训后预期**:completion 应该回到 ≥ extraction 水平,甚至超过(先验真正发挥作用)。

### 8.4 执行建议

1. **completion 重训**:`run.py --task completion --dataset spacenet --gpus 0` + `--dataset didi_xian`,然后 infer+eval
2. extraction 不受影响,现有 ckpt 可继续用
3. 重训后对比 completion vs extraction,验证 completion 回升


---

## 四、下一步行动建议

1. **先修推理侧匹配坐标系**(3.2),这是确证的 bug,修了直接重跑 infer+eval 看指标回升
2. **再验证训练侧 render 坐标**(3.4.1):画一张 road_feature_map 和 GT road_mask 叠加图,看是否对齐。若不对齐,是第二个 bug,需重训
3. 修复后用 `run.py --steps infer,eval --resume-run` 复用现有 ckpt,无需重训(训练侧坐标系是对的)

> **重要**:在修之前,先让服务器上的 completion infer 结果可视化几张,人工确认 known edges 确实连错(对照 partial prior png),作为 bug 的最终人眼确认。
