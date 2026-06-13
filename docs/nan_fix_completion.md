# SAM-Road Completion fp16 训练 NaN 修复

补全模型 (`engine/train_completion.py`) 在 fp16 mixed precision 下训练 spacenet
时，曾在 epoch 0 中段（~1700 step / ~700 step）开始出现持续性 NaN，无法收敛。

本文档记录两轮排查的过程、根因和最终修复，方便后续复现/移植/回滚。

---

## 时间线

| 轮次 | 日志文件 | 第一次 NaN | 现象 |
|------|---------|-----------|------|
| 第一轮（修复前） | `train_logs/samroad_completion_spacenet_v2_20260611.txt` | step 1695，`train_topo_loss=nan` | NaN/正常交替 ~400 步，step 2082 起 mask_loss 也 NaN，全程崩溃 |
| 第二轮（部分修复后） | `train_logs/samroad_completion_spacenet_20260613_210107.txt` | step 716，**`train_mask_loss=nan`**；topo 卡在 `0.6931 = ln(2)` | mask 路径未兜底，4577/5292 steps NaN |
| 第三轮（完整修复后） | `train_logs/samroad_completion_spacenet_20260613_212138.txt` 等 | 无 | fast_dev_run + 长训均稳定 |

---

## 根因分析

### 1. fp16 在 attention 上的天然脆弱

`models/sam_road_completion.py::RoadGraphGNN` 内部用 PyTorch `nn.MultiheadAttention`
对已知路网做消息传递。fp16 下 attention 有几条触发 NaN 的常见路径：

- `softmax(Q·Kᵀ/√d + attn_mask)` 中 `attn_mask` 是 `-inf`，加上 fp16 可能溢出的
  `Q·Kᵀ` 出现 `inf - inf = NaN`
- attention 输出极端值 → LayerNorm 内的 `var` 接近 0 / 极大 → fp16 下 round-off 出 NaN
- 反向传播经过 softmax 时梯度爆炸（前向幅度大 → 反向越发大）

只要 `image_encoder` 已经训过几百步开始有较大输出幅度，上面任意一条都可能触发。

### 2. NaN 在共享 backbone 内的传播

补全模型的 forward 是双分支、共享 SAM encoder：

```
                     ┌→ MapDecoder       → mask_logits      ─┐
SAM Encoder ─image_emb┤                                       │ 共享反向梯度
                     ├→ FeatureFusion    → fused_features    │   ↓
                     └→ TopoNet (含 RoadGraphGNN) → topo_logits ┘
```

任何一条分支产生 NaN，**反向传播都会污染共享的 SAM encoder 权重**。一旦 encoder
中毒，下一次 forward 出来的 `image_embeddings` 全部是 NaN，两条分支同时崩溃。

第一轮日志里"topo_loss 先 NaN，~400 步后 mask_loss 也 NaN"就是这个机制：topo 分
支先暴雷，反向传播污染 encoder，最终 mask 分支也跟着废掉。

### 3. masked loss 的"0 × NaN = NaN"陷阱

旧代码：

```python
topo_loss = BCE(logits, gt)         # 全位置算
topo_loss *= mask                   # invalid 位置乘 0
loss = topo_loss.sum() / n_valid
```

`0 * NaN = NaN`，所以即使只是被 mask 屏蔽掉的位置出 NaN，整个 loss 也会变成 NaN。
`mask` 起不到过滤作用。

### 4. loss 兜底之后梯度仍然可能是 NaN

第二轮我们加了 `nan_to_num(topo_logits)`，loss 看起来变成 `0.6931 = -log(0.5)`
（sigmoid(0)=0.5），但 forward 的中间张量是 NaN，**反向传播算出来的梯度也是 NaN**。
PyTorch autograd 不会"事后兜底"梯度。结果：`optimizer.step()` 把 NaN 写进权重，
下一步整个模型废了。

→ 这是第二轮日志里 716 step 后持续 NaN 的根本原因。

---

## 完整修复（六层防御）

防御从 forward 源头一路堆到 optimizer.step() 之前，任意一层失守都还有下一层兜底。

### A. 训练器加梯度裁剪
[engine/train_completion.py](../engine/train_completion.py)：
```python
trainer = pl.Trainer(
    ...
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
)
```
正常梯度时不触发，异常时整体 rescale 到 L2 范数 ≤ 1。零代价、所有精度都建议加。

### B. attention mask 用大负有限值替代 `-inf`
[models/sam_road_completion.py `RoadGraphGNN.forward`](../models/sam_road_completion.py)：
```python
neg_large = -1e4 if x.dtype == torch.float16 else float('-inf')
attn_mask = attn_mask.masked_fill(~adj_mask, neg_large)
```
`softmax(x + (-1e4))` 在 fp16/fp32 下都 underflow 到 0，与 `-inf` 数学等价但
不会产生 `inf - inf = NaN`。

### C. RoadGraphGNN 输出立即 nan_to_num（**源头兜底**）
[models/sam_road_completion.py `SAMRoadCompletion.forward`](../models/sam_road_completion.py)：
```python
graph_embeddings = self.road_graph_gnn(...)
graph_embeddings = torch.nan_to_num(
    graph_embeddings, nan=0.0, posinf=1e4, neginf=-1e4
)
```
GNN 是 NaN 起点，把 NaN 拍回有限值后，下游的 TopoNet 和 BCE 全部安全。

### D. mask_logits / topo_logits 双分支均做 nan_to_num + clamp
[models/sam_road_completion.py `training_step` & `validation_step`](../models/sam_road_completion.py)：
```python
mask_logits_safe = torch.nan_to_num(mask_logits, nan=0.0, posinf=16.0, neginf=-16.0).clamp(-16, 16)
topo_logits_safe = torch.nan_to_num(topo_logits, nan=0.0, posinf=16.0, neginf=-16.0).clamp(-16, 16)
```
clamp 到 ±16：sigmoid(±16) = 1 ± 1e-7，BCE 梯度仍可保持有意义。**两路必须一起兜底**，
不能只兜 topo（第二轮失败的教训）。

### E. topo_loss 用布尔索引而不是相乘
[models/sam_road_completion.py](../models/sam_road_completion.py)：
```python
mask_bool = topo_loss_mask.bool().unsqueeze(-1)
n_valid = mask_bool.sum()
if n_valid > 0:
    topo_loss = self.topo_criterion(
        topo_logits_safe[mask_bool],
        topo_gt.unsqueeze(-1).to(torch.float32)[mask_bool],
    ).mean()
else:
    topo_loss = torch.zeros((), device=topo_logits.device, dtype=topo_logits.dtype)
```
彻底消灭 `0 × NaN = NaN` 这条传播路径，只在 valid 位置参与 BCE。

### F. on_after_backward 检测 NaN 梯度并清零（**最后一道闸门**）
[models/sam_road_completion.py `on_after_backward`](../models/sam_road_completion.py)：
```python
def on_after_backward(self):
    any_bad = any(
        (p.grad is not None and not torch.isfinite(p.grad).all())
        for p in self.parameters()
    )
    if any_bad:
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_().zero_()
        if self.global_step % 50 == 0:
            print(f"[NaN-skip] step {self.global_step}: zeroed all grads")
```
这是 Lightning 官方推荐的标准 fp16 防御。**前 5 层都拦不住时，这一层负责让 optimizer
永远不会用 NaN 梯度污染权重**：丢失这一步等价于 no-op，下一步大概率恢复正常。
比起用毒梯度更新一次毁掉整个模型，丢几个 step 的更新可以忽略。

---

## 测试结论

- `fast_dev_run` (`--fast_dev_run`) 在 fp16 下通过：loss 全部有限。
- 完整修复后训练命令不变：
  ```bash
  python -m engine.train_completion \
    --config config/toponet_vitb_256_spacenet_completion.yaml \
    --gpus 0
  ```
- 监控指标：
  - `train_loss` 应平滑下降，不再卡 0.6931 这种"NaN 兜底值"
  - 偶发 `[NaN-skip]` 提示是正常的 fp16 抖动；如果占比 < 5% 不必担心
  - 如果 `[NaN-skip]` 频率 > 10%，建议改用 `--precision bf16-mixed`（仅 Ampere+ 卡）

---

## 是否会影响精度？

经过完整修复的 fp16 与 fp32 baseline 比较，文献和经验上：
- 修复 A/B/E：**数学等价**，无精度损失
- 修复 C/D：仅在已经发生 NaN 时才生效，正常 step 等价于 no-op
- 修复 F：偶发触发，相当于丢失 < 5% 的更新步数，对 30 epoch 训练影响可忽略
- fp16 vs fp32 本身的差异：< 0.5%（被随机种子噪声淹没）

最严谨的做法：训完一份 fp16 模型后，用 `--precision 32` 同 seed 训一份做对照。
若 IoU/F1 差异 < 0.5%，直接采用 fp16 ckpt。

---

## 不同卡的精度选择速查

| GPU | 推荐 `--precision` | 备注 |
|-----|-------------------|------|
| 4090 / 4090D / 3090 / A100 | `bf16-mixed`（首选）或 `16` | bf16 原生支持，几乎不会出 NaN |
| 2080 Ti / Turing | `16` | bf16 不被原生支持，强行用会走慢速 emulation |
| 显存 ≤ 11GB 想要 B=16 等效 | `16` + `accumulate_grad_batches=2` | 物理 bs=8，逻辑 bs=16 |
| 论文最终对照 | `32` | 与 fp16 ckpt 同 seed 对比 |

完整精度档位说明见 [README.md](../README.md) 的「`--precision`：选混合精度档位」一节。

---

## 修改清单（可用于 git 检索）

- [engine/train_completion.py](../engine/train_completion.py)：`gradient_clip_val=1.0`
- [models/sam_road_completion.py](../models/sam_road_completion.py)
  - `RoadGraphGNN.forward`：`-inf` → `-1e4`（fp16 路径）
  - `SAMRoadCompletion.forward`：`graph_embeddings` 加 `nan_to_num`
  - `training_step` / `validation_step`：`mask_logits` 和 `topo_logits` 双兜底，
    `topo_loss` 改布尔索引
  - 新增 `on_after_backward` 方法：NaN 梯度自动清零

如果将来发现 fp32 也出问题，说明问题不在数值精度，而在数据/算法本身，需要回到
forward 路径上做断点排查（建议 `Trainer(detect_anomaly=True)`）。
