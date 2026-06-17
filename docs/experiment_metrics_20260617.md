# 2026-06-17 推理 / Metric 实验汇总

本文整理 `save/` 下 6 个 `infer*` 目录对应的实验：任务、数据集、模型 checkpoint、epoch、推理策略与 APLS / TOPO 指标。

> 说明：
> - APLS / TOPO 来自各目录的 `results/apls.json` 与 `results/topo.json`。
> - `n_eval / n_total` 表示 APLS 实际参与均值计算的样本数；部分样本图过小/不连通会得到 NaN，被脚本跳过。
> - Completion 推理若 `run_info.yaml` 中 `input_graph_dir: null`，表示**未提供已知路网先验**，模型走 full extraction fallback。
> - Xian completion 若 `traj_dir: null`，表示**推理阶段未加载 active.png 轨迹热力图**（训练阶段使用了 active.png）。

---

## 总览表

| 输出目录 | 数据集 | 任务 / 模型 | 策略 | Checkpoint | Epoch | APLS | APLS n | TOPO F1 | TOPO P | TOPO R |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `save/infer__20260617_103051` | SpaceNet | 原始 SAM-Road | 纯 RGB extraction | `checkpoints/samroad_spacenet/epoch-epoch=09-val_loss=0.1244.ckpt` | 9 | **0.7025** | 356/382 | **0.7920** | 0.9288 | 0.6903 |
| `save/infer__20260617_115912` | SpaceNet | 原始 SAM-Road | 纯 RGB extraction | `checkpoints/samroad_spacenet/epoch-epoch=00-val_loss=0.1158.ckpt` | 0 | 0.6202 | 352/382 | 0.7340 | 0.9473 | 0.5991 |
| `save/infer_completion__20260617_103201` | SpaceNet | SAM-Road Completion | **无先验 fallback** (`input_graph_dir=null`) | `checkpoints/samroad_completion/completion-epoch=09-val_loss=0.1288.ckpt` | 9 | 0.6857 | 358/382 | 0.7924 | 0.9301 | 0.6903 |
| `save/infer_completion__20260617_120038` | SpaceNet | SAM-Road Completion | **无先验 fallback** (`input_graph_dir=null`) | `checkpoints/samroad_completion/completion-epoch=00-val_loss=0.1273.ckpt` | 0 | 0.6422 | 352/382 | 0.7391 | 0.9401 | 0.6088 |
| `save/infer__20260617_172655` | DiDi Xian | 原始 SAM-Road | 纯 RGB extraction | `checkpoints/samroad_didi_xian/epoch-epoch=09-val_loss=0.2678.ckpt` | 9 | 0.1085 | 53/58 | **0.2574** | 0.6666 | 0.1595 |
| `save/infer_completion__20260617_172815` | DiDi Xian | SAM-Road Completion | **无先验 / 无轨迹 fallback** (`input_graph_dir=null`, `traj_dir=null`) | `checkpoints/samroad_completion_didi_xian/completion-epoch=09-val_loss=0.2270.ckpt` | 9 | **0.1210** | 52/58 | 0.2437 | 0.6682 | 0.1491 |

---

## SpaceNet 结果

### 原始 SAM-Road (`engine.inferencer`)

| 输出目录 | Checkpoint | Epoch | APLS | TOPO F1 | TOPO P | TOPO R | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| `save/infer__20260617_103051` | `epoch-epoch=09-val_loss=0.1244.ckpt` | 9 | **0.7025** | **0.7920** | 0.9288 | 0.6903 | 当前 SpaceNet 原始模型最佳 |
| `save/infer__20260617_115912` | `epoch-epoch=00-val_loss=0.1158.ckpt` | 0 | 0.6202 | 0.7340 | 0.9473 | 0.5991 | epoch 0 对照 |

结论：SpaceNet 原始模型从 epoch 0 到 epoch 9，APLS +0.0823，TOPO F1 +0.0580，主要来自 recall 提升（0.5991 → 0.6903）。

### Completion (`engine.inferencer_completion`)

| 输出目录 | Checkpoint | Epoch | 策略 | APLS | TOPO F1 | TOPO P | TOPO R | 备注 |
|---|---|---:|---|---:|---:|---:|---:|---|
| `save/infer_completion__20260617_103201` | `completion-epoch=09-val_loss=0.1288.ckpt` | 9 | 无先验 fallback | **0.6857** | **0.7924** | 0.9301 | 0.6903 | TOPO 与原始 ep9 基本持平，APLS 略低 |
| `save/infer_completion__20260617_120038` | `completion-epoch=00-val_loss=0.1273.ckpt` | 0 | 无先验 fallback | 0.6422 | 0.7391 | 0.9401 | 0.6088 | epoch 0 对照 |

结论：Completion 在**未提供已知路网先验**时会退化为 extraction 模式。SpaceNet 上 epoch 9 completion 的 TOPO F1 与原始模型 epoch 9 几乎相同（0.7924 vs 0.7920），但 APLS 低约 0.0168。

---

## DiDi Xian 结果

| 输出目录 | 模型 | Checkpoint | Epoch | 策略 | APLS | TOPO F1 | TOPO P | TOPO R | 备注 |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| `save/infer__20260617_172655` | 原始 SAM-Road | `epoch-epoch=09-val_loss=0.2678.ckpt` | 9 | 纯 RGB extraction | 0.1085 | **0.2574** | 0.6666 | 0.1595 | 当前 Xian baseline |
| `save/infer_completion__20260617_172815` | Completion | `completion-epoch=09-val_loss=0.2270.ckpt` | 9 | 无先验 / 无轨迹 fallback | **0.1210** | 0.2437 | 0.6682 | 0.1491 | 推理未传 `input_graph_dir` / `traj_dir` |

结论：Xian 上 completion checkpoint 在无先验/无轨迹 fallback 模式下，APLS 略高（0.1210 vs 0.1085），但 TOPO F1 略低（0.2437 vs 0.2574）。当前结果不能代表 completion 的完整能力，因为推理时没有传入 `--input_graph_dir` 和 `--traj_dir`。

---

## 后续建议

1. **Completion 正式评估应补跑带先验版本**：
   - SpaceNet：`--input_graph_dir datasets/spacenet/RGB_1.0_meter`
   - Xian：`--input_graph_dir datasets/didi/xian/2019_400/xian_2019_400 --traj_dir datasets/didi/xian/2019_400/xian_2019_400`
2. **Xian 指标显著低于 SpaceNet**，建议确认：
   - `--dataset didi_xian` 评估口径是否使用预期 GT（当前 metric 使用 `region_{}_graph_gt.pickle`，训练使用 refine 图）
   - Xian 坐标系 / 保存图坐标变换是否与 metric 完全一致
3. **APLS 样本数低于总数**：SpaceNet 约 352–358 / 382，Xian 约 52–53 / 58。被跳过的样本多数是 GT / pred 图过小或 APLS Go 输出 NaN，属于指标本身的边界情况；比较不同模型时应同时看 `n_eval / n_total`。

---

## 对应命令模板

```bash
# SpaceNet 原始模型
python -m engine.inferencer \
  --config config/toponet_vitb_256_spacenet_local.yaml \
  --checkpoint "checkpoints/samroad_spacenet/<ckpt>"

# SpaceNet Completion (无先验 fallback)
python -m engine.inferencer_completion \
  --config config/toponet_vitb_256_spacenet_completion.yaml \
  --checkpoint "checkpoints/samroad_completion/<ckpt>"

# Xian 原始模型
python -m engine.inferencer \
  --config config/toponet_vitb_256_xian.yaml \
  --checkpoint "checkpoints/samroad_didi_xian/<ckpt>"

# Xian Completion (建议补跑带轨迹/已知图)
python -m engine.inferencer_completion \
  --config config/toponet_vitb_256_xian_completion.yaml \
  --checkpoint "checkpoints/samroad_completion_didi_xian/<ckpt>" \
  --input_graph_dir datasets/didi/xian/2019_400/xian_2019_400 \
  --traj_dir datasets/didi/xian/2019_400/xian_2019_400

# Metric
cd metrics
python eval.py --dataset spacenet   --dir save/<spacenet_output> --workers 16
python eval.py --dataset didi_xian  --dir save/<xian_output>     --workers 16
```
