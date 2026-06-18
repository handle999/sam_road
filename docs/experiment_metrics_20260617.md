# 2026-06-17 / 2026-06-18 推理 / Metric 实验汇总

本文整理 `save/` 下所有 `infer*` 输出目录对应的实验：任务、数据集、模型 checkpoint、epoch、推理策略与 APLS / TOPO 指标。

> 说明：
> - APLS / TOPO 来自各目录的 `results/apls.json` 与 `results/topo.json`。
> - `n_eval / n_total` 表示 APLS 实际参与均值计算的样本数；部分样本图过小/不连通会得到 NaN，被脚本跳过。
> - Completion 推理若 `run_info.yaml` 中 `input_graph_dir: null`，表示**未提供已知路网先验**，模型走 full extraction fallback。
> - Xian completion 若 `traj_dir: null`，表示**推理阶段未加载 active.png 轨迹热力图**（训练阶段使用了 active.png）。
> - 2026-06-18 的新结果是在调整 config thresholds 后重测的结果；相同 checkpoint 下如果指标变化，主要来自阈值 / NMS / 后处理差异。

---

## 总览表

| 输出目录 | 数据集 | 任务 / 模型 | 策略 | Checkpoint | Epoch | APLS | APLS n | TOPO F1 | TOPO P | TOPO R | 备注 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `save/infer__20260617_103051` | SpaceNet | 原始 SAM-Road | 纯 RGB extraction | `checkpoints/samroad_spacenet/epoch-epoch=09-val_loss=0.1244.ckpt` | 9 | **0.7025** | 356/382 | 0.7920 | 0.9288 | 0.6903 | 旧阈值 |
| `save/infer__20260617_115912` | SpaceNet | 原始 SAM-Road | 纯 RGB extraction | `checkpoints/samroad_spacenet/epoch-epoch=00-val_loss=0.1158.ckpt` | 0 | 0.6202 | 352/382 | 0.7340 | 0.9473 | 0.5991 | epoch 0 对照 |
| `save/infer_completion__20260617_103201` | SpaceNet | SAM-Road Completion | **无先验 fallback** (`input_graph_dir=null`) | `checkpoints/samroad_completion/completion-epoch=09-val_loss=0.1288.ckpt` | 9 | 0.6857 | 358/382 | 0.7924 | 0.9301 | 0.6903 | 旧阈值 |
| `save/infer_completion__20260617_120038` | SpaceNet | SAM-Road Completion | **无先验 fallback** (`input_graph_dir=null`) | `checkpoints/samroad_completion/completion-epoch=00-val_loss=0.1273.ckpt` | 0 | 0.6422 | 352/382 | 0.7391 | 0.9401 | 0.6088 | epoch 0 对照 |
| `save/infer_completion__20260618_142452` | SpaceNet | SAM-Road Completion | **无先验 fallback** (`input_graph_dir=null`) | `checkpoints/samroad_completion/completion-epoch=09-val_loss=0.1288.ckpt` | 9 | 0.6931 | 357/382 | **0.7924** | 0.9287 | 0.6910 | 新阈值后重测 |
| `save/infer__20260617_172655` | DiDi Xian | 原始 SAM-Road | 纯 RGB extraction | `checkpoints/samroad_didi_xian/epoch-epoch=09-val_loss=0.2678.ckpt` | 9 | 0.1083 | 53/58 | 0.2574 | 0.6666 | 0.1595 | 旧训练/旧阈值 |
| `save/infer__20260618_001226` | DiDi Xian | 原始 SAM-Road | 纯 RGB extraction | `checkpoints/samroad_didi_xian/epoch-epoch=09-val_loss=0.2281.ckpt` | 9 | **0.4372** | 52/58 | 0.5455 | 0.9139 | 0.3887 | 新训练/阈值 |
| `save/infer__20260618_143325` | DiDi Xian | 原始 SAM-Road | 纯 RGB extraction | `checkpoints/samroad_didi_xian/epoch-epoch=09-val_loss=0.2281.ckpt` | 9 | 0.4008 | 53/58 | **0.5891** | 0.8576 | 0.4486 | 新阈值后重测 |
| `save/infer_completion__20260617_172815` | DiDi Xian | SAM-Road Completion | **无先验 / 无轨迹 fallback** (`input_graph_dir=null`, `traj_dir=null`) | `checkpoints/samroad_completion_didi_xian/completion-epoch=09-val_loss=0.2270.ckpt` | 9 | 0.1175 | 52/58 | 0.2437 | 0.6682 | 0.1491 | 旧训练/旧阈值 |
| `save/infer_completion__20260618_003142` | DiDi Xian | SAM-Road Completion | **无先验 / 无轨迹 fallback** (`input_graph_dir=null`, `traj_dir=null`) | `checkpoints/samroad_completion_didi_xian/completion-epoch=09-val_loss=0.1868.ckpt` | 9 | 0.4106 | 52/58 | 0.5123 | 0.9239 | 0.3544 | 新训练/旧阈值 |
| `save/infer_completion__20260618_143053` | DiDi Xian | SAM-Road Completion | **无先验 / 无轨迹 fallback** (`input_graph_dir=null`, `traj_dir=null`) | `checkpoints/samroad_completion_didi_xian/completion-epoch=09-val_loss=0.1868.ckpt` | 9 | 0.4133 | 53/58 | **0.6203** | 0.8661 | 0.4832 | 新阈值后重测 |

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
| `save/infer_completion__20260617_103201` | `completion-epoch=09-val_loss=0.1288.ckpt` | 9 | 无先验 fallback | 0.6857 | 0.7924 | 0.9301 | 0.6903 | 旧阈值 |
| `save/infer_completion__20260618_142452` | `completion-epoch=09-val_loss=0.1288.ckpt` | 9 | 无先验 fallback | **0.6931** | **0.7924** | 0.9287 | 0.6910 | 新阈值后重测 |
| `save/infer_completion__20260617_120038` | `completion-epoch=00-val_loss=0.1273.ckpt` | 0 | 无先验 fallback | 0.6422 | 0.7391 | 0.9401 | 0.6088 | epoch 0 对照 |

结论：SpaceNet completion 在新阈值后 APLS 由 0.6857 → 0.6931，TOPO F1 基本不变（0.7924）。与原始模型 epoch 9 相比，TOPO 持平，APLS 低约 0.0094。

---

## DiDi Xian 结果

### 原始 SAM-Road (`engine.inferencer`)

| 输出目录 | Checkpoint | Epoch | APLS | TOPO F1 | TOPO P | TOPO R | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| `save/infer__20260617_172655` | `epoch-epoch=09-val_loss=0.2678.ckpt` | 9 | 0.1083 | 0.2574 | 0.6666 | 0.1595 | 旧训练/旧阈值 |
| `save/infer__20260618_001226` | `epoch-epoch=09-val_loss=0.2281.ckpt` | 9 | **0.4372** | **0.5455** | 0.9139 | 0.3887 | 新训练/阈值，当前已评估最佳 |
| `save/infer__20260618_143325` | `epoch-epoch=09-val_loss=0.2281.ckpt` | 9 | 0.4008 | **0.5891** | 0.8576 | 0.4486 | 新阈值后重测，当前 Xian 原始模型 TOPO 最佳 |

结论：Xian 原始模型换用新训练 checkpoint/阈值后大幅提升；最新重测 TOPO F1 达到 0.5891，APLS 最高仍为 0.4372。

### Completion (`engine.inferencer_completion`)

| 输出目录 | Checkpoint | Epoch | 策略 | APLS | TOPO F1 | TOPO P | TOPO R | 备注 |
|---|---|---:|---|---:|---:|---:|---:|---|
| `save/infer_completion__20260617_172815` | `completion-epoch=09-val_loss=0.2270.ckpt` | 9 | 无先验/无轨迹 fallback | 0.1175 | 0.2437 | 0.6682 | 0.1491 | 旧训练/旧阈值 |
| `save/infer_completion__20260618_003142` | `completion-epoch=09-val_loss=0.1868.ckpt` | 9 | 无先验/无轨迹 fallback | 0.4106 | 0.5123 | 0.9239 | 0.3544 | 新训练/旧阈值 |
| `save/infer_completion__20260618_143053` | `completion-epoch=09-val_loss=0.1868.ckpt` | 9 | 无先验/无轨迹 fallback | **0.4133** | **0.6203** | 0.8661 | 0.4832 | 新阈值后重测 |

结论：Xian completion 使用新 checkpoint + 新阈值后显著提升，TOPO F1 从 0.5123 → 0.6203，主要来自 recall 提升（0.3544 → 0.4832），precision 有所下降（0.9239 → 0.8661）。APLS 变化很小（0.4106 → 0.4133）。

---

## 当前关键结论

1. **SpaceNet**：原始模型 ep9 APLS 仍最高（0.7025）；completion ep9 在无先验 fallback 下 TOPO 持平但 APLS 稍低。
2. **DiDi Xian**：新训练 + 新阈值显著提升，completion 在无先验/无轨迹 fallback 下 TOPO F1 最高（0.6203），但原始模型 APLS 最高（0.4372）。
3. **Completion 的完整能力尚未测完**：当前所有 completion 推理的 `input_graph_dir` / `traj_dir` 都是 `null`，也就是没有使用已知路网和 Xian active.png 轨迹先验。后续需要补跑带先验版本。
4. **APLS 样本数低于总数**：SpaceNet 约 352–358 / 382，Xian 约 52–53 / 58。被跳过的样本多数是 GT / pred 图过小或 APLS Go 输出 NaN，比较模型时应同时看 `n_eval / n_total`。

---

## 后续建议

1. **Completion 正式评估应补跑带先验版本**：
   - SpaceNet：`--input_graph_dir datasets/spacenet/RGB_1.0_meter`
   - Xian：`--input_graph_dir datasets/didi/xian/2019_400/xian_2019_400 --traj_dir datasets/didi/xian/2019_400/xian_2019_400`
2. **Xian 指标口径需继续确认**：metric 当前使用 `region_{}_graph_gt.pickle`，训练使用 refine 图；这保持了历史 metric 口径，但会与训练目标略有差异。
3. **已完成全部当前 `save/infer*` 目录评估**：本文列出的 11 个输出目录均已纳入汇总。

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
