# SAM-Road 训练指南

## 目录

- [环境准备](#环境准备)
- [配置文件说明](#配置文件说明)
- [训练命令](#训练命令)
- [输出路径说明](#输出路径说明)
- [命令行参数](#命令行参数)
- [Checkpoint 管理](#checkpoint-管理)
- [日志系统](#日志系统)
- [恢复训练](#恢复训练)
- [常见问题](#常见问题)

---

## 环境准备

```bash
# 激活 conda 环境
conda activate samroad

# 验证依赖
python -c "import torch; import lightning; print(f'torch={torch.__version__}, lightning={lightning.__version__}')"
# 预期输出: torch=2.1.2+cu121, lightning=2.2.1
```

**依赖列表**: torch, lightning (pytorch), torchvision, torchmetrics, opencv-python, rtree, scipy, addict, pyyaml, igraph

**SAM checkpoint**: 确保文件存在 `sam_ckpts/sam_vit_b_01ec64.pth`

---

## 配置文件说明

| 配置文件 | 用途 | BATCH_SIZE | 显存需求 | 模型 |
|---|---|---|---|---|
| `toponet_vitb_256_spacenet.yaml` | 原版 SAMRoad (服务器) | 64 | ~24GB (4xA100) | SAMRoad |
| `toponet_vitb_256_spacenet_local.yaml` | 原版 SAMRoad (本地12GB) | 16 | ~8GB | SAMRoad |
| `toponet_vitb_256_spacenet_completion.yaml` | Completion v2 (本地12GB) | 16 | ~9GB | SAMRoadCompletion |

### 配置项说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DATASET` | `'spacenet'` | 数据集名称 |
| `BATCH_SIZE` | 16/64 | 批大小，12GB 卡建议 16 |
| `DATA_WORKER_NUM` | 1-4 | DataLoader worker 数 |
| `TRAIN_EPOCHS` | 30 | 最大训练轮数 |
| `BASE_LR` | 0.001 | 基础学习率 |
| `FREEZE_ENCODER` | False | 是否冻结 SAM 编码器 |
| `ENCODER_LR_FACTOR` | 0.1 | 编码器学习率缩放 |
| `PATCH_SIZE` | 256 | 输入图像块大小 |
| `TOPO_SAMPLE_NUM` | 128 | 每个 patch 的拓扑采样点数 |
| `FOCAL_LOSS` | False | 是否使用 Focal Loss (否则 BCE) |
| `USE_SAM_DECODER` | False | 是否使用 SAM 原生 decoder |

### Completion 专属参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `GRAPH_DIM` | 32 | GNN 图拓扑嵌入维度 |
| `KEEP_RATIO_MIN` | 0.2 | 动态 keep_ratio 下限 |
| `KEEP_RATIO_MAX` | 0.8 | 动态 keep_ratio 上限 |
| `MODALITY_DROPOUT_PROB` | 0.2 | 全部先验清空概率 |
| `TRAJ_DROPOUT_PROB` | 0.2 | traj 热力图 dropout 概率 |

---

## 训练命令

### 原版 SAMRoad (本地 12GB 卡)

```bash
cd /home/hanhaoyu/workspace/research/sam_road

python engine/train.py \
    --config config/toponet_vitb_256_spacenet_local.yaml \
    --gpus 0 \
    --precision 16
```

### 原版 SAMRoad (服务器多卡)

```bash
python engine/train.py \
    --config config/toponet_vitb_256_spacenet.yaml \
    --gpus 0,1,2,3 \
    --precision 16
```

### Completion v2 (本地 12GB 卡)

```bash
cd /home/hanhaoyu/workspace/research/sam_road

python engine/train_completion.py \
    --config config/toponet_vitb_256_spacenet_completion.yaml \
    --gpus 0 \
    --precision 16
```

### 快速调试 (只跑几个 batch)

```bash
python engine/train.py \
    --config config/toponet_vitb_256_spacenet_local.yaml \
    --gpus 0 \
    --fast_dev_run
```

### 带 Early Stopping

```bash
# 原版: patience=10 (默认关闭, 需手动指定)
python engine/train.py \
    --config config/toponet_vitb_256_spacenet_local.yaml \
    --gpus 0 \
    --patience 10

# Completion: patience=5 (默认开启)
python engine/train_completion.py \
    --config config/toponet_vitb_256_spacenet_completion.yaml \
    --gpus 0 \
    --patience 10
```

### 日志同时保存到文件

```bash
python engine/train.py \
    --config config/toponet_vitb_256_spacenet_local.yaml \
    --gpus 0 \
    2>&1 | tee train_logs/session_$(date +%Y%m%d_%H%M%S).log
```

---

## 输出路径说明

### Checkpoint 输出

| 模式 | 输出目录 | 文件名示例 |
|---|---|---|
| 原版 SAMRoad | `checkpoints/samroad_spacenet/` | `epoch=0-step=5292.ckpt` |
| Completion v2 | `checkpoints/samroad_completion/` | `completion-epoch=02-val_loss=0.4388.ckpt` + `last.ckpt` |

### 日志输出

| 日志类型 | 路径 | 说明 |
|---|---|---|
| **文本日志** | `train_logs/samroad_spacenet_{timestamp}.txt` (原版) / `samroad_completion_spacenet_{timestamp}.txt` (completion) | 每个 step 的 loss，每个 epoch 的 val 指标 |
| **CSV 日志** | `train_logs/csv/version_*/metrics.csv` (原版) / `train_logs/csv_completion/version_*/metrics.csv` (completion) | 结构化指标，可用 pandas 读取画曲线 |

---

## 命令行参数

### 通用参数 (两个脚本共用)

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--config` | str | None | 配置文件路径 (**必填**) |
| `--resume` | str | None | 从 checkpoint 恢复训练 |
| `--precision` | int | 16 | 精度: 16 (fp16/AMP) 或 32 (fp32) |
| `--gpus` | str | `"0"` | GPU id，如 `"0"` 或 `"0,1"` |
| `--fast_dev_run` | flag | False | 只跑 1 个 batch (快速验证) |
| `--dev_run` | flag | False | 使用小数据集调试 |

### 原版 SAMRoad 专属

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--patience` | int | 0 | Early stopping patience (0=关闭) |

### Completion v2 专属

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--patience` | int | 5 | Early stopping patience (0=关闭, 默认5) |

---

## Checkpoint 管理

### 保留策略

| 模式 | save_top_k | save_last | 最大占用 |
|---|---|---|---|
| 原版 SAMRoad | **3** | ❌ | ~1 GB (3×350MB) |
| Completion v2 | **5** | ✅ | ~2.1 GB (5×350MB + last) |

- `save_top_k` 是 FIFO 策略：新 checkpoint 保存后，若已满，自动删除 val_loss 最差的
- `save_last=True` 额外保存最新 epoch 的 checkpoint (可用于恢复训练)
- **不会无限膨胀**，磁盘占用有上限

### 查看 checkpoint

```bash
# 列出所有 checkpoint
ls -lh checkpoints/samroad_spacenet/
ls -lh checkpoints/samroad_completion/

# 查看 checkpoint 内容
python -c "
import torch
ckpt = torch.load('checkpoints/samroad_spacenet/epoch=0-step=5292.ckpt', map_location='cpu')
print('Keys:', list(ckpt.keys()))
print('Epoch:', ckpt.get('epoch'))
print('Global step:', ckpt.get('global_step'))
"
```

---

## 日志系统

训练过程有三种日志输出，全部**无需 wandb/VPN**：

### 1. 文本日志 (TextLogCallback)

实时写入 `.txt` 文件，格式示例：

```
SAM-Road Training Log - 2026-06-12 14:30:00
Epoch 0: 1/5292, train_mask_loss=0.6823, train_topo_loss=0.6934, train_loss=1.3757
Epoch 0: 2/5292, train_mask_loss=0.6701, train_topo_loss=0.6812, train_loss=1.3513
...
Epoch 0 val: val_mask_loss=0.5123, val_topo_loss=0.4921, val_loss=1.0044, keypoint_iou=0.32, road_iou=0.61, topo_f1=0.85
--- End of Epoch 0 ---
```

### 2. CSV 日志 (CSVLogger)

结构化存储，可用 Python 读取画曲线：

```python
import pandas as pd
df = pd.read_csv('train_logs/csv/version_0/metrics.csv')
# 画 loss 曲线
df.dropna(subset=['train_loss']).plot(x='step', y='train_loss')
```

### 3. Terminal 输出

- PL 默认 tqdm 进度条，实时显示 loss
- 验证结束时打印一行 val 指标

---

## 恢复训练

```bash
# 从最后一个 checkpoint 恢复
python engine/train.py \
    --config config/toponet_vitb_256_spacenet_local.yaml \
    --gpus 0 \
    --resume checkpoints/samroad_spacenet/last.ckpt

# 从指定 checkpoint 恢复
python engine/train_completion.py \
    --config config/toponet_vitb_256_spacenet_completion.yaml \
    --gpus 0 \
    --resume checkpoints/samroad_completion/completion-epoch=05-val_loss=0.4388.ckpt
```

---

## 常见问题

### Q: OOM (显存不足)

- 降低 `BATCH_SIZE`：64→32→16→8
- 原版模型 87M params，12GB 卡 batch_size=16 + fp16 约 8GB 显存
- Completion 模型 88M params，12GB 卡 batch_size=16 + fp16 约 9GB 显存

### Q: NaN loss

- 已修复：`topo_loss` 分母加 epsilon (`1e-8`)，`topo_logits` 加 clamp (`-16, 16`) 防止 fp16 BCE 溢出
- 若仍出现，尝试 `--precision 32` 用 fp32 训练

### Q: wandb 报错

- 已完全移除 wandb 依赖，使用 CSVLogger 替代
- 如需 wandb 图像日志，可单独安装 wandb 并用 WandbLogger 替换 CSVLogger

### Q: 多卡训练

```bash
# 双卡 DDP (需相应增大 batch_size)
python engine/train.py \
    --config config/toponet_vitb_256_spacenet.yaml \
    --gpus 0,1
```

注意：多卡自动启用 DDP，`BATCH_SIZE` 是每张卡的 batch size，总 batch = BATCH_SIZE × GPU 数

### Q: 数据集路径

SpaceNet 数据集预期路径 (相对项目根目录):

```
datasets/spacenet/
├── RGB_1.0_meter/           # 卫星影像
│   ├── {tile}__rgb.png      # RGB 图像
│   └── {tile}__gt_graph.p   # GT 图结构
├── processed/               # 预处理标签
│   ├── keypoint_mask_{tile}.png
│   └── road_mask_{tile}.png
└── data_split.json          # 训练/验证/测试划分
```
