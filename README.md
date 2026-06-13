# SAM-Road: 基于 SAM 的路网图提取

"Segment Anything Model for Road Network Graph Extraction" 官方代码库，CVPRW 2024。

📄 [论文](https://arxiv.org/pdf/2403.16051.pdf) | 🏆 CVPR 2024 第二届场景图与图表征学习研讨会 [最佳论文](https://sites.google.com/corp/view/sg2rl/)

> 原始英文 README 见 [docs/README_original_en.md](docs/README_original_en.md)

---

## 模型变体

### 1. SAM-Road（原始模型）

输入 RGB 卫星影像，经 SAM 编码器后分两路：MapDecoder 生成分割 mask，TopoNet 预测图拓扑。

```
RGB [B,H,W,3]
  → SAM ImageEncoderViT (ViT-B/L/H)
  → image_embeddings [B,256,h,w]
      ├→ MapDecoder → mask [B,H,W,2]  (ch0=keypoint, ch1=road)
      └→ TopoNet → topo_scores [B,N_s,N_p,1]
```

- 损失：`mask_loss (BCE/Focal) + topo_loss (BCE, masked)`
- 支持 LoRA 微调 qkv

### 2. SAM-Road 4ch（4通道模型）

在原始模型基础上扩展为 4 通道输入（RGB + Active 先验 mask），用于迭代式路网更新：

| 改动 | 原始 | 4ch |
|------|------|-----|
| 输入通道 | 3 (RGB) | 4 (RGB + Active) |
| `pixel_mean/std` | `[3]` 维 | `[4]` 维 (第4通道 mean=0, std=1) |
| `patch_embed` | `Conv2d(3,...)` | `Conv2d(4,...)`，第4通道零初始化 |
| LoRA 解冻 | 仅 qkv | qkv + `patch_embed` |

**先验增强策略**（仅训练时）：
- 20%：全黑先验（保持纯视觉生成能力）
- 60%：腐蚀先验（随机擦除块，模拟轨迹中断）
- 20%：完美 GT（学习信任准确先验）
- 验证集：始终全黑先验

### 3. SAM-Road Completion（路网补全模型）

专为路网补全设计，输入 RGB + 已知路网特征图：

```
RGB + road_feature_map [B,4,H,W]
  → SAM Encoder ─┬─ image_embeddings
  → RoadGraphEncoder (CNN) ─┬─ road_embeddings
                             └→ FeatureFusion (1×1 Conv) → fused_features
      ├→ MapDecoder (仅用 image_emb，不混入图信息避免过拟合)
      └→ TopoNetCompletion
           + RoadGraphGNN (编码已知图拓扑)
           pair_proj 输入 +2*graph_dim → topo_scores
```

**路网特征图 4 通道**：道路 mask / 距离场 / 方向场 / 关键节点位置

**RoadGraphGNN**：在已知路网边上做 MultiheadAttention 消息传递，为每个节点生成图拓扑嵌入。

---

## 数据集与坐标系

三个数据集的 **pickle 坐标原点不同**，这是最容易踩坑的地方：

|  | CityScale | SpaceNet | Xian (DiDi) |
|---|---|---|---|
| **图片尺寸** | 2048×2048 | 400×400 | 400×400 |
| **pickle 原点** | 左上（图像坐标） | 左下（数学坐标） | 左上（图像坐标） |
| **coord_transform** | `v[:, ::-1]`（swap） | `np.stack([v[:,1], SIZE-v[:,0]])`（swap+flip-y） | `v[:, ::-1]`（swap） |
| **样本数** | 180 | 2549 | ~400 |
| **Active mask** | ✗ | ✗ | ✓ |

> ⚠️ SpaceNet 的 pickle 使用数学坐标系（y 轴从下向上），需要翻转 y 轴。CityScale/Xian 使用图像坐标系（row 从上向下），只需交换。

**IoU 定量验证**（变换后的 road_mask vs GT.png）：

| 变换 | CityScale | SpaceNet | Xian |
|---|:---:|:---:|:---:|
| swap (cityscale) | **~0.60 ✅** | ~0.03 | **~0.60 ✅** |
| flip-y (spacenet) | ~0.03 | **~0.62 ✅** | ~0.10 |
| raw | ~0.03 | ~0.03 | ~0.01 |

详细分析见 [数据集与坐标系分析](docs/数据集与坐标系分析.md)。

---

## 项目结构

```
sam_road/
├── config/            # 训练配置 (YAML)
├── data/              # 数据集类 & DataLoader
│   ├── dataset.py             # 基础: CityScale / SpaceNet / Xian
│   ├── dataset_4ch.py         # 4ch: + active mask 与先验增强
│   └── dataset_completion.py  # Completion: + 已知图 & 特征图
├── datasets/          # 原始数据 & generate_labels.py
│   ├── cityscale/
│   ├── spacenet/
│   └── didi/xian/
├── docs/              # 文档
│   ├── 数据集与坐标系分析.md   # 坐标系详细分析
│   ├── 方案B_路网补全设计.md   # 补全模型设计文档
│   ├── Train.md                # 训练流程详解
│   ├── nan_fix_completion.md  # 补全模型 fp16 NaN 修复总结
│   ├── README_original_en.md  # 原始英文 README
│   └── imgs/                  # 验证图片（已入库）
├── engine/            # 训练 & 推理
│   ├── train.py / train_4ch.py / train_completion.py
│   ├── inferencer.py / inferencer_4ch.py / inferencer_completion.py
│   └── test.py
├── models/            # 模型定义
│   ├── sam_road.py            # 原始模型
│   ├── sam_road_4ch.py        # 4通道模型
│   └── sam_road_completion.py # 补全模型
├── outputs/           # 生成物（gitignore）
│   ├── experiments/   # 实验参数 CSV
│   ├── logs/          # 训练/评估日志
│   └── viz/           # 可视化图片
├── postprocess/       # 图提取、triage
├── scripts/           # 验证 & 可视化脚本
├── sam/               # SAM 子模块
├── tools/             # 工具集 (plot_loss, param_exps 等)
├── metrics/           # APLS & TOPO 评估
├── conda_only.yml
├── myenv.yml
└── pip_requirements.txt
```

---

## 快速开始

### 安装

```bash
# 方式一: conda + pip 分开安装（推荐，避免 pip 卡死）
conda env create -f conda_only.yml
conda activate samroad
pip install -r pip_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方式二: 全量 conda
conda env create -f myenv.yml
```

#### ⚠️ 已知环境坑：`setuptools` 与 `libstdc++` 版本冲突

在某些机器上（尤其是系统 GCC ≤ 11，例如 Debian 12），新建 SAM 环境后第一次 `python -m engine.train ...`
会接连撞两个错误，必须按下面顺序处理。

**坑 1：`ModuleNotFoundError: No module named 'pkg_resources'`**

`setuptools >= 81` 移除了 `pkg_resources`，但 `lightning.fabric` 仍在
`__import__('pkg_resources').declare_namespace(...)`，导致 `import lightning.pytorch` 直接报错。

```bash
conda activate SAM
pip install "setuptools<81"      # 80.x 自带 pkg_resources，长期可用
```

**坑 2：`ImportError: libstdc++.so.6: version 'CXXABI_1.3.15' not found`**

只在导入 `torch` 之后再导入 `scipy` / `lightning` 时触发。原因不是 SAM 环境坏了：
`torch/lib/libnvToolsExt-*.so` 没有正确的 RPATH 指向 `$CONDA_PREFIX/lib`，
`import torch` 时 loader 顺着默认路径加载了系统的旧版 `libstdc++.so.6`
（GCC 11，最高 `CXXABI_1.3.13`），并把这个 SONAME pin 在进程里；
之后 `scipy._highspy` / `pypocketfft` 再 NEEDED `libstdc++.so.6` 时就拿到了旧的，
即使 conda 自己的 `libstdc++.so.6.0.34`（含 `CXXABI_1.3.15`）就在 RPATH 上也用不上。

最干净的修法是给 SAM 环境加一对 conda activate/deactivate 钩子，
让它在 `conda activate SAM` 时自动 `LD_PRELOAD` conda 自己的新版 libstdc++，
deactivate 时自动还原。**只影响 SAM 环境**，新开 tmux 窗口无需任何 export。

```bash
SAM_PREFIX=$(conda env list | awk '$1=="SAM"{print $NF}')
mkdir -p "$SAM_PREFIX/etc/conda/activate.d" "$SAM_PREFIX/etc/conda/deactivate.d"

cat > "$SAM_PREFIX/etc/conda/activate.d/zz-libstdcxx-preload.sh" <<'EOF'
# Force conda's libstdc++ to load before torch pulls in the system one.
if [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ] && [ "${_SAM_LD_PRELOAD_SET:-0}" != "1" ]; then
    export _SAM_OLD_LD_PRELOAD="${LD_PRELOAD-__unset__}"
    export _SAM_LD_PRELOAD_SET=1
    if [ -z "${LD_PRELOAD:-}" ]; then
        export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
    else
        export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$LD_PRELOAD"
    fi
fi
EOF

cat > "$SAM_PREFIX/etc/conda/deactivate.d/zz-libstdcxx-preload.sh" <<'EOF'
if [ "${_SAM_LD_PRELOAD_SET:-0}" = "1" ]; then
    if [ "${_SAM_OLD_LD_PRELOAD-}" = "__unset__" ]; then
        unset LD_PRELOAD
    else
        export LD_PRELOAD="$_SAM_OLD_LD_PRELOAD"
    fi
    unset _SAM_OLD_LD_PRELOAD _SAM_LD_PRELOAD_SET
fi
EOF
```

写完重新 `conda deactivate && conda activate SAM`，`echo $LD_PRELOAD` 应自动指向
`$CONDA_PREFIX/lib/libstdc++.so.6`，从此 `python -m engine.train ...` 直接可用。

> 若以后系统 GCC 升到 12+ 自带 `CXXABI_1.3.15`，可删除这两个钩子；
> 它们只是把 torch 错误 RPATH 的副作用屏蔽掉，治标不治本。

### SAM 权重

下载 [ViT-B 权重](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)，放到：
```
sam_road/sam_ckpts/sam_vit_b_01ec64.pth
```

### 数据准备

按 [RNGDet++](https://github.com/TonyXuQAQ/RNGDetPlusPlus) 说明下载数据集：

| 数据集 | 下载链接 |
|--------|---------|
| SpaceNet | https://drive.google.com/uc?id=1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W |
| CityScale | https://drive.google.com/uc?id=1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H |

放到 `datasets/` 下，然后分别运行 `generate_labels.py`：
```bash
cd datasets/spacenet && python generate_labels.py
cd datasets/cityscale && python generate_labels.py
```

### 训练

```bash
# 原始模型
python engine/train.py --config=config/toponet_vitb_512_cityscale.yaml
python engine/train.py --config=config/toponet_vitb_256_spacenet.yaml

# 4通道模型
python engine/train_4ch.py --config=config/toponet_vitb_256_spacenet.yaml

# 补全模型
python engine/train_completion.py --config=config/toponet_vitb_256_spacenet_completion.yaml
```

#### 多卡机：指定单卡训练

`engine/train.py` 和 `engine/train_completion.py` 内置 `--gpus` 参数（默认 `"0"`，即第 0 块卡），多卡机要训单卡时显式传：

```bash
# 用第 1 块卡训练补全模型
python engine/train_completion.py \
    --config config/toponet_vitb_256_spacenet_completion.yaml \
    --gpus 1

# 用 0,1 两块卡 DDP（如需）
python engine/train_completion.py \
    --config config/toponet_vitb_256_spacenet_completion.yaml \
    --gpus 0,1
```

`engine/train_4ch.py` 暂未实现 `--gpus`，需要用 `CUDA_VISIBLE_DEVICES` 环境变量遮罩：

```bash
CUDA_VISIBLE_DEVICES=1 python engine/train_4ch.py \
    --config config/toponet_vitb_256_spacenet.yaml
```

> 提示：`CUDA_VISIBLE_DEVICES` 也可以叠在 `--gpus` 上，先遮罩再选卡。常用于把整张卡从其他用户那"独占"出来：`CUDA_VISIBLE_DEVICES=2,3 ... --gpus 0` = 物理 GPU 2 被映射成逻辑 0。

#### `--precision`：选混合精度档位

三个 train 脚本都接受 `--precision`（默认 `16`，即 fp16 mixed precision）：

| 取值 | 显存（B=16, ViT-B 256）| 速度 | 硬件门槛 |
|---|---|---|---|
| `16` / `16-mixed` | ~6–8 GB | 1.0× | Pascal 起 |
| `bf16-mixed` | ~6–8 GB | 0.95–1.0× | **Ampere (sm_80) 起** |
| `32` / `32-true` | ~13–16 GB | 0.5–0.6× | 任意 |

按硬件选择经验：

```bash
# 4090 / 3090 / A100：bf16 最稳，零代价
python engine/train_completion.py --config <cfg> --gpus 0 --precision bf16-mixed

# 2080 Ti：bf16 不被原生支持，留 fp16
python engine/train_completion.py --config <cfg> --gpus 0 --precision 16

# 论文最终对照：fp32 (4090 24G 跑 B=16 fp32 约 ~13 GB)
python engine/train_completion.py --config <cfg> --gpus 0 --precision 32
```

> ⚠️ 2080 Ti 强行用 `bf16-mixed` 不会报错但会走慢速 emulation（约 0.4× fp16 速度），不建议。

> 💡 补全模型 (`engine/train_completion.py`) 在 fp16 下已加六层 NaN 防御
> （梯度裁剪 + attention mask 改 -1e4 + RoadGraphGNN/mask/topo 输出 nan_to_num
> + topo_loss 布尔索引 + `on_after_backward` 梯度清零），fast_dev_run 与长训均稳定。
> 完整原因分析、修复细节和测试记录见 [docs/nan_fix_completion.md](docs/nan_fix_completion.md)。

#### 快速冒烟测试（fast dev run）

正式训练前先用 `--fast_dev_run` 跑 1 个 batch（含 1 个 val batch），
用来验证数据管线、模型 forward、loss、checkpoint 目录权限是否都 OK。
这个模式**不会写 ckpt、不会发 logger，只在几秒内跑完一个完整循环**：

```bash
# 原始模型（spacenet/cityscale 任一 config 都行）
python engine/train.py            --config=config/toponet_vitb_256_spacenet.yaml --fast_dev_run --gpus 0

# 4通道模型 (无 --gpus，用 CUDA_VISIBLE_DEVICES)
CUDA_VISIBLE_DEVICES=0 python engine/train_4ch.py --config=config/toponet_vitb_256_spacenet.yaml --fast_dev_run

# 补全模型
python engine/train_completion.py --config=config/toponet_vitb_256_spacenet_completion.yaml --fast_dev_run --gpus 0
```

如果想跑得久一点观察 loss 走势但仍不写 ckpt，可以改用 `--dev_run`
（数据集会在 dev 模式下取小子集，但走的是完整 trainer，会有真实 epoch / 验证）。

### 推理

```bash
python engine/inferencer.py --config=<config> --checkpoint=<ckpt_path>
python engine/inferencer_4ch.py --config=<config> --checkpoint=<ckpt_path>
python engine/inferencer_completion.py --config=<config> --checkpoint=<ckpt_path>
```

### 评估

```bash
cd metrics
python eval.py --dataset spacenet --dir <输出目录>
```

### 坐标系验证

```bash
# 定量验证 (IoU, ~30秒)
python scripts/quantitative_verify.py

# 可视化验证 (对齐图, ~2分钟)
python scripts/visualize_dataset.py --dataset cityscale
python scripts/visualize_dataset.py --dataset spacenet
```

---

## 预训练权重

官方权重: [congrui/sam_road](https://huggingface.co/congrui/sam_road)

---

## 引用

```bibtex
@article{hetang2024segment,
  title={Segment Anything Model for Road Network Graph Extraction},
  author={Hetang, Congrui and Xue, Haoru and Le, Cindy and Yue, Tianwei and Wang, Wenping and He, Yihui},
  journal={arXiv preprint arXiv:2403.16051},
  year={2024}
}
```

## 致谢

- [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [RNGDet++](https://github.com/TonyXuQAQ/RNGDetPlusPlus)
- SAMed, Detectron2
