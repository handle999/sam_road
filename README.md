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
