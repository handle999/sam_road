# SAM-Road: Road Network Graph Extraction with SAM

Official codebase for "Segment Anything Model for Road Network Graph Extraction", CVPRW 2024.

📄 [Paper](https://arxiv.org/pdf/2403.16051.pdf) | 🏆 [Best Paper Award](https://sites.google.com/corp/view/sg2rl/) at CVPR 2024, 2nd Workshop on Scene Graphs and Graph Representation Learning

---

## Model Variants

### 1. SAM-Road (Original)

Input: RGB satellite image → SAM Encoder → MapDecoder (mask) + TopoNet (topology)

```
RGB [B,H,W,3]
  → SAM ImageEncoderViT (ViT-B/L/H)
  → image_embeddings [B,256,h,w]
      ├→ MapDecoder → mask [B,H,W,2]  (keypoint + road)
      └→ TopoNet → topo_scores [B,N_s,N_p,1]
```

- Loss: `mask_loss (BCE/Focal) + topo_loss (BCE, masked)`
- Supports LoRA fine-tuning on qkv

### 2. SAM-Road 4ch

Extends input to 4 channels (RGB + Active Prior Mask) for iterative road graph update:

| Modification | Original | 4ch |
|---|---|---|
| Input channels | 3 (RGB) | 4 (RGB + Active) |
| `pixel_mean/std` | `[3]` | `[4]` (4th: mean=0, std=1) |
| `patch_embed` | `Conv2d(3,...)` | `Conv2d(4,...)`, 4th channel zero-init |
| LoRA unfreeze | qkv only | qkv + `patch_embed` |

**Prior augmentation** (training only):
- 20%: all-black prior (preserve pure visual ability)
- 60%: eroded prior (random block removal, simulate trajectory interruption)
- 20%: perfect GT (learn to trust accurate prior)
- Val set: always all-black prior

### 3. SAM-Road Completion

Designed for road network completion with known partial graph:

```
RGB + road_feature_map [B,4,H,W]
  → SAM Encoder ─┬─ image_embeddings
  → RoadGraphEncoder (CNN) ─┬─ road_embeddings
                             └→ FeatureFusion (1×1 Conv) → fused_features
      ├→ MapDecoder (image_emb only, no graph leakage)
      └→ TopoNetCompletion
           + RoadGraphGNN (known graph topology encoding)
           pair_proj input +2*graph_dim → topo_scores
```

**Road feature map** (4 channels): road mask / distance field / direction field / keypoint positions

**RoadGraphGNN**: MultiheadAttention message passing on known graph edges → graph topology embedding per node.

---

## Datasets & Coordinate Systems

Three datasets with **different pickle coordinate origins**:

| | CityScale | SpaceNet | Xian (DiDi) |
|---|---|---|---|
| **Image size** | 2048×2048 | 400×400 | 400×400 |
| **Pickle origin** | Left-Top (image) | Left-Bottom (math) | Left-Top (image) |
| **coord_transform** | `v[:, ::-1]` (swap) | `np.stack([v[:,1], SIZE-v[:,0]])` (swap+flip-y) | `v[:, ::-1]` (swap) |
| **Samples** | 180 | 2549 | ~400 |
| **Active mask** | ✗ | ✗ | ✓ |

> ⚠️ SpaceNet uses mathematical coordinates (y↑ from bottom), requiring y-axis flip. CityScale/Xian use image coordinates (row↓ from top), only need swap.

**Verified by IoU against GT.png**:

| Transform | CityScale | SpaceNet | Xian |
|---|:---:|:---:|:---:|
| swap (cityscale) | **~0.60 ✅** | ~0.03 | **~0.60 ✅** |
| flip-y (spacenet) | ~0.03 | **~0.62 ✅** | ~0.10 |
| raw | ~0.03 | ~0.03 | ~0.01 |

See [数据集与坐标系分析](docs/数据集与坐标系分析.md) for detailed analysis and verification.

---

## Project Structure

```
sam_road/
├── config/            # Training configs (YAML)
├── data/              # Dataset classes & dataloaders
│   ├── dataset.py             # Base: CityScale / SpaceNet / Xian
│   ├── dataset_4ch.py         # 4ch: + active mask & prior augmentation
│   └── dataset_completion.py  # Completion: + known graph & feature map
├── datasets/          # Raw data & generate_labels.py
│   ├── cityscale/
│   ├── spacenet/
│   └── didi/xian/
├── docs/              # Documentation
├── engine/            # Training & inference scripts
│   ├── train.py / train_4ch.py / train_completion.py
│   ├── inferencer.py / inferencer_4ch.py / inferencer_completion.py
│   └── test.py
├── models/            # Model definitions
│   ├── sam_road.py            # Original
│   ├── sam_road_4ch.py        # 4ch
│   └── sam_road_completion.py # Completion
├── outputs/           # Generated outputs (gitignored)
│   ├── experiments/
│   ├── logs/
│   └── viz/
├── postprocess/       # Graph extraction, triage
├── scripts/           # Verification & visualization scripts
├── sam/               # SAM submodule
├── tools/             # Utilities (plot_loss, param_exps, etc.)
├── metrics/           # APLS & TOPO evaluation
├── conda_only.yml
├── myenv.yml
└── pip_requirements.txt
```

---

## Getting Started

### Installation

```bash
# Option 1: conda + pip
conda env create -f conda_only.yml
conda activate samroad
pip install -r pip_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Option 2: full conda
conda env create -f myenv.yml
```

### SAM Preparation

Download [ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place at:
```
sam_road/sam_ckpts/sam_vit_b_01ec64.pth
```

### Data Preparation

Download datasets following [RNGDet++](https://github.com/TonyXuQAQ/RNGDetPlusPlus) instructions:

| Dataset | Link |
|---------|------|
| SpaceNet | https://drive.google.com/uc?id=1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W |
| CityScale | https://drive.google.com/uc?id=1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H |

Place under `datasets/` and run `generate_labels.py` for each:
```bash
cd datasets/spacenet && python generate_labels.py
cd datasets/cityscale && python generate_labels.py
```

### Training

```bash
# Original model
python engine/train.py --config=config/toponet_vitb_512_cityscale.yaml
python engine/train.py --config=config/toponet_vitb_256_spacenet.yaml

# 4ch model
python engine/train_4ch.py --config=config/toponet_vitb_256_spacenet.yaml

# Completion model
python engine/train_completion.py --config=config/toponet_vitb_256_spacenet_completion.yaml
```

### Inference

```bash
python engine/inferencer.py --config=<config> --checkpoint=<ckpt_path>
python engine/inferencer_4ch.py --config=<config> --checkpoint=<ckpt_path>
python engine/inferencer_completion.py --config=<config> --checkpoint=<ckpt_path>
```

### Evaluation

```bash
cd metrics
python eval.py --dataset spacenet --dir <output_dir>
```

### Coordinate Verification

```bash
# Quantitative (IoU, ~30s)
python scripts/quantitative_verify.py

# Visual (alignment checks, ~2min)
python scripts/visualize_dataset.py --dataset cityscale
python scripts/visualize_dataset.py --dataset spacenet
```

---

## Checkpoints

Pre-trained: [congrui/sam_road](https://huggingface.co/congrui/sam_road)

---

## Citation

```bibtex
@article{hetang2024segment,
  title={Segment Anything Model for Road Network Graph Extraction},
  author={Hetang, Congrui and Xue, Haoru and Le, Cindy and Yue, Tianwei and Wang, Wenping and He, Yihui},
  journal={arXiv preprint arXiv:2403.16051},
  year={2024}
}
```

## Acknowledgement

- [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [RNGDet++](https://github.com/TonyXuQAQ/RNGDetPlusPlus)
- SAMed, Detectron2
