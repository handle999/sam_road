# macOS (Apple Silicon) 环境安装指南

本文档说明在 macOS ARM (Apple Silicon) 上搭建 SAM-Road 开发环境的步骤。

## 与 Linux 环境的主要差异

| 项目 | Linux (原始) | macOS ARM (本文) |
|---|---|---|
| PyTorch | 2.1.2+cu121 (CUDA) | 2.4.0 (CPU) |
| torchvision | 0.16.2+cu121 | 0.19.0 |
| torchaudio | 2.1.2 | 2.4.0 |
| PyG 扩展 wheel 来源 | data.pyg.org (cu121) | data.pyg.org (universal2) |
| pyg_lib | 安装 | 不安装（见下方说明） |
| numpy | >=1.24 | >=1.24,<2.0 |
| opencv-python | >=4.8 | >=4.8,<4.10 |

## 为什么 torch 版本从 2.1.2 升级到 2.4.0？

1. **torch 2.1.2 在 macOS ARM 上无法编译 PyG 扩展包**：`torch-scatter`、`torch-sparse`、`torch-cluster`、`torch-spline-conv` 需要从源码编译，但 torch 2.1.2 的 C++ 头文件与新版 macOS SDK 存在兼容性问题（`is_arithmetic cannot be specialized` 错误）。
2. **PyG 预编译 universal2 wheel 从 torch 2.3.0 起提供**：data.pyg.org 的预编译 wheel 中，包含 arm64 的 universal2 版本从 torch 2.3.0 开始才有，2.1.x 和 2.2.x 只有 x86_64 版本。
3. **选择 2.4.0**：是 2.3.0 之后最稳定的版本，与项目其他依赖兼容性良好。

## 安装步骤

### 1. 创建 conda 环境

```bash
conda create -n samroad python=3.10 -y
conda activate samroad
```

### 2. 安装 PyTorch CPU 版本

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

> 注意：macOS ARM 的 CPU 版本 wheel 从 PyTorch 官方下载，不使用 CUDA index-url。

### 3. 安装 PyG 扩展包

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
```

> 使用 data.pyg.org 的 universal2 预编译 wheel，无需本地编译。

### 4. 安装 torch-geometric

```bash
pip install torch-geometric==2.4.0
```

### 5. 安装其余依赖

```bash
pip install lightning==2.2.1 pytorch-lightning==2.2.1 \
    "torchmetrics>=1.0" "tensorboard>=2.15" \
    "opencv-python>=4.8,<4.10" "Pillow>=10.0" "imageio>=2.31" \
    "scikit-image>=0.21" "numpy>=1.24,<2.0" "scipy>=1.11" \
    "scikit-learn>=1.3" "igraph>=0.11" "rtree>=1.1" "shapely>=2.0" \
    "networkx>=3.1" "addict>=2.4" "pyyaml>=6.0" "tcod>=1.2" \
    "matplotlib>=3.8" "svgwrite>=1.4"
```

> **注意**：`numpy` 必须锁定 `<2.0`，否则 torch 2.4.0 和 torch-scatter 会出现兼容性问题。`opencv-python` 需锁定 `<4.10`，因为 4.10+ 要求 numpy>=2。

### 6. 初始化 SAM 子模块并安装

```bash
cd /path/to/sam_road
git submodule update --init --recursive
cd sam && pip install -e . && cd ..
```

### 7. 验证安装

```bash
python -c "
import torch; print(f'torch: {torch.__version__}')
import torch_scatter; print('torch_scatter: OK')
import torch_sparse; print('torch_sparse: OK')
import torch_cluster; print('torch_cluster: OK')
import torch_geometric; print(f'torch_geometric: {torch_geometric.__version__}')
import lightning; print(f'lightning: {lightning.__version__}')
import segment_anything; print('segment_anything: OK')
print('All imports successful!')
"
```

## 已知问题与注意事项

### pyg_lib 不安装

data.pyg.org 提供的 `pyg_lib` universal2 wheel 链接了系统 Python framework（`/Library/Frameworks/Python.framework/Versions/3.10/Python`），在 conda 环境中加载会报 `OSError: Library not loaded` 错误。卸载 `pyg_lib` 后 `torch_sparse` 等 PyG 扩展包仍可正常工作。

```bash
# 如果已安装 pyg_lib 导致 import 错误，执行：
pip uninstall pyg_lib -y
```

### setuptools 版本

`lightning.fabric` 依赖 `pkg_resources`，需要 `setuptools<81`：

```bash
pip install "setuptools<81"
```

### CUDA 不可用

macOS 上 `torch.cuda.is_available()` 返回 `False`，这是预期行为。所有模型将以 CPU 模式运行，适合做代码验证和简单测试，不适合实际训练。

## 版本兼容矩阵

| 包 | 版本 | 来源 |
|---|---|---|
| python | 3.10 | conda |
| torch | 2.4.0 | download.pytorch.org/whl/cpu |
| torchvision | 0.19.0 | download.pytorch.org/whl/cpu |
| torchaudio | 2.4.0 | download.pytorch.org/whl/cpu |
| torch-scatter | 2.1.2 | data.pyg.org (universal2) |
| torch-sparse | 0.6.18 | data.pyg.org (universal2) |
| torch-cluster | 1.6.3 | data.pyg.org (universal2) |
| torch-spline-conv | 1.2.2 | data.pyg.org (universal2) |
| torch-geometric | 2.4.0 | PyPI |
| lightning | 2.2.1 | PyPI |
| pytorch-lightning | 2.2.1 | PyPI |
| numpy | 1.26.4 | PyPI |
| opencv-python | 4.9.0 | PyPI |
| scipy | 1.15.3 | PyPI |
