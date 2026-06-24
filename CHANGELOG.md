# Changelog

## [2026-06-24] Completion: traj 换真实轨迹 + partial rn 固定采样 (train/infer 一致)

### traj 文件: active.png → traj.png (xian completion)
- completion 模型原先用 `region_{c}_active.png` (DiDi 路网矢量栅格化) 当轨迹输入, 改为
  `region_{c}_traj.png` (DelvMap 真实快递员 GPS 轨迹二值图, 已制备对齐)。
- 训练 [data/dataset_completion.py:547] + 推理 [engine/inferencer_completion.py:112] + config 注释同步。
- spacenet/cityscale 无 traj, 不变。

### partial rn 固定采样 (infer 用, 与训练策略对齐)
- **问题**: 推理 `--input_graph_dir` 原本指向完整的 `refine_gt_graph.p`, 而训练时 rn 是
  随机删边的部分子图 (keep_ratio∈[0.2,0.8], 每 epoch 重采样) —— train/infer 不一致。
- **修复**:
  - [data/generate_partial_prior.py] 加 `--seed` (默认42, 可复现) + 只采样训练/推理实际用的 GT 图
    (didi `refine_gt_graph.p` / spacenet `__gt_graph.p`), 排除评测 GT (graph_gt.pickle) 和 dense 变体。
  - 离线预生成 partial rn (keep_ratio=0.5, seed=42): didi 378 个 + spacenet 2549 个,
    `_partial.p` + `_partial.png` 存原数据目录。
  - [engine/inferencer_completion.py] `load_known_graph` 默认读 `_partial.p` (didi/spacenet 分支),
    与训练"部分子图"策略对齐。想喂完整图做上界测试, 用 `--input_graph` 显式指定。
- **策略**: 训练仍每 epoch 随机 [0.2,0.8] (多样性), infer 固定 0.5 + seed=42 (复现); 两者都是
  "随机选边删边"同一策略, 仅比例/种子固定 —— 符合 train 多样性 / infer 复现的要求。

### 生成 partial rn 命令
```bash
python data/generate_partial_prior.py --dataset didi \
    --input_dir datasets/didi/xian/2019_400 --output_dir datasets/didi/xian/2019_400 \
    --keep_ratio 0.5 --seed 42
python data/generate_partial_prior.py --dataset spacenet \
    --input_dir datasets/spacenet/RGB_1.0_meter --output_dir datasets/spacenet/RGB_1.0_meter \
    --keep_ratio 0.5 --seed 42
```

### 验证
- didi region_0: full edges=184, partial edges=92 (ratio=0.500) ✓
- spacenet: 2549 个 partial 生成, dense 变体正确排除 ✓
- traj.png 加载正常 (region_0 nonzero=71897) ✓

## [2026-06-24] Xian 数据集重建: 对齐 DelvMap traj + 目录扁平化 + NW 编号

### Xian 数据集重建 (didi_xian)
- **bbox 统一到 DelvMap 西安范围**: lat[34.206385, 34.279658] lon[108.917423, 108.99286] (`tools/prepare_dataset/config/xian.json`)，size=400，共 **378 块 (21×18)**（原 575 块作废）。
- **编号改为 NW-first**: tile 从左上角(NW)开始行优先 TL→BR (i=0=最北行, j=0=最西列)。
- **新增真实轨迹模态 `region_{c}_traj.png`**: 从 DelvMap `rawdata/traj_heat.png` (5625×6610 Web Mercator 快递员 GPS 二值热力图) 按经纬度逐像素 Mercator 重投影采样进每个 tile，二值化 + 可选 3×3 闭运算，与 sat 同公式对齐 (traj vs DelvMap 原生栅格 IoU ≈ 0.7-0.8)。
- **卫星图支持本地源**: `download_use_osm.py --sat_source local:<png>` 复用 DelvMap `sat_img.png` (ESRI 不可达时)，按经纬度 Mercator 重投影采样。
- **边缘 tile 补黑一致**: sat/traj/rn/active 超出 DelvMap 大图范围的像素 (下边/右边) 一律补黑；rn/active 用 `clip_bbox` 裁到 DelvMap extent，不画黑边区域的路，四模态黑边对齐。

### 目录结构扁平化 (对齐 cityscale/spacenet)
- region 文件从 `2019_400/xian_2019_400/` 上提到 `2019_400/` (不再双层嵌套)。
- `processed/` 与 `data_split.json` 上提到与 `2019_400/` 同级 (即 `datasets/didi/xian/{2019_400/, processed/, data_split.json}`)。
- 批量更新 9 个 loader/inferencer 路径: `data/dataset.py`, `data/dataset_4ch.py`, `data/dataset_completion.py`, `data/dataset_registry.py`, `data/visualize_coord.py`, `data/img_folder_to_json_list.py`, `engine/inferencer.py`, `engine/inferencer_4ch.py`, `engine/inferencer_completion.py`, `tools/registry.py`。

### 新增脚本
- `tools/prepare_dataset/generate_traj.py`: 生成 `region_{c}_traj.png`，含 sanity 闸 (sat 数量==lat_n×lon_n) 与 `--qc` (extent 闸 + 抽样叠加 + IoU)。
- `tools/prepare_dataset/download_use_osm.py` 新增参数: `--sat_source`, `--sat_local_extent`；新增 NW-first 编号、本地 sat 重投影 (edge zero-pad)、`clip_bbox` 裁 rn/active。
- `datasets/didi/xian/generate_labels.py`: IMAGE_SIZE 修回 400，路径 argparse 化 (注: 该文件在 gitignore 的 datasets/ 下，不入库)。

### 数据划分
- 378 块 → train 302 / val 37 / test 39 (seed=42)，`data_split.json` 重生成。

### 坐标系说明 (澄清)
- didi_xian tile 内部坐标系为 **WGS84 线性像素坐标，左上原点 y-down (北在顶)**，节点坐标 (y, x)，与 CityScale 一致。运行时 loader (`dataset.py:368`, `dataset_4ch.py`) 使用 `coord_transform = v[:, ::-1]` (swap, 无翻转)。
- ⚠️ 注意: `data/dataset_registry.py` 中 `didi_xian` 条目仍标记为 `bottom-left / (y_up,x) / need_y_flip=True` (与 2026-06-06 条目一致)，但运行时 loader 实际走 top-left swap 分支，二者不一致。registry 的 flip 标记目前未被训练主路径使用，属已知遗留，未在本轮修改。

## [2026-06-06] Dataset Restructuring & Coordinate System Fix

### Directory Structure Refactoring
- Consolidated all datasets under `datasets/` directory:
  - `xian/` and `chengdu/` moved to `datasets/didi/xian/` and `datasets/didi/chengdu/`
  - `cityscale/20cities/` and `cityscale/processed/` moved to `datasets/cityscale/ `
  - `spacenet/` metadata moved to `datasets/spacenet/`
- Removed redundant directories: `segment-anything-road/`, root-level `spacenet/`
- Updated `.gitignore` to reflect new dataset paths

### New File: `data/dataset_registry.py`
- Centralized dataset configuration registry to eliminate scattered if/elif chains
- Registered datasets: `cityscale`, `spacenet`, `didi_xian`
- Each entry includes: image_size, sample_margin, coord_origin, coord_format, coord_transform, path patterns, data_partition, need_y_flip_for_sat2graph
- Legacy aliases: `didi` >> `didi_xian`, `xian` >> `didi_xian`
- Utility functions: `get_dataset_config()`, `get_data_partition()`, `convert_pred_nodes_to_sat2graph()`, `convert_gt_nodes_from_sat2graph()`

### Critical Bug Fix: didi_xian Coordinate System
- **Root Cause**: xian `graph_gt.pickle` uses `(y_up, x)` bottom-left coordinate format (same as spacenet), NOT `(row, col)` top-left (like cityscale). This was confirmed via visual comparison of graph overlay on satellite images.
- **Impact**: The wrong `coord_transform = lambda v: v[:, :-1]` (which assumes row,col format) produced incorrect keypoint/road masks during training, leading to misaligned training labels.
- **Fix**:
  - `data/dataset.py`: changed `coord_transform` from `v[:, :-1]` to `np.stack([v[, 1], 400 - v[:, 0]], axis=1)`
  - `data/dataset_4ch.py`: same fix
  - `engine/inferencer_4ch.py`: added `didi_xian`/`xian` to Y-flip conditions (previously only `spacenet` was handled)
  - `engine/inferencer.py`: updated Y-flip conditions to also support `didi_xian` alias
  - `data/dataset_registry.py`: `didi_xian` configured with `coord_origin: bottom-left`, `coord_format: (y_up,x)`, `need_y_flip_for_sat2graph: True`

### Path Updates (all xian/chengdu references updated)
- `data/dataset.py`: `./xian/` >> `datasets/didi/xian/`
- `data/dataset_4ch.py`: `./xian/` >> `datasets/didi/xian/`
- `engine/inferencer.py`: `./xian/` >> `datasets/didi/xian/`
- `engine/inferencer_4ch.py`: `./xian/` >> `datasets/didi/xian/`
- `metrics/eval.py`: `../xian/` >> `../datasets/didi/xian/`
- `metrics/topo/main.py`: `../xian/` >> `../datasets/didi/xian/`
- `metrics/topo/eval_parallel.py`: `../xian/` >> `../datasets/didi/xian/`
- `data/img_folder_to_json_list.py`: `./xian/` >> `./datasets/didi/xian/`
- `tools/crop_png.py`: path updated
- `tools/tptk/common/hhy_txt_to_csv.py`: `./xian/` >> `./datasets/didi/xian/`
- `tools/tptk/common/hhy_path_txt_to_csv.py`: `../xian/` >> `../datasets/didi/xian/`
- `tools/tptk/common/hhy_mm_txt_to_csv.py`: `./xian/` >> `./datasets/didi/xian/`

### Config YAML Updates
- `config/topnet_vitb_256_xian_cityscale.yaml`: `DATASET: didi` >> `DATASET: didi_xian`
- `config/topnet_vitb_256_xian_spacenet.yaml`: `DATASET: xian` >> `DATASET: didi_xian`
- `config/topnet_vitb_512_xian_cityscale.yaml`: `DATASET: xian` >> `DATASET: didi_xian`

### New Tool: `data/visualize_coord.py`
- Utility to visualize `graph_gt.pickle` overlay on satellite images for coordinate system verification
- Supports cityscale (single panel) and didi_xian (3-panel comparison: as-is / Y-flip / swap)
- Usage: `python data/visualize_coord.py --dataset [cityscale|didi_xian] --id [region_id]`

### Coordinate System Reference

| Dataset | Pickle Format | coord_origin | coord_transform to (row,col) | need_y_flip_for_sat2graph |
|---------|------------|---------|-----------------------------------------|-------------------------|
| cityscale | (row, col) | top-left | v[:, :-1] | False |
| spacenet | (y_up, x) | bottom-left | np.stack([v[, 1], 400 - v[:, 0]], axis=1) | True |
| didi_xian | (y_up, x) | bottom-left | np.stack([v[:, 1], 400 - v[:, 0]], axis=1) | True |
