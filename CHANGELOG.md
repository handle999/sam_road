# Changelog

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
