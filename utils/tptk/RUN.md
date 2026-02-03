
```shell
conda install gdal "numpy<2.0"

# 还会报错
Traceback (most recent call last):
  File "/home//miniconda3/envs/SAM/lib/python3.10/site-packages/osgeo/__init__.py", line 30, in swig_import_helper
    return importlib.import_module(mname)
  File "/home//miniconda3/envs/SAM/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 674, in _load_unlocked
  File "<frozen importlib._bootstrap>", line 571, in module_from_spec
  File "<frozen importlib._bootstrap_external>", line 1176, in create_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found (required by /home//miniconda3/envs/SAM/lib/python3.10/site-packages/osgeo/../../../libgdal.so.37)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home//sam_road/utils/tptk/main.py", line 2, in <module>
    from common.road_network import load_rn_shp
  File "/home//sam_road/utils/tptk/common/road_network.py", line 3, in <module>
    from osgeo import ogr
  File "/home//miniconda3/envs/SAM/lib/python3.10/site-packages/osgeo/__init__.py", line 35, in <module>
    _gdal = swig_import_helper()
  File "/home//miniconda3/envs/SAM/lib/python3.10/site-packages/osgeo/__init__.py", line 32, in swig_import_helper
    return importlib.import_module('_gdal')
  File "/home//miniconda3/envs/SAM/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ModuleNotFoundError: No module named '_gdal'
````

报错含义：你新安装的 `GDAL (libgdal.so.37)` 是用较新的 `GCC` 编译器编译的，它需要版本为 `CXXABI_1.3.15` 的 `C++` 标准库。

冲突点：报错显示系统正在加载 `/lib/x86_64-linux-gnu/libstdc++.so.6`，这是你 `Linux` 系统自带的老版本库，它不支持这个新版本标准。而 `Conda` 环境通常应该优先使用自己的新版库。

所以使用
```shell
conda install -c conda-forge libstdcxx-ng
```

还是不行，所以要尝试降低 `GDAL` 版本

```shell
conda remove gdal libgdal
conda install "gdal=3.6.2" "numpy>=1.21"
```

# xian

## compare
| param(min) | time-seg | time-stat |  objects | points | trajectories |
| :-----: | :----: | :----: | :----: | :---: | :---: |
| raw | - | 1:51:24 | 6,600,129 | 600,845,426 | 8,557,131 |
| 5 | 3:14:02 | 1:50:17 | 6,588,432 | 600,639,126 | 9,019,818 |
| 1 | 2:29:21 | 2:29:21 | 6,572,027 | 592,975,777 | 23,826,622 |

## command
```shell
cd ./utils
# (SAM) hanhaoyu@debian:~/sam_road/utils$

python -m tptk.main --phase segment_mm --mm_traj_dir ../xian/filtered_mm_traj --segment_output_dir ../xian/seg_mm_traj_5

python -m tptk.main --phase stat --clean_traj_dir ../xian/seg_mm_traj_5/

python -m tptk.common.hhy_txt_to_csv -i ../xian/seg_mm_traj -o ../xian/seg_mm_traj_csv

python -m tptk.common.hhy_mm_txt_to_csv -i ../xian/filtered_mm_traj -o ../xian/filtered_mm_traj_csv

python -m tptk.common.hhy_mm_txt_to_csv -i ../xian/seg_mm_traj_5 -o ../xian/seg_mm_traj_5_csv

```

## raw

```shell
Namespace(tdrive_root_dir=None, clean_traj_dir='../xian/filtered_mm_traj/', rn_path=None, mm_traj_dir=None, segment_output_dir=None, phase='stat')
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [1:51:24<00:00,  4.68s/it]
#objects:6600129
#points:600845426
#trajectories:8557131

```

## seg-5min

```shell

Namespace(tdrive_root_dir=None, clean_traj_dir=None, rn_path=None, mm_traj_dir='../xian/filtered_mm_traj', segment_output_dir='../xian/seg_mm_traj', phase='segment_mm')
Start segmenting MM trajectories from ../xian/filtered_mm_traj...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [3:14:02<00:00,  8.15s/it]
Segmentation done.


Namespace(tdrive_root_dir=None, clean_traj_dir='../xian/seg_mm_traj/', rn_path=None, mm_traj_dir=None, segment_output_dir=None, phase='stat')
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [1:50:17<00:00,  4.63s/it]
#objects:6588432
#points:600639126
#trajectories:9019818


Namespace(input='../xian/seg_mm_traj', output='../xian/seg_mm_traj_csv')
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [1:59:06<00:00,  5.00s/it]

[INFO] All files processed successfully.
```

## seg-1min

```shell
Namespace(tdrive_root_dir=None, clean_traj_dir=None, rn_path=None, mm_traj_dir='../xian/filtered_mm_traj', segment_output_dir='../xian/seg_mm_traj', phase='segment_mm')
Start segmenting MM trajectories from ../xian/filtered_mm_traj...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [3:09:35<00:00,  7.97s/it]
Segmentation done.

Namespace(tdrive_root_dir=None, clean_traj_dir='../xian/seg_mm_traj/', rn_path=None, mm_traj_dir=None, segment_output_dir=None, phase='stat')
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [2:29:21<00:00,  6.28s/it]
#objects:6572027
#points:592975777
#trajectories:23826622

Namespace(input='../xian/seg_mm_traj', output='../xian/seg_mm_traj_csv')
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [2:06:06<00:00,  5.30s/it]

[INFO] All files processed successfully.

```

# chengdu

```shell
cd ./utils

python -m tptk.main \
    --phase segment_mm \
    --mm_traj_dir ../chengdu/filtered_mm_traj \
    --segment_output_dir ../chengdu/seg_mm_traj

Namespace(tdrive_root_dir=None, clean_traj_dir=None, rn_path=None, mm_traj_dir='../chengdu/filtered_mm_traj', segment_output_dir='../chengdu/seg_mm_traj', phase='segment_mm')
Start segmenting MM trajectories from ../chengdu/filtered_mm_traj...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1451/1451 [4:38:02<00:00, 11.50s/it]
Segmentation done.

```

```shell
cd ./utils

python -m tptk.main --phase stat --clean_traj_dir ../chengdu/seg_mm_traj/

Namespace(tdrive_root_dir=None, clean_traj_dir='../chengdu/seg_mm_traj/', rn_path=None, mm_traj_dir=None, segment_output_dir=None, phase='stat')
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1449/1449 [2:46:45<00:00,  6.90s/it]
#objects:11672503
#points:872526100
#trajectories:14452855

```

```shell
cd ./utils

python -m tptk.common.hhy_txt_to_csv -i ../chengdu/seg_mm_traj -o ../chengdu/seg_mm_traj_csv

Namespace(input='../chengdu/seg_mm_traj', output='../chengdu/seg_mm_traj_csv')
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1449/1449 [2:53:22<00:00,  7.18s/it]

[INFO] All files processed successfully.

```
