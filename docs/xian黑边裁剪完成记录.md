# xian 黑边裁剪完成记录

> **状态**:✅ 已完成,肉眼验证裁剪正确(38 个黑边 region 全部裁剪,裁断处补 node,无悬空边)

## 问题

xian 38 个边缘 region(10%)的 sat 有 no-data 黑边,但 rn graph 在黑边区仍有路。rn 覆盖 > sat 覆盖,导致 mask loss/评测失真。

## 方案(方案A)

与 cityscale/spacenet 的 `graph2RegionCoordinate` 同思路:
1. 检测 sat 有效 bbox(非黑边区)
2. Cohen-Sutherland 裁剪 rn graph 到有效区
3. 裁断处补交点为新 node(无悬空边)

## 验证

- region_373(黑边 65%):裁前 gt 路在黑边 35.9% → 裁后 0.16% ✓
- 38 个黑边 region 裁剪后黑边区 node 全部 = 0 ✓
- 可视化(`docs/imgs/xian_clip_compare_batch{1-4}.png`):肉眼确认裁剪正确,黄线(edge)+红点(node)不再出现在黑边区,裁断处有新 node

## 执行流程

```bash
# 1. 裁剪 rn graph
python tools/clip_rn_to_sat.py --dataset didi --input_dir datasets/didi/xian/2019_400
# 2. 重生成 gt.png
python tools/regenerate_labels_after_clip.py --input_dir datasets/didi/xian/2019_400 --dataset didi
# 3. 重生成 road_mask/keypoint_mask
cd datasets/didi/xian/2019_400 && python ../../datasets/didi/xian/generate_labels.py --root . && cd ~/sam_road
# 4. 重生成 partial (edge_random + component)
python data/generate_partial_prior.py --dataset didi --input_dir datasets/didi/xian/2019_400 --output_dir datasets/didi/xian/2019_400 --keep_ratio 0.5 --seed 42 --strategy edge_random
python data/generate_partial_prior.py --dataset didi --input_dir datasets/didi/xian/2019_400 --output_dir datasets/didi/xian/2019_400/partial_component --keep_ratio 0.5 --seed 42 --strategy component
```

## 影响

改完后需重训 sam_road(extraction + completion),didi_xian 所有结果重测。
