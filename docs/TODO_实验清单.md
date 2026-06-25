# 实验执行清单 (TODO)

> 前提:partial 已更新为"按连通块保"策略(seed 42,破碎度 26→2-4 块)。
> 服务器需先 `git pull` 拿到最新代码,并重新生成 partial(见下方第0步)。

---

## 第0步:服务器准备(必做,一次性)

```bash
cd ~/sam_road
git pull origin main
conda activate samroad

# 重新生成 partial (按连通块保, seed 42, 覆盖旧破碎版)
python data/generate_partial_prior.py --dataset didi \
    --input_dir datasets/didi/xian/2019_400 --output_dir datasets/didi/xian/2019_400 \
    --keep_ratio 0.5 --seed 42 --strategy component

python data/generate_partial_prior.py --dataset spacenet \
    --input_dir datasets/spacenet/RGB_1.0_meter --output_dir datasets/spacenet/RGB_1.0_meter \
    --keep_ratio 0.5 --seed 42 --strategy component
```

---

## 实验1:spacenet completion 重跑重测

partial 变了,需重训 completion + infer + eval。

```bash
python run.py --task completion --dataset spacenet --gpus 0
# 默认 steps=train,infer,eval, ckpt=auto(best)
# 产物: runs/completion_spacenet_<新时间戳>/
```

---

## 实验2:didi_xian completion 重跑重测

```bash
python run.py --task completion --dataset didi_xian --gpus 0
# 产物: runs/completion_didi_xian_<新时间戳>/
```

> 训练侧 `_create_known_graph` 目前仍是旧的按边随机采样。
> 若要训练也用按连通块保(与推理一致),先改 `data/dataset_completion.py` 再训(见备注)。

---

## 实验3:P2CNet 对比复现(spacenet + didi_xian)

用**你的数据集**(不是 P2CNet 自带的 spacenet)复现 P2CNet,做对比基线。
**问题**:P2CNet 数据格式与 sam_road 不同,需先转换。

### 3a. 数据格式转换(需先实现转换脚本)

P2CNet 期望目录结构:
```
data/<dataset>/
  Vegas/         # 仅 spacenet 按城市分; didi_xian 可单目录
    sats/        # 卫星图 PNG (你的 *_sat.png)
    maps/        # 完整GT路网 mask PNG (你的 road_mask_*.png)
    maps_50/     # 50% partial mask PNG (你的 *_partial.png, 已生成)
```
你的数据是 graph pickle + sat PNG,需转成像素级 mask PNG。
**待实现**: `tools/convert_to_p2cnet_format.py`
- sat: 直接复制/重命名 `*_sat.png` → `sats/`
- maps: 用 `road_mask_*.png`(已有)或从 GT graph 渲染 → `maps/`
- maps_50: 用已生成的 `*_partial.png` → `maps_50/`
- 注意坐标系:P2CNet 是像素 mask,需用 `transform_node` 渲染(didi/spacenet 不同)

### 3b. P2CNet 训练 + 测试

```bash
cd ~/research/P2CNet
conda activate <p2cnet环境>

# spacenet
python train_deeplabv3plus_mix_mp_sat_gsam_spacenet.py
python test_deeplabv3plus_mix_mp_sat_gsam_spacenet.py

# didi_xian (用 osm 入口, 改 config data_dir 指向你的 didi 数据)
python train_deeplabv3plus_mix_mp_sat_gsam_osm.py
python test_deeplabv3plus_mix_mp_sat_gsam_osm.py
```

> P2CNet config 需改 `data_dir` 指向转换后的数据;ratio 用 `mix` 或固定 `0.50`。
> P2CNet 的指标是像素级 IoU/F1(非 APLS/TOPO),对比时注意口径。

---

## 备注

1. **训练侧采样统一**:实验1/2 若要训练也用按连通块保,需先改 `data/dataset_completion.py` 的 `_create_known_graph`。开销已验证可忽略(每 epoch +0.3-2s)。不改则训练用旧策略、推理用新策略,有 train/infer gap。
2. **extraction 不用重跑**:extraction 不碰 partial,现有结果(extraction spacenet 0.7012 / didi_xian 0.4288)仍有效。
3. **P2CNet 转换脚本是阻塞项**:实验3 必须先完成 3a 的转换脚本,才能跑 3b。
