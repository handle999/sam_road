有很多需要安装的东西

```shell
# 不影响torch1.13.1
pip install pytorch-lightning==1.9.5
pip install lightning==1.9.5
pip install torchmetrics==0.11.4
```

除了torch相关，还有一些其他

```shell
# graph_extraction.py
pip3 install tcod
pip install igraph
pip install scikit-image==0.24.0
# inferencer.py
pip install imageio
# graph
pip install networkx==2.8.8
# metric/topo
pip install svgwrite
```

下面会有fastapi和pydantic冲突，然后涉及到gradio的问题
```shell
  File "C:\Users\1\.conda\envs\SAM\lib\site-packages\fastapi\params.py", line 4, in <module> 
    from pydantic.fields import FieldInfo, Undefined
ImportError: cannot import name 'Undefined' from 'pydantic.fields' (C:\Users\1\.conda\envs\SAM\lib\site-packages\pydantic\fields.py)
对应版本：
pydantic                2.0
lightning               1.9.5
lightning-cloud         0.5.70
lightning-utilities     0.14.2
pytorch-lightning       1.9.5

pip install "pydantic<2"
gradio 4.39.0 requires pydantic>=2.0, but you have pydantic 1.10.21 which is incompatible.
pip install gradio==3.50.2

```

运行还需要修改

```shell
predictor还改了，
line 10
from sam.segment_anything.modeling import Sam
不能从相对路径直接导入？必须从root？
```

## 官方ckpt
```shell
python inferencer.py --config=config/toponet_vitb_512_cityscale.yaml --checkpoint=./checkpoints/cityscale_vitb_512_e10.ckpt
python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/spacenet_vitb_256_e10.ckpt
```

## 自己的ckpt

```shell
python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/sam_road/epoch=19-step=26460.ckpt --output_dir=sam_road_official_ep20
python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/sam_road/epoch=24-step=33075.ckpt --output_dir=sam_road_official_ep25

```

```shell
python inferencer.py --config=config/toponet_vitb_256_xian_space.yaml --checkpoint=./checkpoints/spacenet_vitb_256_e10.ckpt --output_dir=xian_sam_road_official_ep10_202512292249

python inferencer.py --config=config/toponet_vitb_512_xian_cityscale.yaml --checkpoint=./checkpoints/cityscale_vitb_512_e10.ckpt --output_dir=xian_sam_road_official_cityscale_ep10_202512292303

# spacenet
## official
python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/samroad_official_spacenet/epoch=9-step=13230.ckpt --output_dir=spacenet_official_ep10

## 4 channel train
python inferencer_copy.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/samroad_4c_spacenet/epoch=9-step=13230.ckpt --output_dir=spacenet_4c_ep10

## 4 channel updata
## change `inferencer_copy.py` line 290
python inferencer_copy.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/samroad_4c_spacenet/epoch=9-step=13230.ckpt --output_dir=spacenet_4c_ep10_update

python inferencer_copy.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/samroad_4c_update_spacenet/epoch=0-step=1323.ckpt --output_dir=spacenet_4c_update_train_ep1

python inferencer_copy.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/samroad_4c_update_spacenet/epoch=9-step=13230.ckpt --output_dir=spacenet_4c_update_train_ep10

### extract: total 0 gt
python inferencer_copy.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/samroad_4c_update_spacenet/epoch=9-step=13230.ckpt --output_dir=spacenet_4c_update_train_ep10_extract --task=extraction

# update: 25% gt
python inferencer_copy.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/samroad_4c_update_spacenet/epoch=9-step=13230.ckpt --output_dir=spacenet_4c_update_train_ep10_update_25 --task=update

# update: 50% gt
python inferencer_copy.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/samroad_4c_update_spacenet/epoch=9-step=13230.ckpt --output_dir=spacenet_4c_update_train_ep10_update --task=update

# update: 75% gt
python inferencer_copy.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/samroad_4c_update_spacenet/epoch=9-step=13230.ckpt --output_dir=spacenet_4c_update_train_ep10_update_75 --task=update

# full: 100% gt (extreme version: all gt input)
python inferencer_copy.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/samroad_4c_update_spacenet/epoch=9-step=13230.ckpt --output_dir=spacenet_4c_update_train_ep10_full --task=full

# didi
## xian-2019-400
python inferencer_copy.py --config=config/toponet_vitb_256_xian_cityscale.yaml --checkpoint=./checkpoints/samroad_xian_2019_400/epoch=0-step=1794.ckpt --output_dir=didi_xian_ep0
```

# Train

```shell
# City-scale dataset:  
python train.py --config=config/toponet_vitb_512_cityscale.yaml
python train.py --config=config/toponet_vitb_512_cityscale.yaml  --ckpt_path=./checkpoints/samroad_city 2>&1 | tee ./train_logs/sam_road_official_cityscale.txt
python -u train.py --config=config/toponet_vitb_512_cityscale.yaml --ckpt_path=./checkpoints/samroad_city

# SpaceNet dataset:  
python train.py --config=config/toponet_vitb_256_spacenet.yaml
python train.py --config=config/toponet_vitb_256_spacenet.yaml --ckpt_path=./checkpoints/samroad_spacenet 2>&1 | tee ./train_logs/sam_road_spacenet.txt
python -u train.py --config=config/toponet_vitb_256_spacenet.yaml --ckpt_path=./checkpoints/samroad_spacenet
python train.py --config=config/toponet_vitb_256_spacenet.yaml --ckpt_path=./checkpoints/samroad_4c_update_spacenet 2>&1 | tee ./train_logs/sam_road_spacenet_4c_update.txt

CUDA_VISIBLE_DEVICES=7 python train.py --config=config/toponet_vitb_512_cityscale.yaml 2>&1 | tee ./train_logs/sam_road_official_cityscale.txt

CUDA_VISIBLE_DEVICES=6 python train.py --config=config/toponet_vitb_256_spacenet.yaml 2>&1 | tee ./train_logs/sam_road_official_spacenet.txt

# xian dataset
CUDA_VISIBLE_DEVICES=7 python train.py --config=config/toponet_vitb_256_xian_cityscale.yaml 2>&1 | tee ./train_logs/sam_road_xian_cityscale.txt
```


# sample pickle

```shell
python generate_partial_prior.py --dataset spacenet --input_dir ./spacenet/RGB_1.0_meter --output_dir ./spacenet/sample_0.5 --keep_ratio 0.5 --thickness 3

python generate_partial_prior.py --dataset spacenet --input_dir ./spacenet/RGB_1.0_meter --output_dir ./spacenet/sample_0.25 --keep_ratio 0.25 --thickness 3

python generate_partial_prior.py --dataset spacenet --input_dir ./spacenet/RGB_1.0_meter --output_dir ./spacenet/sample_0.75 --keep_ratio 0.75 --thickness 3
```


# new infer (single and auto-param)

```shell
# single
python inferencer_copy.py --checkpoint ./checkpoints/samroad_4c_update_spacenet/epoch=9-step=13230.ckpt --config ./config/toponet_vitb_256_spacenet.yaml --task update --exp_id test_run_01 --edge 2 --ratio 0.5

# single metric
cd spacenet_metrics
# change 'folder'
eval_sam.cmd


# multi (auto, csv)
tmux new -s ablation_run
python param_exps.py
cd spacenet_metrics
eval_params.cmd
cd ..
python params_aggregate_rsts.py
```

