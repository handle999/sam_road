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

python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/sam_road_contra_lambda001/epoch=19-step=52920.ckpt --output_dir=sam_road_contra_lambda001_ep20
python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/sam_road_contra_lambda001/epoch=24-step=66150.ckpt --output_dir=sam_road_contra_lambda001_ep25
python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/sam_road_contra_lambda001/epoch=29-step=79380.ckpt --output_dir=sam_road_contra_lambda001_ep30
```


```shell
python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/sam_road_graph/epoch=9-step=13230.ckpt
```

```shell
python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/sam_road_graph/epoch=29-step=39690.ckpt
```

```shell
python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/sam_road_contra/epoch=2-step=7938.ckpt
python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/sam_road_contra/epoch=5-step=15876.ckpt
python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/sam_road_contra/epoch=9-step=26460.ckpt
```

```shell
python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/sam_road_contra_lambda01/epoch=9-step=26460.ckpt
```

```shell
python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=./checkpoints/sam_road_contra_lambda001/epoch=9-step=26460.ckpt
```

```shell
python inferencer.py --config=config/toponet_vitb_256_xian_space.yaml --checkpoint=./checkpoints/spacenet_vitb_256_e10.ckpt --output_dir=xian_sam_road_official_ep10_202512292249

python inferencer.py --config=config/toponet_vitb_512_xian_cityscale.yaml --checkpoint=./checkpoints/cityscale_vitb_512_e10.ckpt --output_dir=xian_sam_road_official_cityscale_ep10_202512292303
```

# Train

```shell
# City-scale dataset:  
python train.py --config=config/toponet_vitb_512_cityscale.yaml

# SpaceNet dataset:  
python train.py --config=config/toponet_vitb_256_spacenet.yaml

CUDA_VISIBLE_DEVICES=7 python train.py --config=config/toponet_vitb_512_cityscale.yaml 2>&1 | tee ./train_logs/sam_road_official_cityscale.txt

CUDA_VISIBLE_DEVICES=6 python train.py --config=config/toponet_vitb_256_spacenet.yaml 2>&1 | tee ./train_logs/sam_road_official_spacenet.txt
```
