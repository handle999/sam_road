from argparse import ArgumentParser
import os
import sys

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.config_utils import load_config
from data.dataset import SatMapDataset, graph_collate_fn
from models.sam_road import SAMRoad

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


parser = ArgumentParser()
parser.add_argument(
    "--config",
    default=None,
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the nnU-Net config. See /config for examples.",
)
parser.add_argument(
    "--checkpoint", default=None, help="checkpoint of the model to test."
)
parser.add_argument(
    "--precision", default=32, help="32 or 16 (默认 32, 与训练一致)"
)
parser.add_argument(
    "--device", default="auto", help="auto / cpu / gpu (Mac MPS 兼容性差, 可用 cpu 强制 CPU)"
)


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)

    
    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    

    net = SAMRoad(config)

    val_ds = SatMapDataset(config, is_train=False, dev_run=False)

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )

    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer_kwargs = dict(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=args.precision,
    )
    if args.device == "cpu":
        trainer_kwargs.update(accelerator="cpu", devices=1)
    elif args.device == "gpu":
        trainer_kwargs.update(accelerator="gpu")
    # auto: 不指定, lightning 自动选

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.test(net, dataloaders=val_loader, ckpt_path=args.checkpoint)