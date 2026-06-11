"""
SAM-Road Completion Training Script
====================================
路网补全模型训练脚本

与原版 train.py 的差异:
  - 使用 SAMRoadCompletion 模型
  - 使用 SatMapCompletionDataset 数据集
  - 每个 epoch 刷新已知图 (重新随机删边)
"""

from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.config_utils import load_config
from data.dataset_completion import SatMapCompletionDataset, completion_graph_collate_fn
from models.sam_road_completion import SAMRoadCompletion

import wandb
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor


parser = ArgumentParser()
parser.add_argument("--config", default=None,
                    help="config file (.yml) containing the hyper-parameters for training.")
parser.add_argument("--resume", default=None,
                    help="checkpoint of the last epoch of the model")
parser.add_argument("--precision", default=16, help="32 or 16")
parser.add_argument("--fast_dev_run", default=False, action='store_true')
parser.add_argument("--dev_run", default=False, action='store_true')


class CompletionRefreshCallback(pl.Callback):
    """每个 epoch 开始时刷新已知图 (重新随机删边)"""

    def on_train_epoch_start(self, trainer, pl_module):
        train_loader = trainer.train_dataloader
        if train_loader and hasattr(train_loader, 'dataset'):
            dataset = train_loader.dataset
            if hasattr(dataset, 'refresh_known_graphs'):
                dataset.refresh_known_graphs()
                print(f"[Epoch {trainer.current_epoch}] Refreshed known graphs (re-sampled deleted edges)")


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    dev_run = args.dev_run or args.fast_dev_run

    wandb.init(
        project="sam_road_completion",
        config=config,
        mode='offline'
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    net = SAMRoadCompletion(config)

    train_ds = SatMapCompletionDataset(config, is_train=True, dev_run=dev_run)
    val_ds = SatMapCompletionDataset(config, is_train=False, dev_run=dev_run)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=completion_graph_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=completion_graph_collate_fn,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/samroad_completion/",
        every_n_epochs=1,
        save_top_k=-1
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    refresh_callback = CompletionRefreshCallback()

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor, refresh_callback],
        logger=wandb_logger,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
    )

    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)
