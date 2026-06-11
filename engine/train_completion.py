"""
SAM-Road Completion v2 Training Script
========================================
路网补全模型训练脚本

v2 变更:
  - SAMRoadCompletion 模型支持 4ch 输入 (RGB + traj_heatmap)
  - 数据集返回 traj_heatmap (Xian 有, 其他全零)
  - 动态 keep_ratio (U[0.2, 0.8])
  - 每个 epoch 刷新已知图 (重新随机删边)
  - EarlyStopping + Best-5 Checkpoint
"""

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
from data.dataset_completion import SatMapCompletionDataset, completion_graph_collate_fn
from models.sam_road_completion import SAMRoadCompletion

import wandb
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger


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

    # ---- Checkpoint: 保存 val_loss 最小的 top-5 ----
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/samroad_completion/",
        filename="completion-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,  # 额外保存最后一个 epoch
    )

    # ---- Early Stopping: val_loss 连续 5 epoch 不降则停 ----
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    refresh_callback = CompletionRefreshCallback()

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        accelerator="gpu",
        devices=1,  # 单卡训练
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, refresh_callback],
        logger=wandb_logger,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
    )

    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume)
