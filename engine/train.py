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

from datetime import datetime

import lightning.pytorch as pl
from engine.callbacks import TextLogCallback, EarlyStoppingCallback
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


parser = ArgumentParser()
parser.add_argument("--patience", default=0, type=int, help="Early stopping patience (0=disabled)")
parser.add_argument(
    "--config",
    default=None,
    help="config file (.yml) containing the hyper-parameters for training. "
    "If None, use the nnU-Net config. See /config for examples.",
)
parser.add_argument(
    "--resume", default=None, help="checkpoint of the last epoch of the model"
)
parser.add_argument(
    "--precision", default=16, help="32 or 16"
)
parser.add_argument(
    "--fast_dev_run", default=False, action='store_true'
)
parser.add_argument(
    "--dev_run", default=False, action='store_true'
)
parser.add_argument(
    "--gpus", default="0", type=str,
    help="GPU id(s) to use, e.g. '0' or '0,1'"
)


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    dev_run = args.dev_run or args.fast_dev_run


    net = SAMRoad(config)

    train_ds, val_ds = SatMapDataset(config, is_train=True, dev_run=dev_run), SatMapDataset(config, is_train=False, dev_run=dev_run)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=graph_collate_fn,
    )

    # ---- Checkpoint: save top-5 by val_loss ----
    dataset_name = config.DATASET
    ckpt_dir = f"checkpoints/samroad_{dataset_name}/"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        every_n_epochs=1,
        mode="min",
        save_top_k=5,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    log_dir = "train_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_log_callback = TextLogCallback(log_path=os.path.join(log_dir, f"samroad_{dataset_name}_{timestamp}.txt"))
    csv_logger = CSVLogger(save_dir="train_logs", name="csv", flush_logs_every_n_steps=10)
    callbacks = [checkpoint_callback, lr_monitor, text_log_callback]
    if args.patience > 0:
        callbacks.append(EarlyStoppingCallback(patience=args.patience))

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        accelerator="gpu",
        devices=[int(g) for g in args.gpus.split(',')],
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=callbacks,
        logger=csv_logger,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
    )

    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume)
