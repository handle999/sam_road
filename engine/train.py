from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.config_utils import load_config
from data.dataset import SatMapDataset, graph_collate_fn
from models.sam_road import SAMRoad

import os
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

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/samroad_spacenet/",
        every_n_epochs=1, 
        save_top_k=3, monitor="val_loss", mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    log_dir = "train_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_log_callback = TextLogCallback(log_path=os.path.join(log_dir, f"samroad_spacenet_{timestamp}.txt"))
    csv_logger = CSVLogger(save_dir="train_logs", name="csv", flush_logs_every_n_steps=10)
    callbacks = [checkpoint_callback, lr_monitor, text_log_callback]
    if args.patience > 0:
        callbacks.append(EarlyStoppingCallback(patience=args.patience))

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=callbacks,
        logger=csv_logger,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision,
    )

    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume)
