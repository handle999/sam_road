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

from tools.config_utils import load_config, ensure_run_dirs, mark_step_done
from tools.run_info import dump_run_info, mark_run_finished
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
parser.add_argument("--run-root", default=None,
    help="编排层注入: 若提供, ckpt/log 写到 {run-root}/train/ 下; 否则走默认 checkpoints/ 路径")


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

    # ---- Checkpoint & 日志路径: 优先走 run-root 统一目录, 否则老路径 ----
    dataset_name = config.DATASET
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_root:
        paths = ensure_run_dirs(os.path.basename(args.run_root.rstrip('/')))
        # run_root 形如 runs/{run_id}, 取末段做 run_id
        import os as _os
        run_id = _os.path.basename(args.run_root.rstrip('/'))
        from tools.registry import run_paths
        rp = run_paths(run_id)
        ckpt_dir = rp['ckpt_dir']
        text_log_path = rp['train_log']
        csv_logger = CSVLogger(save_dir=rp['train_csv'], name="csv", flush_logs_every_n_steps=10)
        log_dir = rp['train_dir']
    else:
        ckpt_dir = f"checkpoints/samroad_{dataset_name}/"
        log_dir = "train_logs"
        os.makedirs(log_dir, exist_ok=True)
        text_log_path = os.path.join(log_dir, f"samroad_{dataset_name}_{timestamp}.txt")
        csv_logger = CSVLogger(save_dir="train_logs", name="csv", flush_logs_every_n_steps=10)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        every_n_epochs=1,
        mode="min",
        save_top_k=-1,
        save_last=True,
    )
    text_log_callback = TextLogCallback(log_path=text_log_path)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor, text_log_callback]
    if args.patience > 0:
        callbacks.append(EarlyStoppingCallback(patience=args.patience))

    # 写运行元信息: run_root 模式写到 train_dir, 老模式仍写两份
    os.makedirs(ckpt_dir, exist_ok=True)
    run_info_path = dump_run_info(
        output_dir=ckpt_dir,
        script=__file__,
        args=args,
        config_source=args.config,
        extra={'task': 'train', 'model': 'sam_road', 'text_log': text_log_path},
        filename=f'run_info_{timestamp}.yaml',
    )
    if not args.run_root:
        dump_run_info(
            output_dir=log_dir,
            script=__file__,
            args=args,
            config_source=args.config,
            extra={'task': 'train', 'model': 'sam_road', 'ckpt_dir': ckpt_dir},
            filename=f'run_info_{timestamp}.yaml',
        )

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
    mark_run_finished(run_info_path)
    if args.run_root:
        mark_step_done(os.path.basename(args.run_root.rstrip('/')), 'train')
