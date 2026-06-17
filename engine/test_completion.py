"""
Completion 模型的阈值筛选入口.

原始 SAM-Road 在 models/sam_road.py 的 on_test_end 里用 torchmetrics
BinaryPrecisionRecallCurve 一次性算出整条 PR 曲线, 取 F1 最大点作为最佳阈值,
打印 `Best threshold ..., P=... R=... F1=...`。把这三个值抄进 config 即可。

本脚本对 SAMRoadCompletion 做同样的事 (engine/test.py 只支持 SAMRoad)。

用法 (项目根目录 sam_road/ 下):
    python -m engine.test_completion \
        --config config/toponet_vitb_256_xian_completion.yaml \
        --checkpoint checkpoints/samroad_completion_didi_xian/completion-epoch=09-val_loss=XXXX.ckpt

输出三行:
    Best threshold <itsc>, P=... R=... F1=...   -> ITSC_THRESHOLD
    Best threshold <road>, P=... R=... F1=...   -> ROAD_THRESHOLD
    Best threshold <topo>, P=... R=... F1=...   -> TOPO_THRESHOLD
"""
from argparse import ArgumentParser
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from tools.config_utils import load_config
from data.dataset_completion import SatMapCompletionDataset, completion_graph_collate_fn
from models.sam_road_completion import SAMRoadCompletion

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


parser = ArgumentParser()
parser.add_argument(
    "--config",
    default=None,
    help="config file (.yml) containing the hyper-parameters.",
)
parser.add_argument(
    "--checkpoint", default=None, help="checkpoint of the model to test."
)
parser.add_argument(
    "--precision", default=16, help="32 or 16"
)


if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)

    # Good when model architecture/input shape are fixed.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    net = SAMRoadCompletion(config)

    # 用验证集 (is_train=False) 的 patch 做 PR 曲线, 与 test.py 一致
    val_ds = SatMapCompletionDataset(config, is_train=False, dev_run=False)

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_WORKER_NUM,
        pin_memory=True,
        collate_fn=completion_graph_collate_fn,
    )

    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=config.TRAIN_EPOCHS,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=args.precision,
    )

    trainer.test(net, dataloaders=val_loader, ckpt_path=args.checkpoint)
