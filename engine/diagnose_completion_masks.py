"""
诊断 completion 模型 keypoint/road mask 预测是否正常。

问题: completion test 时 keypoint PR=0 R=0, 但训练时 keypoint_iou=0.11 (spacenet)。
本脚本绕过 PR metric, 直接看模型 mask 输出与 GT 的重叠, 区分"模型预测坏"还是"metric坏"。

用法 (服务器, 项目根 sam_road/):
    CUDA_VISIBLE_DEVICES=0 python -m engine.diagnose_completion_masks \
        --config config/toponet_vitb_256_spacenet_completion.yaml \
        --checkpoint checkpoints/samroad_completion/completion-epoch=09-val_loss=0.1288.ckpt \
        --precision 32 --num-batches 5

关键看输出:
  - keypoint: pred高分点 与 GT正样本 的重叠率。若重叠≈0 → 模型keypoint通道坏了
  - road: 同上作为对照 (road正常, 重叠应高)
  - 对比 train(val)时的keypoint_iou: 若这里也算出~0.11, 说明test_step的PR metric有bug
"""
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.config_utils import load_config
from data.dataset_completion import SatMapCompletionDataset, completion_graph_collate_fn
from models.sam_road_completion import SAMRoadCompletion
from torchmetrics.classification import BinaryJaccardIndex, BinaryPrecisionRecallCurve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--precision", default="32", help="32 or 16 (务必和训练一致)")
    parser.add_argument("--num-batches", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    net = SAMRoadCompletion(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(ckpt["state_dict"], strict=True)
    net.eval()
    net.to(device)

    ds = SatMapCompletionDataset(config, is_train=False, dev_run=False)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.DATA_WORKER_NUM, pin_memory=True,
                        collate_fn=completion_graph_collate_fn)

    # 累积所有 patch 的预测和 GT
    kp_pred_all, kp_gt_all = [], []
    rd_pred_all, rd_gt_all = [], []

    # IoU metric (和 validation_step 一致)
    kp_iou = BinaryJaccardIndex(task='binary')
    rd_iou = BinaryJaccardIndex(task='binary')
    # PR metric (和 test_step 一致)
    kp_pr = BinaryPrecisionRecallCurve(ignore_index=-1)
    rd_pr = BinaryPrecisionRecallCurve(ignore_index=-1)

    use_half = args.precision == "16"
    n_done = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break
            rgb = batch['rgb'].to(device)
            traj = batch.get('traj_heatmap', torch.zeros(rgb.shape[0], rgb.shape[1], rgb.shape[2], 1, device=device)).to(device)
            rfm = batch['road_feature_map'].to(device)
            gp = batch['graph_points'].to(device)
            pairs = batch['pairs'].to(device)
            valid = batch['valid'].to(device)
            kei = batch.get('known_edge_index')
            if kei is not None:
                kei = kei.to(device)

            if use_half:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    mask_logits, mask_scores, topo_logits, topo_scores = net(rgb, traj, rfm, gp, pairs, valid, kei)
            else:
                mask_logits, mask_scores, topo_logits, topo_scores = net(rgb, traj, rfm, gp, pairs, valid, kei)

            kp_mask = batch['keypoint_mask'].to(device)  # [B,H,W] float 0/1
            rd_mask = batch['road_mask'].to(device)
            kp_scores = mask_scores[..., 0]  # [B,H,W]
            rd_scores = mask_scores[..., 1]

            # IoU (val 方式, float target)
            kp_iou.update(kp_scores, kp_mask)
            rd_iou.update(rd_scores, rd_mask)
            # PR (test 方式, int32 target)
            kp_pr.update(kp_scores, kp_mask.to(torch.int32))
            rd_pr.update(rd_scores, rd_mask.to(torch.int32))

            # 直接统计重叠
            kp_pred_bin = (kp_scores > 0.5).float()
            rd_pred_bin = (rd_scores > 0.5).float()
            kp_gt_bin = (kp_mask > 0.5).float()
            rd_gt_bin = (rd_mask > 0.5).float()
            print(f"\n[Batch {batch_idx}]")
            print(f"  keypoint scores: min={kp_scores.min():.4f} max={kp_scores.max():.4f} mean={kp_scores.mean():.4f}")
            print(f"  road scores:     min={rd_scores.min():.4f} max={rd_scores.max():.4f} mean={rd_scores.mean():.4f}")
            print(f"  keypoint: pred>0.5 count={int(kp_pred_bin.sum())}, GT positive={int(kp_gt_bin.sum())}, overlap={int((kp_pred_bin*kp_gt_bin).sum())}")
            print(f"  road:     pred>0.5 count={int(rd_pred_bin.sum())}, GT positive={int(rd_gt_bin.sum())}, overlap={int((rd_pred_bin*rd_gt_bin).sum())}")
            print(f"  keypoint scores where GT positive: mean={kp_scores[kp_gt_bin.bool()].mean().item() if kp_gt_bin.sum()>0 else 0:.4f}")
            print(f"  keypoint scores where GT negative: mean={kp_scores[~kp_gt_bin.bool()].mean().item():.4f}")
            n_done += 1

    print("\n" + "="*60)
    print(f"汇总 ({n_done} batches, precision={args.precision})")
    print("="*60)
    print(f"keypoint IoU (val方式): {kp_iou.compute().item():.4f}  <-- 训练时报告的应该~这个值")
    print(f"road IoU (val方式):     {rd_iou.compute().item():.4f}")
    p, r, t = kp_pr.compute()
    f1 = 2*p*r/(p+r)
    idx = torch.nan_to_num(f1, nan=-1).argmax()
    print(f"keypoint PR (test方式): P={p[idx].item():.4f} R={r[idx].item():.4f} F1={f1[idx].item()} thr={t[idx].item():.4f}")
    p, r, t = rd_pr.compute()
    f1 = 2*p*r/(p+r)
    idx = torch.nan_to_num(f1, nan=-1).argmax()
    print(f"road PR (test方式):     P={p[idx].item():.4f} R={r[idx].item():.4f} F1={f1[idx].item():.4f} thr={t[idx].item():.4f}")


if __name__ == "__main__":
    main()
