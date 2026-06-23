"""
迁移现有 checkpoints/ 到 runs/ 体系, 并补建 profile.yaml
==========================================================
把 checkpoints/samroad_{variant}_{dataset}/ 下的 ckpt + run_info
迁移到 runs/{task}_{dataset}_{ts}/train/checkpoints/, 并按 run_info 反推
补建 profile.yaml, 使其可被 run.py --resume-run 复用继续 infer+eval.

用法:
  python tools/migrate_ckpts_to_runs.py --dry-run   # 预览
  python tools/migrate_ckpts_to_runs.py             # 执行迁移
"""
import os
import sys
import shutil
import argparse
import yaml
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.registry import run_paths
from tools.config_utils import save_config_snapshot, write_profile, select_best_ckpt
from tools.run_info import _try_get_git

# 旧目录 → (task, dataset) 映射 (从 run_info 的 script + config_source 反推)
# 目录名规律:
#   samroad_{dataset}                  = extraction
#   samroad_completion_{dataset}       = completion
#   samroad_4ch_{dataset}              = 4ch
#   samroad_4c_update_{dataset}        = 4ch (历史 update 命名)
# 特例: samroad_completion (无 dataset 后缀) 实际是 spacenet
def parse_old_dir(dirname):
    """从旧 checkpoint 目录名反推 (task, dataset)."""
    if dirname == 'samroad_completion':
        # 历史遗留: 无 dataset 后缀, 实际是 spacenet
        return 'completion', 'spacenet'
    if dirname.startswith('samroad_completion_'):
        ds = dirname[len('samroad_completion_'):]
        return 'completion', ds
    # 注意: 必须在通用 samroad_ 分支之前处理 4ch/4c 历史目录,
    # 否则 samroad_4c_update_spacenet 会被误判为 extraction × 4c_update_spacenet.
    if dirname.startswith('samroad_4ch_'):
        ds = dirname[len('samroad_4ch_'):]
        return '4ch', ds
    if dirname.startswith('samroad_4c_update_'):
        ds = dirname[len('samroad_4c_update_'):]
        return '4ch', ds
    if dirname.startswith('samroad_4c_'):
        ds = dirname[len('samroad_4c_'):]
        return '4ch', ds
    if dirname.startswith('samroad_'):
        ds = dirname[len('samroad_'):]
        return 'extraction', ds
    return None, None


def migrate(dry_run=True):
    old_root = os.path.join(PROJECT_ROOT, 'checkpoints')
    if not os.path.isdir(old_root):
        print('checkpoints/ 不存在, 无需迁移')
        return

    migrated = []
    for dirname in sorted(os.listdir(old_root)):
        old_dir = os.path.join(old_root, dirname)
        if not os.path.isdir(old_dir):
            continue
        task, dataset = parse_old_dir(dirname)
        if task is None:
            print(f'跳过无法识别的目录: {dirname}')
            continue

        # 找 run_info 拿时间戳
        run_infos = [f for f in os.listdir(old_dir) if f.startswith('run_info_') and f.endswith('.yaml')]
        if not run_infos:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            # run_info_20260616_202457.yaml → 20260616_202457
            ri = sorted(run_infos)[0]
            ts = ri[len('run_info_'):-len('.yaml')]

        run_id = f'{task}_{dataset}_{ts}'
        rp = run_paths(run_id)
        new_ckpt_dir = rp['ckpt_dir']
        new_train_dir = rp['train_dir']

        # 找 ckpt 文件
        ckpts = [f for f in os.listdir(old_dir) if f.endswith('.ckpt')]
        print(f'\n{"[DRY] " if dry_run else ""}{dirname} → runs/{run_id}/')
        print(f'  task={task} dataset={dataset} ckpts={ckpts}')

        if dry_run:
            migrated.append((dirname, run_id, ckpts))
            continue

        # 真实迁移
        os.makedirs(new_ckpt_dir, exist_ok=True)
        # 移动 ckpt
        for ckpt in ckpts:
            src = os.path.join(old_dir, ckpt)
            dst = os.path.join(new_ckpt_dir, ckpt)
            shutil.move(src, dst)
            print(f'  moved {ckpt}')
        # 移动 run_info 到 train_dir
        for ri in run_infos:
            src = os.path.join(old_dir, ri)
            dst = os.path.join(new_train_dir, ri)
            shutil.move(src, dst)

        # 补建 profile.yaml
        # 读 run_info 拿 config_source
        config_source = None
        ri_path = os.path.join(new_train_dir, run_infos[0]) if run_infos else None
        if ri_path and os.path.exists(ri_path):
            with open(ri_path) as f:
                ri_data = yaml.safe_load(f)
            config_source = ri_data.get('config_source')

        # 用旧 config 路径的快照 (若文件还在)
        config_snapshot = save_config_snapshot(config_source, run_id)

        profile = {
            'run_id': run_id,
            'task': task,
            'dataset': dataset,
            'created_at': ts,
            'migrated_from': f'checkpoints/{dirname}',
            'config_source': config_source,
            'config_snapshot': config_snapshot,
            'steps_requested': ['train', 'infer', 'eval'],
            'paths': {k: rp[k] for k in ('run_root', 'train_dir', 'ckpt_dir',
                                          'infer_dir', 'eval_dir', 'profile', 'best_ckpt')},
            'git': _try_get_git(PROJECT_ROOT),
            'step_status': {'train': 'done (migrated)'},
            'note': '由 migrate_ckpts_to_runs.py 从旧 checkpoints/ 迁移而来',
        }
        write_profile(run_id, profile)
        # 标记 train 已完成
        from tools.config_utils import mark_step_done
        mark_step_done(run_id, 'train')
        best = select_best_ckpt(run_id)
        if best:
            print(f'  best_ckpt.txt 写入: {best}')
        else:
            print('  ⚠️ 未能选择 best ckpt (train/checkpoints/ 下没有 .ckpt?)')
        print(f'  profile.yaml 写入, train 标记完成')

        migrated.append((dirname, run_id, ckpts))

    print(f'\n迁移完成: {len(migrated)} 个 run')
    if dry_run:
        print('(dry-run, 未实际移动文件. 去掉 --dry-run 执行)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true', default=True, help='预览不执行 (默认)')
    p.add_argument('--execute', action='store_true', help='实际执行迁移')
    args = p.parse_args()
    migrate(dry_run=not args.execute)
