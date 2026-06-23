#!/usr/bin/env python
"""
实验编排入口 (Experiment Orchestration Entry)
================================================
统一管理 train → infer → eval 三步流水线, 单一 run_id 贯穿,
产物归一到 runs/{run_id}/, 支持多步拆分、断点续跑、批量执行.

典型用法:
  # 全流程
  python run.py --task completion --dataset spacenet --steps train,infer,eval --gpus 0

  # 换模型重跑 infer+eval (复用已有 train)
  python run.py --task completion --dataset spacenet --run-id completion_spacenet_20260622_2000 \
      --steps infer,eval --resume-run

  # 批量跑四任务
  python run.py --batch batch.yaml

  # 只看不跑
  python run.py --task completion --dataset spacenet --steps train,infer,eval --dry-run

设计: run.py 是编排层, 不重写底层脚本, 通过 --run-root 注入路径调用各 train/infer/eval.
"""

import argparse
import os
import sys
import subprocess
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.registry import TASKS, DATASETS, get_task, get_dataset, default_config_for, run_paths
from tools.config_utils import (
    ensure_run_dirs, write_profile, read_profile,
    mark_step_done, is_step_done, select_best_ckpt,
    resolve_checkpoint_arg, save_config_snapshot,
)
from tools.run_info import _try_get_git


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description='实验编排: train→infer→eval 一条命令',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 实验标识
    p.add_argument('--task', choices=list(TASKS.keys()), help='模型变体 (注册表键)')
    p.add_argument('--dataset', choices=list(DATASETS.keys()), help='数据集 (注册表键)')
    p.add_argument('--run-id', default=None, help='实验唯一标识; 缺省={task}_{dataset}_{timestamp}')

    # 执行控制
    p.add_argument('--steps', default='train,infer,eval',
                   help='执行步骤, 逗号分隔: train,infer,eval (可子集)')
    p.add_argument('--gpus', default='0', help='GPU id, 如 0 或 0,1')
    p.add_argument('--resume-run', action='store_true',
                   help='续跑: 读已有 profile, 跳过已完成步')
    p.add_argument('--dry-run', action='store_true', help='只打印命令, 不执行')
    p.add_argument('--batch', default=None, help='批量执行配置 yaml (指定后忽略 --task/--dataset)')
    p.add_argument('--parallel', action='store_true', help='批量执行时各 run 按各自 GPU 并行')

    # 覆写 (命令行 > profile > config 默认)
    p.add_argument('--config', default=None, help='覆写 task 的默认 config 路径')
    p.add_argument('--checkpoint', default='auto',
                   help='infer 用哪个 ckpt: auto=自动选best / last / epoch:N 或 epN 或 N (Lightning 0-based epoch编号) / 路径')
    p.add_argument('--precision', default=None, help='16 或 32 (覆写)')
    p.add_argument('--epochs', type=int, default=None, help='覆写 config.TRAIN_EPOCHS')
    p.add_argument('--patience', type=int, default=None, help='early stopping patience')
    p.add_argument('--device', default='cuda', help='推理设备 cuda/cpu')
    p.add_argument('--workers', type=int, default=16, help='eval 并行 worker 数')
    p.add_argument('--infer-opts', default='', help='透传给 inferencer 的额外参数')
    p.add_argument('--eval-opts', default='', help='透传给 eval.py 的额外参数')
    p.add_argument('--no-input-graph', action='store_true',
                   help='completion 推理时不给已知路网 (纯提取退化模式)')
    return p


# ---------------------------------------------------------------------------
# run_id & profile
# ---------------------------------------------------------------------------
def make_run_id(task, dataset, run_id_arg):
    if run_id_arg:
        return run_id_arg
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{task}_{dataset}_{ts}'


def build_profile(args, task, dataset, run_id, config_path):
    """构造 profile.yaml 内容 (实验级声明快照)."""
    task_info = get_task(task)
    ds_info = get_dataset(dataset)
    rp = run_paths(run_id)
    config_snapshot = save_config_snapshot(config_path, run_id)

    profile = {
        'run_id': run_id,
        'task': task,
        'dataset': dataset,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config_source': config_path,
        'config_snapshot': config_snapshot,
        'steps_requested': [s.strip() for s in args.steps.split(',')],
        'gpus': args.gpus,
        'precision': args.precision,
        'epochs': args.epochs,
        'patience': args.patience,
        'paths': {k: rp[k] for k in ('run_root', 'train_dir', 'ckpt_dir',
                                      'infer_dir', 'eval_dir', 'profile', 'best_ckpt')},
        'task_info': {
            'train_script': task_info['train_script'],
            'infer_script': task_info['infer_script'],
            'test_script': task_info['test_script'],
            'model_class': task_info['model_class'],
            'ckpt_prefix': task_info['ckpt_prefix'],
        },
        'infer': {
            'checkpoint': args.checkpoint,
            'input_graph_dir': None if args.no_input_graph else ds_info['infer_input_graph_dir'],
            'traj_dir': ds_info['infer_traj_dir'],
            'device': args.device,
            'extra_opts': args.infer_opts,
        },
        'eval': {
            'dataset': ds_info['eval_dataset_name'],
            'workers': args.workers,
            'extra_opts': args.eval_opts,
        },
        'git': _try_get_git(PROJECT_ROOT),
        'step_status': {},
    }
    return profile


# ---------------------------------------------------------------------------
# 命令构造
# ---------------------------------------------------------------------------
def _gpu_env(gpus):
    """设置 CUDA_VISIBLE_DEVICES, 返回 env dict."""
    env = os.environ.copy()
    # 取第一个 gpu 作为可见设备 (多 gpu 训练时 train 脚本自己处理 devices 列表)
    first_gpu = gpus.split(',')[0].strip()
    env['CUDA_VISIBLE_DEVICES'] = first_gpu
    return env


def build_train_cmd(args, task, run_id, config_path):
    task_info = get_task(task)
    rp = run_paths(run_id)
    cmd = [
        sys.executable, '-m', task_info['train_script'],
        '--config', config_path,
        '--gpus', args.gpus,
        '--run-root', rp['run_root'],
    ]
    if args.precision:
        cmd += ['--precision', str(args.precision)]
    if args.patience is not None and args.patience > 0:
        cmd += ['--patience', str(args.patience)]
    return cmd


def build_infer_cmd(args, task, dataset, run_id, config_path, ckpt_path):
    task_info = get_task(task)
    ds_info = get_dataset(dataset)
    rp = run_paths(run_id)
    cmd = [
        sys.executable, '-m', task_info['infer_script'],
        '--config', config_path,
        '--checkpoint', ckpt_path,
        '--run-root', rp['run_root'],
        '--device', args.device,
    ]
    # completion 类推理需要已知路网和轨迹
    if 'completion' in task or '4ch' in task:
        if not args.no_input_graph and ds_info['infer_input_graph_dir']:
            cmd += ['--input_graph_dir', ds_info['infer_input_graph_dir']]
        if ds_info['infer_traj_dir']:
            cmd += ['--traj_dir', ds_info['infer_traj_dir']]
    if args.infer_opts:
        cmd += args.infer_opts.split()
    return cmd


def build_eval_cmd(args, dataset, run_id):
    ds_info = get_dataset(dataset)
    rp = run_paths(run_id)
    cmd = [
        sys.executable, 'metrics/eval.py',
        '--dataset', ds_info['eval_dataset_name'],
        # eval.py 内部会把 --dir 规范化为相对项目根, 传相对路径即可
        '--dir', rp['infer_dir'],
        '--workers', str(args.workers),
    ]
    if args.eval_opts:
        cmd += args.eval_opts.split()
    return cmd


# ---------------------------------------------------------------------------
# 步骤执行
# ---------------------------------------------------------------------------
def run_step(name, cmd, env, dry_run):
    """执行一个步骤, 返回成功与否."""
    cmd_str = ' '.join(cmd)
    if env.get('CUDA_VISIBLE_DEVICES'):
        cmd_str = f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} {cmd_str}"
    print(f'\n{"="*70}\n▶ [{name}] {cmd_str}\n{"="*70}')
    if dry_run:
        print('  (dry-run, 跳过执行)')
        return True
    t0 = time.time()
    ret = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)
    dt = time.time() - t0
    ok = ret.returncode == 0
    status = '✓ 成功' if ok else f'✗ 失败 (exit={ret.returncode})'
    print(f'\n[{name}] {status}  耗时 {dt:.1f}s')
    return ok


def execute_run(args, task, dataset, run_id):
    """执行一个完整的 run (可能含多步)."""
    task_info = get_task(task)
    config_path = args.config or default_config_for(task, dataset)

    if not os.path.exists(config_path):
        print(f'✗ config 不存在: {config_path}')
        return False

    rp = run_paths(run_id)
    env = _gpu_env(args.gpus)

    # resume-run: 读已有 profile
    if args.resume_run:
        prof = read_profile(run_id)
        if prof is None:
            print(f'✗ --resume-run: runs/{run_id}/profile.yaml 不存在, 无法续跑')
            return False
        print(f'▶ 续跑 {run_id}, 已有 profile 加载')
    else:
        # 新建: 确保目录 + 写 profile
        ensure_run_dirs(run_id)
        prof_dict = build_profile(args, task, dataset, run_id, config_path)
        write_profile(run_id, prof_dict)
        print(f'▶ 新建实验 {run_id}, profile 写入 {rp["profile"]}')

    steps = [s.strip() for s in args.steps.split(',') if s.strip()]
    valid_steps = {'train', 'infer', 'eval'}
    bad = [s for s in steps if s not in valid_steps]
    if bad:
        print(f'✗ 未知步骤: {bad}, 可选: {valid_steps}')
        return False

    # 依赖检查: infer 需要 train 已完成 (除非显式给 checkpoint 路径)
    for step in steps:
        # resume 时跳过已完成步
        if args.resume_run and is_step_done(run_id, step):
            print(f'▶ [{step}] 已完成, 跳过')
            continue

        if step == 'train':
            cmd = build_train_cmd(args, task, run_id, config_path)
            if not run_step('train', cmd, env, args.dry_run):
                return False
            if not args.dry_run:
                mark_step_done(run_id, 'train')
                # 训练完自动选 best ckpt
                best = select_best_ckpt(run_id)
                if best:
                    print(f'▶ 选定 best ckpt: {best}')

        elif step == 'infer':
            # 确定 ckpt 路径
            if args.dry_run and args.checkpoint == 'auto':
                # dry-run 时 train 未真跑, 用占位符展示命令
                ckpt_path = f'{rp["ckpt_dir"]}/<auto-best>.ckpt'
            else:
                ckpt_path = resolve_checkpoint_arg(run_id, args.checkpoint)
                if ckpt_path is None:
                    print(f'✗ infer: 无法解析 checkpoint={args.checkpoint!r}; '
                          f'可用 auto / last / epoch:N / epN / 数字 / 路径')
                    return False
                if not args.dry_run and args.checkpoint != 'auto' and not os.path.exists(ckpt_path):
                    print(f'✗ infer: checkpoint 不存在: {ckpt_path}')
                    return False
                if not args.dry_run and args.checkpoint == 'auto' and not os.path.exists(ckpt_path):
                    print(f'✗ infer: auto 选到的 ckpt 不存在: {ckpt_path}')
                    return False
            cmd = build_infer_cmd(args, task, dataset, run_id, config_path, ckpt_path)
            if not run_step('infer', cmd, env, args.dry_run):
                return False
            if not args.dry_run:
                mark_step_done(run_id, 'infer')

        elif step == 'eval':
            # eval 需要 infer 产物
            graph_dir = os.path.join(rp['infer_dir'], 'graph')
            if not args.dry_run and not os.path.isdir(graph_dir):
                print(f'✗ eval: 推理产物不存在 {graph_dir}, 先跑 infer')
                return False
            cmd = build_eval_cmd(args, dataset, run_id)
            if not run_step('eval', cmd, env, args.dry_run):
                return False
            if not args.dry_run:
                mark_step_done(run_id, 'eval')

    print(f'\n✓ run {run_id} 完成, 产物在 {rp["run_root"]}/')
    return True


# ---------------------------------------------------------------------------
# batch 批量执行
# ---------------------------------------------------------------------------
def execute_batch(args):
    import yaml
    with open(args.batch) as f:
        batch = yaml.safe_load(f)
    runs = batch.get('runs', [])
    default_steps = batch.get('default_steps', 'train,infer,eval')
    parallel = args.parallel

    print(f'▶ 批量执行 {len(runs)} 个 run, steps={default_steps}, parallel={parallel}')

    if parallel:
        # 每个 run 按各自 gpus 分配, 并行启动
        procs = []
        for spec in runs:
            run_args = _batch_spec_to_args(spec, args, default_steps)
            if run_args is None:
                continue
            # 并行模式: 用 subprocess 启动 run.py 自身
            cmd = _build_self_cmd(run_args)
            env = _gpu_env(spec.get('gpus', '0'))
            print(f'  启动 {run_args.run_id} on GPU {spec.get("gpus","0")}')
            p = subprocess.Popen(cmd, env=env, cwd=PROJECT_ROOT)
            procs.append((run_args.run_id, p))
        for rid, p in procs:
            ret = p.wait()
            print(f'  {rid}: exit={ret}')
        return all(p.wait() == 0 for _, p in procs)
    else:
        all_ok = True
        for spec in runs:
            run_args = _batch_spec_to_args(spec, args, default_steps)
            if run_args is None:
                continue
            ok = execute_run(run_args, run_args.task, run_args.dataset, run_args.run_id)
            all_ok = all_ok and ok
            if not ok:
                print(f'  ✗ {run_args.run_id} 失败, 继续下一个')
        return all_ok


def _batch_spec_to_args(spec, base_args, default_steps):
    """把 batch.yaml 里的一项转成 args namespace."""
    import copy
    a = copy.copy(base_args)
    a.batch = None
    a.task = spec['task']
    a.dataset = spec['dataset']
    a.run_id = spec.get('run_id')
    a.gpus = str(spec.get('gpus', '0'))
    # steps 统一成逗号分隔字符串 (default_steps 可能是 list)
    steps_val = spec.get('steps', default_steps)
    if isinstance(steps_val, list):
        steps_val = ','.join(steps_val)
    a.steps = steps_val
    if 'config' in spec:
        a.config = spec['config']
    else:
        a.config = None  # 用默认 config
    if 'checkpoint' in spec:
        a.checkpoint = spec['checkpoint']
    else:
        a.checkpoint = 'auto'
    a.run_id = make_run_id(a.task, a.dataset, a.run_id)
    return a


def _build_self_cmd(args):
    """并行模式下, 构造调用 run.py 自身的命令."""
    cmd = [sys.executable, 'run.py',
           '--task', args.task, '--dataset', args.dataset,
           '--run-id', args.run_id, '--gpus', args.gpus,
           '--steps', args.steps]
    if args.config:
        cmd += ['--config', args.config]
    if args.checkpoint != 'auto':
        cmd += ['--checkpoint', args.checkpoint]
    return cmd


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.batch:
        ok = execute_batch(args)
        sys.exit(0 if ok else 1)

    if not args.task or not args.dataset:
        parser.error('--task 和 --dataset 必填 (或用 --batch)')

    run_id = make_run_id(args.task, args.dataset, args.run_id)
    ok = execute_run(args, args.task, args.dataset, run_id)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
