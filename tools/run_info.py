"""
Run metadata dumper — 把"这次跑用了什么 ckpt / config / 命令 / git 版本"等可复现信息
独立写到 run_info.yaml, 与模型 config.yaml 关注点分离。

Usage in train / inference scripts:

    from tools.run_info import dump_run_info, mark_run_finished

    run_info_path = dump_run_info(
        output_dir=output_dir,            # 推理: save/infer_xxx/, 训练: train_logs/
        script=__file__,                  # 或显式 'engine.train_completion'
        args=args,                        # argparse Namespace
        config_source=args.config,        # 原始 config 文件路径 (与 config.yaml 区分)
        checkpoint=args.checkpoint,       # 仅推理时
        extra={'task': 'inference'}       # 任意其它元信息
    )
    # ... 实际工作 ...
    mark_run_finished(run_info_path)      # 写 end_time + duration_seconds
"""

from __future__ import annotations

import os
import sys
import shlex
import socket
import platform
import subprocess
from datetime import datetime
from typing import Any, Optional


def _try_get_git(cwd: str) -> dict:
    """Best-effort 抓取 git commit / branch / dirty 状态. 出错返回空 dict."""
    info = {}
    try:
        info['commit'] = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return info  # 不在 git 仓库, 直接返回
    try:
        info['branch'] = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        pass
    try:
        # --quiet 退出码 0 = 干净, 1 = 有改动
        subprocess.check_call(
            ['git', 'diff', '--quiet'],
            cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        info['dirty'] = False
    except subprocess.CalledProcessError:
        info['dirty'] = True
    except Exception:
        pass
    return info


def _try_get_torch_info() -> dict:
    """Best-effort 抓取 torch / CUDA 信息. 不强制依赖 torch."""
    info = {}
    try:
        import torch
        # 注意: torch.__version__ 是 TorchVersion 对象, yaml.safe_dump 序列化后会
        # 嵌入 !!python/object 标记导致后续 safe_load 失败. 强制转 str.
        info['torch_version'] = str(torch.__version__)
        info['cuda_available'] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info['cuda_device_count'] = int(torch.cuda.device_count())
            info['cuda_version'] = str(torch.version.cuda) if torch.version.cuda else None
    except Exception:
        pass
    return info


def _reconstruct_command() -> str:
    """重建用户实际执行的命令行 (尽量还原, 包含 CUDA_VISIBLE_DEVICES 等环境前缀)."""
    parts = []
    for env_key in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        val = os.environ.get(env_key)
        if val is not None:
            parts.append(f'{env_key}={shlex.quote(val)}')
    parts.append(sys.executable)
    parts.extend(shlex.quote(a) for a in sys.argv)
    return ' '.join(parts)


def dump_run_info(
    output_dir: str,
    script: str,
    args: Any = None,
    config_source: Optional[str] = None,
    checkpoint: Optional[str] = None,
    extra: Optional[dict] = None,
    filename: str = 'run_info.yaml',
) -> str:
    """
    在 output_dir 下写一份 run_info.yaml, 返回它的路径.

    Args:
        output_dir: 写入目录, 不存在会被创建
        script: 当前脚本标识. 传 __file__ 或 'engine.train_completion' 等都可以
        args: argparse.Namespace, 会被 vars() 转 dict
        config_source: 原始 config 文件路径 (跟 dumped config.yaml 区分)
        checkpoint: 推理用的 ckpt 路径 (仅推理脚本传)
        extra: 任意补充字段, 顶层合并到 yaml
        filename: 默认 run_info.yaml

    Returns:
        写入的 yaml 文件绝对路径
    """
    import yaml  # 推迟到调用时导入, 避免顶层依赖

    os.makedirs(output_dir, exist_ok=True)

    # 标准化 script 字段: 如果传了文件路径, 取相对项目根的形式
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isabs(script) and script.startswith(project_root):
        script_label = os.path.relpath(script, project_root)
    else:
        script_label = script

    payload = {
        'script': script_label,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': None,                       # mark_run_finished() 会填
        'duration_seconds': None,               # mark_run_finished() 会填
        'host': socket.gethostname(),
        'output_dir': os.path.abspath(output_dir),
        'config_source': config_source,
        'checkpoint': checkpoint,
        'command': _reconstruct_command(),
        'cli_args': vars(args) if args is not None else None,
        'env': {
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '<all>'),
            'omp_num_threads': os.environ.get('OMP_NUM_THREADS', '<not set>'),
            'mkl_num_threads': os.environ.get('MKL_NUM_THREADS', '<not set>'),
            'ld_preload': os.environ.get('LD_PRELOAD', '<not set>'),
            'conda_env': os.environ.get('CONDA_DEFAULT_ENV', '<not set>'),
            'python_version': platform.python_version(),
            'platform': platform.platform(),
        },
        'git': _try_get_git(project_root),
        'torch': _try_get_torch_info(),
    }
    if extra:
        payload.update(extra)

    out_path = os.path.join(output_dir, filename)
    with open(out_path, 'w') as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return out_path


def mark_run_finished(run_info_path: str) -> None:
    """脚本结束时调用: 写入 end_time 和 duration_seconds."""
    if not os.path.exists(run_info_path):
        return
    import yaml
    try:
        with open(run_info_path, 'r') as f:
            payload = yaml.safe_load(f) or {}
        end = datetime.now()
        payload['end_time'] = end.strftime('%Y-%m-%d %H:%M:%S')
        try:
            start = datetime.strptime(payload.get('start_time', ''), '%Y-%m-%d %H:%M:%S')
            payload['duration_seconds'] = round((end - start).total_seconds(), 2)
        except Exception:
            pass
        with open(run_info_path, 'w') as f:
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    except Exception:
        # 写入失败不影响主流程
        pass
