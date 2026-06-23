import yaml
from addict import Dict
from datetime import datetime
import os

from tools.registry import run_paths


def load_config(path):
    with open(path) as file:
        config_dict = yaml.safe_load(file)
    return Dict(config_dict)


# ---------------------------------------------------------------------------
# 旧接口 (保留向后兼容): 推理输出目录 + 存 config.yaml
# ---------------------------------------------------------------------------
def create_output_dir_and_save_config(output_dir_prefix, config, specified_dir=None):
    if specified_dir:
        output_dir = specified_dir
    else:
        # Generate the output directory name with the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir_prefix}_{timestamp}"

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the path for the config file
    config_path = os.path.join(output_dir, "config.yaml")

    # Save the config as a YAML file
    with open(config_path, 'w') as file:
        yaml.dump(config.to_dict(), file)

    return output_dir


# ---------------------------------------------------------------------------
# 新接口: 基于 run_id 的统一路径体系
# ---------------------------------------------------------------------------
def ensure_run_dirs(run_id):
    """创建 runs/{run_id}/ 下所有需要的子目录, 返回路径字典."""
    paths = run_paths(run_id)
    for key in ('run_root', 'train_dir', 'ckpt_dir', 'infer_dir', 'eval_dir', 'step_done_dir'):
        os.makedirs(paths[key], exist_ok=True)
    return paths


def save_config_snapshot(config_path, run_id):
    """把 config.yaml 内容内联到 profile 的快照字段, 保证可复现."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return f.read()
    # 没有源文件时, 用 config 对象的 dump
    try:
        return yaml.dump(config_path.to_dict(), allow_unicode=True) if hasattr(config_path, 'to_dict') else None
    except Exception:
        return None


def write_profile(run_id, profile_dict):
    """写 runs/{run_id}/profile.yaml (实验级声明快照)."""
    paths = run_paths(run_id)
    os.makedirs(paths['run_root'], exist_ok=True)
    with open(paths['profile'], 'w') as f:
        yaml.safe_dump(profile_dict, f, sort_keys=False,
                       default_flow_style=False, allow_unicode=True)
    return paths['profile']


def read_profile(run_id):
    """读 runs/{run_id}/profile.yaml, 不存在返回 None."""
    paths = run_paths(run_id)
    if not os.path.exists(paths['profile']):
        return None
    with open(paths['profile']) as f:
        return Dict(yaml.safe_load(f))


def mark_step_done(run_id, step):
    """标记某步完成: 写 runs/{run_id}/.step_done/{step}"""
    paths = run_paths(run_id)
    os.makedirs(paths['step_done_dir'], exist_ok=True)
    with open(os.path.join(paths['step_done_dir'], step), 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def is_step_done(run_id, step):
    """检查某步是否已完成."""
    paths = run_paths(run_id)
    return os.path.exists(os.path.join(paths['step_done_dir'], step))


def select_best_ckpt(run_id):
    """
    扫描 runs/{run_id}/train/checkpoints/, 按 val_loss 选最小的 ckpt.
    返回 ckpt 绝对路径, 写入 best_ckpt.txt. 无可用 ckpt 返回 None.
    """
    paths = run_paths(run_id)
    ckpt_dir = paths['ckpt_dir']
    if not os.path.isdir(ckpt_dir):
        return None

    import re
    best_ckpt = None
    best_loss = float('inf')
    # 文件名形如: completion-epoch=09-val_loss=0.1288.ckpt 或 epoch-epoch=09-val_loss=0.1244.ckpt
    pattern = re.compile(r'val_loss=([0-9.]+)\.ckpt$')
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith('.ckpt') or fname == 'last.ckpt':
            continue
        m = pattern.search(fname)
        if m:
            loss = float(m.group(1))
            if loss < best_loss:
                best_loss = loss
                best_ckpt = os.path.join(ckpt_dir, fname)

    # 老权重文件名可能没有 val_loss (如 epoch=9-step=13230.ckpt)。
    # 这种情况下退到 last.ckpt；若也没有 last, 选 mtime 最新的 ckpt。
    if best_ckpt is None:
        last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
        if os.path.exists(last_ckpt):
            best_ckpt = last_ckpt
        else:
            ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
            if ckpts:
                best_ckpt = max(ckpts, key=os.path.getmtime)

    if best_ckpt is not None:
        with open(paths['best_ckpt'], 'w') as f:
            f.write(best_ckpt + '\n')
    return best_ckpt


def get_best_ckpt(run_id):
    """读 best_ckpt.txt, 若不存在则重新选择."""
    paths = run_paths(run_id)
    if os.path.exists(paths['best_ckpt']):
        with open(paths['best_ckpt']) as f:
            p = f.read().strip()
            if p and os.path.exists(p):
                return p
    return select_best_ckpt(run_id)


def get_ckpt_by_epoch(run_id, epoch):
    """按 Lightning 的 0-based epoch 编号选择 ckpt.

    Args:
        run_id: runs/{run_id}
        epoch: int, 文件名里的 epoch 编号. 例如 epoch=09 表示第 10 个 epoch 后的 ckpt.

    Returns:
        匹配到的 ckpt 路径; 未找到返回 None.

    支持的文件名示例:
        completion-epoch=09-val_loss=0.1288.ckpt
        epoch-epoch=09-val_loss=0.1244.ckpt
        epoch=9-step=13230.ckpt
    """
    paths = run_paths(run_id)
    ckpt_dir = paths['ckpt_dir']
    if not os.path.isdir(ckpt_dir):
        return None

    import re
    try:
        epoch = int(epoch)
    except (TypeError, ValueError):
        return None

    # 抓文件名中最后一个 epoch= / epoch- 编号; 兼容 epoch-epoch=09 这种 PL 模板
    epoch_pat = re.compile(r'epoch[=-](\d+)')
    loss_pat = re.compile(r'val_loss=([0-9.]+)\.ckpt$')
    matches = []
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith('.ckpt'):
            continue
        nums = epoch_pat.findall(fname)
        if not nums:
            continue
        if int(nums[-1]) != epoch:
            continue
        path = os.path.join(ckpt_dir, fname)
        m = loss_pat.search(fname)
        loss = float(m.group(1)) if m else float('inf')
        matches.append((loss, os.path.getmtime(path), path))

    if not matches:
        return None
    # 同一 epoch 理论只有一份; 若有多份, 优先 val_loss 小, 再 mtime 新
    matches.sort(key=lambda x: (x[0], -x[1]))
    return matches[0][2]


def resolve_checkpoint_arg(run_id, checkpoint_arg):
    """解析 run.py --checkpoint 参数.

    支持:
      - auto: best_ckpt.txt / select_best_ckpt
      - last: train/checkpoints/last.ckpt
      - epoch:N / epN / N: 按 Lightning 0-based epoch 编号查找 ckpt
      - 其它: 原样视为路径
    """
    if checkpoint_arg == 'auto':
        return get_best_ckpt(run_id)

    paths = run_paths(run_id)
    if checkpoint_arg == 'last':
        return os.path.join(paths['ckpt_dir'], 'last.ckpt')

    import re
    m = re.fullmatch(r'(?:epoch:|ep)?(\d+)', str(checkpoint_arg))
    if m:
        return get_ckpt_by_epoch(run_id, int(m.group(1)))

    return checkpoint_arg
