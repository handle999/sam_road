#!/usr/bin/env python
"""
服务器重构前置自检 (Server Refactor Preflight Check)
======================================================
服务器上的 Claude 在动手迁移前先跑这个脚本, 一键探测现状, 输出结构化报告。
对应 docs/服务器编排重构手册.md 的 §〇 自检清单。

用法:
  python tools/server_preflight.py
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)


def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"<error: {e}>"


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


print("🔍 服务器编排重构前置自检")
print(f"项目根: {PROJECT_ROOT}")

# ---------------------------------------------------------------------------
section("1. 编排代码是否就位")
code_files = ['run.py', 'tools/registry.py', 'tools/config_utils.py',
              'tools/migrate_ckpts_to_runs.py', 'batch.yaml']
for f in code_files:
    print(f"  {'✓' if os.path.exists(f) else '✗ 缺失'} {f}")

run_root_scripts = run("grep -l 'run-root' engine/train.py engine/train_completion.py "
                       "engine/train_4ch.py engine/inferencer.py engine/inferencer_completion.py "
                       "engine/inferencer_4ch.py 2>/dev/null").split()
print(f"  含 --run-root 的脚本: {len(run_root_scripts)}/6")
if len(run_root_scripts) < 6:
    print("  ⚠️ 脚本未全部加 --run-root, 可能需要 git pull")

# ---------------------------------------------------------------------------
section("2. 现有权重探测")
print("  checkpoints/ 下的 ckpt:")
ckpts = run("find checkpoints -name '*.ckpt' 2>/dev/null | sort")
print(f"  {ckpts if ckpts else '  (无)'}")

print("\n  checkpoints/ 子目录:")
dirs = run("ls -d checkpoints/*/ 2>/dev/null")
print(f"  {dirs if dirs else '  (无 checkpoints/ 目录)'}")

print("\n  save/ (旧推理产物):")
save_dirs = run("ls -d save/*/ 2>/dev/null | head")
print(f"  {save_dirs if save_dirs else '  (无)'}")

print("\n  runs/ 现状:")
runs = run("ls runs/ 2>/dev/null")
print(f"  {runs if runs else '  (runs/ 不存在)'}")

# ---------------------------------------------------------------------------
section("3. 迁移需求判断")
has_old_ckpts = bool(run("find checkpoints -name '*.ckpt' 2>/dev/null | head -1"))
has_runs_ckpts = bool(run("find runs -name '*.ckpt' 2>/dev/null | head -1"))
if has_old_ckpts:
    print("  → 情况 A: 旧权重在 checkpoints/, 需用 migrate_ckpts_to_runs.py 迁移")
elif has_runs_ckpts:
    print("  → 情况 B: 权重已在 runs/, 无需迁移")
else:
    print("  → 无现有权重, 后续用 run.py 直接训练")

# ---------------------------------------------------------------------------
section("4. .gitignore 关键规则")
gi = run("grep -nE 'runs|\\.ckpt|checkpoints|save' .gitignore 2>/dev/null")
print(f"  {gi if gi else '  (无相关规则)'}")
has_runs_rule = 'runs' in (gi or '')
print(f"  runs/ 规则: {'✓ 已配' if has_runs_rule else '✗ 缺失, 需补 (见手册 §三)'}")

# ---------------------------------------------------------------------------
section("5. config 文件检查")
print("  四任务默认 config 是否存在:")
try:
    from tools.registry import default_config_for
    for t in ['extraction', 'completion']:
        for d in ['spacenet', 'didi_xian']:
            c = default_config_for(t, d)
            print(f"    {'✓' if os.path.exists(c) else '✗'} {t} × {d} → {c}")
except Exception as e:
    print(f"  ⚠️ registry 导入失败: {e}")

# ---------------------------------------------------------------------------
section("6. GPU 可用性")
gpu = run("python -c \"import torch; print('CUDA:', torch.cuda.is_available(), '设备数:', torch.cuda.device_count())\" 2>/dev/null")
print(f"  {gpu if gpu else '  (torch 不可用)'}")
nvidia = run("nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null")
if nvidia:
    print("  GPU 列表:")
    for line in nvidia.split('\n'):
        print(f"    {line}")

# ---------------------------------------------------------------------------
section("7. git 状态")
print(f"  分支: {run('git branch --show-current')}")
print(f"  远程: {run('git remote -v').split(chr(10))[0]}")
print(f"  最近3条 commit:")
print(run("git log --oneline -3"))
dirty = run("git status -s")
print(f"  工作区: {'干净' if not dirty else '有未提交改动:'}")
if dirty:
    for line in dirty.split('\n')[:10]:
        print(f"    {line}")

# ---------------------------------------------------------------------------
section("📋 建议下一步")
if len(run_root_scripts) < 6 or not all(os.path.exists(f) for f in code_files):
    print("  1. git pull origin main  (编排代码未就位)")
elif has_old_ckpts:
    print("  1. python tools/migrate_ckpts_to_runs.py --dry-run  (预览迁移)")
    print("  2. 核对 dry-run 映射后: python tools/migrate_ckpts_to_runs.py --execute")
    print("  3. 按 §三 补 .gitignore 的 runs/ 规则")
    print("  4. 按 §四 做端到端冒烟测试")
elif not has_runs_rule and has_runs_ckpts:
    print("  1. 按 §三 补 .gitignore 的 runs/ 规则")
    print("  2. 按 §四 做端到端冒烟测试")
else:
    print("  环境就绪, 可直接用 run.py 跑实验")
print("\n  详细步骤见 docs/服务器编排重构手册.md")
