# TOPO parallel script, same as "main.py" but with multiprocessing support and better Windows compatibility
import os
import argparse
import json
import subprocess
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-savedir', type=str, required=True)
# 与 apls 一致的并发控制
default_workers = max(1, (os.cpu_count() or 4) - 2)
parser.add_argument('-workers', type=int, default=default_workers)
args = parser.parse_args()

target_dir = args.savedir
out_dir = f"../{target_dir}/results/topo"

# ================= 1. 唤醒并执行底层多进程引擎 =================
print(f"Starting TOPO evaluation with {args.workers} workers...")

# 我们不捕获输出，让底层引擎的 tqdm 进度条直接打印到控制台
res = subprocess.run(
    [sys.executable, "topo/eval_parallel.py", "-savedir", target_dir, "-workers", str(args.workers)]
)

if res.returncode != 0:
    print("Error: TOPO parallel evaluation failed.")
    exit(1)

# ================= 2. 聚合并计算最终指标 =================
precision = []
recall = []

if os.path.exists(out_dir):
    for file_name in os.listdir(out_dir):
        if not file_name.endswith('.txt'):
            continue
        with open(os.path.join(out_dir, file_name)) as f:
            lines = f.readlines()
        if not lines:
            continue
        
        p = float(lines[-1].split(' ')[0].split('=')[-1])
        r = float(lines[-1].split(' ')[-1].split('=')[-1])
        if p + r > 0:
            precision.append(p)
            recall.append(r)

mean_p = np.mean(precision) if precision else 0
mean_r = np.mean(recall) if recall else 0
topo_f1 = 2 * mean_p * mean_r / (mean_p + mean_r) if (mean_p + mean_r) > 0 else 0

print(f"TOPO: {topo_f1:.8f} | Precision: {mean_p:.8f} | Recall: {mean_r:.8f}")

out_path = f"../{target_dir}/results/topo.json"
with open(out_path, "w") as jf:
    json.dump({'mean topo': [topo_f1, mean_p, mean_r], 'prec': precision, 'recall': recall, 'f1': topo_f1}, jf)
