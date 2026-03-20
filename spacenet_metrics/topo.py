import os
import argparse
import json
import subprocess
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-savedir', type=str, required=True)
args = parser.parse_args()

target_dir = args.savedir
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "spacenet"
out_dir = f"../{target_dir}/results/topo"

os.makedirs(out_dir, exist_ok=True)

# 清理旧的 txt 文件以确保进度条准确
for f in os.listdir(out_dir):
    if f.endswith('.txt'):
        os.remove(os.path.join(out_dir, f))

# 获取总任务数
try:
    with open(f"../{data_dir}/data_split.json", "r") as f:
        total_files = len(json.load(f).get("test", []))
except Exception:
    graph_dir = f"../{target_dir}/graph"
    total_files = len([f for f in os.listdir(graph_dir) if f.endswith('.p')]) if os.path.exists(graph_dir) else 100

try:
    from tqdm import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False

main_py_path = os.path.join(script_dir, "topo", "main.py")

# ================= 1. 异步运行底层脚本并监控进度 =================
process = subprocess.Popen(
    ["python", main_py_path, "-savedir", target_dir],
    stdout=subprocess.DEVNULL, 
    stderr=subprocess.DEVNULL
)

if has_tqdm:
    pbar = tqdm(total=total_files, desc="TOPO Progress")
    current_count = 0
    while process.poll() is None:
        files_count = len([f for f in os.listdir(out_dir) if f.endswith('.txt')])
        if files_count > current_count:
            pbar.update(files_count - current_count)
            current_count = files_count
        time.sleep(0.5)
        
    final_count = len([f for f in os.listdir(out_dir) if f.endswith('.txt')])
    if final_count > current_count:
        pbar.update(final_count - current_count)
    pbar.close()
else:
    print("TOPO Progress: Running...", end="", flush=True)
    process.wait()
    print("\rTOPO Progress: Done!       ")

# ================= 2. 计算最终指标 =================
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
