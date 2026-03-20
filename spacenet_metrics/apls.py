import os
import argparse
import json
import subprocess
import numpy as np
import shutil
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
# 新增并发参数，默认保留2个核心给系统
default_workers = max(1, (os.cpu_count() or 4) - 2)
parser.add_argument('--workers', type=int, default=default_workers)
args = parser.parse_args()

target_dir = args.dir
data_dir = "spacenet"
out_dir = f"../{target_dir}/results/apls"

os.makedirs(out_dir, exist_ok=True)

# ================= 0. 处理 Go 环境与预编译优化 =================
go_exe = r"E:\Softwares\GO\go1.24.2.windows-amd64\go\bin\go.exe"
if not os.path.exists(go_exe):
    go_exe = shutil.which("go") or "go"

go_bin_name = "eval_bin.exe" if platform.system() == "Windows" else "eval_bin"
go_bin_path = f"apls/{go_bin_name}"

print("Pre-compiling Go evaluation script...")
build_result = subprocess.run(
    [go_exe, "build", "-o", go_bin_name, "main.go"],
    cwd="apls", capture_output=True, text=True
)
if build_result.returncode != 0:
    print(f"Warning: Go compilation failed!\n{build_result.stderr}")
    exit(1)

# 核心修复：获取编译出的 exe 的绝对路径，绕过 Windows subprocess 找不到当前目录执行文件的 Bug
abs_go_bin = os.path.abspath(go_bin_path)

# ================= 1. 读取测试集 =================
with open(f"../{data_dir}/data_split.json", "r") as f:
    test_names = json.load(f).get("test", [])

try:
    from tqdm import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False

# ================= 2. 定义单个图片的处理任务 =================
def process_single_image(name):
    # 严格保持原版的相对路径
    pred_path = f"../{target_dir}/graph/{name}.p"
    gt_path = f"../{data_dir}/RGB_1.0_meter/{name}__gt_graph.p"
    out_txt = f"../../{target_dir}/results/apls/{name}.txt"
    
    if not os.path.exists(pred_path):
        return f"SKIP: Missing {pred_path}"

    # 核心：使用图片名作为独立的临时文件名，避免多线程写入冲突
    temp_gt = f"gt_{name}.json"
    temp_prop = f"prop_{name}.json"
    
    try:
        # 恢复使用原版的 "python"
        res_gt = subprocess.run(["python", "apls/convert.py", gt_path, temp_gt], capture_output=True, text=True)
        if res_gt.returncode != 0:
            return f"Convert GT Error ({name}): {res_gt.stderr}"

        res_prop = subprocess.run(["python", "apls/convert.py", pred_path, temp_prop], capture_output=True, text=True)
        if res_prop.returncode != 0:
            return f"Convert Prop Error ({name}): {res_prop.stderr}"
        
        # 传入绝对路径执行 Go 文件，完美避坑
        res_go = subprocess.run(
            [abs_go_bin, f"../{temp_gt}", f"../{temp_prop}", out_txt, "spacenet"],
            cwd="apls", capture_output=True, text=True
        )
        if res_go.returncode != 0:
            return f"Go Execution Error ({name}): {res_go.stderr}"

        return "SUCCESS"
    except Exception as e:
        return f"Exception ({name}): {str(e)}"
    finally:
        # 清理该线程专属的临时文件
        if os.path.exists(temp_gt):
            os.remove(temp_gt)
        if os.path.exists(temp_prop):
            os.remove(temp_prop)

# ================= 3. 多线程并发执行 =================
print(f"Starting APLS evaluation with {args.workers} workers...")
errors = []

with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = {executor.submit(process_single_image, name): name for name in test_names}
    
    if has_tqdm:
        iterable_futures = tqdm(as_completed(futures), total=len(test_names), desc="APLS Progress")
    else:
        print("APLS Progress: ", end="", flush=True)
        iterable_futures = as_completed(futures)

    for future in iterable_futures:
        res = future.result()
        
        # ================== 调试输出区 ==================
        if res != "SUCCESS":
            print(f"\n[DEBUG] {res}") 
        # ================================================

        if res != "SUCCESS":
            errors.append(res)
        elif not has_tqdm:
            print(".", end="", flush=True)

if not has_tqdm:
    print()

# 清理预编译的 Go 临时文件
if os.path.exists(go_bin_path):
    os.remove(go_bin_path)

# ================= 4. 计算最终指标 =================
apls_vals = []
output_apls = []

if os.path.exists(out_dir):
    for file_name in sorted(os.listdir(out_dir)):
        with open(os.path.join(out_dir, file_name)) as f:
            lines = f.readlines()
        if not lines or 'NaN' in lines[0]:
            continue
        val = float(lines[0].split(' ')[-1])
        apls_vals.append(val)
        output_apls.append([file_name, val])

mean_apls = np.mean(apls_vals) if apls_vals else 0
print(f"APLS: {mean_apls:.8f}")

with open(f"../{target_dir}/results/apls.json", "w") as jf:
    json.dump({'apls': output_apls, 'final_APLS': mean_apls}, jf)
    