import os.path

import pandas as pd
import random
from collections import deque


def sample_large_csv(
		input_file: str,
		output_file: str,
		mode: str = "random",
		sample_ratio: float = 0.001,
		n_rows: int = 1000,
		start: int = 0,
		chunksize: int = 10000,
		has_header: bool = False,
):
	"""
    通用大文件采样函数

    参数：
        input_file: 原始 CSV 文件路径
        output_file: 输出采样文件路径
        mode: 采样模式 ["random", "head", "tail", "offset"]
        sample_ratio: 随机采样比例（random 模式）
        n_rows: 指定行数（head/tail/offset 模式）
        start: 起始行索引（offset 模式）
        chunksize: 每次读入的行数
        has_header: 是否有表头
    """

	header = 0 if has_header else None
	first_write = True

	if mode == "random":
		# ✅ 随机采样模式：每块按比例采样
		with open(output_file, "w", encoding="utf-8") as out_f:
			for chunk in pd.read_csv(input_file, header=header, chunksize=chunksize):
				sampled = chunk.sample(frac=sample_ratio)
				sampled.to_csv(out_f, header=first_write and has_header, index=False)
				first_write = False
		print(f"✅ 随机采样完成，采样比例 = {sample_ratio}")

	elif mode == "head":
		# ✅ 前 n 条模式：读够 n_rows 即停止
		rows_collected = 0
		with open(output_file, "w", encoding="utf-8") as out_f:
			for chunk in pd.read_csv(input_file, header=header, chunksize=chunksize):
				if rows_collected >= n_rows:
					break
				remaining = n_rows - rows_collected
				sampled = chunk.iloc[:remaining]
				sampled.to_csv(out_f, header=first_write and has_header, index=False)
				rows_collected += len(sampled)
				first_write = False
		print(f"✅ 采样前 {n_rows} 条完成")

	elif mode == "tail":
		# ✅ 后 n 条模式：用 deque 保留最后 n 行
		buffer = deque(maxlen=n_rows)
		for chunk in pd.read_csv(input_file, header=header, chunksize=chunksize):
			for _, row in chunk.iterrows():
				buffer.append(row)
		df_tail = pd.DataFrame(list(buffer))
		df_tail.to_csv(output_file, header=has_header, index=False)
		print(f"✅ 采样后 {n_rows} 条完成")

	elif mode == "offset":
		# ✅ 从指定起始行开始取 n 行
		rows_collected = 0
		current_row = 0
		with open(output_file, "w", encoding="utf-8") as out_f:
			for chunk in pd.read_csv(input_file, header=header, chunksize=chunksize):
				end_row = current_row + len(chunk)
				if end_row < start:
					current_row = end_row
					continue
				# 开始行落在本chunk中
				start_idx = max(0, start - current_row)
				sampled = chunk.iloc[start_idx:start_idx + n_rows - rows_collected]
				sampled.to_csv(out_f, header=first_write and has_header, index=False)
				rows_collected += len(sampled)
				first_write = False
				current_row = end_row
				if rows_collected >= n_rows:
					break
		print(f"✅ 从第 {start} 行开始取 {n_rows} 条完成")

	else:
		raise ValueError("mode 必须是 ['random', 'head', 'tail', 'offset'] 之一")


# ✅ 使用示例
if __name__ == "__main__":
	file_path = "../dataset/xianshi_1001_1015"
	input_name = "part-00000-3bbf0a7c-4528-4398-812b-03fe2dde0474-c000.csv"
	input_file = os.path.join(file_path, input_name)
	output_name = "sample_traj_1000.csv"
	output_file = os.path.join(file_path, output_name)
	sample_large_csv(
		input_file=input_file,
		output_file=output_file,
		mode="head",  # "head" / "tail" / "offset" / "random"
		sample_ratio=0.001,  # 仅 random 模式使用
		n_rows=1000,  # 仅 head / tail / offset 模式使用
		start=100000,  # 仅 offset 模式使用
		chunksize=10000,
		has_header=False
	)
