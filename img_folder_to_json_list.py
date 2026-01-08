import json
import random
import re
from pathlib import Path


def generate_data_split_from_cases(
    data_root,
    save_path,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    case_regex=r"region_(\d+)",
    seed=42
):
    """
    根据文件名中的 case id 生成 data_split.json
    同一个 case 的多个文件只计一次

    Args:
        data_root (str): 数据目录
        save_path (str): data_split.json 保存路径
        train_ratio / val_ratio / test_ratio
        case_regex (str): 从文件名中提取 case id 的正则
            默认匹配 region_0_xxx -> 0
        seed (int)
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train / val / test 比例之和必须为 1"

    random.seed(seed)
    data_root = Path(data_root)

    case_set = set()

    for p in data_root.iterdir():
        if not p.is_file():
            continue

        match = re.search(case_regex, p.name)
        if match:
            case_id = match.group(1)
            case_set.add(case_id)

    case_list = sorted(case_set)
    random.shuffle(case_list)

    n_total = len(case_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_list = case_list[:n_train]
    val_list = case_list[n_train:n_train + n_val]
    test_list = case_list[n_train + n_val:]

    split_dict = {
        "train": train_list,
        "validation": val_list,
        "test": test_list
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2, ensure_ascii=False)

    print(" data_split.json 已生成（按 case 级别）")
    print(f"  Total cases: {n_total}")
    print(f"  Train: {len(train_list)}")
    print(f"  Val:   {len(val_list)}")
    print(f"  Test:  {len(test_list)}")


if __name__ == "__main__":
    # ======= 你只需要改这里 =======
    data_root = "./xian/xian_2019_400/"
    save_path = "./xian/data_split.json"

    generate_data_split_from_cases(
        data_root=data_root,
        save_path=save_path,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        case_regex=r"region_(\d+)",  # 关键：定义 case 解析规则
        seed=42
    )
