from PIL import Image
from pathlib import Path


def crop_png_sliding_window(
    img_path,
    save_dir,
    crop_size=(512, 512),
    overlap=0.0,
    start_index=0,
    name_tag=""
):
    """
    PNG 滑窗裁剪
    - 西南 -> 东北（左下 -> 右上）
    - 越界部分自动填充为黑色
    - 文件名包含 row / col 信息
    """

    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    W, H = img.size
    crop_w, crop_h = crop_size

    stride_w = int(crop_w * (1 - overlap))
    stride_h = int(crop_h * (1 - overlap))
    assert stride_w > 0 and stride_h > 0

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    idx = start_index

    # ===============================
    # 关键修改：y 方向从下到上
    # ===============================
    top_list = list(range(0, H, stride_h))[::-1]

    row_id = 0  # row：南 → 北
    for top in top_list:
        col_id = 0  # col：西 → 东
        for left in range(0, W, stride_w):
            right = left + crop_w
            bottom = top + crop_h

            # 全黑 patch
            patch = Image.new("RGB", (crop_w, crop_h), (0, 0, 0))

            # 有效区域
            valid_left = max(left, 0)
            valid_top = max(top, 0)
            valid_right = min(right, W)
            valid_bottom = min(bottom, H)

            if valid_right > valid_left and valid_bottom > valid_top:
                patch_part = img.crop(
                    (valid_left, valid_top, valid_right, valid_bottom)
                )
                paste_x = valid_left - left
                paste_y = valid_top - top
                patch.paste(patch_part, (paste_x, paste_y))

            # ========= 文件命名 =========
            # 示例：tag_12_r03_c07.png
            name_parts = []
            if name_tag:
                name_parts.append(name_tag)

            name_parts.append(f"{idx}")
            name_parts.append(f"r{row_id}")
            name_parts.append(f"c{col_id}")

            save_name = "_".join(name_parts) + ".png"
            patch.save(save_dir / save_name)

            idx += 1
            col_id += 1

        row_id += 1

    return idx


if __name__ == "__main__":
    INPUT_IMG = "../dataset/xian-z18-arcgis.png"
    OUTPUT_DIR = "../dataset/img-xian/arcgis"

    CROP_SIZE = (400, 400)
    OVERLAP = 0

    START_INDEX = 1
    NAME_TAG = ""

    next_idx = crop_png_sliding_window(
        img_path=INPUT_IMG,
        save_dir=OUTPUT_DIR,
        crop_size=CROP_SIZE,
        overlap=OVERLAP,
        start_index=START_INDEX,
        name_tag=NAME_TAG,
    )

    print(f"[INFO] 本批次结束，下一个起始 index = {next_idx}")


if __name__ == "__main__":
    # 方式：左下 -> 右上

    INPUT_IMG = "../dataset/xian-z18-arcgis.png"
    OUTPUT_DIR = "../dataset/img/xian/arcgis"

    # 裁剪参数
    CROP_SIZE = (400, 400)
    OVERLAP = 0

    # ===== 命名与批次控制 =====
    START_INDEX = 1            # 比如 1 → 1000
    NAME_TAG = ""     # 任意字符串

    next_idx = crop_png_sliding_window(
        img_path=INPUT_IMG,
        save_dir=OUTPUT_DIR,
        crop_size=CROP_SIZE,
        overlap=OVERLAP,
        start_index=START_INDEX,
        name_tag=NAME_TAG,
    )

    print(f"[INFO] 本批次结束，下一个起始 index = {next_idx}")
