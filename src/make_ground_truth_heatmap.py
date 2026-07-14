import os
import random
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch


# ======================================================
# 1. 参数配置
# ======================================================
INPUT_DIR = r"./test_samples"                  # 图片文件夹
OUTPUT_PATH = r"./ground_truth_heatmap.png"    # 输出图片路径

# 网格大小
N_ROWS = 3
N_COLS = 3
MAX_IMAGES = N_ROWS * N_COLS

# 手动输入真值标签
# 这里每个数字对应一张图片，顺序与读取到的图片顺序一致
# 默认: 1=已绑扎, 0=未绑扎
MANUAL_LABELS = [
    0, 1, 1,
    1, 0, 0,
    1, 0, 1
]

# 标签含义
LABEL_TO_NAME = {
    1: "已绑扎",
    0: "未绑扎"
}

# 是否随机抽图
RANDOM_SAMPLE = False
RANDOM_SEED = 42

# 缩略图大小
THUMB_SIZE = (95, 95)

# 图背景颜色
FIG_BG_COLOR = "#EDEDED"

# 中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ======================================================
# 2. 工具函数
# ======================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def list_images_safe(folder, recursive=True):
    """
    安全读取图片，自动去重
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []

    if recursive:
        for root, _, filenames in os.walk(folder):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in valid_exts:
                    files.append(os.path.abspath(os.path.join(root, name)))
    else:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                ext = os.path.splitext(name)[1].lower()
                if ext in valid_exts:
                    files.append(os.path.abspath(path))

    files = sorted(set(files))
    return files


def resize_with_padding(pil_img, target_size=(95, 95)):
    """
    缩放并填充到固定大小
    """
    target_w, target_h = target_size
    w, h = pil_img.size
    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    img_resized = pil_img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))

    return np.array(canvas)


def make_thumbnail_grid(thumbnails, n_rows, n_cols, thumb_size):
    """
    将缩略图拼成左侧大图
    """
    total_cells = n_rows * n_cols

    while len(thumbnails) < total_cells:
        blank = np.ones((thumb_size[1], thumb_size[0], 3), dtype=np.uint8) * 255
        thumbnails.append(blank)

    rows = []
    idx = 0
    for _ in range(n_rows):
        row_imgs = []
        for _ in range(n_cols):
            row_imgs.append(thumbnails[idx])
            idx += 1
        row_strip = np.hstack(row_imgs)
        rows.append(row_strip)

    canvas = np.vstack(rows)
    return canvas


# ======================================================
# 3. 主程序
# ======================================================
def main():
    set_seed(RANDOM_SEED)

    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")

    image_paths = list_images_safe(INPUT_DIR, recursive=True)

    if len(image_paths) == 0:
        print("没有找到图片，请检查 INPUT_DIR。")
        return

    print("实际找到的图片数量:", len(image_paths))
    for i, p in enumerate(image_paths):
        print(f"{i+1:02d}: {p}")

    # 选择要展示的图片
    if RANDOM_SAMPLE:
        image_paths = random.sample(image_paths, min(MAX_IMAGES, len(image_paths)))
    else:
        image_paths = image_paths[:MAX_IMAGES]

    actual_n = len(image_paths)
    print(f"\n本次用于绘图的图片数量: {actual_n}")

    # 检查手动标签数量
    if len(MANUAL_LABELS) != actual_n:
        raise ValueError(
            f"MANUAL_LABELS 数量与图片数量不一致。\n"
            f"当前图片数量: {actual_n}\n"
            f"当前标签数量: {len(MANUAL_LABELS)}"
        )

    # 读取缩略图
    thumbnails = []
    gt_values = []

    print("\n图片与真值对应关系如下：")
    for idx, img_path in enumerate(image_paths):
        pil_img = Image.open(img_path).convert("RGB")
        thumb = resize_with_padding(pil_img, THUMB_SIZE)
        thumbnails.append(thumb)

        label = MANUAL_LABELS[idx]
        if label not in [0, 1]:
            raise ValueError(f"第 {idx+1} 个标签不是 0 或 1，请检查 MANUAL_LABELS。")

        gt_values.append(label)

        print(f"{idx+1:02d}. {os.path.basename(img_path)} -> {label} ({LABEL_TO_NAME[label]})")

    # 补空白
    total_cells = N_ROWS * N_COLS
    while len(gt_values) < total_cells:
        gt_values.append(np.nan)

    thumb_canvas = make_thumbnail_grid(thumbnails, N_ROWS, N_COLS, THUMB_SIZE)
    gt_matrix = np.array(gt_values).reshape(N_ROWS, N_COLS)

    # ==================================================
    # 4. 绘图
    # ==================================================
    fig = plt.figure(figsize=(12.5, 6.0), facecolor=FIG_BG_COLOR)
    outer = gridspec.GridSpec(
        1, 4,
        width_ratios=[1.20, 0.16, 1.0, 0.10],
        wspace=0.18
    )

    # 左侧样本图
    ax_img = plt.subplot(outer[0])
    ax_img.imshow(thumb_canvas)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_visible(False)

    # 中间箭头
    ax_arrow = plt.subplot(outer[1])
    ax_arrow.axis("off")
    arrow = FancyArrowPatch(
        (0.08, 0.5), (0.92, 0.5),
        arrowstyle="simple",
        mutation_scale=34,
        fc="red",
        ec="black",
        linewidth=1.0
    )
    ax_arrow.add_patch(arrow)
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)

    # 右侧真值热力图
    ax_hm = plt.subplot(outer[2])

    cmap = plt.cm.YlGnBu_r.copy()
    cmap.set_bad(color="white")

    im = ax_hm.imshow(gt_matrix, cmap=cmap, vmin=0, vmax=1)
    ax_hm.set_xticks([])
    ax_hm.set_yticks([])
    for spine in ax_hm.spines.values():
        spine.set_visible(False)

    # 在格子中写数字
    for i in range(N_ROWS):
        for j in range(N_COLS):
            val = gt_matrix[i, j]
            if np.isnan(val):
                continue
            ax_hm.text(
                j, i, f"{int(val)}",
                ha="center", va="center",
                fontsize=14, color="black"
            )

    # 色条
    cax = plt.subplot(outer[3])
    cb = plt.colorbar(im, cax=cax)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["0", "1"])

    # 顶部和底部标签
    cax.text(
        0.5, 1.06, LABEL_TO_NAME[1],
        ha="center", va="bottom",
        transform=cax.transAxes,
        fontsize=13,
        bbox=dict(boxstyle="square,pad=0.35", facecolor="#F2C200", edgecolor="black")
    )
    cax.text(
        0.5, -0.08, LABEL_TO_NAME[0],
        ha="center", va="top",
        transform=cax.transAxes,
        fontsize=13,
        bbox=dict(boxstyle="square,pad=0.35", facecolor="#C9EAF4", edgecolor="black")
    )

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\n[OK] 真值图已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()