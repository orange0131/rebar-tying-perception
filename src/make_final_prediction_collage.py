import os
import math
import random

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch

from model_ablation import AblationModel


# ======================================================
# 1. 参数配置
# ======================================================
CKPT_PATH = r"./runs/ablation_cbam/best_model.pth"   # 最终模型权重
INPUT_DIR = r"./test_samples"                            # 输入图片目录
OUTPUT_PATH = r"./probability_heatmap.png"               # 输出图路径

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 类别名称，可按你的实际情况修改
CLASS_NAMES = ["已绑扎", "未绑扎"]

# 模型结构配置
USE_CBAM = True
USE_GEM = True

# 热力图显示哪一类的概率
# 0 -> 显示“已绑扎”概率
# 1 -> 显示“未绑扎”概率
TARGET_CLASS_INDEX = 0

# 左侧样本展示网格大小
N_ROWS = 3
N_COLS = 3
MAX_IMAGES = N_ROWS * N_COLS

# 是否随机抽图
RANDOM_SAMPLE = False
RANDOM_SEED = 42

# 缩略图尺寸
THUMB_SIZE = (95, 95)

# 图像整体风格
FIG_BG_COLOR = "#EDEDED"

# matplotlib 中文设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ======================================================
# 2. 工具函数
# ======================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])


def smart_load_state_dict(model, ckpt_path):
    """
    兼容不同保存格式的权重
    """
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    return model


def list_images_safe(folder, recursive=True):
    """
    安全读取图片，自动去重，避免 Windows 下 jpg/JPG 重复统计
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

    # 去重 + 排序
    files = sorted(set(files))
    return files


def build_model():
    model = AblationModel(
        num_classes=2,
        pretrained=False,
        use_cbam=USE_CBAM,
        use_gem=USE_GEM,
        dropout=0.2
    )
    return model


@torch.no_grad()
def predict_one(model, img_path, transform):
    pil_img = Image.open(img_path).convert("RGB")
    x = transform(pil_img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    return pil_img, probs


def resize_with_padding(pil_img, target_size=(95, 95)):
    """
    缩放并居中填充到固定大小
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

    # 读取图片
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

    # 建模
    model = build_model()
    model = smart_load_state_dict(model, CKPT_PATH)
    model.to(DEVICE)
    model.eval()

    transform = build_transform(IMG_SIZE)

    # 推理
    thumbnails = []
    prob_values = []

    target_class_name = CLASS_NAMES[TARGET_CLASS_INDEX]
    opposite_class_name = CLASS_NAMES[1 - TARGET_CLASS_INDEX]

    print(f"\n显示的热力图概率类别: {target_class_name}\n")

    for img_path in image_paths:
        pil_img, probs = predict_one(model, img_path, transform)

        thumb = resize_with_padding(pil_img, THUMB_SIZE)
        thumbnails.append(thumb)
        prob_values.append(float(probs[TARGET_CLASS_INDEX]))

        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]
        pred_score = float(probs[pred_idx])

        print(
            f"[OK] {os.path.basename(img_path)} | "
            f"预测: {pred_name}({pred_score:.3f}) | "
            f"{target_class_name}概率: {probs[TARGET_CLASS_INDEX]:.3f}"
        )

    # 若数量不足则补空白
    total_cells = N_ROWS * N_COLS
    while len(prob_values) < total_cells:
        prob_values.append(np.nan)

    # 左侧缩略图拼图
    thumb_canvas = make_thumbnail_grid(thumbnails, N_ROWS, N_COLS, THUMB_SIZE)

    # 概率矩阵
    prob_matrix = np.array(prob_values).reshape(N_ROWS, N_COLS)

    # ==================================================
    # 4. 绘图
    # ==================================================
    fig = plt.figure(figsize=(12.5, 6.0), facecolor=FIG_BG_COLOR)
    outer = gridspec.GridSpec(
        1, 4,
        width_ratios=[1.20, 0.16, 1.0, 0.10],
        wspace=0.18
    )

    # ---------------- 左侧样本图 ----------------
    ax_img = plt.subplot(outer[0])
    ax_img.imshow(thumb_canvas)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_visible(False)

    # ---------------- 中间箭头 ----------------
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

    # ---------------- 右侧热力图 ----------------
    ax_hm = plt.subplot(outer[2])

    cmap = plt.cm.YlGnBu_r.copy()
    cmap.set_bad(color="white")

    im = ax_hm.imshow(prob_matrix, cmap=cmap, vmin=0, vmax=1)
    ax_hm.set_xticks([])
    ax_hm.set_yticks([])
    for spine in ax_hm.spines.values():
        spine.set_visible(False)

    # 在每个格子上写数值
    for i in range(N_ROWS):
        for j in range(N_COLS):
            val = prob_matrix[i, j]
            if np.isnan(val):
                continue
            ax_hm.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=12, color="black"
            )

    # ---------------- 色条 ----------------
    cax = plt.subplot(outer[3])
    cb = plt.colorbar(im, cax=cax)
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    # 顶部和底部类别标签
    cax.text(
        0.5, 1.06, target_class_name,
        ha="center", va="bottom",
        transform=cax.transAxes,
        fontsize=13,
        bbox=dict(boxstyle="square,pad=0.35", facecolor="#F2C200", edgecolor="black")
    )
    cax.text(
        0.5, -0.08, opposite_class_name,
        ha="center", va="top",
        transform=cax.transAxes,
        fontsize=13,
        bbox=dict(boxstyle="square,pad=0.35", facecolor="#C9EAF4", edgecolor="black")
    )

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\n[OK] 概率热力图已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()