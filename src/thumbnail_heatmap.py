import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm


# =========================
# 模块定义
# =========================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        hidden = max(in_planes // ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = torch.mean(x, dim=(-1, -2), keepdim=False)
        return x.pow(1.0 / self.p)


class OursCBAMGeM(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,)
        )
        channels = self.backbone.feature_info.channels()[-1]
        self.cbam = CBAM(channels)
        self.pool = GeM()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[0]
        feat = self.cbam(feat)
        feat = self.pool(feat)
        feat = self.dropout(feat)
        return self.fc(feat)


class AblationHeadModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,)
        )
        channels = self.backbone.feature_info.channels()[-1]
        self.cbam = CBAM(channels)
        self.pool = GeM()
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)[0]
        feat = self.cbam(feat)
        feat = self.pool(feat)
        return self.classifier(feat)


# =========================
# 构建模型
# =========================
def create_model(model_name):
    if model_name == "resnet18":
        return timm.create_model("resnet18", pretrained=False, num_classes=2)

    elif model_name == "efficientnet_b0":
        return timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)

    elif model_name == "convnext_tiny":
        return timm.create_model("convnext_tiny", pretrained=False, num_classes=2)

    elif model_name == "ablation_cbam":
        backbone = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            features_only=True,
            out_indices=(3,)
        )
        channels = backbone.feature_info.channels()[-1]

        class CBAMModel(nn.Module):
            def __init__(self, backbone, channels):
                super().__init__()
                self.backbone = backbone
                self.cbam = CBAM(channels)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.dropout = nn.Dropout(0.2)
                self.fc = nn.Linear(channels, 2)

            def forward(self, x):
                feat = self.backbone(x)[0]
                feat = self.cbam(feat)
                feat = self.pool(feat)
                feat = torch.flatten(feat, 1)
                feat = self.dropout(feat)
                return self.fc(feat)

        return CBAMModel(backbone, channels)

    elif model_name == "ablation_gem":
        return OursCBAMGeM(num_classes=2, pretrained=False, dropout=0.2)

    elif model_name == "ablation_head":
        return AblationHeadModel(num_classes=2, pretrained=False, dropout=0.2)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


# =========================
# 工具函数
# =========================
def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_image_paths(test_dir):
    class_names = sorted([d.name for d in Path(test_dir).iterdir() if d.is_dir()])
    image_infos = []
    for cls in class_names:
        for p in sorted((Path(test_dir) / cls).glob("*.*")):
            image_infos.append((str(p), cls))
    return image_infos, class_names


@torch.no_grad()
def predict_one(model, img_path, transform, class_names, device):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_name = class_names[pred_idx]
    return pred_name, probs


def center_crop_square(img: Image.Image):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def make_thumbnail(img_path, size=72):
    img = Image.open(img_path).convert("RGB")
    img = center_crop_square(img)
    img = img.resize((size, size))
    return np.array(img)


def draw_thumbnail_heatmap(thumbnails, labels, matrix, col_labels, save_path, title):
    n_rows = len(labels)
    n_cols = len(col_labels)

    fig_w = 2.8 + 2.2 + max(6, n_cols * 1.2)
    fig_h = max(6, n_rows * 0.9)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        nrows=n_rows + 1,
        ncols=3,
        width_ratios=[1.2, 2.2, max(4, n_cols * 1.3)],
        height_ratios=[0.6] + [1] * n_rows,
        figure=fig
    )

    # 标题行
    ax_title_img = fig.add_subplot(gs[0, 0])
    ax_title_txt = fig.add_subplot(gs[0, 1])
    ax_title_heat = fig.add_subplot(gs[0, 2])

    for ax in [ax_title_img, ax_title_txt]:
        ax.axis("off")
    ax_title_img.text(0.5, 0.5, "Image", ha="center", va="center", fontsize=11, fontweight="bold")
    ax_title_txt.text(0.5, 0.5, "Sample / GT", ha="center", va="center", fontsize=11, fontweight="bold")

    ax_title_heat.imshow(np.zeros((1, n_cols)), aspect='auto', vmin=0, vmax=1, cmap='viridis')
    ax_title_heat.set_xticks(range(n_cols))
    ax_title_heat.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=10, fontweight="bold")
    ax_title_heat.set_yticks([])
    ax_title_heat.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    for spine in ax_title_heat.spines.values():
        spine.set_visible(False)

    # 每一行
    for i in range(n_rows):
        ax_img = fig.add_subplot(gs[i + 1, 0])
        ax_txt = fig.add_subplot(gs[i + 1, 1])
        ax_heat = fig.add_subplot(gs[i + 1, 2])

        # 缩略图
        ax_img.imshow(thumbnails[i])
        ax_img.axis("off")

        # 文本
        ax_txt.axis("off")
        ax_txt.text(0.02, 0.5, labels[i], ha="left", va="center", fontsize=10)

        # 热力格子
        row_data = np.array(matrix[i]).reshape(1, -1)
        im = ax_heat.imshow(row_data, aspect='auto', vmin=0, vmax=1, cmap='viridis')
        ax_heat.set_xticks([])
        ax_heat.set_yticks([])

        for j in range(n_cols):
            value = row_data[0, j]
            text_color = "white" if value < 0.55 else "black"
            ax_heat.text(j, 0, f"{value:.3f}", ha="center", va="center", fontsize=9, color=text_color)

        for spine in ax_heat.spines.values():
            spine.set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.97, 0.98])

    # 颜色条
    cax = fig.add_axes([0.98, 0.15, 0.015, 0.7])
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    # ===== 改成你的测试集目录 =====
    test_dir = "examples/test_samples"

    # ===== 改成你的模型权重路径 =====
    model_configs = {
        "ResNet18": r"./shiyan1/resnet18/best_model.pth",
        "EffNet-B0": r"./shiyan1/efficientnet_b0/best_model.pth",
        "ConvNeXt": r"./shiyan1/convnext_tiny/best_model.pth",
        "CBAM": r"./shiyan1/ablation_cbam/best_model.pth",
        "CBAM+GeM": r"./shiyan1/ablation_gem/best_model.pth",
        "CBAM+GeM+Head": r"./shiyan1/ablation_head/best_model.pth",
    }

    model_name_map = {
        "ResNet18": "resnet18",
        "EffNet-B0": "efficientnet_b0",
        "ConvNeXt": "convnext_tiny",
        "CBAM": "ablation_cbam",
        "CBAM+GeM": "ablation_gem",
        "CBAM+GeM+Head": "ablation_head",
    }

    save_dir = Path("./compare_probs")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform(img_size=224)

    image_infos, class_names = load_image_paths(test_dir)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # ===== 建议你手动挑一些困难样本，这里先取前12张 =====
    max_samples = 12
    image_infos = image_infos[:max_samples]

    # 加载模型
    models = {}
    for show_name, weight_path in model_configs.items():
        model = create_model(model_name_map[show_name])
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models[show_name] = model
        print(f"Loaded: {show_name}")

    thumbnails = []
    labels = []
    gt_prob_matrix = []
    rows = []

    for img_path, gt_name in image_infos:
        image_name = os.path.basename(img_path)
        gt_idx = class_to_idx[gt_name]

        thumbnails.append(make_thumbnail(img_path, size=72))
        labels.append(f"{image_name}\nGT: {gt_name}")

        row = {"image": image_name, "gt": gt_name}
        gt_probs = []

        for show_name, model in models.items():
            pred_name, probs = predict_one(model, img_path, transform, class_names, device)
            gt_prob = float(probs[gt_idx])

            row[f"{show_name}_pred"] = pred_name
            row[f"{show_name}_gt_prob"] = round(gt_prob, 4)
            gt_probs.append(gt_prob)

        rows.append(row)
        gt_prob_matrix.append(gt_probs)

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "thumbnail_heatmap_table.csv", index=False, encoding="utf-8-sig")
    df.to_excel(save_dir / "thumbnail_heatmap_table.xlsx", index=False)

    draw_thumbnail_heatmap(
        thumbnails=thumbnails,
        labels=labels,
        matrix=gt_prob_matrix,
        col_labels=list(models.keys()),
        save_path=save_dir / "thumbnail_gt_prob_heatmap.png",
        title="Model Comparison on Same Samples (Ground-Truth Class Probability)"
    )


if __name__ == "__main__":
    main()
