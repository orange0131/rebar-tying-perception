import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


def draw_heatmap(matrix, row_labels, col_labels, save_path, title):
    matrix = np.array(matrix, dtype=float)

    fig_h = max(6, 0.45 * len(row_labels))
    fig_w = max(8, 1.2 * len(col_labels))
    plt.figure(figsize=(fig_w, fig_h))

    im = plt.imshow(matrix, aspect='auto', vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.03, pad=0.02)

    plt.xticks(range(len(col_labels)), col_labels, rotation=30, ha='right')
    plt.yticks(range(len(row_labels)), row_labels)

    # 写数值
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value < 0.5 else "black"
            plt.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8, color=text_color)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap: {save_path}")


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
    print("class_names =", class_names)

    # ===== 可选：只取前20张，避免图太长 =====
    max_samples = 10
    image_infos = image_infos[:max_samples]

    # 预加载模型
    models = {}
    for show_name, weight_path in model_configs.items():
        model = create_model(model_name_map[show_name])
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models[show_name] = model
        print(f"Loaded: {show_name}")

    rows = []
    gt_prob_matrix = []
    pred_prob_matrix = []
    row_labels = []

    for img_path, gt_name in image_infos:
        image_name = os.path.basename(img_path)
        gt_idx = class_to_idx[gt_name]
        row_labels.append(f"{image_name} | {gt_name}")

        row = {
            "image": image_name,
            "gt": gt_name
        }

        gt_prob_row = []
        pred_prob_row = []

        for show_name, model in models.items():
            pred_name, probs = predict_one(model, img_path, transform, class_names, device)

            gt_prob = float(probs[gt_idx])
            pred_idx = int(np.argmax(probs))
            pred_prob = float(probs[pred_idx])

            row[f"{show_name}_pred"] = pred_name
            row[f"{show_name}_gt_prob"] = round(gt_prob, 4)
            row[f"{show_name}_pred_prob"] = round(pred_prob, 4)

            gt_prob_row.append(gt_prob)
            pred_prob_row.append(pred_prob)

        gt_prob_matrix.append(gt_prob_row)
        pred_prob_matrix.append(pred_prob_row)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "heatmap_table.csv", index=False, encoding="utf-8-sig")
    df.to_excel(save_dir / "heatmap_table.xlsx", index=False)

    col_labels = list(models.keys())

    draw_heatmap(
        gt_prob_matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        save_path=save_dir / "gt_prob_heatmap.png",
        title="Probability of Ground-Truth Class"
    )

    draw_heatmap(
        pred_prob_matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        save_path=save_dir / "pred_prob_heatmap.png",
        title="Confidence of Predicted Class"
    )


if __name__ == "__main__":
    main()
