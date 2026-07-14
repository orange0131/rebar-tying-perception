import os
import csv
import json
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import timm

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================
# 基础工具
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_dataloaders(data_dir, img_size=224, batch_size=16, num_workers=4):
    train_tf, val_tf = build_transforms(img_size)

    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_tf)
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_set.classes


# =========================
# 模型模块
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


class CBAMGeMEncoder(nn.Module):
    def __init__(self, pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,)
        )
        channels = self.backbone.feature_info.channels()[-1]
        self.channels = channels

        self.cbam = CBAM(channels)
        self.pool = GeM()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        feat = self.backbone(x)[0]
        feat = self.cbam(feat)
        feat = self.pool(feat)
        feat = self.dropout(feat)
        return feat


class LocalConsistencyModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.2, local_crop_ratio=0.6):
        super().__init__()
        self.local_crop_ratio = local_crop_ratio
        self.encoder = CBAMGeMEncoder(pretrained=pretrained, dropout=dropout)
        self.classifier = nn.Linear(self.encoder.channels, num_classes)

    def center_crop_and_resize(self, x):
        b, c, h, w = x.shape
        crop_h = int(h * self.local_crop_ratio)
        crop_w = int(w * self.local_crop_ratio)

        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2

        x_local = x[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
        x_local = F.interpolate(x_local, size=(h, w), mode='bilinear', align_corners=False)
        return x_local

    def forward(self, x):
        # 全图
        feat_global = self.encoder(x)
        logits_global = self.classifier(feat_global)

        # 局部图
        x_local = self.center_crop_and_resize(x)
        feat_local = self.encoder(x_local)
        logits_local = self.classifier(feat_local)

        return logits_global, logits_local


# =========================
# 损失函数
# =========================
def consistency_loss(logits_a, logits_b):
    """
    让全图和局部图的输出概率尽量一致
    """
    prob_a = torch.softmax(logits_a, dim=1)
    prob_b = torch.softmax(logits_b, dim=1)
    return F.mse_loss(prob_a, prob_b)


# =========================
# 训练与评估
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device, lambda_cons=0.5, scaler=None):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits_global, logits_local = model(images)

                loss_global = criterion(logits_global, labels)
                loss_local = criterion(logits_local, labels)
                loss_cons = consistency_loss(logits_global, logits_local)

                loss = loss_global + loss_local + lambda_cons * loss_cons

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits_global, logits_local = model(images)

            loss_global = criterion(logits_global, labels)
            loss_local = criterion(logits_local, labels)
            loss_cons = consistency_loss(logits_global, logits_local)

            loss = loss_global + loss_local + lambda_cons * loss_cons
            loss.backward()
            optimizer.step()

        preds = logits_global.argmax(dim=1)   # 评估主分支
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0

    all_probs = []
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits_global, logits_local = model(images)

        loss_global = criterion(logits_global, labels)
        loss_local = criterion(logits_local, labels)
        loss_cons = consistency_loss(logits_global, logits_local)
        loss = loss_global + loss_local + 0.5 * loss_cons

        probs = torch.softmax(logits_global, dim=1)[:, 1]
        preds = logits_global.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return (
        running_loss / total,
        running_correct / total,
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


# =========================
# 结果可视化
# =========================
def plot_curves(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure()
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, history['train_acc'], label='train_acc')
    plt.plot(epochs, history['val_acc'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'acc_curve.png'), dpi=300)
    plt.close()


def save_history_csv(history, save_dir):
    csv_path = os.path.join(save_dir, 'history.csv')
    keys = list(history.keys())
    rows = zip(*[history[k] for k in keys])

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()


def plot_roc_pr(y_true, y_prob, save_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision, label=f'AP = {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=300)
    plt.close()

    return roc_auc, ap


def save_report(y_true, y_pred, class_names, save_dir):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)


# =========================
# 主训练流程
# =========================
def main():
    data_dir = "data/rebar_dataset"
    output_dir = r"./runs/cbam_gem_local_consistency"

    img_size = 224
    batch_size = 16
    epochs = 20
    lr = 1e-4
    weight_decay = 1e-4
    num_workers = 4
    seed = 42
    early_stop = 10
    amp = True
    lambda_cons = 0.5
    local_crop_ratio = 0.6

    seed_everything(seed)
    ensure_dir(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers
    )

    model = LocalConsistencyModel(
        num_classes=2,
        pretrained=True,
        dropout=0.2,
        local_crop_ratio=local_crop_ratio
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if (amp and device.type == 'cuda') else None

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    best_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            lambda_cons=lambda_cons, scaler=scaler
        )
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(
            f"[cbam_gem_local_consistency] Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(output_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    plot_curves(history, output_dir)
    save_history_csv(history, output_dir)

    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))
    test_loss, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, device)

    plot_confusion_matrix(y_true, y_pred, class_names, output_dir)
    roc_auc, ap = plot_roc_pr(y_true, y_prob, output_dir)
    save_report(y_true, y_pred, class_names, output_dir)

    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    result = {
        'model_name': 'cbam_gem_local_consistency',
        'loss_name': 'ce + local_consistency',
        'best_val_acc': float(best_acc),
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'ap': float(ap),
        'class_names': class_names,
        'lambda_cons': lambda_cons,
        'local_crop_ratio': local_crop_ratio
    }

    with open(os.path.join(output_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
