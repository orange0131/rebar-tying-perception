import torch
import torch.nn as nn
import timm
from dataset_utils import run_training


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


class AblationCBAMModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,)
        )
        channels = self.backbone.feature_info.channels()[-1]

        self.cbam = CBAM(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[0]
        feat = self.cbam(feat)
        feat = self.pool(feat).flatten(1)
        feat = self.dropout(feat)
        out = self.fc(feat)
        return out


def main():
    data_dir = "data/rebar_dataset"
    output_dir = r"./runs/ablation_cbam"

    model = AblationCBAMModel(num_classes=2, pretrained=True)

    run_training(
        model=model,
        model_name='ablation_cbam',
        data_dir=data_dir,
        output_dir=output_dir,
        loss_name='ce',
        img_size=224,
        batch_size=32,
        epochs=50,
        lr=1e-4,
        weight_decay=1e-4,
        num_workers=4,
        seed=42,
        early_stop=10,
        amp=True,
        use_weighted_sampler=False
    )


if __name__ == "__main__":
    main()
