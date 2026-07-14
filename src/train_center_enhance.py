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


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = torch.mean(x, dim=(-1, -2), keepdim=False)
        return x.pow(1.0 / self.p)


class CenterPriorEnhance(nn.Module):
    def __init__(self, alpha=0.4, sigma=0.5):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w, device=device),
            indexing='ij'
        )
        dist2 = xx ** 2 + yy ** 2
        gaussian_map = torch.exp(-dist2 / (2 * self.sigma ** 2))
        gaussian_map = gaussian_map.unsqueeze(0).unsqueeze(0)

        weight = 1.0 + self.alpha * gaussian_map
        return x * weight


class CenterEnhanceModel(nn.Module):
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
        self.center_enhance = CenterPriorEnhance(alpha=0.4, sigma=0.5)
        self.pool = GeM()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[0]
        feat = self.cbam(feat)
        feat = self.center_enhance(feat)
        feat = self.pool(feat)
        feat = self.dropout(feat)
        out = self.fc(feat)
        return out


def main():
    data_dir = "data/rebar_dataset"
    output_dir = r"./runs/center_enhance"

    model = CenterEnhanceModel(num_classes=2, pretrained=True, dropout=0.2)

    run_training(
        model=model,
        model_name='center_enhance',
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
