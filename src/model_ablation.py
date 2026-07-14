import torch
import torch.nn as nn
import timm


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


class AblationModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, use_cbam=False, use_gem=False, dropout=0.2):
        super().__init__()

        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,)
        )
        channels = self.backbone.feature_info.channels()[-1]

        self.use_cbam = use_cbam
        self.use_gem = use_gem

        self.cbam = CBAM(channels) if use_cbam else nn.Identity()
        self.pool = GeM() if use_gem else nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        feat = self.backbone(x)[0]         # [B, C, H, W]
        feat = self.cbam(feat)

        if self.use_gem:
            feat = self.pool(feat)         # [B, C]
        else:
            feat = self.pool(feat)         # [B, C, 1, 1]
            feat = torch.flatten(feat, 1)  # [B, C]

        feat = self.dropout(feat)
        out = self.fc(feat)
        return out