import torch
import torch.nn as nn
import torch.nn.functional as F
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


class LocalGlobalDualBranchModel(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        dropout=0.2,
        local_crop_ratio=0.6
    ):
        super().__init__()
        self.local_crop_ratio = local_crop_ratio

        # 共享 backbone，更稳，也更省参数
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,)
        )
        channels = self.backbone.feature_info.channels()[-1]

        # 全局分支
        self.global_cbam = CBAM(channels)
        self.global_pool = GeM()

        # 局部分支
        self.local_cbam = CBAM(channels)
        self.local_pool = GeM()

        fusion_dim = channels * 2

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fusion_dim, num_classes)

    def center_crop_and_resize(self, x):
        """
        x: [B, C, H, W]
        从中心裁剪出一个局部区域，再 resize 回原大小
        """
        b, c, h, w = x.shape
        crop_h = int(h * self.local_crop_ratio)
        crop_w = int(w * self.local_crop_ratio)

        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2

        x_local = x[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
        x_local = F.interpolate(
            x_local, size=(h, w), mode='bilinear', align_corners=False
        )
        return x_local

    def extract_branch_feature(self, x, cbam, pool):
        feat = self.backbone(x)[0]
        feat = cbam(feat)
        feat = pool(feat)
        return feat

    def forward(self, x):
        # 全局输入
        x_global = x

        # 局部输入：中心裁剪再放大
        x_local = self.center_crop_and_resize(x)

        # 分别提特征
        feat_global = self.extract_branch_feature(
            x_global, self.global_cbam, self.global_pool
        )
        feat_local = self.extract_branch_feature(
            x_local, self.local_cbam, self.local_pool
        )

        # 融合
        feat = torch.cat([feat_global, feat_local], dim=1)
        feat = self.dropout(feat)
        out = self.fc(feat)
        return out


def main():
    data_dir = "data/rebar_dataset"
    output_dir = r"./runs/local_global_dual_branch"

    model = LocalGlobalDualBranchModel(
        num_classes=2,
        pretrained=True,
        dropout=0.2,
        local_crop_ratio=0.7   # 推荐先用 0.6
    )

    run_training(
        model=model,
        model_name='local_global_dual_branch',
        data_dir=data_dir,
        output_dir=output_dir,
        loss_name='ce',
        img_size=224,
        batch_size=16,          # 双分支更占显存，先用16更稳
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
