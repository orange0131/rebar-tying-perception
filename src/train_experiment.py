import argparse
from pathlib import Path

import timm
import torch
import torch.nn as nn

from dataset_utils import run_training
from model_ablation import AblationModel, CBAM, GeM


class AblationHeadModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,),
        )
        channels = self.backbone.feature_info.channels()[-1]
        self.cbam = CBAM(channels)
        self.pool = GeM()
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)[0]
        feat = self.cbam(feat)
        feat = self.pool(feat)
        return self.classifier(feat)


def build_model(model_name, pretrained=True, dropout=0.2):
    if model_name == "resnet18":
        return timm.create_model("resnet18", pretrained=pretrained, num_classes=2)
    if model_name == "efficientnet_b0":
        return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=2)
    if model_name == "densenet121":
        return timm.create_model("densenet121", pretrained=pretrained, num_classes=2)
    if model_name == "mobilenetv3_small":
        return timm.create_model("mobilenetv3_small_100", pretrained=pretrained, num_classes=2)
    if model_name == "convnext_tiny":
        return timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=2)
    if model_name == "convnext_cbam":
        return AblationModel(
            num_classes=2,
            pretrained=pretrained,
            use_cbam=True,
            use_gem=False,
            dropout=dropout,
        )
    if model_name == "convnext_gem":
        return AblationModel(
            num_classes=2,
            pretrained=pretrained,
            use_cbam=False,
            use_gem=True,
            dropout=dropout,
        )
    if model_name == "convnext_cbam_gem":
        return AblationModel(
            num_classes=2,
            pretrained=pretrained,
            use_cbam=True,
            use_gem=True,
            dropout=dropout,
        )
    if model_name == "convnext_cbam_gem_head":
        return AblationHeadModel(num_classes=2, pretrained=pretrained, dropout=dropout)

    raise ValueError(f"Unsupported model: {model_name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train rebar-tying state recognition models."
    )
    parser.add_argument(
        "--model",
        default="convnext_cbam_gem",
        choices=[
            "resnet18",
            "efficientnet_b0",
            "densenet121",
            "mobilenetv3_small",
            "convnext_tiny",
            "convnext_cbam",
            "convnext_gem",
            "convnext_cbam_gem",
            "convnext_cbam_gem_head",
        ],
    )
    parser.add_argument("--data-dir", default="data/rebar_dataset")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--loss", default="ce", choices=["ce", "focal"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path("runs") / args.model)

    model = build_model(
        args.model,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    )

    run_training(
        model=model,
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=output_dir,
        loss_name=args.loss,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        early_stop=args.early_stop,
        amp=not args.no_amp,
        use_weighted_sampler=args.weighted_sampler,
    )


if __name__ == "__main__":
    main()
