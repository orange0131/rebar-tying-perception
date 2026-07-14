from dataset_utils import run_training
from model_ablation import AblationModel


def main():
    data_dir = "data/rebar_dataset"
    output_dir = r"./runs/ablation_focal"

    model = AblationModel(
        num_classes=2,
        pretrained=True,
        use_cbam=True,
        use_gem=True
    )

    run_training(
        model=model,
        model_name='ablation_focal',
        data_dir=data_dir,
        output_dir=output_dir,
        loss_name='focal',
        img_size=224,
        batch_size=32,
        epochs=50,
        lr=1e-4,
        weight_decay=1e-4,
        num_workers=4,
        seed=42,
        early_stop=10,
        amp=True,
        use_weighted_sampler=True
    )


if __name__ == "__main__":
    main()
