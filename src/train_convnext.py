import timm
from dataset_utils import run_training

def main():
    data_dir = "data/rebar_dataset"
    output_dir = r"./runs/convnext_tiny"

    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=2)

    run_training(
        model=model,
        model_name='convnext_tiny',
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
