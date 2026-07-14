import os
import shutil
import random
import argparse
from pathlib import Path

def split_dataset(src_dir, dst_dir, train_ratio=0.6, val_ratio=0.2, seed=42):
    random.seed(seed)

    classes = ['tied', 'untied']

    for c in classes:
        src_path = Path(src_dir) / c
        images = list(src_path.glob('*.*'))

        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }

        for split in splits:
            dst_path = Path(dst_dir) / split / c
            dst_path.mkdir(parents=True, exist_ok=True)

            for img in splits[split]:
                shutil.copy(img, dst_path / img.name)

        print(f"{c}: total={n}, train={n_train}, val={n_val}, test={n-n_train-n_val}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split cropped rebar images into train/val/test folders."
    )
    parser.add_argument("--src-dir", default="data/rebar_crops")
    parser.add_argument("--dst-dir", default="data/rebar_dataset")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
