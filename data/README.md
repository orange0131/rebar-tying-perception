# Data Directory

This directory is a placeholder for local datasets. The full image dataset is not tracked by Git.

Expected training layout:

```text
data/rebar_dataset/
├── train/
│   ├── tied/
│   └── untied/
├── val/
│   ├── tied/
│   └── untied/
└── test/
    ├── tied/
    └── untied/
```

Expected crop layout before splitting:

```text
data/rebar_crops/
├── tied/
└── untied/
```

Use `python src/split_dataset.py --src-dir data/rebar_crops --dst-dir data/rebar_dataset` to create the train/validation/test split.
