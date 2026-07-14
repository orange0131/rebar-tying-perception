# Task-Oriented RGB-D Perception and Rebar-Tying State Recognition

This repository contains the code used for the manuscript **"Task-Oriented RGB-D Perception and Execution Planning Framework for Autonomous Robotic Rebar Tying"**.

The project focuses on autonomous robotic rebar tying in structured construction scenes. The code in this repository covers the image-based rebar intersection state recognition part of the pipeline, including dataset preparation, model training, ablation experiments, result summarization, and Grad-CAM visualization.

![Overview of the proposed task-oriented RGB-D perception and execution planning framework](assets/paper_framework_overview.png)

## Highlights

- ConvNeXt-Tiny based tied/untied rebar intersection classifier.
- CBAM attention and GeM pooling variants for ablation studies.
- Baseline models including ResNet-18, EfficientNet-B0, DenseNet-121, MobileNetV3-Small, and ConvNeXt-Tiny.
- Training, evaluation, ROC/PR curves, confusion matrix, and summary table generation.
- Grad-CAM visualization for model interpretability.
- Interactive crop tool for preparing tied/untied intersection samples.

## Paper Figures

The complete manuscript framework integrates RGB-D rebar intersection perception, binding state recognition, robot-frame coordinate mapping, and task-constrained execution planning.

![Task semantic recognition and perception-to-execution mapping](assets/paper_semantic_mapping.png)

The recognition module classifies cropped intersection patches as tied or untied, filters tied intersections from the executable task set, and maps retained untied targets into the robot coordinate frame.

![Grad-CAM comparison for binding state recognition](assets/paper_module_heatmaps.png)

![Ground-truth labels and predicted probability heatmaps](assets/paper_probability_heatmaps.png)

For execution planning, untied intersections are visited under grid-constrained path optimization.

![Path planning comparison](assets/paper_path_planning.png)

## Repository Structure

```text
.
├── assets/                  # Lightweight figures for the README
├── data/                    # Local data placeholder; full dataset is not tracked
├── examples/                # Small demo inputs
├── results/                 # Lightweight paper/result figures
├── src/
│   ├── dataset_utils.py      # Training, evaluation, metrics, and plotting utilities
│   ├── model_ablation.py     # ConvNeXt + CBAM/GeM model components
│   ├── train_experiment.py   # Recommended unified training entry point
│   ├── split_dataset.py      # Train/val/test splitting utility
│   ├── gradcam_batch_visualize_final.py
│   ├── summarize_results.py
│   └── ...
└── requirements.txt
```

Large training artifacts, raw image datasets, virtual environments, and model weights are intentionally excluded from Git.

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you train with CUDA, install the PyTorch build that matches your GPU driver and CUDA version.

## Data Layout

The training code expects an ImageFolder-style dataset:

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

If you start from cropped tied/untied samples, organize them as:

```text
data/rebar_crops/
├── tied/
└── untied/
```

Then split them into train/validation/test sets:

```bash
python src/split_dataset.py \
  --src-dir data/rebar_crops \
  --dst-dir data/rebar_dataset \
  --train-ratio 0.6 \
  --val-ratio 0.2
```

## Interactive Cropping

The interactive crop tool can be used to extract tied and untied rebar intersections from raw images:

```bash
python src/main.py \
  --input-dir data/raw_images \
  --output-dir data/rebar_crops \
  --crop-size 224 \
  --padding 10
```

Controls:

- Drag a box around an intersection.
- Press `t` to save as tied.
- Press `u` to save as untied.
- Press `n` / `p` to move to the next/previous image.
- Press `z` to undo the last displayed annotation.
- Press `q` or `Esc` to quit.

## Training

The recommended training entry point is `src/train_experiment.py`.

Train the proposed ConvNeXt + CBAM + GeM model:

```bash
python src/train_experiment.py \
  --model convnext_cbam_gem \
  --data-dir data/rebar_dataset \
  --output-dir runs/convnext_cbam_gem \
  --epochs 50 \
  --batch-size 32
```

Train common baselines:

```bash
python src/train_experiment.py --model resnet18 --data-dir data/rebar_dataset --output-dir runs/resnet18
python src/train_experiment.py --model efficientnet_b0 --data-dir data/rebar_dataset --output-dir runs/efficientnet_b0
python src/train_experiment.py --model densenet121 --data-dir data/rebar_dataset --output-dir runs/densenet121
python src/train_experiment.py --model mobilenetv3_small --data-dir data/rebar_dataset --output-dir runs/mobilenetv3_small
python src/train_experiment.py --model convnext_tiny --data-dir data/rebar_dataset --output-dir runs/convnext_tiny
```

Train ablation variants:

```bash
python src/train_experiment.py --model convnext_cbam --data-dir data/rebar_dataset --output-dir runs/convnext_cbam
python src/train_experiment.py --model convnext_gem --data-dir data/rebar_dataset --output-dir runs/convnext_gem
python src/train_experiment.py --model convnext_cbam_gem_head --data-dir data/rebar_dataset --output-dir runs/convnext_cbam_gem_head
```

Each run writes metrics and figures to the selected output directory, including:

- `best_model.pth`
- `history.csv`
- `result.json`
- `loss_curve.png`
- `acc_curve.png`
- `confusion_matrix.png`
- `roc_curve.png`
- `pr_curve.png`
- `classification_report.txt`

## Result Summary

After training multiple models, summarize results with:

```bash
python src/summarize_results.py \
  --runs-dir runs \
  --save-dir results/runs_summary
```

## Grad-CAM Visualization

Generate Grad-CAM panels from a trained checkpoint:

```bash
python src/gradcam_batch_visualize_final.py \
  --checkpoint runs/convnext_cbam_gem/best_model.pth \
  --input examples/test_images \
  --output-dir gradcam_outputs \
  --use-cbam \
  --use-gem
```

The script exports the original image, heatmap, overlay, thresholded mask, contour visualization, and a combined panel for each input image.

## Reported Manuscript Results

The manuscript reports that the ConvNeXt-Tiny semantic recognition model enhanced with CBAM attention and GeM pooling achieved:

- Accuracy: 98.24%
- Precision: 98.84%
- F1-score: 98.28%

Confusion matrices from the manuscript are included below for the baseline and ablation variants.

![Confusion matrices for binding state recognition variants](assets/paper_confusion_matrices.png)

The complete perception-to-execution framework also includes RGB-D target perception, robot-frame spatial mapping, and task-constrained execution planning. This repository currently focuses on the classification and visualization code used for the semantic recognition experiments.

## Notes for Submission

- Keep the GitHub repository private while the manuscript is under review if the target journal has anonymity or prior-publication constraints.
- Do not upload raw field images, full datasets, trained `.pth` checkpoints, or reviewer-sensitive files unless the journal explicitly requires public availability.
- If the code must be made public later, add a dataset access statement and a license before release.

## Citation

If this code is useful for your research, please cite the corresponding manuscript:

```bibtex
@article{ma2026rebarTying,
  title   = {Task-Oriented RGB-D Perception and Execution Planning Framework for Autonomous Robotic Rebar Tying},
  author  = {Ma, Zhanguo and Fan, Shicheng and Sheikder, Chandan and Yin, Zhiwei and He, Haotong and Liu, Pengyang and Yang, Guan},
  year    = {2026},
  note    = {Manuscript}
}
```

## License

No open-source license has been selected yet. Please add a license before making the repository public.
