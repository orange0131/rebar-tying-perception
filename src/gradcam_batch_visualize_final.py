import os
import glob
import argparse
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from model_ablation import AblationModel


# =========================
# 1. 配置区
# =========================
CKPT_PATH = r"./runs/ablation_gem/best_model.pth"   # 模型权重路径
INPUT_PATH = r"./test_images"                       # 可以是单张图片，也可以是文件夹
OUTPUT_DIR = r"./gradcam_outputs"                  # 输出路径

IMG_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ["已绑扎", "未绑扎"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型结构开关
USE_CBAM = True
USE_GEM = True

# 热力图阈值（0~1）
HEATMAP_THRESHOLD = 0.5

# 轮廓最小面积，小于这个面积的区域不画，避免很多小噪声点
MIN_CONTOUR_AREA = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM panels for rebar-tying state recognition."
    )
    parser.add_argument("--checkpoint", default=CKPT_PATH)
    parser.add_argument("--input", default=INPUT_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--use-cbam", action=argparse.BooleanOptionalAction, default=USE_CBAM)
    parser.add_argument("--use-gem", action=argparse.BooleanOptionalAction, default=USE_GEM)
    parser.add_argument("--threshold", type=float, default=HEATMAP_THRESHOLD)
    parser.add_argument("--min-contour-area", type=float, default=MIN_CONTOUR_AREA)
    return parser.parse_args()


# =========================
# 2. 图像预处理
# =========================
def build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])


# =========================
# 3. 构建模型
# =========================
def build_model():
    model = AblationModel(
        num_classes=NUM_CLASSES,
        pretrained=False,
        use_cbam=USE_CBAM,
        use_gem=USE_GEM,
        dropout=0.2
    )
    return model


# =========================
# 4. 加载权重
# =========================
def smart_load_state_dict(model, ckpt_path):
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    # 去掉可能存在的 module. 前缀
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    return model


# =========================
# 5. 获取 Grad-CAM 目标层
# =========================
def get_target_layer(model):
    """
    优先取 ConvNeXt 最后一阶段的最后一个 block；
    如果失败，则退化为最后一个 Conv2d。
    """
    # 尝试几种常见写法
    try:
        return model.backbone.stages_3.blocks[-1]
    except Exception:
        pass

    try:
        return model.backbone.stages[-1].blocks[-1]
    except Exception:
        pass

    # 从 named_modules 中找最后一个包含 blocks 的层
    candidate = None
    for name, module in model.named_modules():
        if "blocks" in name:
            candidate = module
    if candidate is not None:
        return candidate

    # 最后兜底：找最后一个 Conv2d
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m

    if last_conv is None:
        raise RuntimeError("无法找到可用于 Grad-CAM 的目标层。")

    return last_conv


# =========================
# 6. Grad-CAM 类
# =========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, x, class_idx=None):
        self.model.zero_grad()

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(torch.argmax(probs, dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward()

        acts = self.activations   # [B, C, H, W]
        grads = self.gradients    # [B, C, H, W]

        weights = grads.mean(dim=(2, 3), keepdim=True)   # [B, C, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        pred_idx = int(torch.argmax(probs, dim=1).item())
        pred_score = float(probs[0, pred_idx].item())

        return cam, pred_idx, pred_score


# =========================
# 7. 可视化工具
# =========================
def denormalize_image(tensor_img):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    img = tensor_img.cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
    return img


def apply_heatmap_on_image(rgb_img, cam):
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.float32(heatmap) * 0.4 + np.float32(rgb_img) * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return heatmap, overlay


def cam_to_binary_mask(cam, threshold=0.5):
    mask = (cam >= threshold).astype(np.uint8) * 255
    return mask


def apply_threshold_mask_on_image(rgb_img, mask_u8):
    overlay = rgb_img.copy()

    color_mask = np.zeros_like(rgb_img, dtype=np.uint8)
    color_mask[:, :, 0] = mask_u8   # 红色通道

    overlay = np.float32(overlay) * 0.7 + np.float32(color_mask) * 0.3
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def draw_mask_contours(rgb_img, mask_u8, min_area=20):
    img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            valid_contours.append(cnt)

    cv2.drawContours(img_bgr, valid_contours, -1, (0, 0, 255), 2)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def put_title(img, text):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    canvas = cv2.copyMakeBorder(
        img_bgr, 40, 0, 0, 0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )
    cv2.putText(
        canvas, text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 0, 0), 2
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def save_panel(original, heatmap, overlay, thresh_overlay, save_path, pred_name, pred_score):
    original = put_title(original, "Original")
    heatmap = put_title(heatmap, "Heatmap")
    overlay = put_title(overlay, f"Overlay | Pred: {pred_name} ({pred_score:.3f})")
    thresh_overlay = put_title(thresh_overlay, f"Thresholded | thr={HEATMAP_THRESHOLD:.2f}")

    panel = np.concatenate([original, heatmap, overlay, thresh_overlay], axis=1)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, panel_bgr)


# =========================
# 8. 获取输入图片列表
# =========================
def list_images(input_path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(input_path, ext)))
            files.extend(glob.glob(os.path.join(input_path, ext.upper())))
        files = sorted(files)
        return files

    raise FileNotFoundError(f"输入路径不存在: {input_path}")


# =========================
# 9. 单张图片处理
# =========================
def process_one_image(model, cam_engine, image_path, transform):
    pil_img = Image.open(image_path).convert("RGB")
    x = transform(pil_img).unsqueeze(0).to(DEVICE)

    cam, pred_idx, pred_score = cam_engine.generate(x)

    rgb_resized = denormalize_image(x[0])
    heatmap, overlay = apply_heatmap_on_image(rgb_resized, cam)

    # 阈值掩膜
    mask_u8 = cam_to_binary_mask(cam, threshold=HEATMAP_THRESHOLD)

    # 阈值叠加图
    thresh_overlay = apply_threshold_mask_on_image(rgb_resized, mask_u8)

    # 轮廓图
    contour_overlay = draw_mask_contours(rgb_resized, mask_u8, min_area=MIN_CONTOUR_AREA)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    pred_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{base_name}_original.png"),
        cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{base_name}_heatmap.png"),
        cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{base_name}_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{base_name}_mask_thr_{HEATMAP_THRESHOLD:.2f}.png"),
        mask_u8
    )
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{base_name}_overlay_thr_{HEATMAP_THRESHOLD:.2f}.png"),
        cv2.cvtColor(thresh_overlay, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{base_name}_contour_thr_{HEATMAP_THRESHOLD:.2f}.png"),
        cv2.cvtColor(contour_overlay, cv2.COLOR_RGB2BGR)
    )

    save_panel(
        rgb_resized,
        heatmap,
        overlay,
        thresh_overlay,
        os.path.join(OUTPUT_DIR, f"{base_name}_panel.png"),
        pred_name,
        pred_score
    )

    print(f"[OK] {base_name} -> {pred_name} ({pred_score:.4f}), threshold={HEATMAP_THRESHOLD}")


# =========================
# 10. 主函数
# =========================
def main():
    global CKPT_PATH, INPUT_PATH, OUTPUT_DIR, IMG_SIZE
    global USE_CBAM, USE_GEM, HEATMAP_THRESHOLD, MIN_CONTOUR_AREA

    args = parse_args()
    CKPT_PATH = args.checkpoint
    INPUT_PATH = args.input
    OUTPUT_DIR = args.output_dir
    IMG_SIZE = args.img_size
    USE_CBAM = args.use_cbam
    USE_GEM = args.use_gem
    HEATMAP_THRESHOLD = args.threshold
    MIN_CONTOUR_AREA = args.min_contour_area

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = build_model()
    model = smart_load_state_dict(model, CKPT_PATH)
    model.to(DEVICE)
    model.eval()

    target_layer = get_target_layer(model)
    print("Grad-CAM target layer:", target_layer)

    transform = build_transform(IMG_SIZE)
    image_list = list_images(INPUT_PATH)

    if len(image_list) == 0:
        print("没有找到图片，请检查 INPUT_PATH。")
        return

    print(f"共找到 {len(image_list)} 张图片。")

    cam_engine = GradCAM(model, target_layer)

    for img_path in image_list:
        try:
            process_one_image(model, cam_engine, img_path, transform)
        except Exception as e:
            print(f"[FAILED] {img_path}: {e}")

    cam_engine.remove()
    print("全部处理完成，结果保存在：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
