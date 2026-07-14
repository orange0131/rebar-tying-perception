import os
import glob
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# 这里改成你自己的模型文件
from model_ablation import AblationModel


# =========================
# 1. 配置
# =========================
CKPT_PATH = r"./runs/ablation_gem/best_model.pth"   # 你的权重
INPUT_PATH = r"./test_images"                       # 可以是单张图片，也可以是文件夹
OUTPUT_DIR = r"./gradcam_outputs"

IMG_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ["tied", "untied"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 对应你的模型配置
USE_CBAM = True
USE_GEM = True


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

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model


# =========================
# 5. 找 Grad-CAM 目标层
# =========================
def get_target_layer(model):
    # 你的模型里 backbone 是 timm features_only=True 的 convnext_tiny
    # 一般最后一层块这样取
    try:
        return model.backbone.stages_3.blocks[-1]
    except Exception:
        pass

    # 兜底：找最后一个 Conv2d
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("无法找到可用于 Grad-CAM 的目标层。")
    return last_conv


# =========================
# 6. Grad-CAM
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


def put_title(img, text):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    canvas = cv2.copyMakeBorder(
        img_bgr, 40, 0, 0, 0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )
    cv2.putText(
        canvas, text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (0, 0, 0), 2
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def save_panel(original, heatmap, overlay, save_path, pred_name, pred_score):
    original = put_title(original, "Original")
    heatmap  = put_title(heatmap, "Heatmap")
    overlay  = put_title(overlay, f"Overlay | Pred: {pred_name} ({pred_score:.3f})")

    panel = np.concatenate([original, heatmap, overlay], axis=1)
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
# 9. 单张处理
# =========================
def process_one_image(model, cam_engine, image_path, transform):
    pil_img = Image.open(image_path).convert("RGB")
    x = transform(pil_img).unsqueeze(0).to(DEVICE)

    cam, pred_idx, pred_score = cam_engine.generate(x)

    rgb_resized = denormalize_image(x[0])
    heatmap, overlay = apply_heatmap_on_image(rgb_resized, cam)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    pred_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

    save_dir = OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    cv2.imwrite(
        os.path.join(save_dir, f"{base_name}_original.png"),
        cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        os.path.join(save_dir, f"{base_name}_heatmap.png"),
        cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        os.path.join(save_dir, f"{base_name}_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    save_panel(
        rgb_resized,
        heatmap,
        overlay,
        os.path.join(save_dir, f"{base_name}_panel.png"),
        pred_name,
        pred_score
    )

    print(f"[OK] {base_name} -> {pred_name} ({pred_score:.4f})")


# =========================
# 10. 主程序
# =========================
def main():
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