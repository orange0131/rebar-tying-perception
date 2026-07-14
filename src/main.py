import os
import cv2
import glob
import argparse

IMG_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]


class RebarCropTool:
    def __init__(self, input_dir, output_dir, win_name="Rebar Crop Tool", crop_size=None, padding=10):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.win_name = win_name
        self.crop_size = crop_size
        self.padding = padding

        self.image_paths = self._load_images()
        if len(self.image_paths) == 0:
            raise ValueError(f"输入文件夹中没有找到图片: {input_dir}")

        self.index = 0
        self.image = None
        self.image_show = None
        self.clone = None

        self.drawing = False
        self.ix, self.iy = -1, -1
        self.fx, self.fy = -1, -1

        self.current_rect = None
        self.saved_rects = []

        self.class_dirs = {
            "untied": os.path.join(output_dir, "untied"),
            "tied": os.path.join(output_dir, "tied")
        }
        os.makedirs(self.class_dirs["untied"], exist_ok=True)
        os.makedirs(self.class_dirs["tied"], exist_ok=True)

    def _load_images(self):
        paths = []
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(self.input_dir, ext)))
        paths.sort()
        return paths

    def _read_image(self):
        path = self.image_paths[self.index]
        self.image = cv2.imread(path)
        if self.image is None:
            raise ValueError(f"读取失败: {path}")
        self.clone = self.image.copy()
        self.image_show = self.image.copy()
        self.current_rect = None
        self.saved_rects = []

    def _draw_ui(self):
        self.image_show = self.clone.copy()

        for rect in self.saved_rects:
            x1, y1, x2, y2, label = rect
            color = (0, 0, 255) if label == "untied" else (0, 255, 0)
            text = "untied" if label == "untied" else "tied"
            cv2.rectangle(self.image_show, (x1, y1), (x2, y2), color, 2)
            cv2.putText(self.image_show, text, (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if self.current_rect is not None:
            x1, y1, x2, y2 = self.current_rect
            cv2.rectangle(self.image_show, (x1, y1), (x2, y2), (255, 0, 0), 2)

        info1 = f"[{self.index + 1}/{len(self.image_paths)}] {os.path.basename(self.image_paths[self.index])}"
        info2 = "Drag box | u: untied | t: tied | z: undo | n: next | p: prev | r: reset | q: quit"
        cv2.putText(self.image_show, info1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(self.image_show, info2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    def _clamp_rect(self, x1, y1, x2, y2, w, h):
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        return x1, y1, x2, y2

    def _get_crop_from_rect(self, rect):
        x1, y1, x2, y2 = rect
        h, w = self.image.shape[:2]

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        x1 -= self.padding
        y1 -= self.padding
        x2 += self.padding
        y2 += self.padding

        x1, y1, x2, y2 = self._clamp_rect(x1, y1, x2, y2, w, h)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = self.image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        if self.crop_size is not None:
            crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)

        return crop, (x1, y1, x2, y2)

    def _save_crop(self, label):
        if self.current_rect is None:
            print("请先框选一个区域，再按 u 或 t 保存。")
            return

        result = self._get_crop_from_rect(self.current_rect)
        if result is None:
            print("当前框无效，无法保存。")
            return

        crop, rect = result
        x1, y1, x2, y2 = rect

        img_name = os.path.splitext(os.path.basename(self.image_paths[self.index]))[0]
        save_dir = self.class_dirs[label]

        existing = glob.glob(os.path.join(save_dir, f"{img_name}_{label}_*.jpg"))
        save_id = len(existing) + 1
        save_name = f"{img_name}_{label}_{save_id:03d}.jpg"
        save_path = os.path.join(save_dir, save_name)

        cv2.imwrite(save_path, crop)
        self.saved_rects.append((x1, y1, x2, y2, label))
        print(f"已保存: {save_path}")

        self.current_rect = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.fx, self.fy = x, y
            self.current_rect = (self.ix, self.iy, self.fx, self.fy)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.fx, self.fy = x, y
                self.current_rect = (self.ix, self.iy, self.fx, self.fy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.fx, self.fy = x, y
            self.current_rect = (self.ix, self.iy, self.fx, self.fy)

    def run(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_name, self.mouse_callback)

        self._read_image()

        while True:
            self._draw_ui()
            cv2.imshow(self.win_name, self.image_show)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('u'):
                self._save_crop("untied")
            elif key == ord('t'):
                self._save_crop("tied")
            elif key == ord('z'):
                if len(self.saved_rects) > 0:
                    removed = self.saved_rects.pop()
                    print(f"撤销标注: {removed}")
                else:
                    print("没有可撤销的标注。")
            elif key == ord('r'):
                self.current_rect = None
                self.saved_rects = []
                print("已重置当前图片的显示框记录（不会删除已保存小图）。")
            elif key == ord('n'):
                if self.index < len(self.image_paths) - 1:
                    self.index += 1
                    self._read_image()
                else:
                    print("已经是最后一张图片。")
            elif key == ord('p'):
                if self.index > 0:
                    self.index -= 1
                    self._read_image()
                else:
                    print("已经是第一张图片。")
            elif key == ord('q') or key == 27:
                break

        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively crop tied and untied rebar intersections."
    )
    parser.add_argument("--input-dir", default="data/raw_images")
    parser.add_argument("--output-dir", default="data/rebar_crops")
    parser.add_argument("--crop-size", type=int, default=224, help="Use 0 to keep original crop size.")
    parser.add_argument("--padding", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    crop_size = args.crop_size
    padding = args.padding

    crop_size = None if crop_size == 0 else crop_size

    tool = RebarCropTool(
        input_dir=input_dir,
        output_dir=output_dir,
        crop_size=crop_size,
        padding=padding
    )
    tool.run()


if __name__ == "__main__":
    main()
