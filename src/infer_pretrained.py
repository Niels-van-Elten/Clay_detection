from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pick_best_mask(result):
    """
    Heuristic to pick one mask from the model output.
    We pick the detection with the highest confidence.
    """
    if result.masks is None or result.masks.data is None:
        return None

    masks = result.masks.data.cpu().numpy()  # shape: [N, H, W]

    best_idx = 0
    if result.boxes is not None and result.boxes.conf is not None:
        confs = result.boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))

    return masks[best_idx]


def main():
    model = YOLO("yolov8n-seg.pt")

    input_dir = Path("input_images")
    overlay_dir = Path("output_images")
    mask_dir = Path("output_masks")

    ensure_dir(input_dir)
    ensure_dir(overlay_dir)
    ensure_dir(mask_dir)

    exts = {".jpg", ".jpeg", ".png", ".webp"}

    files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in exts]
    if not files:
        print("No images found in input_images. Add images then run again.")
        return

    for img_file in files:
        results = model.predict(source=str(img_file), imgsz=640, conf=0.25)
        r = results[0]

        overlay_bgr = r.plot()
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(overlay_rgb).save(overlay_dir / f"{img_file.stem}_overlay.png")

        best_mask = pick_best_mask(r)
        if best_mask is None:
            print("No mask found:", img_file.name)
            continue

        mask_u8 = (best_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(mask_dir / f"{img_file.stem}_mask.png"), mask_u8)

        print("Saved overlay and mask for:", img_file.name)


if __name__ == "__main__":
    main()