import json
from pathlib import Path

import cv2


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def contour_features(cnt):
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / float(h) if h else 0.0

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    convexity = float(area / hull_area) if hull_area else 0.0

    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), angle = rect
    min_area_aspect = float(max(rw, rh) / min(rw, rh)) if min(rw, rh) else 0.0

    return {
        "area_px": area,
        "perimeter_px": perimeter,
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(w),
        "bbox_h": int(h),
        "aspect_ratio": aspect_ratio,
        "convexity": convexity,
        "min_area_rect_center_x": float(cx),
        "min_area_rect_center_y": float(cy),
        "min_area_rect_angle": float(angle),
        "min_area_rect_aspect": min_area_aspect,
    }


def main():
    mask_dir = Path("output_masks")
    out_dir = Path("output_features")
    ensure_dir(out_dir)

    exts = {".png", ".jpg", ".jpeg"}
    masks = [p for p in sorted(mask_dir.iterdir()) if p.suffix.lower() in exts]
    if not masks:
        print("No masks found in output_masks. Run inference first.")
        return

    for mask_file in masks:
        m = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue

        _, bin_m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contour:", mask_file.name)
            continue

        cnt = max(contours, key=cv2.contourArea)
        feats = contour_features(cnt)

        payload = {"mask_file": mask_file.name, "features": feats}
        out_path = out_dir / f"{mask_file.stem}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf8")

        print("Saved features:", out_path.name)


if __name__ == "__main__":
    main()