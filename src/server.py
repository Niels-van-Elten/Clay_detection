import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO


app = FastAPI()
model = YOLO("yolov8n-seg.pt")


def features_from_mask(mask_u8):
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / float(h) if h else 0.0

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    convexity = float(area / hull_area) if hull_area else 0.0

    return {
        "area_px": area,
        "perimeter_px": perimeter,
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(w),
        "bbox_h": int(h),
        "aspect_ratio": aspect_ratio,
        "convexity": convexity,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    img_np = np.array(img)

    results = model.predict(source=img_np, imgsz=640, conf=0.25)
    r = results[0]

    if r.masks is None or r.masks.data is None:
        return JSONResponse({"ok": True, "found": False})

    masks = r.masks.data.cpu().numpy()

    best_idx = 0
    if r.boxes is not None and r.boxes.conf is not None:
        confs = r.boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))

    m = masks[best_idx]
    mask_u8 = (m.astype(np.uint8) * 255)

    feats = features_from_mask(mask_u8)
    return JSONResponse({"ok": True, "found": feats is not None, "features": feats})