import cv2
import yaml
import os
import numpy as np
from ultralytics import YOLO

MODEL_REGULAR = './best.pt'
MODEL_OBB     = './best_obb.pt'
DATASET_IMAGES_DIR = './dataset_v13_relabeled_obb/train/images'
OUTPUT_DIR_REG = 'cropped_boxes_nonobb'
OUTPUT_DIR_OBB = 'cropped_boxes_obb'
PADDING = 2
CONFIDENCE_THRESHOLD = 0.84

with open('./dataset_v13_relabeled_obb/data.yaml', "r") as f:
    class_names = yaml.safe_load(f)["names"]

regular_model = YOLO(MODEL_REGULAR)
regular_model.export(format="ncnn")
ncnn_regular = YOLO("best_ncnn_model")

obb_model = YOLO(MODEL_OBB)
obb_model.export(format="ncnn")
ncnn_obb = YOLO("best_obb_ncnn_model", task='obb')

os.makedirs(OUTPUT_DIR_REG, exist_ok=True)
os.makedirs(OUTPUT_DIR_OBB, exist_ok=True)

with open('images.txt', 'r') as f:
    selected_images = [line.strip() for line in f.readlines()]

print(f"Running inference on {len(selected_images)} images...")

def crop_axis_aligned(frame, box, padding=2):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    x1 = max(0, x1 + padding)
    y1 = max(0, y1 + padding)
    x2 = min(frame.shape[1], x2 - padding)
    y2 = min(frame.shape[0], y2 - padding)
    return frame[y1:y2, x1:x2]

def crop_rotated(frame, points, padding=2):
    pts = points.astype(np.float32)
    pts = pts[[3, 2, 1, 0]]
    w = int(np.linalg.norm(pts[1] - pts[0]) + padding * 2)
    h = int(np.linalg.norm(pts[3] - pts[0]) + padding * 2)
    if h > w:
        w, h = h, w
        dst_pts = np.array([
            [padding,     padding + h],
            [padding,     padding    ],
            [padding + w, padding    ],
            [padding + w, padding + h],
        ], dtype=np.float32)
    else:
        dst_pts = np.array([
            [padding,     padding    ],
            [padding + w, padding    ],
            [padding + w, padding + h],
            [padding,     padding + h],
        ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    return cv2.warpPerspective(frame, M, (w + padding * 2, h + padding * 2))

counter = 0
saved   = 0
skipped_no_reg = 0
skipped_no_obb = 0

for img_filename in selected_images:
    frame = cv2.imread(os.path.join(DATASET_IMAGES_DIR, img_filename))
    if frame is None:
        print(f"Skipping {img_filename} — could not read")
        continue

    reg_crop = None
    reg_results = ncnn_regular(frame, verbose=False)
    for result in reg_results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            if box.conf[0].item() < CONFIDENCE_THRESHOLD:
                continue
            crop = crop_axis_aligned(frame, box, PADDING)
            if crop.size > 0:
                reg_crop = crop
                break  # take highest confidence only
        if reg_crop is not None:
            break

    obb_crop = None
    obb_results = ncnn_obb(frame, verbose=False)
    for result in obb_results:
        if result.obb is None:
            continue
        for obb in result.obb:
            if obb.conf[0].item() < CONFIDENCE_THRESHOLD:
                continue
            points = obb.xyxyxyxy[0].cpu().numpy().astype(np.int32)
            crop = crop_rotated(frame, points.astype(np.float32), PADDING)
            if crop.size > 0:
                obb_crop = crop
                break
        if obb_crop is not None:
            break

    # save IF both models detected a plate
    if reg_crop is None:
        print(f"[SKIP] {img_filename} — regular model missed it")
        skipped_no_reg += 1
        continue
    if obb_crop is None:
        print(f"[SKIP] {img_filename} — obb model missed it")
        skipped_no_obb += 1
        continue

    fname = f"{counter:04d}.jpg"
    cv2.imwrite(f"{OUTPUT_DIR_REG}/{fname}", reg_crop)
    cv2.imwrite(f"{OUTPUT_DIR_OBB}/{fname}", obb_crop)
    print(f"[{counter:04d}] Saved: {img_filename}")
    counter += 1
    saved += 1

print(f"\nDone.")
print(f"  Paired crops saved : {saved}")
print(f"  Skipped (reg miss) : {skipped_no_reg}")
print(f"  Skipped (obb miss) : {skipped_no_obb}")