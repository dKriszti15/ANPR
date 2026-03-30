import cv2
import yaml
import os
import random
import numpy as np
from ultralytics import YOLO

MODEL_NAME = './best_obb.pt'
DATASET_IMAGES_DIR = './dataset/train/images'
OUTPUT_DIR = 'cropped_boxes'
NUM_IMAGES = 50
PADDING = 2
CONFIDENCE_THRESHOLD = 0.9

with open('./dataset/data.yaml', "r") as f:
    data = yaml.safe_load(f)
    class_names = data["names"]

model = YOLO(MODEL_NAME)
model.export(format="ncnn")
ncnn_model = YOLO("best_obb_ncnn_model", task='obb')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pick random images
all_images = [f for f in os.listdir(DATASET_IMAGES_DIR) if f.lower().endswith('.jpg')]
selected_images = random.sample(all_images, min(NUM_IMAGES, len(all_images)))

print(f"Running inference on {len(selected_images)} images...")

def crop_rotated(frame, points, padding=0):
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
    cropped = cv2.warpPerspective(frame, M, (w + padding * 2, h + padding * 2))
    return cropped

total_detections = 0
counter = 0
for img_filename in selected_images:
    img_path = os.path.join(DATASET_IMAGES_DIR, img_filename)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Skipping {img_filename} — could not read")
        continue

    results = ncnn_model(frame, verbose=False)

    for i, result in enumerate(results):
        if result.obb is None:
            continue

        for j, obb in enumerate(result.obb):
            confidence = obb.conf[0].item()
            class_id = int(obb.cls[0])
            label_name = class_names[class_id]

            points = obb.xyxyxyxy[0].cpu().numpy().astype(np.int32)

            if confidence > CONFIDENCE_THRESHOLD:
                cropped_img = crop_rotated(frame, points.astype(np.float32), padding=PADDING)
                if cropped_img.size > 0:
                    # source image name in filename
                    base_name = os.path.splitext(img_filename)[0]
                    filename = f"{OUTPUT_DIR}/{counter:04d}.jpg"
                    counter += 1
                    cv2.imwrite(filename, cropped_img)
                    total_detections += 1
                    print(f"Saved: {filename}")

print(f"\nDone. Total crops saved: {total_detections}")