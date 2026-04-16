import cv2
import os
import random
from collections import defaultdict

IMAGES_DIR  = "./dataset_v12_nonobb/train/images"
LABELS_DIR  = "./dataset_v12_nonobb/train/labels"
OUTPUT_DIR  = "gt_crops_nonobb"
PADDING     = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("images_gt_crop.txt", "r") as f:
    selected_prefixes = set(name.split("_")[0] for name in f.read().splitlines())

valid_pairs = []

for label_filename in os.listdir(LABELS_DIR):
    if not label_filename.endswith(".txt"):
        continue

    base_name = os.path.splitext(label_filename)[0]

    for ext in ['.jpg']:
        image_path = os.path.join(IMAGES_DIR, base_name + ext)
        if os.path.exists(image_path):
            valid_pairs.append((label_filename, image_path))
            break

grouped = defaultdict(list)

for lf, ip in valid_pairs:
    prefix = os.path.basename(ip).split("_")[0]
    if prefix in selected_prefixes:
        grouped[prefix].append((lf, ip))

selected = [random.choice(v) for v in grouped.values()]

print(f"Processing {len(selected)} images...")

counter = 0

for label_filename, image_path in selected:
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Skipping {image_path} — could not read")
        continue

    h_img, w_img = frame.shape[:2]

    with open(os.path.join(LABELS_DIR, label_filename), "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        cx, cy, bw, bh = [float(p) for p in parts[1:]]

        x1 = max(0, int((cx - bw / 2) * w_img) - PADDING)
        y1 = max(0, int((cy - bh / 2) * h_img) - PADDING)
        x2 = min(w_img, int((cx + bw / 2) * w_img) + PADDING)
        y2 = min(h_img, int((cy + bh / 2) * h_img) + PADDING)

        cropped = frame[y1:y2, x1:x2]

        if cropped.size > 0:
            filename = f"{OUTPUT_DIR}/{counter:04d}.jpg"
            cv2.imwrite(filename, cropped)
            print(f"Saved: {filename} (class {class_id})")
            counter += 1

print(f"\nDone. Total crops saved: {counter}")