import random
import cv2
import os
import numpy as np
from collections import defaultdict

IMAGES_DIR  = "./dataset/train/images"
LABELS_DIR  = "./dataset/train/labels"
OUTPUT_DIR  = "cropped_gt_obb"
NUM_IMAGES  = 100
PADDING     = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

def crop_rotated(frame, points, padding=0):
    pts = points.astype(np.float32)

    center = pts.mean(axis=0)

    def angle_from_center(pt):
        return np.arctan2(pt[1] - center[1], pt[0] - center[0])

    sorted_pts = sorted(pts, key=angle_from_center)
    pts = np.array(sorted_pts, dtype=np.float32)

    corner_sums = pts.sum(axis=1)
    tl_idx = np.argmin(corner_sums)
    pts = np.roll(pts, -tl_idx, axis=0)

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

# gruop by prefix
grouped = defaultdict(list)

for label_filename in os.listdir(LABELS_DIR):
    if not label_filename.endswith(".txt"):
        continue

    base_name = os.path.splitext(label_filename)[0]


    prefix = base_name.split("_")[0]

    for ext in ['.jpg']:
        candidate = os.path.join(IMAGES_DIR, base_name + ext)
        if os.path.exists(candidate):
            grouped[prefix].append((label_filename, candidate))
            break

# Pick 1
valid_pairs = [random.choice(v) for v in grouped.values()]

selected = random.sample(valid_pairs, min(NUM_IMAGES, len(valid_pairs)))

with open("images_gt_crop.txt", "w") as f:
    for label_filename, image_path in selected:
        f.write(os.path.basename(image_path) + "\n")

print(f"Processing {len(selected)} unique images...")


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
        if len(parts) != 9:
            continue

        class_id = int(parts[0])
        coords = [float(p) for p in parts[1:]]

        points = np.array([
            [coords[0] * w_img, coords[1] * h_img],
            [coords[2] * w_img, coords[3] * h_img],
            [coords[4] * w_img, coords[5] * h_img],
            [coords[6] * w_img, coords[7] * h_img],
        ], dtype=np.float32)

        cropped = crop_rotated(frame, points, padding=PADDING)

        if cropped.size > 0:
            filename = f"{OUTPUT_DIR}/{counter:04d}.jpg"
            cv2.imwrite(filename, cropped)
            print(f"Saved: {filename} (class {class_id})")
            counter += 1

print(f"\nDone. Total crops saved: {counter}")