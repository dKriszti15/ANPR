import random
import cv2
import os
import numpy as np
from collections import defaultdict

IMAGES_DIR = "./tilted_50_obb/train/images"
LABELS_DIR = "./tilted_50_obb/train/labels"
OUTPUT_DIR = "cropped_gt_TILTED_obb"
NUM_IMAGES = 100
PADDING = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)


def crop_rotated(frame, points, padding=0):
    pts = points.astype(np.float32)

    # Deterministic corner order: top-left, top-right, bottom-right, bottom-left.
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]

    width_a = np.linalg.norm(ordered[2] - ordered[3])
    width_b = np.linalg.norm(ordered[1] - ordered[0])
    height_a = np.linalg.norm(ordered[1] - ordered[2])
    height_b = np.linalg.norm(ordered[0] - ordered[3])

    w = max(1, int(round(max(width_a, width_b))))
    h = max(1, int(round(max(height_a, height_b))))

    dst_pts = np.array([
        [padding, padding],
        [padding + w, padding],
        [padding + w, padding + h],
        [padding, padding + h],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst_pts)
    cropped = cv2.warpPerspective(frame, M, (w + padding * 2, h + padding * 2))

    return cropped


def normalize_plate_orientation(cropped):
    if cropped is None or cropped.size == 0:
        return cropped

    # Keep output horizontal.
    if cropped.shape[0] > cropped.shape[1]:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    # EU-style Romanian plates should have the blue strip on the left.
    h, w = cropped.shape[:2]
    strip_w = max(2, int(w * 0.18))
    left = cropped[:, :strip_w]
    right = cropped[:, w - strip_w:]

    left_blue = np.mean(left[:, :, 0].astype(np.float32) - np.maximum(left[:, :, 1], left[:, :, 2]).astype(np.float32))
    right_blue = np.mean(right[:, :, 0].astype(np.float32) - np.maximum(right[:, :, 1], right[:, :, 2]).astype(np.float32))

    if right_blue > left_blue + 5.0:
        cropped = cv2.rotate(cropped, cv2.ROTATE_180)

    return cropped

valid_pairs = []

for label_file in os.listdir(LABELS_DIR):
    if not label_file.endswith(".txt"):
        continue

    image_file = label_file.replace(".txt", ".jpg")
    image_path = os.path.join(IMAGES_DIR, image_file)

    if os.path.exists(image_path):
        valid_pairs.append((label_file, image_path))

if not valid_pairs:
    raise RuntimeError("No valid image-label pairs found.")


grouped = defaultdict(list)

for lf, ip in valid_pairs:
    prefix = os.path.basename(ip).split("_")[0]
    grouped[prefix].append((lf, ip))

selected_prefixes = list(grouped.keys())

valid_pairs = [
    random.choice(v)
    for k, v in grouped.items()
    if k in selected_prefixes and len(v) > 0
]

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

    label_path = os.path.join(LABELS_DIR, label_filename)
    if not os.path.exists(label_path):
        print(f"Missing label: {label_path}")
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 9:
            continue

        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        points = np.array([
            [coords[0] * w_img, coords[1] * h_img],
            [coords[2] * w_img, coords[3] * h_img],
            [coords[4] * w_img, coords[5] * h_img],
            [coords[6] * w_img, coords[7] * h_img],
        ], dtype=np.float32)

        cropped = crop_rotated(frame, points, padding=PADDING)

        if cropped is None or cropped.size == 0:
            continue

        cropped = normalize_plate_orientation(cropped)

        filename = os.path.join(OUTPUT_DIR, f"{counter:04d}.jpg")
        cv2.imwrite(filename, cropped)

        print(f"Saved: {filename} (class {class_id})")
        counter += 1

print(f"\nDone. Total crops saved: {counter}")