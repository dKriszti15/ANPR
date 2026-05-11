import os
import re
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO


MODEL_NAME = "./best_obb.pt"
NCNN_MODEL_DIR = "best_obb_ncnn_model"
DATA_YAML = "./dataset/data.yaml"
LOG_FILE = "plate_reads.txt"
CAMERA_INDEX = 0
DETECTION_CONFIDENCE_THRESHOLD = 0.80
PADDING = 2
OCR_LOG_COOLDOWN_SECONDS = 2.0
TROCR_MODEL_NAME = "microsoft/trocr-small-printed"

COUNTY_CODES = sorted(
    [
        "AB", "AR", "AG", "BC", "BH", "BN", "BT", "BV", "BR", "B",
        "CL", "CS", "CJ", "CT", "CV", "DB", "DJ", "GL", "GR", "GJ",
        "HR", "HD", "IL", "IS", "IF", "MM", "MH", "MS", "NT", "OT",
        "PH", "SJ", "SM", "SB", "SV", "TR", "TM", "TL", "VS", "VL", "VN",
    ],
    key=len,
    reverse=True,
)


def apply_length_filter(text, is_red=False):
    return text[:8] if is_red else text[:7]


def is_valid_plate(text):
    """Check if a plate is valid for logging."""
    if not text or len(text) < 5:
        return False
    
    # length chek
    if len(text) not in [7, 8]:
        return False
    
    # starts with county code
    for code in COUNTY_CODES:
        if text.startswith(code):
            remainder = text[len(code):]
            if remainder and any(c.isdigit() for c in remainder):
                return True
    
    return False


def apply_plate_structure(text):
    if len(text) < 3:
        return text
    body = text[:-3]
    tail = text[-3:]
    tail = tail.replace("0", "O")
    return body + tail


def apply_county_filter(text):
    for i in range(len(text)):
        chunk = text[i:]
        for code in COUNTY_CODES:
            if chunk.startswith(code):
                remainder = chunk[len(code):]
                if remainder and remainder[0].isdigit():
                    return chunk
    return text


def clean_pred(text, is_red=False):
    raw = re.sub(r"[^A-Z0-9]", "", text.strip().upper())
    filtered = apply_county_filter(raw)
    fixed = apply_plate_structure(filtered)
    trimmed = apply_length_filter(fixed, is_red=is_red)
    if raw != trimmed:
        print(f"    [FILTER] {raw} -> {trimmed}")
    return trimmed


def load_class_names():
    with open(DATA_YAML, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data["names"]


def load_obb_model():
    if not os.path.isdir(NCNN_MODEL_DIR):
        model = YOLO(MODEL_NAME)
        model.export(format="ncnn")
    return YOLO(NCNN_MODEL_DIR, task="obb")


def order_points(points):
    pts = points.astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def crop_rotated(frame, points, padding=0):
    ordered = order_points(points)

    width_a = np.linalg.norm(ordered[2] - ordered[3])
    width_b = np.linalg.norm(ordered[1] - ordered[0])
    height_a = np.linalg.norm(ordered[1] - ordered[2])
    height_b = np.linalg.norm(ordered[0] - ordered[3])

    width = max(1, int(round(max(width_a, width_b))))
    height = max(1, int(round(max(height_a, height_b))))

    dst_points = np.array(
        [
            [padding, padding],
            [padding + width, padding],
            [padding + width, padding + height],
            [padding, padding + height],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(ordered, dst_points)
    return cv2.warpPerspective(frame, transform, (width + padding * 2, height + padding * 2))


def normalize_plate_orientation(cropped):
    if cropped is None or cropped.size == 0:
        return cropped

    if cropped.shape[0] > cropped.shape[1]:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    h, w = cropped.shape[:2]
    strip_w = max(2, int(w * 0.18))
    left = cropped[:, :strip_w]
    right = cropped[:, w - strip_w:]

    left_blue = np.mean(left[:, :, 0].astype(np.float32) - np.maximum(left[:, :, 1], left[:, :, 2]).astype(np.float32))
    right_blue = np.mean(right[:, :, 0].astype(np.float32) - np.maximum(right[:, :, 1], right[:, :, 2]).astype(np.float32))

    if right_blue > left_blue + 5.0:
        cropped = cv2.rotate(cropped, cv2.ROTATE_180)

    return cropped


def run_trocr(processor, model, image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = processor(Image.fromarray(image_rgb), return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def append_plate_log(log_path, plate_text, raw_text, det_conf):
    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} | plate={plate_text} | raw={raw_text} | det_conf={det_conf:.2f}\n")


def main():
    class_names = load_class_names()

    print("Loading NCNN OBB model...")
    ncnn_model = load_obb_model()

    print("Loading TrOCR model...")
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
    trocr_model.eval()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    print("Camera opened")

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_count = 0
    start_time = time.time()
    fpm = 0
    last_logged_at = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame; exiting.")
            break

        frame_count += 1
        results = ncnn_model(frame, verbose=False)

        for result in results:
            if result.obb is None:
                continue

            for obb in result.obb:
                det_conf = obb.conf[0].item()
                class_id = int(obb.cls[0])
                label_name = class_names[class_id]

                points = obb.xyxyxyxy[0].cpu().numpy().astype(np.float32)
                cv2.polylines(frame, [points.astype(np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 1)

                if det_conf < DETECTION_CONFIDENCE_THRESHOLD:
                    continue

                cropped = crop_rotated(frame, points, padding=PADDING)
                cropped = normalize_plate_orientation(cropped)

                if cropped is None or cropped.size == 0:
                    continue

                raw_text = run_trocr(processor, trocr_model, cropped)
                plate_text = clean_pred(raw_text)

                if not plate_text:
                    continue

                label = f"{label_name}: {det_conf:.2f} | {plate_text}"
                (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                top_center_x = int(points[:, 0].mean()) - text_w // 2
                top_y = int(points[:, 1].min()) - 8
                cv2.putText(frame, label, (top_center_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # log valid plates only
                if is_valid_plate(plate_text):
                    now = time.time()
                    last_logged = last_logged_at.get(plate_text, 0.0)
                    if now - last_logged >= OCR_LOG_COOLDOWN_SECONDS:
                        append_plate_log(LOG_FILE, plate_text, raw_text, det_conf)
                        last_logged_at[plate_text] = now
                        print(f"[READ] {plate_text} | raw={raw_text} | conf={det_conf:.2f}")

        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            fpm = fps * 60
            cv2.putText(frame, f"FPM: {fpm:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Original", frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or elapsed_time >= 30:
            print("Model:", MODEL_NAME, "| Elapsed time:", elapsed_time, "| FPM:", fpm)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()