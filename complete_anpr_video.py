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
VIDEO_PATH = "./IMG_6866.mov"

LOG_FILE = "plate_reads.txt"

DETECTION_CONFIDENCE_THRESHOLD = 0.80
INFERENCE_SCALE = 0.8
PADDING = 2

TRACK_EXPIRE_SECONDS = 3
LOG_COOLDOWN_SECONDS = 25
CONFIRMATION_COUNT = 2
FRAME_SKIP = 1

TROCR_MODEL_NAME = "microsoft/trocr-small-printed"

COUNTY_CODES = sorted(
    [
        "AB", "AR", "AG", "BC", "BH", "BN", "BT", "BV",
        "BR", "B", "CL", "CS", "CJ", "CT", "CV", "DB",
        "DJ", "GL", "GR", "GJ", "HR", "HD", "IL", "IS",
        "IF", "MM", "MH", "MS", "NT", "OT", "PH", "SJ",
        "SM", "SB", "SV", "TR", "TM", "TL", "VS", "VL", "VN",
    ],
    key=len,
    reverse=True,
)

tracked = {}


def apply_length_filter(text, is_red=False):
    return text[:8] if is_red else text[:7]


def apply_county_filter(text):
    for i in range(len(text)):
        candidate = text[i:]
        for county in COUNTY_CODES:
            if candidate.startswith(county):
                rem = candidate[len(county):]
                if rem and rem[0].isdigit():
                    return candidate
    return text


def apply_plate_structure(text):
    if len(text) < 4:
        return text

    text = text.upper()

    tail = text[-3:]

    # Bucuresti
    if text.startswith("B"):

        county = "B"
        body = text[1:-3]

    else:
        county = text[:2]
        body = text[2:-3]

        county = (
            county
            .replace("8", "B")
            .replace("0", "O")
            .replace("1", "I")
        )

    body = (
        body
        .replace("O", "0")
        .replace("I", "1")
        .replace("Z", "2")
        .replace("S", "5")
        .replace("B", "8")
    )

    tail = (
        tail
        .replace("0", "O")
        .replace("1", "I")
        .replace("2", "Z")
        .replace("5", "S")
        .replace("8", "B")
    )

    fixed = county + body + tail

    print(
        f"county={county} | body={body} | tail={tail} | final={fixed}"
    )

    return fixed


def clean_pred(text, is_red=False):
    raw = re.sub(r"[^A-Z0-9]", "", text.upper().strip())
    raw = apply_county_filter(raw)
    raw = apply_plate_structure(raw)
    return apply_length_filter(raw, is_red=is_red)


def is_valid_plate(text):

    if not text:
        return False

    text = text.upper()

    # Bucuresti
    if text.startswith("B"):

        rem = text[1:]

        # must end with 3 letters
        if len(rem) < 5:
            return False

        digits = rem[:-3]
        letters = rem[-3:]

        return (
            digits.isdigit()
            and 2 <= len(digits) <= 3
            and letters.isalpha()
        )

    # Other counties
    for county in COUNTY_CODES:

        if county == "B":
            continue

        if text.startswith(county):

            rem = text[len(county):]

            if len(rem) < 5:
                return False

            digits = rem[:-3]
            letters = rem[-3:]

            return (
                digits.isdigit()
                and 2 <= len(digits) <= 3
                and letters.isalpha()
            )

    return False


def load_class_names():
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["names"]


def load_model():
    if not os.path.isdir(NCNN_MODEL_DIR):
        print("Exporting NCNN...")
        YOLO(MODEL_NAME).export(format="ncnn")
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

def crop_simple(frame, points, padding=0):
    x1 = max(0, int(points[:, 0].min()) - padding)
    y1 = max(0, int(points[:, 1].min()) - padding)
    x2 = min(frame.shape[1], int(points[:, 0].max()) + padding)
    y2 = min(frame.shape[0], int(points[:, 1].max()) + padding)
    return frame[y1:y2, x1:x2]

def crop_rotated(frame, points, padding=0):
    pts = points.astype(np.float32)

    pts = pts[[3,2,1,0]]

    w = int(np.linalg.norm(pts[1] - pts[0]) + padding * 2)
    h = int(np.linalg.norm(pts[3] - pts[0]) + padding * 2)

    if h > w:
        w, h = h, w
        dst_pts = np.array([
            [padding, padding + h],
            [padding, padding],
            [padding + w, padding],
            [padding + w, padding + h],
        ], dtype=np.float32)
    else:
        dst_pts = np.array([
            [padding, padding],
            [padding + w, padding],
            [padding + w, padding + h],
            [padding, padding + h],
        ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst_pts)
    cropped = cv2.warpPerspective(frame, M, (w + padding * 2, h + padding * 2))
    return cropped


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


device = "cuda" if torch.cuda.is_available() else "cpu"

def run_trocr(processor, model, img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel = processor(Image.fromarray(rgb), return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(pixel)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def boxes_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (a[2] - a[0]) * (a[3] - a[1])
    area2 = (b[2] - b[0]) * (b[3] - b[1])
    union = area1 + area2 - inter
    return inter / union if union else 0

def find_track(bbox):
    best = None
    score = 0
    for tid, t in tracked.items():
        iou = boxes_iou(bbox, t["bbox"])
        if iou > score:
            score = iou
            best = tid
    return best if score > 0.5 else None


def append_log(plate, raw, conf):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{ts} | raw={raw} -> {plate} | conf={conf:.2f}\n")
    print("[LOGGED]", plate)


def main():
    classes = load_class_names()
    model = load_model()

    print("Loading TrOCR...")
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
    trocr = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
    trocr.to(device)
    trocr.eval()

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0
    start = time.time()
    plate_count = {}
    last_logged = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        frame_count += 1

        small = cv2.resize(frame, None, fx=INFERENCE_SCALE, fy=INFERENCE_SCALE)
        results = model(small, verbose=False)

        for result in results:
            if result.obb is None:
                continue

            for obb in result.obb:
                conf = obb.conf[0].item()
                if conf < DETECTION_CONFIDENCE_THRESHOLD:
                    continue

                cls = int(obb.cls[0])
                is_red = classes[cls] == "red_carplate"

                pts = obb.xyxyxyxy[0].cpu().numpy() / INFERENCE_SCALE

                bbox = (
                    int(pts[:, 0].min()),
                    int(pts[:, 1].min()),
                    int(pts[:, 0].max()),
                    int(pts[:, 1].max()),
                )

                cv2.polylines(frame, [pts.astype(np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 1)

                tid = find_track(bbox)
                if tid is None:
                    tid = str(time.time())
                    tracked[tid] = {
                        "bbox": bbox,
                        "plate": None,
                        "raw": "",
                        "last": now,
                    }
                tr = tracked[tid]
                tr["bbox"] = bbox
                tr["last"] = now

                # skip OCR if we already have a result for this plate
                if tr["plate"] is not None:
                    plate = tr["plate"]
                    raw = tr["raw"]

                else:
                    # only run OCR every FRAME_SKIP frames to avoid blocking
                    if frame_count % FRAME_SKIP != 0:
                        continue

                    crop = crop_rotated(frame, pts, padding=PADDING)
                    crop = normalize_plate_orientation(crop)

                    cv2.imshow("cropped", crop)

                    if crop is None or crop.size == 0:
                        continue

                    raw = run_trocr(processor, trocr, crop)
                    plate = clean_pred(raw, is_red)
                    print(f"[OCR] {raw} -> {plate} | valid={is_valid_plate(plate)}")

                    tr["plate"] = plate
                    tr["raw"] = raw

                    if crop is not None and crop.size > 0:
                        os.makedirs("debug_crops", exist_ok=True)
                        debug_path = f"debug_crops/{frame_count:06d}.jpg"
                        cv2.imwrite(debug_path, crop)

                        debug_crop = cv2.resize(crop, (300, 100), interpolation=cv2.INTER_NEAREST)
                        cv2.putText(debug_crop, f"{plate}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.imshow("OCR Input", debug_crop)

                if not plate:
                    continue

                cv2.putText(frame, plate, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if not is_valid_plate(plate):
                    continue

                plate_count[plate] = plate_count.get(plate, 0) + 1
                cooldown = now - last_logged.get(plate, 0)

                if plate_count[plate] >= CONFIRMATION_COUNT and cooldown >= LOG_COOLDOWN_SECONDS:
                    append_log(plate, raw, conf)
                    last_logged[plate] = now

        for k in list(tracked.keys()):
            if now - tracked[k]["last"] > TRACK_EXPIRE_SECONDS:
                del tracked[k]

        fps = frame_count / (time.time() - start)
        cv2.putText(frame, f"FPS:{fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Original", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()