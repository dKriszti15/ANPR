import os
import cv2
import pandas as pd
from torchmetrics.text import CharErrorRate
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re

EVAL_CONFIGS = {
    "REGULAR_MODEL": {
        "csv_path": "nonobb_tilted.csv",
        "folder": "gt_crops_TILTED_nonobb",
    },
    "OBB_MODEL": {
        "csv_path": "obb_tilted.csv",
        "folder": "cropped_gt_TILTED_obb",
    },
}

MODEL_NAME = "microsoft/trocr-base-printed"

COUNTY_CODES = [
    "AB","AR","AG","BC","BH","BN","BT","BV","BR","B",
    "CL","CS","CJ","CT","CV","DB","DJ","GL","GR","GJ",
    "HR","HD","IL","IS","IF","MM","MH","MS","NT","OT",
    "PH","SJ","SM","SB","SV","TR","TM","TL","VS","VL","VN"
]

COUNTY_CODES = sorted(COUNTY_CODES, key=len, reverse=True)

def normalize(name):
    return name.split(".jpg")[0] + ".jpg"

def apply_length_filter(text, is_red=False):
    return text[:8] if is_red else text[:7]

def apply_plate_structure(text):
    if len(text) < 3:
        return text
    body = text[:-3]
    tail = text[-3:]
    tail = tail.replace('0', 'O')
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

print("Loading TrOCR model...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.eval()

def clean_pred(text, is_red=False):
    raw      = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
    filtered = apply_county_filter(raw)
    fixed    = apply_plate_structure(filtered)
    trimmed  = apply_length_filter(fixed, is_red=is_red)
    if raw != trimmed:
        print(f"    [FILTER] {raw} -> {trimmed}")
    return trimmed

def run_trocr(image):
    image_rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = processor(Image.fromarray(image_rgb), return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def evaluate_folder(name, folder, df, cer_metric):
    predictions   = []
    ground_truths = []
    total         = 0
    exact         = 0

    print(f"\n{'='*55}")
    print(f"  Evaluating: {name} ({folder})")
    print(f"{'='*55}")

    for _, row in df.iterrows():
        image_path = os.path.join(folder, row["image_filename"])
        image = cv2.imread(image_path)

        if image is None:
            print(f"  Skipping {row['image_filename']}")
            continue

        is_red = str(row.get("is_red", "False")).lower() == "true"

        pred = clean_pred(run_trocr(image), is_red=is_red)
        gt   = str(row["ground_truth"]).strip().upper().replace(" ", "")

        predictions.append(pred)
        ground_truths.append(gt)

        total += 1
        if pred == gt:
            exact += 1

        print(f"  [{total:03d}] GT: {gt:12} | Pred: {pred:12} | {'✅' if pred == gt else '❌'}")

    cer = cer_metric(predictions, ground_truths).item()
    rate = exact / total if total else 0

    correct_chars = 0
    total_chars   = 0
    for p, g in zip(predictions, ground_truths):
        total_chars   += len(g)
        correct_chars += sum(pc == gc for pc, gc in zip(p, g))

    recall = correct_chars / total_chars if total_chars else 0

    return {
        "total": total,
        "exact": exact,
        "rate": rate,
        "cer": cer,
        "recall": recall,
    }

def print_summary(results):
    print(f"\n{'='*55}")
    print(f"{'Metric':<25}", end="")
    for name in results:
        print(f"{name:>12}", end="")
    print()
    print(f"{'-'*55}")

    def best(metric, lower=False):
        vals = [results[k][metric] for k in results]
        return min(vals) if lower else max(vals)

    for label, key, lower in [
        ("Recognition Rate", "rate", False),
        ("Char Error Rate", "cer", True),
        ("Recall Ratio", "recall", False),
    ]:
        print(f"{label:<25}", end="")
        best_val = best(key, lower)
        for name in results:
            val = results[name][key]
            mark = " ✅" if val == best_val else ""
            fmt = f"{val:.4f}" if key == "cer" else f"{val:.2%}"
            print(f"{fmt + mark:>12}", end="")
        print()

    print(f"{'='*55}")

cer_metric = CharErrorRate()

results = {}

for name, config in EVAL_CONFIGS.items():
    csv_path = config["csv_path"]
    folder = config["folder"]
    df = pd.read_csv(csv_path)
    results[name] = evaluate_folder(name, folder, df, cer_metric)

print_summary(results)