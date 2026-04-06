import os
import cv2
import pandas as pd
from torchmetrics.text import CharErrorRate
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re

CSV_PATH       = "ocr_test_data_2.csv"
MODEL_NAME     = "microsoft/trocr-base-printed"
DIR_NONOBB     = "cropped_boxes_nonobb_2"
DIR_OBB        = "cropped_boxes_obb_2"
COUNTY_CODES = [
    "AB", "AR", "AG", "BC", "BH", "BN", "BT", "BV", "BR", "B",
    "CL", "CS", "CJ", "CT", "CV", "DB", "DJ", "GL", "GR", "GJ",
    "HR", "HD", "IL", "IS", "IF", "MM", "MH", "MS", "NT", "OT",
    "PH", "SJ", "SM", "SB", "SV", "TR", "TM", "TL", "VS", "VL", "VN"
]

COUNTY_CODES = sorted(COUNTY_CODES, key=len, reverse=True)

def apply_length_filter(text: str) -> str:
    return text[:7]

def apply_plate_structure(text: str) -> str:
    if len(text) < 3:
        return text

    body = text[:-3]
    tail = text[-3:]

    tail = tail.replace('0', 'O')

    return body + tail

def apply_county_filter(text: str) -> str:
    # cut first char until a valid county code is found, followed by a number
    for i in range(len(text)):
        chunk = text[i:]
        for code in COUNTY_CODES:
            if chunk.startswith(code):
                remainder = chunk[len(code):]
                # make sure what follows the county code starts with a digit
                if remainder and remainder[0].isdigit():
                    return chunk

    return text  # no valid match found, return as-is

print("Loading TrOCR model...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.eval()

def clean_pred(text: str) -> str:
    raw      = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
    filtered = apply_county_filter(raw)
    fixed    = apply_plate_structure(filtered)
    trimmed  = apply_length_filter(fixed)
    if raw != trimmed:
        print(f"    [FILTER] {raw} -> {trimmed}")
    return trimmed

def run_trocr(image_cv2) -> str:
    image_rgb    = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    pixel_values = processor(Image.fromarray(image_rgb), return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def evaluate_folder(folder, df, cer_metric):
    predictions   = []
    ground_truths = []
    total_plates  = 0
    exact_matches = 0

    for _, row in df.iterrows():
        image = cv2.imread(os.path.join(folder, row["image_filename"]))
        if image is None:
            print(f"  Skipping {row['image_filename']} — could not read")
            continue

        pred_text = clean_pred(run_trocr(image))
        gt_text   = str(row["ground_truth"]).strip().upper().replace(" ", "")

        predictions.append(pred_text)
        ground_truths.append(gt_text)

        total_plates += 1
        if pred_text == gt_text:
            exact_matches += 1

        print(f"  [{total_plates:03d}] GT: {gt_text:12} | Pred: {pred_text:12} | {'✅' if pred_text == gt_text else '❌'}")

    cer_score        = cer_metric(predictions, ground_truths)
    recognition_rate = exact_matches / total_plates if total_plates > 0 else 0

    correct_chars = 0
    total_chars   = 0
    for pred, gt in zip(predictions, ground_truths):
        total_chars   += len(gt)
        correct_chars += sum(p == g for p, g in zip(pred, gt))
    recall_ratio = correct_chars / total_chars if total_chars > 0 else 0

    return {
        "predictions":   predictions,
        "ground_truths": ground_truths,
        "total":         total_plates,
        "exact":         exact_matches,
        "rate":          recognition_rate,
        "cer":           cer_score.item(),
        "recall":        recall_ratio,
    }

def print_metrics(name, m):
    print(f"\n  {'Total plates':<25} {m['total']}")
    print(f"  {'Exact matches':<25} {m['exact']}")
    print(f"  {'Recognition Rate':<25} {m['rate']:.2%}")
    print(f"  {'Character Error Rate':<25} {m['cer']:.4f}")
    print(f"  {'Recall Ratio':<25} {m['recall']:.2%}")

df         = pd.read_csv(CSV_PATH)
cer_metric = CharErrorRate()

print(f"\n{'='*55}")
print(f"  Evaluating: NON-OBB ({DIR_NONOBB})")
print(f"{'='*55}")
nonobb = evaluate_folder(DIR_NONOBB, df, cer_metric)
print_metrics("NON-OBB", nonobb)

print(f"\n{'='*55}")
print(f"  Evaluating: OBB ({DIR_OBB})")
print(f"{'='*55}")
obb = evaluate_folder(DIR_OBB, df, cer_metric)
print_metrics("OBB", obb)

# summary
print(f"\n{'='*55}")
print(f"  {'Metric':<25} {'NON-OBB':>12} {'OBB':>12}")
print(f"  {'-'*53}")
metrics = [
    ("Recognition Rate", f"{nonobb['rate']:.2%}",  f"{obb['rate']:.2%}",  obb['rate']  > nonobb['rate'],  False),
    ("Char Error Rate",  f"{nonobb['cer']:.4f}",   f"{obb['cer']:.4f}",   obb['cer']   < nonobb['cer'],   True),
    ("Recall Ratio",     f"{nonobb['recall']:.2%}", f"{obb['recall']:.2%}", obb['recall'] > nonobb['recall'], False),
]
for label, r_val, o_val, obb_wins, lower_is_better in metrics:
    r_str = r_val + (" ✅" if not obb_wins else "")
    o_str = o_val + (" ✅" if obb_wins else "")
    print(f"  {label:<25} {r_str:>14} {o_str:>14}")
print(f"{'='*55}")