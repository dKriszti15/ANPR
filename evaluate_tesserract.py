import os
import re
import cv2
import pandas as pd
from torchmetrics.text import CharErrorRate
import pytesseract

CSV_PATH = "ocr_test_data.csv"
CROPPED_BOXES_DIR = "cropped_boxes_nonobb"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CONFIG = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

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
    for i in range(len(text)):
        chunk = text[i:]
        for code in COUNTY_CODES:
            if chunk.startswith(code):
                remainder = chunk[len(code):]
                if remainder and remainder[0].isdigit():
                    return chunk
    return text

def clean_pred(text: str) -> str:
    raw      = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
    filtered = apply_county_filter(raw)
    fixed    = apply_plate_structure(filtered)
    trimmed  = apply_length_filter(fixed)
    if raw != trimmed:
        print(f"    [FILTER] {raw} -> {trimmed}")
    return trimmed

def run_tesseract(image_cv2) -> str:
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    width = int(gray.shape[1] * 3)
    height = int(gray.shape[0] * 3)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    blur = cv2.bilateralFilter(resized, 11, 17, 17)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(thresh, config=TESSERACT_CONFIG)

def evaluate():
    df = pd.read_csv(CSV_PATH)
    cer_metric = CharErrorRate()

    predictions = []
    ground_truths = []
    total_plates = 0
    exact_matches = 0

    for _, row in df.iterrows():
        image = cv2.imread(os.path.join(CROPPED_BOXES_DIR, row["image_filename"]))
        if image is None:
            print(f"Skipping {row['image_filename']} — could not read")
            continue

        pred_text = clean_pred(run_tesseract(image))
        gt_text = str(row["ground_truth"]).strip().upper().replace(" ", "")

        predictions.append(pred_text)
        ground_truths.append(gt_text)

        total_plates += 1
        if pred_text == gt_text:
            exact_matches += 1

        print(f"[{total_plates}] GT: {gt_text} | Pred: {pred_text} | {'✅' if pred_text == gt_text else '❌'}")

    # Recognition rate
    recognition_rate = exact_matches / total_plates
    print(f"\nTotal plates:      {total_plates}")
    print(f"Exact matches:     {exact_matches}")
    print(f"Recognition Rate:  {recognition_rate:.2%}")

    # CER
    cer_score = cer_metric(predictions, ground_truths)
    print(f"CER:               {cer_score:.4f}")

    # Recall ratio
    correct_chars = 0
    total_chars = 0
    for pred, gt in zip(predictions, ground_truths):
        total_chars += len(gt)
        correct_chars += sum(p == g for p, g in zip(pred, gt))
    recall_ratio = correct_chars / total_chars if total_chars > 0 else 0
    print(f"Recall Ratio:      {recall_ratio:.2%}")


if __name__ == "__main__":
    evaluate()