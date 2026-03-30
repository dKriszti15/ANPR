import os

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics.text import CharErrorRate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re

CSV_PATH = "ocr_test_data.csv"
MODEL_NAME = "microsoft/trocr-base-printed"
ALLOWED_CHARACTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

print("Loading TrOCR model...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.eval()

def clean_pred(text: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', text.strip().upper())

def run_trocr(image_cv2) -> str:
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    pixel_values = processor(image_pil, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def evaluate():
    df = pd.read_csv(CSV_PATH)
    cer_metric = CharErrorRate()

    predictions = []
    ground_truths = []
    total_plates = 0
    exact_matches = 0

    for _, row in df.iterrows():
        image = cv2.imread(os.path.join("cropped_boxes", row["image_filename"]))
        if image is None:
            print(f"Skipping {row['image_filename']} — could not read")
            continue

        pred_text = clean_pred(run_trocr(image))
        gt_text = str(row["ground_truth"]).strip().upper().replace(" ", "")

        predictions.append(pred_text)
        ground_truths.append(gt_text)

        total_plates += 1
        if pred_text == gt_text:
            exact_matches += 1

        print(f"[{total_plates}] GT: {gt_text} | Pred: {pred_text} | {'✅' if pred_text == gt_text else '❌'}")

    # Recognition rate
    recognition_rate = exact_matches / total_plates
    print(f"\nTotal plates:  {total_plates}")
    print(f"Exact matches:  {exact_matches}")
    print(f"Recognition Rate:  {recognition_rate:.2%}")

    # Character Error Rate
    cer_score = cer_metric(predictions, ground_truths)
    print(f"Character Error Rate:  {cer_score:.4f}")

    # Recall ratio
    correct_chars = 0
    total_chars = 0
    for pred, gt in zip(predictions, ground_truths):
        total_chars += len(gt)
        correct_chars += sum(p == g for p, g in zip(pred, gt))
    recall_ratio = correct_chars / total_chars if total_chars > 0 else 0
    print(f"Recall Ratio:  {recall_ratio:.2%}")

    # Confusion matrix
    all_pred_chars = []
    all_gt_chars = []
    for pred, gt in zip(predictions, ground_truths):
        for p, g in zip(pred, gt):
            all_pred_chars.append(p)
            all_gt_chars.append(g)

    labels = sorted(set(all_gt_chars))
    cm = confusion_matrix(all_gt_chars, all_pred_chars, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(16, 16))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("conf_matrix.png", dpi=150)
    plt.show()
    print("Saved: conf_matrix.png")

if __name__ == "__main__":
    evaluate()