from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import pandas as pd
import os

#Training on: cpu
#Epoch 1/10 | Loss: 7.3009
#Epoch 2/10 | Loss: 4.1542
#Epoch 3/10 | Loss: 2.7080
#Epoch 4/10 | Loss: 1.5706
#Epoch 5/10 | Loss: 1.0332
#Epoch 6/10 | Loss: 1.2567
#Epoch 7/10 | Loss: 0.5927
#Epoch 8/10 | Loss: 0.4956
#Epoch 9/10 | Loss: 0.4069
#Epoch 10/10 | Loss: 0.4032
#Writing model shards: 100%|██████████| 1/1 [00:00<00:00,  6.20it/s]
#Saved to trocr-finetuned-plates


CROPS_DIR = "cropped_gt_obb"
CSV_PATH = "obb_ocr_dataset.csv"
OUTPUT_DIR = "trocr-finetuned-plates"
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 5e-5

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

class PlateDataset(Dataset):
    def __init__(self, csv_path, crops_dir):
        self.df = pd.read_csv(csv_path)
        self.crops_dir = crops_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.crops_dir, row["image_filename"])
        image = Image.open(image_path).convert("RGB")

        pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze()

        labels = processor.tokenizer(
            str(row["ground_truth"]),
            return_tensors="pt",
            padding="max_length",
            max_length=16,
        ).input_ids.squeeze()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return pixel_values, labels

dataset = PlateDataset(CSV_PATH, CROPS_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        loss = model(pixel_values=pixel_values, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")