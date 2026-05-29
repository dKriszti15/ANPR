from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import time

image_path = "cropped_gt_obb/0081.jpg"
image = Image.open(image_path).convert("RGB")

for model_name in ["microsoft/trocr-small-printed", "trocr-finetuned-plates"]:
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.eval()
    pixel_values = processor(image, return_tensors="pt").pixel_values
    start = time.time()
    generated_ids = model.generate(pixel_values)
    elapsed = time.time() - start
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"{model_name}: {text} | {elapsed:.3f}s")