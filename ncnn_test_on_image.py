from ultralytics import YOLO
import cv2
import yaml
import os

with open('dataset\\data.yaml', "r") as f:
    data = yaml.safe_load(f)
    class_names = data["names"]

model = YOLO("best.pt")

model.export(format="ncnn")

image_path = "test.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image.")
    exit()

ncnn_model = YOLO("best_ncnn_model")

results = ncnn_model(image)

output_dir = "cropped_boxes"
os.makedirs(output_dir, exist_ok=True)

for i, result in enumerate(results):
    for j, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = box.conf[0].item()

        cropped_img = image[y1:y2, x1:x2]

        label_name = class_names[class_id]
        filename = f"{output_dir}/{label_name}_{i}_{j}_conf_{confidence:.2f}.jpg"

        cv2.imwrite(filename, cropped_img)

        label = f"{label_name}: {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

resized = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
cv2.imshow("Detected Image", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
