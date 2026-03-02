import cv2
import yaml
import os
import time
import numpy as np
from ultralytics import YOLO

model_name = './best_obb.pt'

with open('./dataset/data.yaml', "r") as f:
    data = yaml.safe_load(f)
    class_names = data["names"]

model = YOLO(model_name)
model.export(format="ncnn")

ncnn_model = YOLO("best_obb_ncnn_model", task='obb')

cap = cv2.VideoCapture(0)
print("megvan a kamera")

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_count = 0
start_time = time.time()
fpm = 0
PADDING = 2

while True:
    ret, frame = cap.read()

    if not ret:
        print("break")
        break

    frame_count += 1
    results = ncnn_model(frame)

    output_dir = "cropped_boxes"
    os.makedirs(output_dir, exist_ok=True)

    for i, result in enumerate(results):
        if result.obb is None:
            continue

        for j, obb in enumerate(result.obb):
            confidence = obb.conf[0].item()
            class_id = int(obb.cls[0])
            label_name = class_names[class_id]

            points = obb.xyxyxyxy[0].cpu().numpy().astype(np.int32)  # [4, 2]

            # Draw the rotated bounding box ------------- SAVING NON-ROTATED boxes for now
            cv2.polylines(frame, [points.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=1)

            label = f"{label_name}: {confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            top_center_x = points[:, 0].mean().astype(int) - text_w // 2
            top_y = points[:, 1].min() - 8
            cv2.putText(frame, label, (top_center_x, top_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if confidence > 0.8:
                x1 = max(0, points[:, 0].min() + PADDING)
                y1 = max(0, points[:, 1].min() + PADDING)
                x2 = min(frame.shape[1], points[:, 0].max() - PADDING)
                y2 = min(frame.shape[0], points[:, 1].max() - PADDING)

                cropped_img = frame[y1:y2, x1:x2]
                if cropped_img.size > 0:
                    filename = f"{output_dir}/{label_name}_{i}_{j}_conf_{confidence:.2f}.jpg"
                    cv2.imwrite(filename, cropped_img)

    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
        fpm = fps * 60
        cv2.putText(frame, f"FPM: {fpm:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Original', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or elapsed_time >= 30:
        print('Model:', model_name, ' | Elapsed time:', elapsed_time, ' | FPM:', fpm)
        break

cap.release()
cv2.destroyAllWindows()