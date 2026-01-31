import cv2
import yaml
import os
import time
from ultralytics import YOLO

with open('./dataset/data.yaml', "r") as f:
    data = yaml.safe_load(f)
    class_names = data["names"]

model = YOLO('./best.pt')

cap = cv2.VideoCapture(0)
print("megvan a kamera")

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("break")
        break

    frame_count += 1

    results = model(frame)

    output_dir = "cropped_boxes"
    os.makedirs(output_dir, exist_ok=True)

    for i, result in enumerate(results):
        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0].item()

            cropped_img = frame[y1:y2, x1:x2]

            label_name = class_names[class_id]
            filename = f"{output_dir}/{label_name}_{i}_{j}_conf_{confidence:.2f}.jpg"

            if confidence > 0.8:
                cv2.imwrite(filename, cropped_img)

            label = f"{label_name}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
        fpm = fps * 60

        cv2.putText(
            frame,
            f"FPM: {fpm:.0f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow('Original', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
