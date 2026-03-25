import cv2
import yaml
import os
import numpy as np
from ultralytics import YOLO

model_name = './best_obb.pt'
image_path = './grr.jpeg'
PADDING = 2
CONFIDENCE_THRESHOLD = 0.8

with open('./dataset/data.yaml', "r") as f:
    data = yaml.safe_load(f)
    class_names = data["names"]

model = YOLO(model_name)
model.export(format="ncnn")

ncnn_model = YOLO("best_obb_ncnn_model", task='obb')

frame = cv2.imread(image_path)
if frame is None:
    print("Error: Could not read the image.")
    exit()

output_dir = "cropped_boxes"
os.makedirs(output_dir, exist_ok=True)


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

results = ncnn_model(frame)

for i, result in enumerate(results):
    if result.obb is None:
        continue

    for j, obb in enumerate(result.obb):
        confidence = obb.conf[0].item()
        class_id = int(obb.cls[0])
        label_name = class_names[class_id]

        points = obb.xyxyxyxy[0].cpu().numpy().astype(np.int32)
        print(f"Box {j} points: {points}")

        # Draw the rotated bounding box
        cv2.polylines(frame, [points.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=1)

        label = f"{label_name}: {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        top_center_x = int(points[:, 0].mean()) - text_w // 2
        top_y = points[:, 1].min() - 8
        cv2.putText(frame, label, (top_center_x, top_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if confidence > CONFIDENCE_THRESHOLD:
            cropped_img = crop_rotated(frame, points.astype(np.float32), padding=PADDING)
            if cropped_img.size > 0:
                filename = f"{output_dir}/{label_name}_{i}_{j}_conf_{confidence:.2f}.jpg"
                cv2.imwrite(filename, cropped_img)
                print(f"Saved: {filename}")

total_detections = sum(len(r.obb) for r in results if r.obb is not None)
print(f"Total detections: {total_detections}")

resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
cv2.imshow("Detected Image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()