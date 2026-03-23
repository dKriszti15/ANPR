import cv2
from ultralytics import YOLO
import supervision as sv

model_name = './best_obb.pt'
image_path = './test2.jpg'

model = YOLO(model_name)
model.export(format="ncnn")

ncnn_model = YOLO("best_obb_ncnn_model",task="obb")

results = ncnn_model(image_path)

detections = sv.Detections.from_ultralytics(results[0])

oriented_box_annotator = sv.OrientedBoxAnnotator()
annotated_frame = oriented_box_annotator.annotate(
    scene=cv2.imread(image_path),
    detections=detections
)

sv.plot_image(image=annotated_frame, size=(16, 16))