import torch
from ultralytics import YOLO


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))

    model = YOLO('yolo11s.pt').to(device)

    # next training:
    model.train(
        data="D:\\CarplateRecognition\\sandbox_cameraTest\\dataset_v12_yolov11n\\data.yaml",
        epochs=200,
        batch=16,
        imgsz=640,
        amp=True,
        patience=25,
        lr0=0.0075,
        cos_lr=True,
        project='D:\\CarplateRecognition\\sandbox_cameraTest',
        name="train_results_v12_yolov11n_lr0075_coslrTrue_batch16",
        device=device
    )


if __name__ == "__main__":
    main()