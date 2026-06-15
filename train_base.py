
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/voc_baseline_50ep/weights/last.pt")
    results = model.train(
        data="VOC.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
    )