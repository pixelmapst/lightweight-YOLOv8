
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("models/yolov8n_base.yaml")
    results = model.train(
        data="coco128.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
    )