import sys
sys.path.insert(0, ".")

from models.attention import SE
import ultralytics.nn.tasks as tasks
tasks.SE = SE

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("models/yolov8n.yaml")
    results = model.train(
        data="coco128.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
    )