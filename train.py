import sys
sys.path.insert(0, ".")

from models.attention import SE
import ultralytics.nn.tasks as tasks
tasks.SE = SE

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/train-6/weights/last.pt")          
    results = model.train(
        data="VOC.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
    )