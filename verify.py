import sys
sys.path.insert(0, ".")

import torch
from ultralytics import YOLO

model = YOLO("models/yolov8n.yaml")

x = torch.randn(1, 3, 640, 640)
model.model.eval()
with torch.no_grad():
    out = model.model(x)

if isinstance(out, (list, tuple)):
    print(f"✓ 模型跑通，输出 {len(out)} 个尺度")
    for i, o in enumerate(out):
        print(f"  尺度 {i}: {o.shape}")
else:
    print(f"✓ 模型跑通，输出: {out.shape}")

total = sum(p.numel() for p in model.model.parameters())
print(f"  总参数量: {total:,}")