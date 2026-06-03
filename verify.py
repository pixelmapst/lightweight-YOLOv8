import sys
sys.path.insert(0, ".")

import torch
from models.attention import SE

# 把 SE 注册到 ultralytics 模块表中，否则 yaml 解析时找不到
import ultralytics.nn.tasks as tasks
tasks.SE = SE

from ultralytics import YOLO

model = YOLO("models/yolov8n.yaml")

x = torch.randn(1, 3, 640, 640)
model.model.eval()
with torch.no_grad():
    out = model.model(x)

if isinstance(out, (list, tuple)):
    print(f"✓ 模型跑通，输出 {len(out)} 个值")
    for i, o in enumerate(out):
        if hasattr(o, 'shape'):
            print(f"  [{i}]: {o.shape}")
        elif isinstance(o, dict):
            print(f"  [{i}]: dict (keys: {list(o.keys())[:3]}...)")
        else:
            print(f"  [{i}]: {type(o).__name__}")
else:
    print(f"✓ 模型跑通，输出: {out.shape}")

total = sum(p.numel() for p in model.model.parameters())
print(f"  总参数量: {total:,}")