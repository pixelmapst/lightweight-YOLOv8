# lightweight-YOLOv8

YOLOv8 模型改进实验：引入注意力机制并配合通道剪枝，在保持检测精度的前提下压缩模型体积、提升推理速度。

## 动机

YOLOv8 作为当前主流的目标检测模型，在精度和速度之间取得了良好平衡。但针对边缘设备（如无人机、移动端）部署场景，仍有轻量化空间。本项目尝试：

1. **注意力机制**：向 backbone/neck 中引入 SE、CBAM、ECA 等注意力模块，增强特征表达
2. **通道剪枝**：基于 BN 层缩放因子对冗余通道进行结构化剪枝，压缩模型体积
3. **组合验证**：对比"纯剪枝"与"注意力 + 剪枝"两种路线，探索精度-效率上界

## 环境

- Python 3.10+
- PyTorch 2.x（CUDA 推荐）
- ultralytics

```bash
pip install ultralytics torch torchvision
```

## 项目结构（规划）

```
lightweight-YOLOv8/
├── models/
│   ├── attention.py      # 注意力模块（SE / CBAM / ECA）
│   └── yolo_att.yaml     # 带注意力的模型配置
├── prune/
│   └── prune.py          # 通道剪枝脚本
├── train.py              # 训练入口
├── experiments/          # 实验日志
└── README.md
```

## 实验计划

| 实验 | 模型 | mAP@0.5 | 参数量 | 推理速度(ms) | 备注 |
|------|------|---------|--------|-------------|------|
| 基线 | YOLOv8n | - | - | - | 官方权重 |
| 注意力 | YOLOv8n + SE | - | - | - | - |
| 剪枝 | YOLOv8n pruned | - | - | - | - |
| 组合 | YOLOv8n + SE + prune | - | - | - | - |

## 参考

- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- [SENet (CVPR 2018)](https://arxiv.org/abs/1709.01507)
- [CBAM (ECCV 2018)](https://arxiv.org/abs/1807.06521)
- [Learning Efficient CNNs via Network Slimming (ICCV 2017)](https://arxiv.org/abs/1708.06519)
- [Torch-Pruning](https://github.com/VainF/Torch-Pruning)

## License

MIT
