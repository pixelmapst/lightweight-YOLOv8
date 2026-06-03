import torch
import torch.nn as nn

class SE(nn.Module):

    def __init__(self, channels, reduction = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
        # 初始化为恒等映射：bias=1 → sigmoid(1)≈0.73 → 接近 x*1，不破坏预训练特征
        nn.init.zeros_(self.fc[-2].weight)
        nn.init.constant_(self.fc[-2].bias, 2.0)

    def forward(self, x):
        b, c, h, w = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w