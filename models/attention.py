import torch
import torch.nn as nn

class SE(nn.Module):

    def __init__(self, channels, reduction = 16):
        super.__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.linear(channels, channels / reduction),
            nn.ReLU(implace = True),
            nn.linear(channels / reduction, channels),
            nn.sigmoid(),
        )
    
    def forward(self, x):
        b, c, h, w = x.shape()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w