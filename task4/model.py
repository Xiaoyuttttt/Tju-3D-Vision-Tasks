# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        # 第一层卷积：3 -> 32
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # 第二层卷积：32 -> 64
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # 第三层卷积：64 -> 128 (新增)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 经过3次池化，32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256) 
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x))) # [B, 64, 8, 8]
        x = self.pool(F.relu(self.conv3(x))) # [B, 128, 4, 4]
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x