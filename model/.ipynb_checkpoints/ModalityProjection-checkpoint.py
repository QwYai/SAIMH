import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 定义投影网络（增加多个全连接层，并在投影后进行L2归一化）
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=512, hidden_dims=[1024, 512]):
        super(ProjectionHead, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Dropout层，有助于防止过拟合
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        # 通过全连接层进行投影
        x = self.fc(x)
        # 投影后的特征进行L2归一化
        x = F.normalize(x, p=2, dim=1)
        return x
