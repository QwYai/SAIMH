import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImageMLP, self).__init__()
        
        # 第一层 MLP (FC + BN + ReLU)
        self.fc1 = nn.Linear(input_dim, hidden_dim)    # 输入到隐藏层
        self.bn1 = nn.BatchNorm1d(hidden_dim)          # 第一层 BatchNorm
        self.act1 = nn.Tanh()                         # 第一层 ReLU 激活

        # 第二层 MLP (FC + BN + ReLU)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)    # 隐藏层到中间层，输出维度是 hidden_dim
        self.bn2 = nn.BatchNorm1d(hidden_dim)           # 第二层 BatchNorm
        self.act2 = nn.Tanh()                         # 第二层 ReLU 激活

        # 第三层全连接层，用于将 hidden_dim 转换为 output_dim
        self.fc3 = nn.Linear(hidden_dim, output_dim)    # 输出维度是 output_dim

    def forward(self, x):
        # 第一层 MLP
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)

        # 第二层 MLP
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)

        # 最后一层全连接层，用于恢复 output_dim
        x = self.fc3(x)

        # L2 归一化
        x = F.normalize(x, p=2, dim=1)

        return x
