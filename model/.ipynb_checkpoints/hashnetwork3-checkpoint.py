import torch
import torch.nn as nn
import torch.nn.functional as F

class HashingNetwork(nn.Module):
    def __init__(self, image_dim=512, text_dim=512, hidden_dim=8192, hash_size=128):
        super(HashingNetwork, self).__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.hash_size = hash_size

        # 特征提取网络 (对应论文中的FeaExtractor)
        self.image_encoder = nn.Sequential(
            nn.Linear(self.image_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 门控网络（对应论文中的特征细化，这里是一个简化版本）
        self.image_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        self.text_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )

        # 融合后的哈希网络（多阶段哈希）
        self.fusion_hashing_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hash_size),
            nn.BatchNorm1d(self.hash_size),
            nn.Tanh() # Tanh函数作为哈希层的激活函数
        )
    
    def forward(self, image_features, text_features):
        # 1. 独立特征提取
        image_h = self.image_encoder(image_features)
        text_h = self.text_encoder(text_features)

        # 2. 特征细化 (通过门控机制)
        image_gate_output = self.image_gate(image_h)
        text_gate_output = self.text_gate(text_h)
        
        image_refined_h = image_h * image_gate_output
        text_refined_h = text_h * text_gate_output
        
        # 3. 模态融合 (这里使用加法融合，也可以尝试拼接)
        fused_h = image_refined_h + text_refined_h
        
        # 4. 多阶段哈希
        final_hash_code = self.fusion_hashing_network(fused_h)
        
        return final_hash_code