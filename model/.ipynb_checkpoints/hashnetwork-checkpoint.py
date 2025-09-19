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

        # 图像特征的全连接映射网络（生成一半哈希码）
        self.fc_image = nn.Sequential(
            nn.Linear(self.image_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 文本特征的全连接映射网络（生成另一半哈希码）
        self.fc_text = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 拼接后的全连接层，用于将拼接后的哈希码映射到最终的哈希空间
        self.fc_combined = nn.Sequential(
            # nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            # nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 2, self.hash_size),
            nn.BatchNorm1d(self.hash_size),
            nn.Tanh()
            # nn.ReLU(inplace=True)
        )


    def forward(self, image_features, text_features):
        # 图像特征通过全连接层
        image_hash_code = self.fc_image(image_features)

        # 文本特征通过全连接层
        text_hash_code = self.fc_text(text_features)

        # 拼接图像和文本的哈希码
        combined_features = torch.cat((image_hash_code, text_hash_code), dim=1)  # 在特征维度拼接

        # 拼接后的全连接层映射到最终的哈希码
        final_hash_code = self.fc_combined(combined_features)
        # final_hash_code = torch.tanh(final_hash_code)

        return final_hash_code  # 返回拼接后的哈希码
