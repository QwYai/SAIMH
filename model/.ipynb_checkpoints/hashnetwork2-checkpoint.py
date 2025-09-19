import torch
import torch.nn as nn
import torch.nn.functional as F

class HashingNetwork(nn.Module):
    def __init__(
        self,
        image_dim=512,
        text_dim=512,
        hidden_dims=None,
        hash_size=128,
        device='cpu'
    ):
        """
        image_dim:    输入图像特征维度
        text_dim:     输入文本特征维度
        hidden_dims:  两塔 MLP 隐层维度列表，例如 [4096, 512]
        hash_size:    最终哈希码长度
        device:       模型和数据所在设备，如 'cpu' 或 'cuda:0'
        """
        super(HashingNetwork, self).__init__()
        if hidden_dims is None:
            hidden_dims = [4096, 512]
        self.image_dim   = image_dim
        self.text_dim    = text_dim
        self.hidden_dims = hidden_dims
        self.final_dim   = hidden_dims[-1]
        self.hash_size   = hash_size
        self.device      = torch.device(device)

        # —— 构造两塔 MLP —— #
        def make_mlp(in_dim, hidden_dims):
            layers = []
            prev = in_dim
            for h in hidden_dims:
                layers += [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True)
                ]
                prev = h
            return nn.Sequential(*layers)

        self.fc_image = make_mlp(self.image_dim, self.hidden_dims).to(self.device)
        self.fc_text  = make_mlp(self.text_dim,  self.hidden_dims).to(self.device)

        # —— 门控网络 —— #
        self.ifeat_gate = nn.Sequential(
            nn.Linear(self.final_dim, self.final_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.final_dim * 2, self.final_dim),
            nn.Sigmoid()
        ).to(self.device)
        self.tfeat_gate = nn.Sequential(
            nn.Linear(self.final_dim, self.final_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.final_dim * 2, self.final_dim),
            nn.Sigmoid()
        ).to(self.device)

        # —— 拼接后映射到哈希码 —— #
        self.fc_combined = nn.Sequential(
            nn.Linear(self.final_dim * 2, self.hash_size),
            nn.BatchNorm1d(self.hash_size),
            nn.Tanh()
        ).to(self.device)

    def forward(self, image_features, text_features):
        # 确保输入在同一 device
        image_features = image_features.to(self.device)
        text_features  = text_features.to(self.device)

        # 1) 两塔映射
        img_h = self.fc_image(image_features)   # → [B, final_dim]
        txt_h = self.fc_text(text_features)     # → [B, final_dim]

        # # 2) 门控加权
        img_h = img_h * self.ifeat_gate(img_h)
        txt_h = txt_h * self.tfeat_gate(txt_h)

        # 3) 拼接 & 哈希映射
        combined = torch.cat([img_h, txt_h], dim=1)     # → [B, 2*final_dim]
        hash_code = self.fc_combined(combined)          # → [B, hash_size]

        return hash_code
