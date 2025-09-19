import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionGate(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, q, kv):
        B, D = q.shape
        H = self.num_heads
        # 线性映射
        q = self.query(q).view(B, H, self.head_dim).transpose(1, 2)  # [B, head_dim, H]
        k = self.key(kv).view(B, H, self.head_dim).transpose(1, 2)   # [B, head_dim, H]
        v = self.value(kv).view(B, H, self.head_dim).transpose(1, 2) # [B, head_dim, H]

        # 注意力权重
        attn_scores = torch.matmul(q.transpose(-2, -1), k) * self.scale  # [B, H, H]
        attn_probs = F.softmax(attn_scores, dim=-1)                     # [B, H, H]

        # 加权求和
        out = torch.matmul(attn_probs, v.transpose(1, 2))  # [B, H, head_dim]

        # 还原形状
        out = out.transpose(1, 2).contiguous().view(B, D)
        out = self.out(out)
        return out


class HashingNetwork(nn.Module):
    def __init__(self, image_dim=512, text_dim=512, hidden_dims=None, hash_size=128, device='cpu'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [4096, 512]
        self.device = torch.device(device)
        self.final_dim = hidden_dims[-1]

        def make_mlp(in_dim, hidden_dims):
            layers = []
            prev = in_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
                prev = h
            return nn.Sequential(*layers)

        self.fc_image = make_mlp(image_dim, hidden_dims).to(self.device)
        self.fc_text = make_mlp(text_dim, hidden_dims).to(self.device)

        # 替换门控网络为交叉注意力门控
        self.cross_gate_img = CrossAttentionGate(self.final_dim).to(self.device)
        self.cross_gate_txt = CrossAttentionGate(self.final_dim).to(self.device)

        self.fc_combined = nn.Sequential(
            nn.Linear(self.final_dim * 2, hash_size),
            nn.BatchNorm1d(hash_size),
            nn.Tanh()
        ).to(self.device)

    def forward(self, image_features, text_features):
        image_features = image_features.to(self.device)
        text_features = text_features.to(self.device)

        img_h = self.fc_image(image_features)  # [B, final_dim]
        txt_h = self.fc_text(text_features)    # [B, final_dim]

        # 交叉注意力融合，增强特征
        img_h_attn = self.cross_gate_img(img_h, txt_h)
        txt_h_attn = self.cross_gate_txt(txt_h, img_h)

        # 残差连接（可选）
        img_h = img_h + img_h_attn
        txt_h = txt_h + txt_h_attn

        combined = torch.cat([img_h, txt_h], dim=1)
        hash_code = self.fc_combined(combined)
        return hash_code
