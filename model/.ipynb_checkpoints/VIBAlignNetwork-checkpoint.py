import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        # x: [batch, dim]
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        # 残差连接 + BatchNorm
        x = x + residual
        return self.norm(x)

class VIBAlignHead(nn.Module):
    def __init__(self,
                 input_dim,
                 bottleneck_dim=128,
                 hidden_dims=[1024, 512],
                 num_attention_layers=2,
                 num_heads=4):
        super().__init__()
        # 初始 MLP + BatchNorm
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(dims[i+1]))
        self.initial_mlp = nn.Sequential(*layers)

        # Transformer 自注意力模块
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dims[-1],
            nhead=num_heads,
            dim_feedforward=hidden_dims[-1] * 2,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attention_layers
        )

        # 残差 MLP 块
        self.res_blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dims[-1]) for _ in range(2)]
        )

        # 输出 mu 和 logvar
        self.fc_mu = nn.Linear(hidden_dims[-1], bottleneck_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], bottleneck_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        x: [batch, input_dim]
        返回:
          z: [batch, bottleneck_dim]
          mu, logvar: [batch, bottleneck_dim]
        """
        # 初始 MLP
        x = self.initial_mlp(x)               # [batch, hidden_dims[-1]]

        # Transformer 编码 (视为序列长度1)
        x = x.unsqueeze(1)                    # [batch, 1, dim]
        x = self.transformer(x)               # [batch, 1, dim]
        x = x.squeeze(1)                      # [batch, dim]

        # 残差 MLP
        x = self.res_blocks(x)                # [batch, dim]

        # mu & logvar
        mu = self.fc_mu(x)                    # [batch, bottleneck_dim]
        logvar = self.fc_logvar(x)            # [batch, bottleneck_dim]

        # 重参数化 & 归一化
        z = self.reparameterize(mu, logvar)
        z = F.normalize(z, p=2, dim=1)

        return z, mu, logvar

def kl_divergence(mu, logvar):
    # per-sample KL, then mean
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())