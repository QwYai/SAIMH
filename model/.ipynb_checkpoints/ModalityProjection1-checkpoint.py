import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

class AlignProjectionTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, dim_feedforward=1024, num_layers=1, dropout=0.1, final_dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_fc = nn.Linear(input_dim, output_dim)
        # self.final_dropout = nn.Dropout(final_dropout)  # 新增的最终 dropout

    def forward(self, x):
        x = x.unsqueeze(1)                  # (B, 1, input_dim)
        x = self.input_proj(x)               # (B, 1, input_dim)
        x = self.transformer_encoder(x)      # (B, 1, input_dim)
        x = x.squeeze(1)                     # (B, input_dim)
        h = self.output_fc(x)                # (B, output_dim)
        # h = self.final_dropout(h)             # 加在输出层之后
        z = F.normalize(h - h.mean(0, keepdim=True), p=2, dim=1)
        return z




