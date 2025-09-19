import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

# AlignProjectionHead 同前（Linear→BN→ReLU→Linear(d) + center+normalize）
class AlignProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[1024,512]):
        super().__init__()
        layers=[]; prev=input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.fc=nn.Sequential(*layers)
    def forward(self,x):
        h=self.fc(x)
        z=F.normalize(h - h.mean(0,True),p=2,dim=1)
        return z

# ——— NDA 损失 ———
class NDALoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature
    def forward(self, z_i, z_t):
        B = z_i.size(0)
        # compute similarity matrices
        sim_i = (z_i @ z_i.t()) / self.t  # [B,B]
        sim_t = (z_t @ z_t.t()) / self.t
        # mask diag
        mask = ~torch.eye(B, device=z_i.device).bool()
        # compute row‐wise softmax distributions
        p_i = F.softmax(sim_i[mask].view(B, B-1), dim=1)
        p_t = F.softmax(sim_t[mask].view(B, B-1), dim=1)
        # KL divergence symmetrized
        loss = (F.kl_div(p_i.log(), p_t, reduction='batchmean') +
                F.kl_div(p_t.log(), p_i, reduction='batchmean')) * 0.5
        return loss

class BatchGLRLoss(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k
        from sklearn.neighbors import NearestNeighbors
        self.nbrs = NearestNeighbors(n_neighbors=k+1)

    def forward(self, z):
        """
        输入 z: [B,d]，只在当前 batch 上构建 Laplacian
        """
        with torch.no_grad():
            Z_np = z.cpu().numpy()
            # 找到每个点的 kNN（含自身）
            self.nbrs.fit(Z_np)
            _, idxs = self.nbrs.kneighbors(Z_np)
        B = z.size(0)
        # 构建局部权重矩阵
        W = torch.zeros(B, B, device=z.device)
        for i in range(B):
            for j in idxs[i,1:]:
                W[i,j] = W[j,i] = 1.0
        D = torch.diag(W.sum(dim=1))
        L = D - W
        # 计算 trace(Z^T L Z) / B
        return torch.trace(z.t() @ L @ z) / B

