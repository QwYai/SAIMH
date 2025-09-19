import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

def structure_reg(x, a, tau=0.1, L=3, eps=1e-6):
    """
    x: [B, d]  原始（或预训练后不变的）模态特征
    a: [B, k]  对齐后投影到共享空间的特征（未归一化）
    tau: 温度
    L: 正则层数
    返回：标量 STRUCTURE 损失
    """
    B = x.size(0)
    # 1) 归一化并去中心
    xb = F.normalize(x, p=2, dim=1)             # [B,d]
    ab = F.normalize(a, p=2, dim=1)             # [B,k]
    xb = xb - xb.mean(dim=0, keepdim=True)      # 去全局偏置
    ab = ab - ab.mean(dim=0, keepdim=True)

    # 2) 计算相似度矩阵并 row-wise Softmax
    Sx = torch.matmul(xb, xb.t()) / tau         # [B,B]
    Sa = torch.matmul(ab, ab.t()) / tau         # [B,B]
    Px = F.softmax(Sx, dim=1) + eps             # 概率矩阵
    Pa = F.softmax(Sa, dim=1) + eps

    # 3) 多层次幂运算并累加 JS 散度
    loss = 0.0
    for l in range(1, L+1):
        Pxl = Px.matrix_power(l)
        Pal = Pa.matrix_power(l)
        M = 0.5 * (Pxl + Pal)
        # DKL(Pxl || M) + DKL(Pal || M)
        loss += 0.5 * (F.kl_div((Pxl+eps).log(), M, reduction='batchmean')
                     + F.kl_div((Pal+eps).log(), M, reduction='batchmean')) / l
    return loss

# 对比损失：InfoNCE
class InfoNCE_Loss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCE_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        # 计算cosine similarity
        similarity_matrix = torch.mm(image_features, text_features.t())
        
        # 计算对比损失（InfoNCE loss）
        labels = torch.arange(len(image_features)).to(image_features.device)
        logits = similarity_matrix / self.temperature
        loss = F.cross_entropy(logits, labels)

        return loss

# 文本相似度损失（基于欧式距离）
class TextSimilarityLoss(nn.Module):
    def __init__(self, pmargin=1.0, nmargin = 10.0):
        super(TextSimilarityLoss, self).__init__()
        self.positive_margin = pmargin
        self.negative_margin = nmargin

    def forward(self, text_features, labels):
        # 获取文本特征的数量
        batch_size = text_features.size(0)
        
        # 计算文本之间的欧式距离
        dist_matrix = torch.cdist(text_features, text_features, p=2)  # 计算欧式距离矩阵

        # 同标签文本样本的掩码
        positive_mask = torch.matmul(labels, labels.T) > 0  # 同标签样本
        negative_mask = ~positive_mask  # 不同标签样本

        # 提取正样本（同标签）和负样本（不同标签）的欧式距离
        positive_dist = dist_matrix[positive_mask]  # 同标签样本的欧式距离
        negative_dist = dist_matrix[negative_mask]  # 不同标签样本的欧式距离

        # 确保正样本的欧式距离尽量小，负样本的欧式距离尽量大
        positive_loss = torch.mean((positive_dist - self.positive_margin) ** 2)  # 强制正样本距离小于margin
        negative_loss = torch.mean((self.negative_margin - negative_dist) ** 2)  # 强制负样本距离大于margin

        # 总的文本相似度损失，正样本与负样本的损失之和
        loss = positive_loss + negative_loss
        return loss

def negative_log_likelihood_similarity_loss(hash_codes, labels, bit=128, device='cuda:0'):
    """
    负对数似然相似损失（NLLS）函数，适用于二进制哈希码任务
    
    :param hash_codes: 当前批次的哈希码 (batch_size, hash_size)，二进制哈希码
    :param labels: 当前批次的标签 (batch_size, num_labels)，多标签
    :param bit: 哈希码的位数，用于归一化计算
    :param device: 用于计算的设备（'cuda:0' 或 'cpu'）
    :return: 负对数似然相似损失
    """
    
    # 确保所有张量都在同一设备上
    hash_codes = hash_codes.to(device)
    labels = labels.to(device)
    
    batch_size = labels.size(0)
    
    # 将哈希码转换为 double 类型，以便进行更精确的计算
    hash_codes = hash_codes.double()
    
    # 计算样本对之间的相似度（使用内积）
    similarity_matrix = torch.mm(hash_codes, hash_codes.T)  # shape: [batch_size, batch_size]
    
    # 使用标签之间的相似度构建标签匹配矩阵
    label_similarity_matrix = torch.matmul(labels, labels.T)  # shape: [batch_size, batch_size]
    
    # 通过标签向量的相似性构建标签匹配矩阵
    # 1 表示标签相似，0 表示标签不相似
    labels_matrix = (label_similarity_matrix > 0).float()  # 转化为0或1的矩阵
    
    # 归一化相似度矩阵
    omega = similarity_matrix / (bit / 18)
    
    # 计算损失
    loss = -((labels_matrix > 0).float() * omega - torch.log(1 + torch.exp(omega)))
    
    # 对所有样本对进行平均，得到最终的损失值
    loss = torch.mean(loss)
    
    return loss

def contrastive_loss_nuswide(hash_code, labels, margin=10.0, margin2 = 2.0):
    """
    计算对比损失（Contrastive Loss）。要求输入的 `hash_code` 是连续的，可以计算梯度。
    
    Args:
        hash_code (torch.Tensor): 哈希码，形状为 (batch_size, hash_size)，连续值 [-1, 1]
        labels (torch.Tensor): 样本标签，形状为 (batch_size, num_classes)，one-hot 向量
        margin (float): 设定的最小间隔，决定不相似样本之间的最大允许距离
        
    Returns:
        loss (torch.Tensor): 对比损失
    """
    batch_size = hash_code.size(0)
    
    # # 计算样本对之间的欧式距离（L2 范数）
    dist_matrix = torch.cdist(hash_code, hash_code, p=2)  # 计算哈希码之间的欧式距离
    # 计算样本对的标签相似度（如果两个样本属于同一类，则它们是相似的，标签相同）
    labels = labels.float().cuda()
    sim_matrix = torch.matmul(labels, labels.t())  # 计算标签相似性，1表示相似，0表示不相似
    sim_matrix = (sim_matrix > 0).float()  # If labels are the same, set Phi_ij = 1, else 0
    # 提取上三角部分并去除对角线
    sim_matrix = torch.triu(sim_matrix, diagonal=1)

    # 对比损失公式
    # 对于相似样本，最小化距离
    # 对于不相似样本，最大化距离，距离大于 margin 时损失为0，否则损失为 (margin - dist) ** 2

    # 相似样本损失 (sim_matrix == 1)
    similar_loss = (sim_matrix * (margin2 - dist_matrix) ** 2).sum()

    # 不相似样本损失 (sim_matrix == 0)
    dissimilar_loss = ((1 - sim_matrix) * (margin - dist_matrix) ** 2).sum()

    # 归一化损失
    total_loss = (similar_loss + dissimilar_loss) / (batch_size * (batch_size - 1) / 2)
    # total_loss = (similar_loss) / (batch_size * (batch_size - 1) / 2)
    
    return total_loss

def contrastive_loss2(hash_code, labels, margin1=10.0, margin2 = 1.0):
    batch_size = hash_code.size(0)
    
    # # 计算样本对之间的欧式距离（L2 范数）
    dist_matrix = torch.cdist(hash_code, hash_code, p=2)  # 计算哈希码之间的欧式距离
    # dist_matrix = torch.matmul(hash_code, hash_code.T)  # 计算哈希码之间的欧式距离
    # 计算样本对的标签相似度（如果两个样本属于同一类，则它们是相似的，标签相同）
    labels = labels.float().cuda()
    sim_matrix = torch.matmul(labels, labels.t())  # 计算标签相似性，1表示相似，0表示不相似
    sim_matrix = (sim_matrix > 0).float()  # If labels are the same, set Phi_ij = 1, else 0
    # 提取上三角部分并去除对角线
    sim_matrix = torch.triu(sim_matrix, diagonal=1)

    # 对比损失公式
    # 对于相似样本，最小化距离
    # 对于不相似样本，最大化距离，距离大于 margin 时损失为0，否则损失为 (margin - dist) ** 2

    # 相似样本损失 (sim_matrix == 1)
    similar_loss = (sim_matrix * (margin2 - dist_matrix) ** 2).sum()

    # 不相似样本损失 (sim_matrix == 0)
    dissimilar_loss = ((1 - sim_matrix) * (margin1 - dist_matrix) ** 2).sum()

    # 归一化损失
    total_loss = (similar_loss + dissimilar_loss) / (batch_size * (batch_size - 1) / 2)
    # total_loss = (similar_loss) / (batch_size * (batch_size - 1) / 2)
    
    return total_loss


def contrastive_loss(hash_code, labels, margin=10.0):
    """
    计算对比损失（Contrastive Loss）。要求输入的 `hash_code` 是连续的，可以计算梯度。
    
    Args:
        hash_code (torch.Tensor): 哈希码，形状为 (batch_size, hash_size)，连续值 [-1, 1]
        labels (torch.Tensor): 样本标签，形状为 (batch_size, num_classes)，one-hot 向量
        margin (float): 设定的最小间隔，决定不相似样本之间的最大允许距离
        
    Returns:
        loss (torch.Tensor): 对比损失
    """
    batch_size = hash_code.size(0)
    
    # 计算样本对之间的欧式距离（L2 范数）
    dist_matrix = torch.cdist(hash_code, hash_code, p=2)  # 计算哈希码之间的欧式距离
    
    # 计算样本对的标签相似度（如果两个样本属于同一类，则它们是相似的，标签相同）
    labels = labels.float()
    sim_matrix = torch.matmul(labels, labels.t())  # 计算标签相似性，1表示相似，0表示不相似
    
    # 对比损失公式
    # 对于相似样本，最小化距离
    # 对于不相似样本，最大化距离，距离大于 margin 时损失为0，否则损失为 (margin - dist) ** 2
    loss = 0.0
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if sim_matrix[i, j] == 1:
                # 相似对，最小化距离
                loss += dist_matrix[i, j] ** 2
            else:
                # 不相似对，最大化距离, 接近距离margin
                loss += torch.max(torch.tensor(1.0).to(hash_code.device), margin - dist_matrix[i, j]) ** 2
    
    return loss / (batch_size * (batch_size - 1) / 2)  # 归一化

def distillation_loss(student_hash, teacher_hash, T=2.0, alpha=0.5):
    """
    计算教师-学生网络的蒸馏损失。目标是使得学生网络的输出接近教师网络的输出。
    """
    # 计算学生和教师输出之间的欧式距离
    student_hash = student_hash / student_hash.norm(dim=1, keepdim=True)
    teacher_hash = teacher_hash / teacher_hash.norm(dim=1, keepdim=True)
    
    # Softmax计算蒸馏损失
    distillation_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_hash / T, dim=1),
        F.softmax(teacher_hash / T, dim=1)
    )
    
    return distillation_loss * alpha

# 量化损失（Quantization Loss）
def quantization_loss(hash_code_continuous, lambda_param, b):
    # 计算 I 的索引
    I_1 = torch.arange(0, int(lambda_param * b), dtype=torch.long)  # 前 λb 个样本
    I_2 = torch.arange(int((1 - lambda_param) * b), b, dtype=torch.long)  # 后 (1-λ)b 到 b 个样本
    I = torch.cat((I_1, I_2), dim=0)  # 合并两个子集
    # 从哈希码中选择样本
    selected_hash_codes = hash_code_continuous[I]

    # 计算量化损失：|hi| - 1 的 L2 范数
    quant_loss = torch.mean(torch.norm(torch.abs(selected_hash_codes) - 1, p=2, dim=1) ** 2)

    return quant_loss

def quantization_loss1(outputs):
    BCELoss = torch.nn.BCELoss()
    loss = BCELoss((outputs + 1) / 2, (torch.sign(outputs) + 1) / 2)
    return loss

def metric_loss2(hash_code_continuous, labels, lambda_param, delta, device):
    """
    Metric Loss for multi-label setting.
    L_m = (1 / (lambda * b)^2) * sum of [delta * log(1 + exp(Theta_ij)) - Phi_ij * Theta_ij]
    """
    labels = labels.to(device)  # Ensure labels are on the same device
    b = hash_code_continuous.size(0)
    hash_code_continuous = hash_code_continuous.to(device)  # Ensure hash codes are on the same device

    N = hash_code_continuous.size(0)  # Total number of samples (batch size)

    # Calculate pairwise similarity (inner product) between hash codes
    similarity_matrix = torch.matmul(hash_code_continuous, hash_code_continuous.T)  # Shape: (N, N)

    # Calculate Phi_ij: Similarity matrix based on labels
    Phi = torch.matmul(labels, labels.T)  # Shape: (N, N)
    Phi = (Phi > 0).float()  # If labels are the same, set Phi_ij = 1, else 0
    Phi = 1 - Phi

    # Mask out the upper triangle of the matrix to avoid redundant calculations
    mask = torch.triu(torch.ones_like(Phi), diagonal=1)  # Upper triangular matrix (excluding diagonal)

    # Apply the mask to ignore redundant pairs
    similarity_matrix = similarity_matrix * mask
    Phi = Phi * mask

    # Compute metric loss
    loss_term_1 = delta * torch.log(1 + torch.exp(similarity_matrix))  # delta * log(1 + exp(Θij))
    loss_term_2 = Phi * similarity_matrix  # Phi_ij * Θ_ij

    # Sum up the loss terms for all pairs
    loss = torch.sum(loss_term_1 - loss_term_2)

    # Scale the loss by the normalization factor
    loss = (1 / (lambda_param * b) ** 2) * loss

    return loss


def metric_loss(hash_code_continuous, labels, lambda_param, b, delta, device):
    """
    度量损失（Metric Loss），考虑多标签情况
    L_m = (1 / (lambda * b)^2) * sum of [delta * log(1 + exp(Theta_ij)) - Phi_ij * Theta_ij]
    """
    labels = labels.to(device)  # 确保 labels 在相同设备上
    hash_code_continuous = hash_code_continuous.to(device)  # 确保哈希码也在相同设备上

    N = hash_code_continuous.size(0)  # 总样本数量
    # 计算哈希码之间的内积（相似度）
    distance = torch.matmul(hash_code_continuous, hash_code_continuous.T)  # 内积计算

    # 计算 Phi_ij，基于标签判断是否是相似的样本对
    Phi = torch.matmul(labels, labels.T)  # 标签重叠度，计算两个样本的标签相似性
    Phi = (Phi > 0).float()  # 如果有标签重叠则设为1，否则设为0

    # 为避免数值不稳定，限制 distance 的最大值
    distance = torch.clamp(distance, max=10)  # 限制 distance 的值，防止梯度爆炸

    # 计算度量损失 L_m
    Lm = (1 / (lambda_param * b) ** 2) * torch.sum(delta * torch.log(1 + torch.exp(distance)) - Phi * distance)

    return Lm


# 计算余弦相似度
def cosine_similarity(x1, x2):
    return F.cosine_similarity(x1, x2, dim=1)
