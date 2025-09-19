import torch
import numpy as np
import matplotlib.pyplot as plt

def mean_average_precision(query_code,
                           retrieval_code,
                           query_targets,
                           retrieval_targets,
                           device,
                           topk=None):
    """
    Calculate mean average precision (mAP).
    
    Args:
        query_code (torch.Tensor): Query data hash code.
        retrieval_code (torch.Tensor): Retrieval data hash code.
        query_targets (torch.Tensor): Query data targets, one-hot
        retrieval_targets (torch.Tensor): Retrieval data targets, one-hot
        device (torch.device): Using CPU or GPU.
        topk (int, optional): Number of top results to consider for each query.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    query_code = query_code.to(device)
    retrieval_code = retrieval_code.to(device)
    query_targets = query_targets.to(device)
    retrieval_targets = retrieval_targets.to(device)
    num_query = query_targets.shape[0]
    if topk is None:
        topk = retrieval_targets.shape[0]

    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from the database
        retrieval = (query_targets[i, :] @ retrieval_targets.t() > 0).float()

        # Calculate Hamming distance
        hamming_dist = (query_code[i, :] != retrieval_code).sum(dim=1).float()

        # Sort retrieval results based on Hamming distance
        _, sorted_indices = torch.sort(hamming_dist)
        retrieval = retrieval[sorted_indices][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Skip if no valid retrievals
        if retrieval_cnt == 0:
            continue

        # Generate score for each position
        if retrieval_cnt > 0:
            score = torch.arange(1, retrieval_cnt + 1, device=device).float()
        else:
            score = torch.zeros(0, device=device)

        # Get the indices of the valid retrievals
        index = torch.nonzero(retrieval, as_tuple=True)[0] + 1.0

        # Update mean average precision
        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP.item()

import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_pr_curve(query_code, retrieval_code, query_targets, retrieval_targets, device, num_points=10):
    """
    Compute Precision-Recall curve.
    
    Args:
        query_code (torch.Tensor): Query data hash code.
        retrieval_code (torch.Tensor): Retrieval data hash code.
        query_targets (torch.Tensor): Query data targets, one-hot.
        retrieval_targets (torch.Tensor): Retrieval data targets, one-hot.
        device (torch.device): Using CPU or GPU.
        num_points (int): Number of points for the PR curve.

    Returns:
        recall_list (numpy.ndarray): Recall values.
        precision_list (numpy.ndarray): Precision values.
    """
    query_code = query_code.to(device)
    retrieval_code = retrieval_code.to(device)
    query_targets = query_targets.to(device)
    retrieval_targets = retrieval_targets.to(device)

    num_query = query_targets.shape[0]
    recall_list = np.linspace(0, 1, num_points)  # 100 个 Recall 采样点
    precision_list = np.zeros(num_points)

    for i in range(num_query):
        # 计算检索结果的相关性
        relevant = (query_targets[i, :] @ retrieval_targets.t() > 0).float()

        # 计算 Hamming 距离
        hamming_dist = (query_code[i, :] != retrieval_code).sum(dim=1).float()

        # 根据 Hamming 距离排序
        sorted_indices = torch.argsort(hamming_dist)
        relevant = relevant[sorted_indices]

        # 计算累计 TP 和 FP
        cumulative_tp = torch.cumsum(relevant, dim=0)
        total_relevant = relevant.sum().item()

        if total_relevant == 0:
            continue  # 跳过没有相关检索的查询样本

        precision = cumulative_tp / torch.arange(1, len(relevant) + 1, device=device).float()
        recall = cumulative_tp / total_relevant

        # 插值计算 PR 曲线
        for j in range(num_points):
            recall_threshold = recall_list[j]
            valid_indices = recall >= recall_threshold
            if valid_indices.any():
                precision_list[j] += precision[valid_indices].max().item()

    precision_list /= num_query  # 计算平均 Precision

    return recall_list, precision_list

def plot_pr_curve(recall_list, precision_list):
    """ 绘制 Precision-Recall 曲线 """
    plt.figure(figsize=(8, 6))
    plt.plot(recall_list, precision_list, marker='o', linestyle='-')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.savefig("PR-curve-nuswide.png")



