import torch
import numpy as np


def compute_cosine_similarity_matrix(emb1, emb2):
    """
    计算两个蛋白质序列的余弦相似度矩阵

    参数:
        emb1: 序列1的嵌入向量
        emb2: 序列2的嵌入向量

    返回:
        余弦相似度矩阵
    """
    cos = torch.nn.CosineSimilarity(dim=0)
    m, n = emb1.shape[0], emb2.shape[0]
    similarity_matrix = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            sim = cos(torch.tensor(emb1[i], dtype=torch.float32), torch.tensor(emb2[j], dtype=torch.float32)).item()
            similarity_matrix[i, j] = sim

    cosine_similarity_result = torch.from_numpy(similarity_matrix).to(
        device=emb1.device,
        dtype=emb1.dtype
    )
    return cosine_similarity_result


def compute_pearson_similarity_matrix(embedding1, embedding2,):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the similarity matrix
        with the signal enhancement based on Z-scores.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2
        :param l: scalar that can be use to regularize the similarity matrix (no effect with l=1)
        :param p: Minkowski distance order (ex. p=1:Manhattan, p=2:Euclidean)

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
        :type l: float
        :type p: integer
    """
    pearson_similarity_result = pearson_similarity_matrix_np(embedding1, embedding2)
    return pearson_similarity_result


def compute_pearson_new_similarity_matrix(embedding1, embedding2,):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the similarity matrix
        with the signal enhancement based on Z-scores.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2
        :param l: scalar that can be use to regularize the similarity matrix (no effect with l=1)
        :param p: Minkowski distance order (ex. p=1:Manhattan, p=2:Euclidean)

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
        :type l: float
        :type p: integer
    """
    pearson_similarity_result = pearson_similarity_matrix_np(embedding1, embedding2)
    pearson_mormalized_result = z_score_global(pearson_similarity_result)
    return pearson_similarity_result, pearson_mormalized_result


def z_score_similarity_matrix_np(similarity_matrix):
    sm = similarity_matrix
    # print(sm)
    # print(sm_pearson)
    columns_avg = torch.sum(sm, 0) / sm.shape[0]
    rows_avg = torch.sum(sm, 1) / sm.shape[1]

    columns_std = torch.std(sm, 0)
    rows_std = torch.std(sm, 1)

    z_rows = (sm - rows_avg.unsqueeze(1)) / rows_std.unsqueeze(1)
    z_columns = (sm - columns_avg) / columns_std

    z_score_mormalized_result = (z_rows + z_columns) / 2
    return z_score_mormalized_result


def z_score_global(matrix):
    mean = matrix.mean()
    # std = matrix.std()
    std = matrix.std(unbiased=False)  # 使用除以 n 的标准差
    if std == 0:
        std = torch.tensor(1.0, device=matrix.device)
    return (matrix - mean) / std


def pearson_similarity_matrix_np(
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Pearson correlation coefficient matrix between each pair of rows
    in two residue‑level embeddings, using numpy.corrcoef.

    Args:
        embedding1: Tensor of shape (M, D)
        embedding2: Tensor of shape (N, D)

    Returns:
        Tensor of shape (M, N) where [i, j] is the Pearson r between
        embedding1[i, :] and embedding2[j, :], with the same dtype and device
        as embedding1.
    """
    # Move to CPU and convert to numpy
    arr1 = embedding1.detach().cpu().numpy()  # shape (M, D)
    arr2 = embedding2.detach().cpu().numpy()  # shape (N, D)

    # Stack them so that rows 0..M-1 are emb1, rows M..M+N-1 are emb2
    stacked = np.vstack([arr1, arr2])  # shape (M+N, D)

    # Compute the full (M+N)x(M+N) correlation matrix over rows
    corr_full = np.corrcoef(stacked)  # rowvar=True by default

    M = arr1.shape[0]
    # Extract the cross‑correlation block of shape (M, N)
    corr_block = corr_full[:M, M:]  # shape (M, N)

    # 把值域从 [-1, 1] 线性映射到 [0, 1]
    # corr_block = (corr_block + 1) / 2  # 现在所有值都在 0~1 之间

    # Convert back to torch.Tensor, preserving dtype and device
    result = torch.from_numpy(corr_block).to(
        device=embedding1.device,
        dtype=embedding1.dtype
    )
    return result


def compute_euclidean_similarity_matrix(embedding1, embedding2, l=1, p=2):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the similarity matrix
        with the signal enhancement based on Z-scores.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2
        :param l: scalar that can be use to regularize the similarity matrix (no effect with l=1)
        :param p: Minkowski distance order (ex. p=1:Manhattan, p=2:Euclidean)

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
        :type l: float
        :type p: integer
    """
    sm = compute_similarity_matrix_plain(embedding1, embedding2, l=l, p=p)
    # print(sm)
    # print(sm_pearson)
    columns_avg = torch.sum(sm, 0) / sm.shape[0]
    rows_avg = torch.sum(sm, 1) / sm.shape[1]

    columns_std = torch.std(sm, 0)
    rows_std = torch.std(sm, 1)

    z_rows = (sm - rows_avg.unsqueeze(1)) / rows_std.unsqueeze(1)
    z_columns = (sm - columns_avg) / columns_std

    euclidean_similarity_result = (z_rows + z_columns) / 2

    return euclidean_similarity_result


def compute_similarity_matrix_plain(embedding1, embedding2, l=1, p=2):
    """ Take as input 2 sequence embeddings (at a residue level) and returns the plain
        similarity matrix.

        :param embedding1: residues embedding representation for sequence 1
        :param embedding2: residues embedding representation for sequence 2
        :param l: scalar that can be use to regularize the similarity matrix (no effect wit l=1)
        :param p: Minkowski distance order (ex. p=1:Manhattan, p=2:Euclidean)

        :type embedding1: pytorch tensor
        :type embedding2: pytorch tensor
        :type l: float
        :type p: integer
    """
    # result = torch.exp(-l * torch.cdist(embedding1, embedding2, p=p))
    # print(type(result))
    # return result
    return torch.exp(-l * torch.cdist(embedding1, embedding2, p=p))
    # return np.corrcoef(embedding1[i - 1].cpu().numpy(), embedding2[j - 1].cpu().numpy())[0, 1]
