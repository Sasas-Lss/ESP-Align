import torch
import numpy as np


def compute_cosine_similarity_matrix(emb1, emb2):
    """
    Compute the cosine similarity matrix between two protein embeddings.

    Args:
        emb1 (torch.Tensor): Residue embeddings for sequence 1, shape (M, D)
        emb2 (torch.Tensor): Residue embeddings for sequence 2, shape (N, D)

    Returns:
        torch.Tensor: Cosine similarity matrix of shape (M, N)
    """
    cos = torch.nn.CosineSimilarity(dim=0)
    m, n = emb1.shape[0], emb2.shape[0]
    similarity_matrix = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            sim = cos(
                torch.tensor(emb1[i], dtype=torch.float32),
                torch.tensor(emb2[j], dtype=torch.float32)
            ).item()
            similarity_matrix[i, j] = sim

    return torch.from_numpy(similarity_matrix).to(
        device=emb1.device,
        dtype=emb1.dtype
    )


def pearson_similarity_matrix_np(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
    """
    Compute the Pearson correlation coefficient matrix between two residue-level embeddings.

    Args:
        embedding1 (torch.Tensor): Tensor of shape (M, D)
        embedding2 (torch.Tensor): Tensor of shape (N, D)

    Returns:
        torch.Tensor: Pearson correlation matrix of shape (M, N)
    """
    arr1 = embedding1.detach().cpu().numpy()
    arr2 = embedding2.detach().cpu().numpy()
    stacked = np.vstack([arr1, arr2])
    corr_full = np.corrcoef(stacked)
    M = arr1.shape[0]
    corr_block = corr_full[:M, M:]
    return torch.from_numpy(corr_block).to(
        device=embedding1.device,
        dtype=embedding1.dtype
    )


def compute_pearson_similarity_matrix(embedding1, embedding2):
    """
    Compute the Pearson similarity matrix between two residue embeddings.

    Args:
        embedding1 (torch.Tensor): Residue embeddings for sequence 1
        embedding2 (torch.Tensor): Residue embeddings for sequence 2

    Returns:
        torch.Tensor: Pearson similarity matrix
    """
    return pearson_similarity_matrix_np(embedding1, embedding2)


def z_score_global(matrix):
    """
    Apply global Z-score normalization to a similarity matrix.

    Args:
        matrix (torch.Tensor): Input matrix

    Returns:
        torch.Tensor: Z-score normalized matrix
    """
    mean = matrix.mean()
    std = matrix.std(unbiased=False)
    if std == 0:
        std = torch.tensor(1.0, device=matrix.device)
    return (matrix - mean) / std


def compute_pearson_new_similarity_matrix(embedding1, embedding2):
    """
    Compute both raw and Z-score normalized Pearson similarity matrices.

    Args:
        embedding1 (torch.Tensor): Residue embeddings for sequence 1
        embedding2 (torch.Tensor): Residue embeddings for sequence 2

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (raw Pearson matrix, Z-score normalized matrix)
    """
    pearson_result = pearson_similarity_matrix_np(embedding1, embedding2)
    normalized_result = z_score_global(pearson_result)
    return pearson_result, normalized_result


def compute_similarity_matrix_plain(embedding1, embedding2, l=1, p=2):
    """
    Compute a plain exponential similarity matrix based on Minkowski distance.

    Args:
        embedding1 (torch.Tensor): Residue embeddings for sequence 1
        embedding2 (torch.Tensor): Residue embeddings for sequence 2
        l (float): Regularization factor
        p (int): Minkowski distance order (1: Manhattan, 2: Euclidean)

    Returns:
        torch.Tensor: Similarity matrix
    """
    return torch.exp(-l * torch.cdist(embedding1, embedding2, p=p))


def compute_euclidean_similarity_matrix(embedding1, embedding2, l=1, p=2):
    """
    Compute the Euclidean-based similarity matrix with Z-score normalization.

    Args:
        embedding1 (torch.Tensor): Residue embeddings for sequence 1
        embedding2 (torch.Tensor): Residue embeddings for sequence 2
        l (float): Regularization factor
        p (int): Minkowski distance order

    Returns:
        torch.Tensor: Normalized Euclidean similarity matrix
    """
    sm = compute_similarity_matrix_plain(embedding1, embedding2, l=l, p=p)
    columns_avg = torch.sum(sm, 0) / sm.shape[0]
    rows_avg = torch.sum(sm, 1) / sm.shape[1]
    columns_std = torch.std(sm, 0)
    rows_std = torch.std(sm, 1)

    z_rows = (sm - rows_avg.unsqueeze(1)) / rows_std.unsqueeze(1)
    z_columns = (sm - columns_avg) / columns_std
    return (z_rows + z_columns) / 2
