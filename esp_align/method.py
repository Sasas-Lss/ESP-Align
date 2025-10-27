import NW_SS
import numpy as np


def extend_gap_penalty(gap_open):
    """
    Extend the gap open penalty vector (length m) to length m+1,
    representing penalties for inserting gaps between residues.
    """
    m = len(gap_open)
    gap_open_ext = np.zeros(m + 1, dtype=np.float64)

    # Penalty for gap insertion at sequence ends
    gap_open_ext[0] = 0.5 * gap_open[0]
    gap_open_ext[m] = 0.5 * gap_open[-1]

    # Penalty for gap insertion between residues
    for i in range(1, m):
        gap_open_ext[i] = 0.5 * (gap_open[i - 1] + gap_open[i])

    return gap_open_ext


def Needleman_Wunsch_New(similarity_matrix, z_score_similarity_matrix, structure_score_matrix,
                         gap_open_vector1, gap_open_vector2, pearson_weight, gap_extend_penalty):
    """
    Structure-aware Needleman-Wunsch alignment integrating similarity,
    z-score normalization, and structural score matrices.
    """
    gap_open_vector1_m_1 = extend_gap_penalty(gap_open_vector1)
    gap_open_vector2_n_1 = extend_gap_penalty(gap_open_vector2)

    aln_1, aln_2, z_score_score, score_raw = NW_SS.structure_aware_nw(
        similarity_matrix.cpu().numpy(),
        z_score_similarity_matrix.cpu().numpy(),
        structure_score_matrix.cpu().numpy(),
        gap_open_vector1_m_1,
        gap_open_vector2_n_1,
        pearson_weight,
        g_ext=gap_extend_penalty
    )

    l_max = max(z_score_similarity_matrix.shape)
    score_global = score_raw / l_max

    # print("score_raw", score_raw)

    return {
        "score_raw": score_raw,
        "z_score_score": z_score_score,
        "score_global": score_global,
        "aln_1": aln_1,
        "aln_2": aln_2,
    }
