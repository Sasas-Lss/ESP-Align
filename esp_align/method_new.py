import NW_SS_new
import numpy as np

def extend_gap_penalty(gap_open):
    """
    将长度为 m 的 gap_open 转换为 m+1 的 gap_open_ext，
    表示插入 gap 在两个残基之间的惩罚值。

    参数：
        gap_open: 长度为 m 的数组

    返回：
        gap_open_ext: 长度为 m+1 的数组
    """
    m = len(gap_open)
    gap_open_ext = np.zeros(m + 1, dtype=np.float64)

    gap_open_ext[0] = 0.5 * gap_open[0]  # 开头插入
    gap_open_ext[m] = 0.5 * gap_open[-1]  # 结尾插入

    for i in range(1, m):
        gap_open_ext[i] = 0.5 * (gap_open[i - 1] + gap_open[i])  # 中间插入
        # gap_open_ext[i] = min(gap_open[i - 1], gap_open[i])  # 中间插入

    return gap_open_ext

def Needleman_Wunsch_New(similarity_matrix, z_score_similarity_matrix, structure_score_matrix, gap_open_vector1,
                         gap_open_vector2, pearson_weight, gap_extend_penalty):
    gap_open_vector1_m_1 = extend_gap_penalty(gap_open_vector1)
    gap_open_vector2_n_1 = extend_gap_penalty(gap_open_vector2)

    # print(gap_open_vector1_m_1)
    # print(gap_open_vector2_n_1)

    aln_1, aln_2, z_score_score, score_raw = NW_SS_new.structure_aware_nw(similarity_matrix.cpu().numpy(), z_score_similarity_matrix.cpu().numpy(),
                                                                                 structure_score_matrix.cpu().numpy(), gap_open_vector1_m_1,
                                                                                 gap_open_vector2_n_1, pearson_weight, g_ext=gap_extend_penalty)

    l_min = min(z_score_similarity_matrix.shape[0], z_score_similarity_matrix.shape[1])
    l_max = max(z_score_similarity_matrix.shape[0], z_score_similarity_matrix.shape[1])

    score_global = score_raw / l_max
    print('score_raw', score_raw)
    # 明天从这里开始撰写新的分数计算逻辑吧。

    return {'score_raw': score_raw, 'z_score_score': z_score_score, 'score_global': score_global, 'aln_1': aln_1, 'aln_2': aln_2}
