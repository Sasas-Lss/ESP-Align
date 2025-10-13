import numpy as np
import numba as nb

MATCH, INSERT, DELETE = 0, 1, 2


@nb.njit
def _init_affine_structure(m, n, gap_open1, gap_open2, g_ext):
    # Initialize the affine structure with matrices M, L, and U
    M = np.full((m + 1, n + 1), -np.inf, dtype=np.float64)
    L = np.full((m + 1, n + 1), -np.inf, dtype=np.float64)
    U = np.full((m + 1, n + 1), -np.inf, dtype=np.float64)
    changes = np.empty((m + 1, n + 1), dtype=np.uint8)

    # Set the initial value of M[0, 0] to 0.0
    M[0, 0] = 0.0

    # Initialize the first row of L and M
    for i in range(1, m + 1):
        L[i, 0] = gap_open1[0] + (i - 1) * g_ext
        M[i, 0] = L[i, 0]
        changes[i, 0] = DELETE

    # Initialize the first column of U and M
    for j in range(1, n + 1):
        U[0, j] = gap_open2[0] + (j - 1) * g_ext
        M[0, j] = U[0, j]
        changes[0, j] = INSERT

    # Return the initialized matrices
    return M, L, U, changes


@nb.njit
def structure_aware_nw(similarity, z_score_similarity, structure_score, gap_open1, gap_open2, pearson_weight, g_ext):
    m, n = z_score_similarity.shape
    M, L, U, changes = _init_affine_structure(m, n, gap_open1, gap_open2, g_ext)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            delete = max(M[i - 1, j] + gap_open2[j],
                         U[i - 1, j] + g_ext)
            insert = max(M[i, j - 1] + gap_open1[i],
                         L[i, j - 1] + g_ext)
            # match = M[i - 1, j - 1] + z_score_similarity[i - 1, j - 1] + structure_score[i - 1, j - 1]
            # pearson_weight = 0.8
            match = M[i - 1, j - 1] + pearson_weight * z_score_similarity[i - 1, j - 1] + (1 - pearson_weight) * structure_score[i - 1, j - 1]
            # match = M[i - 1, j - 1] + structure_score[i - 1, j - 1]
            # match = M[i - 1, j - 1] + z_score_similarity[i - 1, j - 1]
            M[i, j] = max(match, delete, insert)
            U[i, j] = delete
            L[i, j] = insert

            if M[i, j] == match:
                changes[i, j] = MATCH
            elif M[i, j] == delete:
                changes[i, j] = DELETE
            elif M[i, j] == insert:
                changes[i, j] = INSERT

    # Backtrack to recover the alignment
    align1, align2 = [], []
    raw_alignment_score = 0.0
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and (j == 0 or changes[i, j] == DELETE):
            align1.append(i - 1)
            align2.append(-1)
            i -= 1
        elif j > 0 and (i == 0 or changes[i, j] == INSERT):
            align1.append(-1)
            align2.append(j - 1)
            j -= 1
        else:
            align1.append(i - 1)
            align2.append(j - 1)
            raw_alignment_score += similarity[i - 1, j - 1]
            i -= 1
            j -= 1

    return align1[::-1], align2[::-1], M[m, n], raw_alignment_score


@nb.njit
def structure_aware_nw_wlu(similarity, z_score_similarity, structure_score, gap_open1, gap_open2, g_ext):
    m, n = z_score_similarity.shape
    M, L, U = _init_affine_structure(m, n, gap_open1, gap_open2, g_ext)
    bt = np.zeros((m + 1, n + 1), dtype=np.int64)
    print(gap_open1[0])
    print(gap_open1[m])
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # gap in seq1 (insert gap in seq1, skip seq2[j-1])
            gap1_open = M[i, j - 1] + gap_open1[i]
            gap1_ext = L[i, j - 1] + g_ext
            L[i, j] = gap1_open if gap1_open >= gap1_ext else gap1_ext

            # gap in seq2 (insert gap in seq2, skip seq1[i-1])
            gap2_open = M[i - 1, j] + gap_open2[j]
            gap2_ext = U[i - 1, j] + g_ext
            U[i, j] = gap2_open if gap2_open >= gap2_ext else gap2_ext

            # match scores
            score_diag = structure_score[i - 1, j - 1]
            # score_diag = z_score_similarity[i - 1, j - 1] + structure_score[i - 1, j - 1]
            # score_diag = z_score_similarity[i - 1, j - 1]

            # compute M[i,j] as max from M, L, U diagonals
            m_diag = M[i - 1, j - 1] + score_diag
            l_diag = L[i - 1, j - 1] + score_diag
            u_diag = U[i - 1, j - 1] + score_diag

            # take max and record move
            max_val = m_diag
            move = 0
            if u_diag > max_val:
                max_val = u_diag
                move = 1
            if l_diag > max_val:
                max_val = l_diag
                move = 2

            M[i, j] = max_val
            bt[i, j] = move
    print("M\n", M)
    print("L\n", L)
    print("U\n", U)

    aln1 = np.empty(m + n, dtype=np.int64)
    aln2 = np.empty(m + n, dtype=np.int64)
    idx = 0
    i, j = m, n

    while i > 0 or j > 0:
        move = bt[i, j]
        if move == 0:
            if i > 0 and j > 0:
                aln1[idx] = i - 1
                aln2[idx] = j - 1
                i -= 1
                j -= 1
            else:
                break
        elif move == 1:
            if i > 0:
                aln1[idx] = i - 1
                aln2[idx] = -1
                i -= 1
            else:
                break
        else:
            if j > 0:
                aln1[idx] = -1
                aln2[idx] = j - 1
                j -= 1
            else:
                break
        idx += 1

    while i > 0:
        aln1[idx] = i - 1
        aln2[idx] = -1
        i -= 1
        idx += 1

    while j > 0:
        aln1[idx] = -1
        aln2[idx] = j - 1
        j -= 1
        idx += 1

    res1 = np.empty(idx, dtype=np.int64)
    res2 = np.empty(idx, dtype=np.int64)
    raw_alignment_score = 0.0
    for k in range(idx):
        r1 = aln1[idx - 1 - k]
        r2 = aln2[idx - 1 - k]
        res1[k] = r1
        res2[k] = r2
        if r1 != -1 and r2 != -1:
            raw_alignment_score += similarity[r1, r2]

    final_score = M[m, n]

    return res1, res2, final_score, raw_alignment_score
