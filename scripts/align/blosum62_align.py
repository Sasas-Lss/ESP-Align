import os
import random
from tqdm import tqdm
from itertools import combinations
from Bio import SeqIO, AlignIO
import numpy as np
import csv
from blosum import BLOSUM
import argparse

blosum62 = BLOSUM(62)  # 加载 BLOSUM62 矩阵

def load_fasta_nor(fasta_path):
    """读取 tfa 文件，返回 [(id, seq), ...] 格式列表"""
    records = []
    for r in SeqIO.parse(fasta_path, "fasta"):
        records.append((r.id, str(r.seq)))
    return records

def load_reference_alignment(ref_fasta_path, id1, id2):
    """从参考比对中提取两条序列（参考比对格式为多序列FASTA）"""
    ref_dict = {r.id: str(r.seq) for r in SeqIO.parse(ref_fasta_path, "fasta")}
    if id1 in ref_dict and id2 in ref_dict:
        return ref_dict[id1], ref_dict[id2]
    else:
        raise ValueError(f"Missing reference alignment for {id1} or {id2}")


def nw_single_align(tfa_path, output_path, gap_open=-10, gap_extend=-1):
    seq_records = load_fasta_nor(tfa_path)
    seq1 = seq_records[0][1]
    seq2 = seq_records[1][1]
    # 运行 BLOSUM62 + NW 算法
    pred_align1, pred_align2, _ = needleman_wunsch_affine(seq1, seq2, gap_open, gap_extend)
    print(pred_align1)
    print("\n")
    print(pred_align2)
    with open(output_path, "w") as f:
        f.write(seq_records[0][0] + "\n")
        f.write(pred_align1 + "\n")
        f.write(seq_records[1][0] + "\n")
        f.write(pred_align2 + "\n")

def get_reference_alignment(ref_fasta_path, seq_name1, seq_name2):
    alignment = AlignIO.read(ref_fasta_path, "fasta")
    seq_dict = {record.id: str(record.seq) for record in alignment}
    if seq_name1 not in seq_dict or seq_name2 not in seq_dict:
        raise ValueError(f"{seq_name1} or {seq_name2} not found in reference MSA.")
    aln1 = seq_dict[seq_name1]
    aln2 = seq_dict[seq_name2]
    cleaned1, cleaned2 = [], []
    for a, b in zip(aln1, aln2):
        if a == '-' and b == '-':
            continue
        cleaned1.append(a)
        cleaned2.append(b)
    return ''.join(cleaned1), ''.join(cleaned2)


def extract_matching_pairs(seq1_aligned, seq2_aligned):
    i_idx, j_idx = 0, 0
    match_pairs = []
    for a, b in zip(seq1_aligned, seq2_aligned):
        if a != '-' and b != '-':
            match_pairs.append((i_idx, j_idx))
        if a != '-':
            i_idx += 1
        if b != '-':
            j_idx += 1
    return set(match_pairs)


def get_match_labels_by_alignment_pairs(pred_seq1, pred_seq2, ref_seq1, ref_seq2):
    pred_pairs = extract_matching_pairs(pred_seq1, pred_seq2)
    ref_pairs = extract_matching_pairs(ref_seq1, ref_seq2)
    tp = len(pred_pairs & ref_pairs)
    fp = len(pred_pairs - ref_pairs)
    fn = len(ref_pairs - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


# def needleman_wunsch(seq1, seq2, gap_open=-10, gap_extend=-1):
#     """Needleman-Wunsch algorithm with BLOSUM62 and affine gap penalties."""
#     mat = blosum62
#     n, m = len(seq1), len(seq2)
#     score = np.zeros((n+1, m+1))
#     traceback = np.zeros((n+1, m+1), dtype=int)
#
#     for i in range(1, n+1):
#         score[i][0] = gap_open + (i - 1) * gap_extend
#     for j in range(1, m+1):
#         score[0][j] = gap_open + (j - 1) * gap_extend
#
#     for i in range(1, n+1):
#         for j in range(1, m+1):
#             match_score = mat[seq1[i - 1]].get(seq2[j - 1], -4)  # 忽略不存在情况默认-4
#             match = score[i - 1][j - 1] + match_score
#             delete = score[i-1][j] + gap_extend
#             insert = score[i][j-1] + gap_extend
#             score[i][j] = max(match, delete, insert)
#             if score[i][j] == match:
#                 traceback[i][j] = 0  # diag
#             elif score[i][j] == delete:
#                 traceback[i][j] = 1  # up
#             else:
#                 traceback[i][j] = 2  # left
#
#     # traceback
#     align1, align2 = "", ""
#     i, j = n, m
#     while i > 0 or j > 0:
#         if i > 0 and j > 0 and traceback[i][j] == 0:
#             align1 = seq1[i-1] + align1
#             align2 = seq2[j-1] + align2
#             i -= 1
#             j -= 1
#         elif i > 0 and (j == 0 or traceback[i][j] == 1):
#             align1 = seq1[i-1] + align1
#             align2 = "-" + align2
#             i -= 1
#         else:
#             align1 = "-" + align1
#             align2 = seq2[j-1] + align2
#             j -= 1
#
#     return align1, align2, score[n][m]

def needleman_wunsch_affine(seq1, seq2, gap_open=-10, gap_extend=-1):
    """严格标准的 Needleman-Wunsch 算法，使用 affine gap penalties"""
    n, m = len(seq1), len(seq2)

    # 三个 DP 矩阵
    M = np.full((n+1, m+1), -np.inf)  # match/mismatch
    X = np.full((n+1, m+1), -np.inf)  # gap in seq2
    Y = np.full((n+1, m+1), -np.inf)  # gap in seq1

    # 回溯矩阵：0=M, 1=X, 2=Y
    trace = np.zeros((n+1, m+1), dtype=int)

    # 初始化
    M[0][0] = 0
    for i in range(1, n+1):
        X[i][0] = gap_open + (i-1) * gap_extend
    for j in range(1, m+1):
        Y[0][j] = gap_open + (j-1) * gap_extend

    # DP 填表
    for i in range(1, n+1):
        for j in range(1, m+1):
            # substitution score
            s = blosum62.get(seq1[i-1], {}).get(seq2[j-1], -4)

            # M[i][j]
            M[i][j] = max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]) + s

            # X[i][j] (gap in seq2)
            X[i][j] = max(
                M[i-1][j] + gap_open + gap_extend,
                X[i-1][j] + gap_extend
            )

            # Y[i][j] (gap in seq1)
            Y[i][j] = max(
                M[i][j-1] + gap_open + gap_extend,
                Y[i][j-1] + gap_extend
            )

    # 选择最优得分
    matrices = [M[n][m], X[n][m], Y[n][m]]
    state = int(np.argmax(matrices))
    score = matrices[state]

    # 回溯
    align1, align2 = "", ""
    i, j = n, m
    while i > 0 or j > 0:
        if state == 0:  # M
            s = blosum62.get(seq1[i-1], {}).get(seq2[j-1], -4)
            if M[i][j] == M[i-1][j-1] + s:
                state = 0
            elif M[i][j] == X[i-1][j-1] + s:
                state = 1
            else:
                state = 2
            align1 = seq1[i-1] + align1
            align2 = seq2[j-1] + align2
            i -= 1
            j -= 1

        elif state == 1:  # X (gap in seq2)
            if X[i][j] == M[i-1][j] + gap_open + gap_extend:
                state = 0
            else:
                state = 1
            align1 = seq1[i-1] + align1
            align2 = "-" + align2
            i -= 1

        else:  # Y (gap in seq1)
            if Y[i][j] == M[i][j-1] + gap_open + gap_extend:
                state = 0
            else:
                state = 2
            align1 = "-" + align1
            align2 = seq2[j-1] + align2
            j -= 1

    return align1, align2, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Needleman-Wunsch alignment on a TFA file.")
    parser.add_argument("tfa_path", type=str, help="Path to the input TFA file")
    parser.add_argument("output_path", type=str, help="Path to save the aligned result")
    parser.add_argument("--gap_open", type=float, default=-10, help="Gap opening penalty (default: -10)")
    parser.add_argument("--gap_extend", type=float, default=-1, help="Gap extension penalty (default: -1)")

    args = parser.parse_args()

    nw_single_align(
        tfa_path=args.tfa_path,
        output_path=args.output_path,
        gap_open=args.gap_open,
        gap_extend=args.gap_extend
    )
