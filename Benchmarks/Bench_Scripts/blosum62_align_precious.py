import os
import random
from tqdm import tqdm
from itertools import combinations
from Bio import SeqIO, AlignIO
import numpy as np
import csv
from blosum import BLOSUM

blosum62 = BLOSUM(62)  # 加载 BLOSUM62 矩阵

def load_fasta_nor(path):
    """
    加载FASTA文件，过滤非法氨基酸字符的序列。

    返回：
        合法的 (name, sequence) 元组列表
    """
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")  # 20种标准氨基酸
    sequences = []

    for record in SeqIO.parse(path, "fasta"):
        name = record.id
        sequence = str(record.seq).upper()
        invalid_chars = set(sequence) - valid_aas

        if invalid_chars:
            print(f"[WARNING] Illegal characters found in sequence '{name}' in {path}: {''.join(invalid_chars)}")
            continue  # 跳过含非法字符的序列

        sequences.append((name, sequence))

    return sequences

def load_reference_alignment(ref_fasta_path, id1, id2):
    """从参考比对中提取两条序列（参考比对格式为多序列FASTA）"""
    ref_dict = {r.id: str(r.seq) for r in SeqIO.parse(ref_fasta_path, "fasta")}
    if id1 in ref_dict and id2 in ref_dict:
        return ref_dict[id1], ref_dict[id2]
    else:
        raise ValueError(f"Missing reference alignment for {id1} or {id2}")


def process_all_pairs_blosum(tfa_dir, fasta_dir, output_csv_path, gap_open=-10, gap_extend=-1):
    total_f1, total_precision, total_recall = [], [], []
    tfa_files = [f for f in os.listdir(tfa_dir) if f.endswith(".tfa")]
    tfa_files.sort()

    # 如果 CSV 文件不存在，写入表头
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["TFA_File", "Avg_F1", "Avg_Precision", "Avg_Recall"])

    for tfa_file in tqdm(tfa_files, desc="Processing all TFA files"):
        tfa_path = os.path.join(tfa_dir, tfa_file)
        seq_records = load_fasta_nor(tfa_path)
        if len(seq_records) < 2:
            continue

        pairs = list(combinations(seq_records, 2))
        random.seed(17)
        random.shuffle(pairs)
        chosen_pairs = pairs[:100]

        f1_scores_this_file, precision_this_file, recall_this_file = [], [], []

        for (id1, seq1), (id2, seq2) in tqdm(chosen_pairs, desc=f"Processing {tfa_file}", leave=False):
            ref_fasta = os.path.join(fasta_dir, tfa_file.replace('.tfa', '.fasta'))
            try:
                # 运行 BLOSUM62 + NW 算法
                # 修改了
                pred_align1, pred_align2, _ = needleman_wunsch_affine(seq1, seq2, gap_open, gap_extend)
                print(pred_align1)
                print("\n")
                print(pred_align2)

                # 读取参考比对并获取对齐
                ref_align1, ref_align2 = get_reference_alignment(ref_fasta, id1, id2)

                # 计算 F1
                f1_score, precision_score, recall_score = get_match_labels_by_alignment_pairs(pred_align1, pred_align2, ref_align1, ref_align2)

                if f1_score is not None:
                    print(f"Pair {id1}-{id2} in {tfa_file}. F1 score: {f1_score:.4f}")
                    # ##
                    f1_scores_this_file.append(f1_score)
                    precision_this_file.append(precision_score)
                    recall_this_file.append(recall_score)
                else:
                    print(f"Pair {id1}-{id2} in {tfa_file} skipped.")

            except Exception as e:
                print(f"Failed on pair {id1}-{id2} in {tfa_file}: {e}")

        if f1_scores_this_file:
            avg_f1 = sum(f1_scores_this_file) / len(f1_scores_this_file)
            avg_precision = sum(precision_this_file) / len(precision_this_file)
            avg_recall = sum(recall_this_file) / len(recall_this_file)
        else:
            avg_f1, avg_precision, avg_recall = 0, 0, 0
        print(f"✅ Average F1 for {tfa_file}: {avg_f1:.4f}")

        total_f1.extend(f1_scores_this_file)
        total_precision.extend(precision_this_file)
        total_recall.extend(recall_this_file)

        # 立即写入该 tfa 的结果
        with open(output_csv_path, 'a', newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([tfa_file, avg_f1, avg_precision, avg_recall])

    if total_f1:
        # ##
        final_avg_f1 = sum(total_f1) / len(total_f1)
        final_avg_precision = sum(total_precision) / len(total_precision)
        final_avg_recall = sum(total_recall) / len(total_recall)
        print(
            f"\n✅ Overall Avg F1: {final_avg_f1:.4f}, Precision: {final_avg_precision:.4f}, Recall: {final_avg_recall:.4f}")
    else:
        final_avg_f1, final_avg_precision, final_avg_recall = 0, 0, 0
        print("❌ No successful scores computed.")

    # ##
    return final_avg_f1, final_avg_precision, final_avg_recall

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
    return f1, precision, recall


def needleman_wunsch(seq1, seq2, gap_open=-10, gap_extend=-1):
    """Needleman-Wunsch algorithm with BLOSUM62 and affine gap penalties."""
    mat = blosum62
    n, m = len(seq1), len(seq2)
    score = np.zeros((n+1, m+1))
    traceback = np.zeros((n+1, m+1), dtype=int)

    for i in range(1, n+1):
        score[i][0] = gap_open + (i - 1) * gap_extend
    for j in range(1, m+1):
        score[0][j] = gap_open + (j - 1) * gap_extend

    for i in range(1, n+1):
        for j in range(1, m+1):
            match_score = mat[seq1[i - 1]].get(seq2[j - 1], -4)  # 忽略不存在情况默认-4
            match = score[i - 1][j - 1] + match_score
            delete = score[i-1][j] + gap_extend
            insert = score[i][j-1] + gap_extend
            score[i][j] = max(match, delete, insert)
            if score[i][j] == match:
                traceback[i][j] = 0  # diag
            elif score[i][j] == delete:
                traceback[i][j] = 1  # up
            else:
                traceback[i][j] = 2  # left

    # traceback
    align1, align2 = "", ""
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and traceback[i][j] == 0:
            align1 = seq1[i-1] + align1
            align2 = seq2[j-1] + align2
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or traceback[i][j] == 1):
            align1 = seq1[i-1] + align1
            align2 = "-" + align2
            i -= 1
        else:
            align1 = "-" + align1
            align2 = seq2[j-1] + align2
            j -= 1

    return align1, align2, score[n][m]

# def needleman_wunsch_affine(seq1, seq2, gap_open=-10, gap_extend=-1):
#     """Needleman-Wunsch algorithm with affine gap penalties."""
#     n, m = len(seq1), len(seq2)
#
#     # 初始化三个矩阵
#     M = np.full((n+1, m+1), -np.inf)
#     X = np.full((n+1, m+1), -np.inf)
#     Y = np.full((n+1, m+1), -np.inf)
#
#     # 回溯矩阵 (0=M,1=X,2=Y)
#     trace_M = np.zeros((n+1, m+1), dtype=int)
#     trace_X = np.zeros((n+1, m+1), dtype=int)
#     trace_Y = np.zeros((n+1, m+1), dtype=int)
#
#     # 初始化
#     M[0][0] = 0
#     for i in range(1, n+1):
#         X[i][0] = gap_open + (i-1)*gap_extend
#     for j in range(1, m+1):
#         Y[0][j] = gap_open + (j-1)*gap_extend
#
#     # 递推
#     for i in range(1, n+1):
#         for j in range(1, m+1):
#             # substitution score
#             s = blosum62.get(seq1[i-1], {}).get(seq2[j-1], -4)
#
#             # M[i][j]
#             scores = [M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]]
#             best = np.argmax(scores)
#             M[i][j] = scores[best] + s
#             trace_M[i][j] = best
#
#             # X[i][j] (gap in seq2)
#             scores = [
#                 M[i-1][j] + gap_open + gap_extend,
#                 X[i-1][j] + gap_extend,
#                 Y[i-1][j] + gap_open + gap_extend
#             ]
#             best = np.argmax(scores)
#             X[i][j] = scores[best]
#             trace_X[i][j] = best
#
#             # Y[i][j] (gap in seq1)
#             scores = [
#                 M[i][j-1] + gap_open + gap_extend,
#                 Y[i][j-1] + gap_extend,
#                 X[i][j-1] + gap_open + gap_extend
#             ]
#             best = np.argmax(scores)
#             Y[i][j] = scores[best]
#             trace_Y[i][j] = best
#
#     # 最优得分
#     matrices = [M[n][m], X[n][m], Y[n][m]]
#     state = np.argmax(matrices)
#     score = matrices[state]
#
#     # 回溯
#     align1, align2 = "", ""
#     i, j = n, m
#     while i > 0 or j > 0:
#         if state == 0:  # M
#             prev_state = trace_M[i][j]
#             align1 = seq1[i-1] + align1
#             align2 = seq2[j-1] + align2
#             i -= 1
#             j -= 1
#             state = prev_state
#         elif state == 1:  # X (gap in seq2)
#             prev_state = trace_X[i][j]
#             align1 = seq1[i-1] + align1
#             align2 = "-" + align2
#             i -= 1
#             state = prev_state
#         else:  # Y (gap in seq1)
#             prev_state = trace_Y[i][j]
#             align1 = "-" + align1
#             align2 = seq2[j-1] + align2
#             j -= 1
#             state = prev_state
#
#     return align1, align2, score

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

CDD_sets = {
    "CDD_Tier1": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/NW/Tire1.csv"
    },
    "CDD_Tier2": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/NW/Tire2.csv"
    },
    "CDD_Tier3": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/NW/Tire3.csv"
        },
    "CDD_Tier4": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/NW/Tire4.csv"
        },
}


test_sets = {
    # "CDD": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD/CDD_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD/CDD_ref",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/BLOSUM/CDD_affine_new.csv"
    # },
    # "RV11": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV11_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV11_12_ref",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/BLOSUM/RV11_affine_new.csv"
    # },
    # "RV12": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV12_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV11_12_ref",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/BLOSUM/RV12_affine_new.csv"
    # },
    # "RV911": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV911/RV911_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV911/RV911_ref",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/BLOSUM/RV911_affine_new.csv"
    # },
    # "RV912": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV912/RV912_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV912/RV912_ref",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/BLOSUM/RV912_affine_new.csv"
    # },
    # "RV913": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_ref",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/BLOSUM/RV913_affine_new.csv"
    # },
    # "CDD": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_25/CDD_25_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD/CDD_25_ref",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/BLOSUM/CDD_25.csv"
    # },
    "CDD_fix": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_fix/CDD_fix_tfa",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_fix/CDD_fix_ref",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/BLOSUM/CDD_fix.csv"
    },
    # "RV913": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/BOX214",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_pdb",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_emb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/ESP_Align/BOX214.csv"
    # },
    # "Case_protein": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Case/PLD",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Case_protein/Case_protein_pdb",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Case",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Case_protein/Case_protein_emb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Case/PLD.csv"
    #     },
    # "Covid_protein": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Case/Covid_19",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Case_protein/Case_protein_pdb",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Case",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Case_protein/Case_protein_emb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Case/Covid.csv"
    # },
}

import concurrent.futures

def run_one_blosum_set(set_name, paths, output_file):
    print(f"\n>>> Running test set: {set_name}")
    try:
        avg_f1, avg_precision, avg_recall = process_all_pairs_blosum(
            tfa_dir=paths["tfa_dir"],
            fasta_dir=paths["fasta_dir"],
            output_csv_path=paths["output"],
        )

        result_line = (
            f"{set_name} average scores -> "
            f"F1: {avg_f1:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}"
        )
    except Exception as e:
        result_line = f"{set_name} failed due to error: {e}"

    print(result_line)
    # 写入文件，加锁避免多进程写冲突
    with open(output_file, "a") as f:
        f.write("### " + result_line + "\n\n")
    return result_line


if __name__ == "__main__":
    output_file = "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/BLOSUMOSUM_result_new_precious_affine_new.txt"

    # 并行执行 test_sets
    max_workers = 6  # 根据CPU核数调整
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_one_blosum_set, set_name, paths, output_file)
            for set_name, paths in CDD_sets.items()
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                _ = future.result()  # 触发异常抛出
            except Exception as e:
                print(f"⚠️ One set failed: {e}")

    print(f"\n✅ All test sets complete. Results written to {output_file}")


