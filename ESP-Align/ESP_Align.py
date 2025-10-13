import torch
import method_new
import extract_emd
import similarity_matrix as sm
import os
import argparse
import pandas as pd
import Second_structure as S_s
from Bio import AlignIO
import matplotlib.pyplot as plt
import numpy as np


def compute_similarity_score(seq1, seq2, pdb_path, pearson_weight, Helix, Strand, Coil, gap_ext):

    similarity_matrix, z_score_similarity_matrix = sm.compute_pearson_new_similarity_matrix(seq1[2], seq2[2])
    print(similarity_matrix.shape)
    if pdb_path in [None, "", "None"]:
        print("[INFO] Using ESMFold to predict structures on-the-fly...")
        ss_string1, ss_string2, structure_score_matrix, gap_open_vector1, gap_open_vector2 = S_s.Stride_run(seq1[0], seq2[0], Helix, Strand, Coil)
    else:
        pdb1 = os.path.join(pdb_path, f"{seq1[1]}.pdb")
        pdb2 = os.path.join(pdb_path, f"{seq2[1]}.pdb")
        print(f"[INFO] Using precomputed PDBs: {pdb1}, {pdb2}")
        ss_string1, ss_string2, structure_score_matrix, gap_open_vector1, gap_open_vector2 = S_s.Stride_run_pdb(pdb1, pdb2, Helix, Strand, Coil)
    p_min_clamp = max(z_score_similarity_matrix.min().item(), -3)
    p_max_clamp = min(z_score_similarity_matrix.max().item(), +3)
    structure_score_matrix = p_min_clamp + (structure_score_matrix + 3.0) / 6.0 * (p_max_clamp - p_min_clamp)
    # structure_score_matrix = z_score_global(structure_score_matrix)
    results = method_new.Needleman_Wunsch_New(similarity_matrix, z_score_similarity_matrix, structure_score_matrix, gap_open_vector1, gap_open_vector2, pearson_weight, gap_ext)

    # combine_and_save_heatmap(z_score_similarity_matrix, structure_score_matrix)
    # return results
    return results, similarity_matrix, z_score_similarity_matrix, ss_string1, ss_string2, structure_score_matrix, gap_open_vector1, gap_open_vector2


def combine_and_save_heatmap(similarity_matrix, structure_score_matrix,
                             output_path="/public/home/lss/Protein_embedding/PLEA/ESP-Align/combined_heatmap.svg",
                             sim_weight=0.8, struct_weight=0.2):
    """
    根据 0.8*similarity_matrix + 0.2*structure_score_matrix 生成融合矩阵并保存热力图为 SVG
    行作为 x 轴，列作为 y 轴，颜色采用蓝-红渐变
    """
    # Step 1: 加权融合
    combined = sim_weight * similarity_matrix + struct_weight * structure_score_matrix

    # Step 2: 归一化到 [0, 1]
    min_val = combined.min()
    max_val = combined.max()
    norm_combined = (combined - min_val) / (max_val - min_val + 1e-8)

    # Step 3: 绘制热力图 (行作为x轴, 列作为y轴 -> 转置)
    plt.figure(figsize=(6, 6))
    plt.imshow(norm_combined.T.cpu().numpy(), cmap='viridis', vmin=0, vmax=1, origin="lower")
    plt.colorbar(label="Normalized Score")
    # plt.title("Combined Similarity + Structure Heatmap")
    plt.xlabel("Homo_sapiens_p53")
    plt.ylabel("Gallus_gallus_p53")
    plt.xticks(np.arange(0, combined.shape[0], 50))
    plt.yticks(np.arange(0, combined.shape[1], 50))
    plt.tight_layout()

    # 保存为 SVG
    plt.savefig(output_path, format="svg")
    print(f"✅ Heatmap saved to: {output_path}")
    plt.close()

    return norm_combined


def save_alignment_to_file(aln, seq1, seq2, saving_dir, seqs_path, ss_string1=None, ss_string2=None, width=60):
    """
    以带编号的块状格式将对齐结果保存到文本文件。
    每块宽度为 width，包含 Seq1、匹配行、Seq2（可选加结构注释行）。

    :param aln: 对齐结果，包含 'aln_1', 'aln_2' 等键
    :param seq1: 原始序列1字符串
    :param seq2: 原始序列2字符串
    :param ss_string1: 序列1的二级结构字符串（与seq1等长）
    :param ss_string2: 序列2的二级结构字符串（与seq2等长）
    :param saving_dir: 保存目录
    :param seqs_path: 原始输入序列文件路径（用于命名）
    :param width: 每块最大碱基数，默认60
    """
    import os

    # 如果保存目录不存在，则创建
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    # 生成保存文件名
    filename = os.path.join(
        saving_dir,
        seqs_path.split("/")[-1].split(".")[-2] + "_pearson_NW_SS.txt"
    )

    # 从对齐结果中提取各个参数
    aln1 = aln['aln_1']
    aln2 = aln['aln_2']
    score_raw = aln['score_raw']
    z_score_score = aln['z_score_score']
    score_global = aln['score_global']

    # 将对齐结果中的 -1 替换为 -，得到对齐后的序列
    aligned1 = ''.join('-' if i1 == -1 else seq1[i1] for i1 in aln1)
    aligned2 = ''.join('-' if i2 == -1 else seq2[i2] for i2 in aln2)
    # print(aln1)

    # 结构注释对齐（若提供）
    aligned_ss1 = ''.join('-' if i1 == -1 else ss_string1[i1] for i1 in aln1) if ss_string1 else None
    aligned_ss2 = ''.join('-' if i2 == -1 else ss_string2[i2] for i2 in aln2) if ss_string2 else None

    print(aligned1)
    print(aligned_ss1)
    print(aligned2)
    print(aligned_ss2)

    # 将对齐结果写入文件
    with open(filename, 'w') as f:
        f.write("Score:\n")
        f.write(f"Raw_Score: {score_raw}\n")
        f.write(f"Z_Score_Score: {z_score_score}\n")
        f.write(f"Score_Global: {score_global}\n\n")

        length = len(aligned1)
        for block_start in range(0, length, width):
            block_end = min(block_start + width, length)

            block_indices1 = [i for i in aln1[block_start:block_end] if i != -1]
            block_indices2 = [j for j in aln2[block_start:block_end] if j != -1]
            seq1_start = block_indices1[0] + 1 if block_indices1 else block_start + 1
            seq1_end = block_indices1[-1] + 1 if block_indices1 else block_start
            seq2_start = block_indices2[0] + 1 if block_indices2 else block_start + 1
            seq2_end = block_indices2[-1] + 1 if block_indices2 else block_start

            sub1 = aligned1[block_start:block_end]
            sub2 = aligned2[block_start:block_end]
            mid = ''.join(c1 if c1 == c2 else '-' for c1, c2 in zip(sub1, sub2))
            sub_ss1 = aligned_ss1[block_start:block_end] if aligned_ss1 else ''
            sub_ss2 = aligned_ss2[block_start:block_end] if aligned_ss2 else ''

            f.write(f"Seq 1 : {seq1_start:<4d} {sub1} {seq1_end:>4d}\n")
            if aligned_ss1:
                f.write(f"SS  1 :      {sub_ss1}\n")
            # f.write(f"            {mid}\n")
            f.write(f"Seq 2 : {seq2_start:<4d} {sub2} {seq2_end:>4d}\n")
            if aligned_ss2:
                f.write(f"SS  2 :      {sub_ss2}\n")
            f.write("\n")


ALIGNMENT_ALGORITHM = ["DWT", "SW", "NW"]

def extract_matching_pairs(seq1, seq2):
    """
    从对齐后的两条序列中提取 residue 对的 index 映射（跳过 gap）
    返回一个集合，包含形式如 (i, j) 的 tuple，表示原始序列中第 i 和第 j 个残基对齐
    """
    # 初始化两个索引，分别指向两个序列的起始位置
    i_idx, j_idx = 0, 0
    # 初始化一个空列表，用于存储匹配的 residue 对的 index
    match_pairs = []

    # 遍历两个序列的每一个字符
    for a, b in zip(seq1, seq2):
        # 如果两个字符都不是 gap，则将它们的 index 添加到 match_pairs 中
        if a != '-' and b != '-':
            match_pairs.append((i_idx, j_idx))
        # 如果第一个序列的字符不是 gap，则将第一个序列的索引加一
        if a != '-':
            i_idx += 1
        # 如果第二个序列的字符不是 gap，则将第二个序列的索引加一
        if b != '-':
            j_idx += 1
    # 返回一个集合，包含匹配的 residue 对的 index
    return set(match_pairs)

def get_match_labels_by_alignment_pairs(pred_seq1, pred_seq2, ref_seq1, ref_seq2):
    """
    使用 residue 索引的映射方式构建 y_true 和 y_pred
    """
    pred_pairs = extract_matching_pairs(pred_seq1, pred_seq2)
    ref_pairs = extract_matching_pairs(ref_seq1, ref_seq2)

    # 预测对中正确的数量：TP
    tp = len(pred_pairs & ref_pairs)
    # 预测对中不在参考中的：FP
    fp = len(pred_pairs - ref_pairs)
    # 参考中存在但预测缺失的：FN
    fn = len(ref_pairs - pred_pairs)

    # 避免除零
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1

def get_reference_alignment(ref_fasta_path, seq_name1, seq_name2):
    """
    从参考 MSA 中提取指定两个序列的对齐结果，并去除全 gap 列。
    """
    alignment = AlignIO.read(ref_fasta_path, "fasta")

    # 提取两个序列
    seq_dict = {record.id: str(record.seq) for record in alignment}
    if seq_name1 not in seq_dict or seq_name2 not in seq_dict:
        raise ValueError(f"{seq_name1} or {seq_name2} not found in reference MSA.")

    aln1 = seq_dict[seq_name1]
    aln2 = seq_dict[seq_name2]

    # 删除两个序列都是gap的位置
    cleaned1 = []
    cleaned2 = []
    for a, b in zip(aln1, aln2):
        if a == '-' and b == '-':
            continue
        cleaned1.append(a)
        cleaned2.append(b)

    ref1 = ''.join(cleaned1)
    ref2 = ''.join(cleaned2)
    return ref1, ref2


def main(seqs_path, pdb_path, ref_path, result_dir, pearson_weight=0.8 ,Helix=-5.0, Strand=-3.0, Coil=-1.00, gap_ext=0.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 提取序列
    try:
        seq1, seq2 = extract_emd.extract_emd(seqs_path)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    # 计算相似度分数
    alignment, sim_D, sim_D_z_score, ss_string1, ss_string2, ss_matrix, gap_open_vector1, gap_open_vector2 = compute_similarity_score(seq1, seq2, pdb_path, pearson_weight, Helix, Strand, Coil, gap_ext)
    print("ss_matrix shape before save:", ss_matrix.shape)
    print("seq1 length before save:", len(seq1[0]))
    print("seq2 length before save:", len(seq2[0]))
    aln1 = alignment['aln_1']
    aln2 = alignment['aln_2']
    pred1 = ''.join('-' if i == -1 else seq1[0][i] for i in aln1)
    pred2 = ''.join('-' if i == -1 else seq2[0][i] for i in aln2)

    if ref_path not in [None, "", "None"]:
        ref1, ref2 = get_reference_alignment(ref_path, seq1[1], seq2[1])
        f1 = get_match_labels_by_alignment_pairs(pred1, pred2, ref1, ref2)
        print("\n")
        print("ref1:", ref1)
        print("\n")
        print("ref2:", ref2)
        print(f"Match labels F1 score: {f1}")
        save_alignment_to_file(alignment, seq1[0], seq2[0], result_dir, seqs_path, ss_string1, ss_string2)
        return f1
    print("pred1:", pred1)
    print("\n")
    print("pred2:", pred2)

    # 保存结果
    save_alignment_to_file(alignment, seq1[0], seq2[0], result_dir, seqs_path, ss_string1, ss_string2)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Compute protein sequence similarity and alignment")
    parser.add_argument("--seqs_path", type=str, required=True, help="Path to the input sequences file (fasta format)")
    parser.add_argument("--pdb_path", type=str, default="", help="Path to the PDB files (leave blank to use ESMFold)")
    parser.add_argument("--ref_path", type=str, default="", help="Path to the reference fasta files")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save the results")
    args = parser.parse_args()

    # 调用主函数
    main(args.seqs_path, args.pdb_path, args.ref_path, args.result_dir)    # 写一段代码测试一下使用sw算法的比分结果


