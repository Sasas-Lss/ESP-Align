import subprocess
import tempfile
import os
from Bio import SeqIO

import os
import random
from tqdm import tqdm
from itertools import combinations
from Bio import SeqIO, AlignIO
import numpy as np
import csv
import shutil


def get_alignment_from_foldseek(pdb1_path, pdb2_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        q_pdb = os.path.join(tmpdir, "q.pdb")
        t_pdb = os.path.join(tmpdir, "t.pdb")
        aln_output = os.path.join(tmpdir, "alignment.txt")

        shutil.copy(pdb1_path, q_pdb)
        shutil.copy(pdb2_path, t_pdb)

        subprocess.run([
            "foldseek", "easy-search",
            q_pdb, t_pdb, aln_output, tmpdir,
            "--alignment-type", "1",
            "--format-output", "qaln,taln",
            "-c", "0.0",
            "--cov-mode", "2"
        ], check=True)

        with open(aln_output, "r") as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) >= 2:
                    return fields[0], fields[1]

        raise RuntimeError("❌ No alignment found in Foldseek output.")


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


def process_all_pairs_foldseek(tfa_dir, fasta_dir, pdb_dir, output_csv_path):
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
                pdb1_path = os.path.join(pdb_dir, tfa_file.replace('.tfa', ''), f"{id1}.pdb")
                pdb2_path = os.path.join(pdb_dir, tfa_file.replace('.tfa', ''), f"{id2}.pdb")

                pred_align1, pred_align2 = get_alignment_from_foldseek(pdb1_path, pdb2_path)
                ref_align1, ref_align2 = get_reference_alignment(ref_fasta, id1, id2)
                ref_subseq1, ref_subseq2 = crop_reference_to_foldseek_region(
                    ref_align1, ref_align2, seq1, seq2, pred_align1, pred_align2
                )

                raw_f1, raw_precision, raw_recall = get_match_labels_by_alignment_pairs(
                    pred_align1, pred_align2, ref_subseq1, ref_subseq2
                )

                coverage1 = len(pred_align1.replace("-", "")) / len(seq1)
                coverage2 = len(pred_align2.replace("-", "")) / len(seq2)
                coverage = (coverage1 + coverage2) / 2

                f1_score = raw_f1 * coverage
                precision = raw_precision * coverage
                recall = raw_recall * coverage

                f1_scores_this_file.append(f1_score)
                precision_this_file.append(precision)
                recall_this_file.append(recall)

            except Exception as e:
                print(f"❌ Failed on pair {id1}-{id2} in {tfa_file}: {e}")

        # 当前文件的平均分
        if f1_scores_this_file:
            avg_f1 = sum(f1_scores_this_file) / len(f1_scores_this_file)
            avg_precision = sum(precision_this_file) / len(precision_this_file)
            avg_recall = sum(recall_this_file) / len(recall_this_file)
        else:
            avg_f1, avg_precision, avg_recall = 0, 0, 0

        total_f1.extend(f1_scores_this_file)
        total_precision.extend(precision_this_file)
        total_recall.extend(recall_this_file)

        # 写入每个 TFA 文件的结果
        with open(output_csv_path, 'a', newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([tfa_file, avg_f1, avg_precision, avg_recall])

    # 输出总平均值
    if total_f1:
        final_avg_f1 = sum(total_f1) / len(total_f1)
        final_avg_precision = sum(total_precision) / len(total_precision)
        final_avg_recall = sum(total_recall) / len(total_recall)
        print(f"\n✅ Overall Avg F1: {final_avg_f1:.4f}, Precision: {final_avg_precision:.4f}, Recall: {final_avg_recall:.4f}")
    else:
        final_avg_f1, final_avg_precision, final_avg_recall = 0, 0, 0
        print("❌ No successful scores computed.")

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


def crop_reference_to_foldseek_region(ref_seq1, ref_seq2, full_seq1, full_seq2, qaln, taln):
    frag1 = qaln.replace("-", "")
    frag2 = taln.replace("-", "")

    start1 = full_seq1.find(frag1)
    start2 = full_seq2.find(frag2)

    if start1 == -1 or start2 == -1:
        raise ValueError("Foldseek fragment not found in full sequence")

    # 截取参考比对中相同区域
    i_idx, j_idx = 0, 0
    ref_subseq1, ref_subseq2 = "", ""
    for a, b in zip(ref_seq1, ref_seq2):
        in_range = (start1 <= i_idx < start1 + len(frag1)) and (start2 <= j_idx < start2 + len(frag2))
        if a != "-": i_idx += 1
        if b != "-": j_idx += 1
        if in_range:
            ref_subseq1 += a
            ref_subseq2 += b

    return ref_subseq1, ref_subseq2


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

CDD_sets = {
    "CDD_Tier1": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/Foldseek/Tire1.csv"
    },
    "CDD_Tier2": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/Foldseek/Tire2.csv"
    },
    "CDD_Tier3": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/Foldseek/Tire3.csv"
        },
    "CDD_Tier4": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/Foldseek/Tire4.csv"
        },
}


test_sets = {
    # "CDD": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD/CDD_tfa",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD/CDD_pdb",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD/CDD_ref",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/Foldseek/CDD.csv"
    # },
    # "RV11": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV11_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV11_12_ref",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/esmfold_results_RV11_12",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/Foldseek/RV11_new.csv"
    # },
    # "RV12": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV12_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV11_12_ref",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/esmfold_results_RV11_12",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/Foldseek/RV12_new.csv"
    # },
    # "RV911": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV911/RV911_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV911/RV911_ref",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV911/RV911_pdb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/Foldseek/RV911_new.csv"
    # },
    # "RV912": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV912/RV912_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV912/RV912_ref",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV912/RV912_pdb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/Foldseek/RV912_new.csv"
    # },
    # "RV913": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_ref",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_pdb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/Foldseek/RV913_new.csv"
    # },
    # "CDD": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_25/CDD_25_tfa",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_25/CDD_25_pdb",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_25/CDD_25_ref",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/Foldseek/CDD_new.csv"
    # },
    "CDD_fix": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_fix/CDD_fix_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_fix/CDD_fix_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_fix/CDD_fix_ref",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/Foldseek/CDD_fix.csv"
    }
}

import concurrent.futures

def run_one_set(set_name, paths, output_file):
    print(f"\n>>> Running test set: {set_name}")
    try:
        avg_f1, avg_precision, avg_recall = process_all_pairs_foldseek(
            tfa_dir=paths["tfa_dir"],
            fasta_dir=paths["fasta_dir"],
            pdb_dir=paths["pdb_dir"],
            output_csv_path=paths["output"],
        )

        result_line = (
            f"{set_name} average scores -> "
            f"F1: {avg_f1:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}"
        )
    except Exception as e:
        result_line = f"{set_name} failed due to error: {e}"

    print(result_line)
    with open(output_file, "a") as f:
        f.write("### " + result_line + "\n\n")
    return result_line


if __name__ == "__main__":
    output_file = "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/Foldseek_result_new.txt"

    # 使用多进程并行运行所有 test_sets
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(run_one_set, set_name, paths, output_file)
            for set_name, paths in CDD_sets.items()
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                _ = future.result()  # 触发异常抛出
            except Exception as e:
                print(f"⚠️ One set failed: {e}")

    print(f"\n✅ All test sets complete. Results written to {output_file}")
