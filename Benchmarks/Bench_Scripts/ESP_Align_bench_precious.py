import sys
sys.path.append('/public/home/lss/Protein_embedding/PLEA/ESP-Align/ESP-Align')

import torch
import os
import random
from Bio import SeqIO, AlignIO
from itertools import combinations
from method_new import Needleman_Wunsch_New
from similarity_matrix import compute_pearson_new_similarity_matrix
from Second_structure import Stride_run_pdb
from tqdm import tqdm
import csv


def extract_matching_pairs(seq1, seq2):
    i_idx, j_idx = 0, 0
    match_pairs = []
    for a, b in zip(seq1, seq2):
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
    # ##
    return f1, precision, recall

def load_fasta(file_path):
    records = list(SeqIO.parse(file_path, "fasta"))
    return [(rec.id, str(rec.seq)) for rec in records]

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

def load_precomputed_embedding(seq1, seq2, id1, id2, tfa_file, embedding_dir):
    sub_dir = os.path.join(embedding_dir, tfa_file.replace('.tfa', ''))
    emb1_path = os.path.join(sub_dir, f"{id1}.pt")
    emb2_path = os.path.join(sub_dir, f"{id2}.pt")

    if not os.path.exists(emb1_path) or not os.path.exists(emb2_path):
        raise FileNotFoundError(f"Missing embedding for {id1} or {id2} in {sub_dir}")

    emb1 = torch.load(emb1_path)
    emb2 = torch.load(emb2_path)
    return [seq1, emb1], [seq2, emb2]

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

def compute_valid_ratio(ref1, ref2):
    assert len(ref1) == len(ref2)
    valid_count = sum((a != '-' and b != '-') for a, b in zip(ref1, ref2))
    return valid_count / len(ref1)
def process_alignment(seq1, seq2, emb1, emb2, pdb1_path, pdb2_path, ref_fasta_path, pearson_weight, Helix, Strand, Coil, gap_ext):
    pdb1_name = os.path.basename(pdb1_path).replace('.pdb', '')
    pdb2_name = os.path.basename(pdb2_path).replace('.pdb', '')
    ref1, ref2 = get_reference_alignment(ref_fasta_path, pdb1_name, pdb2_name)
    # ratio = compute_valid_ratio(ref1, ref2)
    # if ratio < 0.5:
    #     print(f"Skipping pair {pdb1_name}-{pdb2_name} due to low valid ratio: {ratio:.2f}")
    #     return None
    sim_mat, z_sim_mat = compute_pearson_new_similarity_matrix(emb1, emb2)
    ss1, ss2, ss_mat, gap1, gap2 = Stride_run_pdb(pdb1_path, pdb2_path, Helix, Strand, Coil)
    p_min, p_max = max(z_sim_mat.min().item(), -3), min(z_sim_mat.max().item(), 3)
    ss_mat = p_min + (ss_mat + 3.0) / 6.0 * (p_max - p_min)
    aln = Needleman_Wunsch_New(sim_mat, z_sim_mat, ss_mat, gap1, gap2, pearson_weight, gap_ext)
    aln1 = aln['aln_1']
    aln2 = aln['aln_2']
    pred1 = ''.join('-' if i == -1 else seq1[i] for i in aln1)
    pred2 = ''.join('-' if i == -1 else seq2[i] for i in aln2)
    # ss1_aln = ''.join('-' if i == -1 else ss1[i] for i in aln1)
    # ss2_aln = ''.join('-' if i == -1 else ss2[i] for i in aln2)
    # print(pred1)
    # print(ss1_aln)
    # print(pred2)
    # print(ss2_aln)
    # print(ref1)
    # print(ref2)

    # ##
    f1, precision, recall = get_match_labels_by_alignment_pairs(pred1, pred2, ref1, ref2)
    return f1, precision, recall


def process_all_pairs(tfa_dir, pdb_dir, fasta_dir, embedding_dir,
                      output_csv_path, pearson_weight,
                      Helix, Strand, Coil, gap_ext):
    # ##
    total_f1, total_precision, total_recall = [], [], []
    tfa_files = [f for f in os.listdir(tfa_dir) if f.endswith(".tfa")]
    tfa_files.sort()

    # 如果 CSV 文件不存在，写入表头
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv)
            # ##
            writer.writerow(["TFA_File", "Avg_F1", "Avg_Precision", "Avg_Recall"])

    for tfa_file in tqdm(tfa_files, desc="Processing all TFA files"):

        tfa_path = os.path.join(tfa_dir, tfa_file)
        seq_records = load_fasta_nor(tfa_path)
        if len(seq_records) < 2:
            print(f"❌ Not enough sequences in {tfa_file}. Skipping.")
            continue

        pairs = list(combinations(seq_records, 2))
        random.seed(17)
        random.shuffle(pairs)
        chosen_pairs = pairs[:100]
        # ##
        f1_scores_this_file, precision_this_file, recall_this_file = [], [], []

        for (id1, seq1), (id2, seq2) in tqdm(chosen_pairs, desc=f"Processing {tfa_file}", leave=False):
            pdb1 = os.path.join(pdb_dir, tfa_file.replace('.tfa', ''), f"{id1}.pdb")
            pdb2 = os.path.join(pdb_dir, tfa_file.replace('.tfa', ''), f"{id2}.pdb")
            ref_fasta = os.path.join(fasta_dir, tfa_file.replace('.tfa', '.fasta'))

            try:
                emb1_pack, emb2_pack = load_precomputed_embedding(seq1, seq2, id1, id2, tfa_file, embedding_dir)
                # ##
                f1_score, precision_score, recall_score = process_alignment(
                    emb1_pack[0], emb2_pack[0],
                    emb1_pack[1], emb2_pack[1],
                    pdb1, pdb2, ref_fasta, pearson_weight,
                    Helix, Strand, Coil, gap_ext
                )
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
        # ##
        if f1_scores_this_file:
            avg_f1 = sum(f1_scores_this_file) / len(f1_scores_this_file)
            avg_precision = sum(precision_this_file) / len(precision_this_file)
            avg_recall = sum(recall_this_file) / len(recall_this_file)
        else:
            avg_f1, avg_precision, avg_recall = 0, 0, 0
        print(f"✅ Average F1 for {tfa_file}: {avg_f1:.4f}")
        # ##
        total_f1.extend(f1_scores_this_file)
        total_precision.extend(precision_this_file)
        total_recall.extend(recall_this_file)

        # 立即写入该 tfa 的结果
        with open(output_csv_path, 'a', newline='') as f_csv:
            writer = csv.writer(f_csv)
            # ##
            writer.writerow([tfa_file, avg_f1, avg_precision, avg_recall])

    # ##
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


CDD_sets = {
    "CDD_Tier1": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/ESP_Align/Tire1.csv"
    },
    "CDD_Tier2": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/ESP_Align/Tire2.csv"
    },
    "CDD_Tier3": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/ESP_Align/Tire3.csv"
        },
    "CDD_Tier4": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/ESP_Align/Tire4.csv"
        },
}

test_sets = {
    # "CDD": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD/CDD_tfa",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD/CDD_pdb",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD/CDD_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD/CDD_emb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/ESP_Align/CDD.csv"
    # },
    # "RV11": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV11_tfa",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/esmfold_results_RV11_12",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV11_12_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/Embeddings",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/ESP_Align/RV11_new.csv"
    # },
    # "RV12": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV12_tfa",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/esmfold_results_RV11_12",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV11_12_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/Embeddings",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/ESP_Align/RV12_new.csv"
    # },
    # "RV911": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV911/RV911_tfa",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV911/RV911_pdb",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV911/RV911_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV911/RV911_emb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/ESP_Align/RV911_new.csv"
    # },
    # "RV912": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV912/RV912_tfa",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV912/RV912_pdb",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV912/RV912_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV912/RV912_emb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/ESP_Align/RV912_new.csv"
    # },
    # "RV913": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_tfa",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_pdb",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_emb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/ESP_Align/RV913_new.csv"
    # },
    # "CDD": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_25/CDD_25_tfa",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_25/CDD_25_pdb",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_25/CDD_25_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_25/CDD_25_emb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/ESP_Align/CDD_25.csv"
    # },
    "CDD_fix": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_fix/CDD_fix_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_fix/CDD_fix_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_fix/CDD_fix_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_fix/CDD_fix_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/ESP_Align/CDD_fix.csv"
    }
    # "RV913": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/BOX214",
    #     "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_pdb",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_emb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/ESP_Align/BOX214.csv"
    # },
}

if __name__ == "__main__":


    # ##
    output_file = "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/ESP_align_new_precious.txt"

    # 逐个测试并写入
    for set_name, paths in CDD_sets.items():
        print(f"\n>>> Running test set: {set_name}")

        try:
            # ##
            avg_f1, avg_precision, avg_recall = process_all_pairs(
                tfa_dir=paths["tfa_dir"],
                pdb_dir=paths["pdb_dir"],
                fasta_dir=paths["fasta_dir"],
                embedding_dir=paths["embedding_dir"],
                output_csv_path=paths["output"],
                pearson_weight=0.8, Helix=-5.0, Strand=-3.0, Coil=-1.0, gap_ext=0.0
            )
            # ##
            result_line = (
                f"{set_name} average scores -> "
                f"F1: {avg_f1:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}"
            )
        except Exception as e:
            result_line = f"{set_name} failed due to error: {e}"

        print(result_line)

        # 每个测试集完成后立即写入文件
        with open(output_file, "a") as f:
            f.write("### " + result_line + "\n\n")

    print(f"\n✅ All test sets complete. Results written to {output_file}")
