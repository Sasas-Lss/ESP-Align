import os
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import AlignIO

import method
import extract_emd
import similarity_matrix as sm
import Second_structure as S_s


def compute_similarity_score(seq1, seq2, pdb_path, pearson_weight, Helix, Strand, Coil, gap_ext):
    """
    Compute similarity and structure matrices, then run alignment using structure-aware Needleman-Wunsch.
    """
    similarity_matrix, z_score_similarity_matrix = sm.compute_pearson_new_similarity_matrix(seq1[2], seq2[2])
    print(similarity_matrix.shape)

    if pdb_path in [None, "", "None"]:
        print("[INFO] Using ESMFold to predict structures on-the-fly...")
        ss_string1, ss_string2, structure_score_matrix, gap_open_vector1, gap_open_vector2 = S_s.Stride_run(
            seq1[0], seq2[0], Helix, Strand, Coil
        )
    else:
        pdb1 = os.path.join(pdb_path, f"{seq1[1]}.pdb")
        pdb2 = os.path.join(pdb_path, f"{seq2[1]}.pdb")
        print(f"[INFO] Using precomputed PDBs: {pdb1}, {pdb2}")
        ss_string1, ss_string2, structure_score_matrix, gap_open_vector1, gap_open_vector2 = S_s.Stride_run_pdb(
            pdb1, pdb2, Helix, Strand, Coil
        )

    # Scale structure matrix to match Pearson range
    p_min_clamp = max(z_score_similarity_matrix.min().item(), -3)
    p_max_clamp = min(z_score_similarity_matrix.max().item(), +3)
    structure_score_matrix = p_min_clamp + (structure_score_matrix + 3.0) / 6.0 * (p_max_clamp - p_min_clamp)

    results = method.Needleman_Wunsch_New(
        similarity_matrix, z_score_similarity_matrix, structure_score_matrix,
        gap_open_vector1, gap_open_vector2, pearson_weight, gap_ext
    )

    return results, similarity_matrix, z_score_similarity_matrix, ss_string1, ss_string2, structure_score_matrix, gap_open_vector1, gap_open_vector2


def save_alignment_to_file(aln, seq1, seq2, saving_dir, seqs_path, ss_string1=None, ss_string2=None, width=60):
    """
    Save alignment result in a block format with optional secondary structure annotation.
    """
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    filename = os.path.join(
        saving_dir,
        seqs_path.split("/")[-1].split(".")[-2] + "ESP_Align_Result.txt"
    )

    aln1 = aln['aln_1']
    aln2 = aln['aln_2']
    score_raw = aln['score_raw']
    z_score_score = aln['z_score_score']
    score_global = aln['score_global']
    print("score_raw:", score_raw)
    print("score_global:", score_global)
    aligned1 = ''.join('-' if i1 == -1 else seq1[i1] for i1 in aln1)
    aligned2 = ''.join('-' if i2 == -1 else seq2[i2] for i2 in aln2)
    aligned_ss1 = ''.join('-' if i1 == -1 else ss_string1[i1] for i1 in aln1) if ss_string1 else None
    aligned_ss2 = ''.join('-' if i2 == -1 else ss_string2[i2] for i2 in aln2) if ss_string2 else None

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
            sub_ss1 = aligned_ss1[block_start:block_end] if aligned_ss1 else ''
            sub_ss2 = aligned_ss2[block_start:block_end] if aligned_ss2 else ''

            f.write(f"Seq 1 : {seq1_start:<4d} {sub1} {seq1_end:>4d}\n")
            if aligned_ss1:
                f.write(f"SS  1 :      {sub_ss1}\n")
            f.write(f"Seq 2 : {seq2_start:<4d} {sub2} {seq2_end:>4d}\n")
            if aligned_ss2:
                f.write(f"SS  2 :      {sub_ss2}\n")
            f.write("\n")


def extract_matching_pairs(seq1, seq2):
    """
    Extract residue index pairs (i, j) for aligned residues (excluding gaps).
    """
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
    """
    Compute F1-score by comparing predicted and reference alignment residue pairs.
    """
    pred_pairs = extract_matching_pairs(pred_seq1, pred_seq2)
    ref_pairs = extract_matching_pairs(ref_seq1, ref_seq2)

    tp = len(pred_pairs & ref_pairs)
    fp = len(pred_pairs - ref_pairs)
    fn = len(ref_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


def get_reference_alignment(ref_fasta_path, seq_name1, seq_name2):
    """
    Extract reference alignment for the two target sequences and remove all-gap columns.
    """
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


def main(seqs_path, pdb_path, result_dir,
         pearson_weight=0.8, Helix=-5.0, Strand=-3.0, Coil=-1.0, gap_ext=0.0):
    """
    Main pipeline: extract embeddings, compute similarity, perform structure-aware alignment, and save results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        seq1, seq2 = extract_emd.extract_emd(seqs_path)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    alignment, sim_D, sim_D_z, ss_string1, ss_string2, ss_matrix, gap_open_vector1, gap_open_vector2 = \
        compute_similarity_score(seq1, seq2, pdb_path, pearson_weight, Helix, Strand, Coil, gap_ext)

    # aln1, aln2 = alignment['aln_1'], alignment['aln_2']
    # pred1 = ''.join('-' if i == -1 else seq1[0][i] for i in aln1)
    # pred2 = ''.join('-' if i == -1 else seq2[0][i] for i in aln2)
    # if ref_path not in [None, "", "None"]:
    #     ref1, ref2 = get_reference_alignment(ref_path, seq1[1], seq2[1])
    #     f1 = get_match_labels_by_alignment_pairs(pred1, pred2, ref1, ref2)
    #     print(f"\nF1-score against reference: {f1}\n")
    #     save_alignment_to_file(alignment, seq1[0], seq2[0], result_dir, seqs_path, ss_string1, ss_string2)
    #     return f1

    # Save alignment result
    save_alignment_to_file(alignment, seq1[0], seq2[0], result_dir, seqs_path, ss_string1, ss_string2)


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Run structure-aware alignment pipeline.")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file path")
    parser.add_argument("-p", "--pdb_path", default=None, help="Path to PDB files (optional, default: None)")
    parser.add_argument("-o", "--output", required=True, help="Directory to save results")

    # 新增的可调参数（带默认值）
    parser.add_argument("--pearson_weight", type=float, default=0.8, help="Weight for Pearson similarity (default: 0.8)")
    parser.add_argument("--Helix", type=float, default=-5.0, help="Score for Helix structure (default: -5.0)")
    parser.add_argument("--Strand", type=float, default=-3.0, help="Score for Strand structure (default: -3.0)")
    parser.add_argument("--Coil", type=float, default=-1.0, help="Score for Coil structure (default: -1.0)")
    parser.add_argument("--gap_ext", type=float, default=0.0, help="Gap extension penalty (default: 0.0)")

    args = parser.parse_args()

    main(
        seqs_path=args.input,
        pdb_path=args.pdb_path,
        result_dir=args.output,
        pearson_weight=args.pearson_weight,
        Helix=args.Helix,
        Strand=args.Strand,
        Coil=args.Coil,
        gap_ext=args.gap_ext
    )

