import subprocess
import torch


def predict_structure_with_model(sequence, model, out_pdb="example_protein.pdb"):
    """
    Predict 3D protein structure using ESMFold and save as PDB file.

    Args:
        sequence (str): Protein amino acid sequence.
        model: Preloaded ESMFold model.
        out_pdb (str): Output PDB file path.

    Returns:
        str: Output PDB file path.
    """
    print("[INFO] Predicting structure with ESMFold...")
    model = model.eval().cuda() if torch.cuda.is_available() else model.eval().float()

    with torch.no_grad():
        pdb_str = model.infer_pdb(sequence)

    with open(out_pdb, "w") as f:
        f.write(pdb_str)

    print(f"[INFO] Structure written to {out_pdb}")
    return out_pdb


def run_stride(pdb_file, stride_path="stride", temp_out="stride_temp.out"):
    """
    Extract residue-level secondary structure annotations using STRIDE.

    Args:
        pdb_file (str): Path to input PDB file.
        stride_path (str): Path to STRIDE executable.
        temp_out (str): Temporary output file path.

    Returns:
        list[tuple]: List of (index, res_name, chain_id, res_num, structure_label).
    """
    with open(temp_out, "w") as f_out:
        subprocess.run([stride_path, pdb_file], stdout=f_out, check=True)

    residues = []
    with open(temp_out, "r") as f:
        index = 0
        for line in f:
            if line.startswith("ASG"):
                tokens = line.split()
                if len(tokens) >= 7:
                    res_name = tokens[1]
                    chain_id = tokens[2]
                    res_num = int(tokens[3])
                    full_structure = tokens[5]
                    residues.append((index, res_name, chain_id, res_num, full_structure))
                    index += 1
    return residues


def build_structure_score_matrix(ss1, ss2, Helix, Strand, Coil):
    """
    Build structure-aware match score matrix and dynamic gap penalties.

    Args:
        ss1 (str): Secondary structure sequence 1 (e.g., "HHHCCEE...").
        ss2 (str): Secondary structure sequence 2.
        Helix (float): Gap open penalty for helix region.
        Strand (float): Gap open penalty for strand region.
        Coil (float): Gap open penalty for coil region.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            structure_score_matrix, gap_open_vector1, gap_open_vector2
    """
    class_map = {
        "H": "Helix", "G": "Helix", "I": "Helix",
        "E": "Strand", "B": "Strand",
        "T": "Coil", "C": "Coil", "S": "Coil"
    }

    gap_penalty_by_class = {
        "Helix": Helix,
        "Strand": Strand,
        "Coil": Coil,
        "Other": Coil,
    }

    m, n = len(ss1), len(ss2)
    score_matrix = torch.zeros((m, n))
    gap_open_vector1 = torch.zeros(m)
    gap_open_vector2 = torch.zeros(n)

    for i in range(m):
        gap_open_vector1[i] = gap_penalty_by_class.get(class_map.get(ss1[i], "Other"))
    for j in range(n):
        gap_open_vector2[j] = gap_penalty_by_class.get(class_map.get(ss2[j], "Other"))

    for i in range(m):
        for j in range(n):
            c1 = class_map.get(ss1[i], "Other")
            c2 = class_map.get(ss2[j], "Other")

            if c1 == c2:
                score_matrix[i, j] = 3.0 if ss1[i] == ss2[j] else 1.0
            else:
                pair = {c1, c2}
                if pair == {"Helix", "Strand"}:
                    score_matrix[i, j] = -3.0
                elif pair == {"Helix", "Coil"}:
                    score_matrix[i, j] = -2.0
                elif pair == {"Strand", "Coil"}:
                    score_matrix[i, j] = -1.0
                else:
                    score_matrix[i, j] = -1.0

    return score_matrix, gap_open_vector1, gap_open_vector2


def Stride_run(seq1, seq2, Helix, Strand, Coil):
    """
    Predict structures for two sequences, extract secondary structure strings,
    and build structure-aware score matrices.

    Args:
        seq1 (str): Protein sequence 1.
        seq2 (str): Protein sequence 2.
        Helix, Strand, Coil (float): Gap penalties.

    Returns:
        tuple[str, str, torch.Tensor, torch.Tensor, torch.Tensor]
    """
    from esm import pretrained
    print("[INFO] Loading ESMFold model once...")
    model = pretrained.esmfold_v1()
    model = model.eval().cuda() if torch.cuda.is_available() else model.eval().float()

    pdb_path1 = predict_structure_with_model(seq1, model, out_pdb="protein1.pdb")
    ss_string1 = "".join([x[4] for x in run_stride(pdb_path1)])

    pdb_path2 = predict_structure_with_model(seq2, model, out_pdb="protein2.pdb")
    ss_string2 = "".join([x[4] for x in run_stride(pdb_path2)])

    structure_score_matrix, gap_open_vector1, gap_open_vector2 = build_structure_score_matrix(ss_string1, ss_string2,
                                                                                              Helix, Strand, Coil)
    return ss_string1, ss_string2, structure_score_matrix, gap_open_vector1, gap_open_vector2


def Stride_run_pdb(pdb_path1, pdb_path2, Helix, Strand, Coil):
    """
    Extract secondary structure info directly from two PDB files.

    Args:
        pdb_path1 (str): Path to first PDB file.
        pdb_path2 (str): Path to second PDB file.

    Returns:
        tuple[str, str, torch.Tensor, torch.Tensor, torch.Tensor]
    """
    ss_string1 = "".join([x[4] for x in run_stride(pdb_path1)])
    ss_string2 = "".join([x[4] for x in run_stride(pdb_path2)])
    structure_score_matrix, gap_open_vector1, gap_open_vector2 = build_structure_score_matrix(ss_string1, ss_string2,
                                                                                              Helix, Strand, Coil)
    return ss_string1, ss_string2, structure_score_matrix, gap_open_vector1, gap_open_vector2


if __name__ == "__main__":
    seq1 = "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWGDKSAVRALYDAIKKVIAEKTKPKG"
    seq2 = "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWGDKSAV"
    ss1, ss2, mat, gap1, gap2 = Stride_run(seq1, seq2, Helix=-3.0, Strand=-2.0, Coil=-1.0)
    print("[RESULT] Secondary structure 1:", ss1)
    print("[RESULT] Secondary structure 2:", ss2)
