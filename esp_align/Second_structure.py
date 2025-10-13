import subprocess
import torch

def predict_structure_with_model(sequence, model, out_pdb="example_protein.pdb"):
    print("[INFO] Predicting structure with ESMFold...")
    model = model.eval().cuda() if torch.cuda.is_available() else model.eval().float()

    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open(out_pdb, "w") as f:
        f.write(output)

    print(f"[INFO] Structure written to {out_pdb}")
    return out_pdb
def run_stride(pdb_file, stride_path='stride', temp_out='stride_temp.out'):
    """
    使用 STRIDE 提取每个残基的完整二级结构注释，并返回从0开始的索引排序。

    参数：
        pdb_file: 输入的 PDB 文件路径。
        stride_path: STRIDE 可执行文件路径。
        temp_out: 暂存 STRIDE 输出文件路径。

    返回：
        residues: 一个列表，每个元素是 (index, res_name, chain_id, res_num, full_structure) 元组。
                  index 从 0 开始编号，对应序列中位置。
    """
    # 调用 STRIDE 并将输出重定向
    with open(temp_out, 'w') as f_out:
        subprocess.run([stride_path, pdb_file], stdout=f_out, check=True)

    residues = []
    with open(temp_out, 'r') as f:
        index = 0
        for line in f:
            if line.startswith("ASG"):
                tokens = line.split()
                if len(tokens) >= 7:
                    res_name = tokens[1]
                    chain_id = tokens[2]
                    res_num = int(tokens[3])
                    full_structure = tokens[5]  # 如 Coil, AlphaHelix, Turn 等

                    residues.append((index, res_name, chain_id, res_num, full_structure))
                    index += 1

    return residues


def build_structure_score_matrix(ss1, ss2, Helix, Strand, Coil):
    """
    构建结构感知 match score 矩阵和 gap_open 向量。
    返回：
        - structure_score_matrix: 结构感知匹配得分矩阵
        - gap_open_vector1: 序列1的结构感知gap open惩罚
        - gap_open_vector2: 序列2的结构感知gap open惩罚
    """
    import torch

    class_map = {
        'H': 'Helix', 'G': 'Helix', 'I': 'Helix',
        'E': 'Strand', 'B': 'Strand',
        'T': 'Coil', 'C': 'Coil', 'S': 'Coil'
    }

    # gap_penalty_by_class = {
    #     'Helix': -3.5,
    #     'Strand': -2.0,
    #     'Coil': -0.75,
    #     'Other': 0
    # }
    gap_penalty_by_class = {
        'Helix': Helix,
        'Strand': Strand,
        'Coil': Coil,
        'Other': Coil,
    }

    m, n = len(ss1), len(ss2)
    score_matrix = torch.zeros((m, n))
    gap_open_vector1 = torch.zeros(m)
    gap_open_vector2 = torch.zeros(n)

    # gap_open_vector1
    for i in range(m):
        class_i = class_map.get(ss1[i], 'Other')
        gap_open_vector1[i] = gap_penalty_by_class[class_i]

    # gap_open_vector2
    for j in range(n):
        class_j = class_map.get(ss2[j], 'Other')
        gap_open_vector2[j] = gap_penalty_by_class[class_j]

    # match score matrix
    for i in range(m):
        for j in range(n):
            a1, a2 = ss1[i], ss2[j]
            c1 = class_map.get(a1, 'Other')
            c2 = class_map.get(a2, 'Other')

            if c1 == c2:
                score_matrix[i, j] = 3.0 if a1 == a2 else 1
            else:
                pair = {c1, c2}
                if pair == {'Helix', 'Strand'}:
                    score_matrix[i, j] = -3.0
                elif pair == {'Helix', 'Coil'}:
                    score_matrix[i, j] = -2.0
                elif pair == {'Strand', 'Coil'}:
                    score_matrix[i, j] = -1.0
                else:
                    score_matrix[i, j] = -1.0
    print("len(ss1):", len(ss1))
    print("len(ss2):", len(ss2))
    print("score_matrix shape:", score_matrix.shape)

    return score_matrix, gap_open_vector1, gap_open_vector2


def Stride_run(seq1, seq2, Helix, Strand, Coil):
    from esm import pretrained

    # 只加载一次模型
    print("[INFO] Loading ESMFold model once...")
    model = pretrained.esmfold_v1()
    model = model.eval().cuda() if torch.cuda.is_available() else model.eval().float()

    # 对第一个序列预测并提取结构
    pdb_path1 = predict_structure_with_model(seq1, model, out_pdb="protein1.pdb")
    ss_tuples1 = run_stride(pdb_path1)
    ss_string1 = "".join([entry[4] for entry in ss_tuples1])

    # 对第二个序列预测并提取结构
    pdb_path2 = predict_structure_with_model(seq2, model, out_pdb="protein2.pdb")
    ss_tuples2 = run_stride(pdb_path2)
    ss_string2 = "".join([entry[4] for entry in ss_tuples2])

    structure_score_matrix, gap_open_vector1, gap_open_vector2 = build_structure_score_matrix(ss_string1, ss_string2, Helix, Strand, Coil)

    return ss_string1, ss_string2, structure_score_matrix, gap_open_vector1, gap_open_vector2


def Stride_run_pdb(pdb_path1, pdb_path2, Helix, Strand, Coil):

    # 对第一个序列预测并提取结构
    ss_tuples1 = run_stride(pdb_path1)
    ss_string1 = "".join([entry[4] for entry in ss_tuples1])

    ss_tuples2 = run_stride(pdb_path2)
    ss_string2 = "".join([entry[4] for entry in ss_tuples2])

    structure_score_matrix, gap_open_vector1, gap_open_vector2 = build_structure_score_matrix(ss_string1, ss_string2, Helix, Strand, Coil)

    return ss_string1, ss_string2, structure_score_matrix, gap_open_vector1, gap_open_vector2


if __name__ == "__main__":
    sequence_1 = "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWGDKSAVRALYDAIKKVIAEKTKPKG"
    sequence_2 = "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWGDKSAV"
    structure = Stride_run(sequence_1, sequence_2)  # ←请替换为实际路径
    print(f"[RESULT] Secondary structure:\n{structure}")
    # sequence = "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWGDKSAVRALYDAIKKVIAEKTKPKG"  # 你的PDB文件路径
    # pdb_path = predict_structure_with_esmfold(sequence)
    # ss = run_stride(pdb_path)
    # print(f"Predicted secondary structure sequence (H/E/C):\n{ss}")
