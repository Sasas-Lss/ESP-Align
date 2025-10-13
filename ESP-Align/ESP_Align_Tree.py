import numpy as np
import extract_emd
from scipy.spatial.distance import pdist, squareform
from ESP_Align import compute_similarity_score


# 4. 计算相似度矩阵（假设已有方法，返回0到1的相似度评分）
def calculate_similarity(emb1, emb2):
    # 计算两个嵌入之间的余弦相似度（这里只是一个简单示例）
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return cos_sim


def extract_emd_model(seq, Model, Model_tokenizer):
    seq_1 = seq[1]
    name1 = seq[0]

    # Model, Model_tokenizer = ESM2_initialize()
    emb1 = extract_emd.get_embs_ESM2(Model, Model_tokenizer, [seq_1], 1)[0]

    return [seq_1, name1, emb1]



# 5. 从序列文件中提取所有的嵌入并计算相似度矩阵
def extract_embeddings_and_generate_distance_matrix(seqs_path):
    # 读取并加载序列
    sequences = extract_emd.load_fasta(seqs_path)
    # 初始化 ESM2 模型
    ESM2, batch_converter = extract_emd.ESM2_initialize()

    # 提取所有序列的嵌入
    seqs = []
    for seq in sequences:
        seq_name_emb = extract_emd_model(seq, ESM2, batch_converter)
        seqs.append(seq_name_emb)

    print(sequences)
    # 计算相似度矩阵
    num_seqs = len(seqs)
    similarity_matrix = np.zeros((num_seqs, num_seqs))

    for i in range(num_seqs):
        for j in range(i + 1, num_seqs):
            sim = compute_similarity_score(seqs[i], seqs[j], "", pearson_weight=0.8, Helix=-5.0, Strand=-3.0, Coil=-1.00, gap_ext=0.0)[0]['score_global']
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim  # 相似度矩阵是对称的

    # 根据相似度矩阵生成距离矩阵
    distance_matrix = 1 - similarity_matrix  # 距离矩阵 = 1 - 相似度矩阵

    return distance_matrix, [seq[0] for seq in sequences]


# 6. 使用邻接矩阵法（NJ）生成树
def neighbor_joining(D, labels):
    """
    使用邻接法（NJ）从距离矩阵 D 和对应的标签列表 labels 构建 Newick 树。
    D: 对称距离矩阵（numpy 数组），对角线应为 0
    labels: 与 D 大小一致的标签列表，如 ["seq1", "seq2", ...]
    返回值：Newick 格式的树字符串
    """
    D = np.array(D, float)
    n = len(D)
    tree = []

    # 迭代合并直到只剩两个节点
    while n > 2:
        # 1. 计算 Q 矩阵
        Q = np.zeros((n, n), float)
        row_sums = D.sum(axis=1)
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] = (n - 2) * D[i, j] - row_sums[i] - row_sums[j]
                Q[j, i] = Q[i, j]

        # 2. 找到最小的 Q 值对应的 i, j
        i, j = np.unravel_index(np.argmin(Q + np.diag([np.inf]*n)), Q.shape)

        # 3. 保存合并前的距离 dist_ij
        dist_ij = D[i, j]

        # 4. 计算新节点到其他节点的距离
        new_dist = []
        for k in range(n):
            if k != i and k != j:
                d_new = (D[i, k] + D[j, k] - dist_ij) / 2
                new_dist.append(d_new)

        # 5. 更新距离矩阵：删除 i, j 行列，再拼接新行新列
        D = np.delete(D, [i, j], axis=0)
        D = np.delete(D, [i, j], axis=1)
        new_row = np.array(new_dist, float).reshape(1, -1)
        D = np.hstack((D, new_row.T))              # 添加新列
        D = np.vstack((D, np.append(new_row, 0)))  # 添加新行，对角线补 0

        # 6. 构建新的 Newick 标签
        new_label = f"({labels[i]}:{dist_ij/2:.6f},{labels[j]}:{dist_ij/2:.6f})"

        # 7. 更新标签列表：先删除高索引再删除低索引，最后 append 新标签
        labels.pop(max(i, j))
        labels.pop(min(i, j))
        labels.append(new_label)

        # 8. 计数减 1
        n -= 1

    # 剩余两个节点时，直接拼接根节点
    final_tree = f"({labels[0]}:{D[0,1]/2:.6f},{labels[1]}:{D[0,1]/2:.6f});"
    return final_tree


# 7. 主函数，整合上述步骤并生成树文件
def generate_tree_from_fasta(seqs_path, tree_file_path):
    # 提取距离矩阵和序列标签
    distance_matrix, labels = extract_embeddings_and_generate_distance_matrix(seqs_path)

    print(distance_matrix, labels)

    # 使用邻接法生成树
    newick_tree = neighbor_joining(distance_matrix, labels)

    # 将结果写入文件
    with open(tree_file_path, "w") as f:
        f.write(newick_tree)
    print(f"Newick Tree saved to {tree_file_path}")


# 示例：调用函数生成树
if __name__ == "__main__":
    generate_tree_from_fasta('./Tree/13_Hemoglobin_named.fasta', './Tree/Hemoglobin.tree')




