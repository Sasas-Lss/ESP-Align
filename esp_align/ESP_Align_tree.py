import numpy as np
import extract_emd
from collections import deque
from scipy.optimize import minimize
from ESP_Align import compute_similarity_score
from io import StringIO
from Bio import Phylo
from scipy.optimize import nnls
import argparse


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
def extract_embeddings_and_generate_distance_matrix(seqs_path, pdb_path):
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
            sim = compute_similarity_score(seqs[i], seqs[j], pdb_path, pearson_weight=0.8, Helix=-5.0, Strand=-3.0, Coil=-1.00, gap_ext=0.0)[0]['score_global']
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim  # 相似度矩阵是对称的

    # 根据相似度矩阵生成距离矩阵
    distance_matrix = 1 - similarity_matrix  # 距离矩阵 = 1 - 相似度矩阵

    return distance_matrix, [seq[0] for seq in sequences], num_seqs

def neighbor_joining_topology_new(D, labels):
    import numpy as np

    D = np.array(D, float)
    n_leaves = len(D)

    # 当前活跃的节点 ID 列表；初始为叶节点 0..n_leaves-1
    node_ids = list(range(n_leaves))
    next_node_id = n_leaves

    edges = []
    lengths = []

    # 迭代合并直到只剩两个活跃节点
    while len(node_ids) > 2:
        n = len(node_ids)
        # 1. 计算 Q 矩阵
        Q = np.zeros((n, n), float)
        row_sums = D.sum(axis=1)
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] = (n - 2) * D[i, j] - row_sums[i] - row_sums[j]
                Q[j, i] = Q[i, j]

        # 2. 找到最小的 Q 值对应的 i, j
        i, j = np.unravel_index(np.argmin(Q + np.diag([np.inf] * n)), Q.shape)

        # 3. 计算分支长度（严格 NJ 公式）
        dist_ij = D[i, j]
        Li = 0.5 * dist_ij + (row_sums[i] - row_sums[j]) / (2 * (n - 2))
        Lj = dist_ij - Li

        # 4. 将新节点与 i, j 对应的旧节点连边
        u = node_ids[i]
        v = node_ids[j]
        edges.append((u, next_node_id))
        lengths.append(Li)
        edges.append((v, next_node_id))
        lengths.append(Lj)

        # 5. 计算新节点与其它节点的距离
        new_dist = [
            (D[i, k] + D[j, k] - dist_ij) / 2
            for k in range(n) if k not in (i, j)
        ]

        # 6. 更新距离矩阵：删除 i, j 行列，再拼接新行新列
        D = np.delete(D, [i, j], axis=0)
        D = np.delete(D, [i, j], axis=1)
        new_row = np.array(new_dist, float).reshape(1, -1)
        D = np.hstack((D, new_row.T))
        D = np.vstack((D, np.append(new_row, 0)))

        # 7. 更新活跃节点 ID 列表
        for idx in sorted((i, j), reverse=True):
            node_ids.pop(idx)
        node_ids.append(next_node_id)

        next_node_id += 1

    # 剩两个节点，连到最终根节点
    u, v = node_ids
    w = D[0, 1] / 2
    edges.append((u, next_node_id))
    lengths.append(w)
    edges.append((v, next_node_id))
    lengths.append(w)

    return edges, lengths



# 6. 使用邻接矩阵法（NJ）生成树
def neighbor_joining_topology(D, labels):
    """
    使用 NJ 算法，从距离矩阵 D 构建树的拓扑 edges 和初始分支长度 lengths。

    参数
    ----
    D : numpy.ndarray
        对称距离矩阵，shape (n_leaves, n_leaves)，对角线应为 0。
    labels : list
        与 D 行/列对应的叶节点标签列表（仅用于映射到节点 ID，不影响结果）。

    返回
    ----
    edges : List[Tuple[int,int]]
        边列表，每条 (u, v) 表示节点 u 与节点 v 相连。
        叶节点 ID 为 0..n_leaves-1；内部节点依次为 n_leaves, n_leaves+1, ...
    lengths : List[float]
        与 edges 一一对应的分支长度。
    """
    D = np.array(D, float)
    n_leaves = len(D)

    # 当前活跃的节点 ID 列表；初始为叶节点 0..n_leaves-1
    node_ids = list(range(n_leaves))
    next_node_id = n_leaves

    edges = []
    lengths = []

    # 迭代合并直到只剩两个活跃节点
    while len(node_ids) > 2:
        n = len(node_ids)
        # 1. 计算 Q 矩阵
        Q = np.zeros((n, n), float)
        row_sums = D.sum(axis=1)
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] = (n - 2) * D[i, j] - row_sums[i] - row_sums[j]
                Q[j, i] = Q[i, j]

        # 2. 找到最小的 Q 值对应的 i, j
        i, j = np.unravel_index(np.argmin(Q + np.diag([np.inf] * n)), Q.shape)

        # 3. 保存 i, j 之间的距离并计算每条新分支长度
        dist_ij = D[i, j]
        w = dist_ij / 2

        # 4. 将新节点与 i, j 对应的旧节点连边
        u = node_ids[i]
        v = node_ids[j]
        edges.append((u, next_node_id))
        lengths.append(w)
        edges.append((v, next_node_id))
        lengths.append(w)

        # 5. 计算新节点与其它节点的距离
        new_dist = [
            (D[i, k] + D[j, k] - dist_ij) / 2
            for k in range(n) if k not in (i, j)
        ]

        # 6. 更新距离矩阵：删除 i, j 行列，再拼接新行新列
        D = np.delete(D, [i, j], axis=0)
        D = np.delete(D, [i, j], axis=1)
        new_row = np.array(new_dist, float).reshape(1, -1)
        D = np.hstack((D, new_row.T))
        D = np.vstack((D, np.append(new_row, 0)))

        # 7. 更新活跃节点 ID 列表：删除 i, j 后加入 next_node_id
        for idx in sorted((i, j), reverse=True):
            node_ids.pop(idx)
        node_ids.append(next_node_id)

        next_node_id += 1

    # 剩两个节点，连到最终根节点
    u, v = node_ids
    w = D[0, 1] / 2
    edges.append((u, next_node_id))
    lengths.append(w)
    edges.append((v, next_node_id))
    lengths.append(w)

    return edges, lengths


# 7. 主函数，整合上述步骤并生成树文件
def generate_tree_from_fasta(seqs_path, pdb_path, tree_file_path):
    # 提取距离矩阵和序列标签
    distance_matrix, labels, n_leaves = extract_embeddings_and_generate_distance_matrix(seqs_path, pdb_path)
    print(distance_matrix, labels)

    edges, init_params = neighbor_joining_topology_new(distance_matrix, labels)
    newick_tree = build_newick_from_edges(edges, init_params, labels)
    final_tree, rnorm = reestimate_branch_lengths_nnls(newick_tree, distance_matrix, labels)
    rearrange_zero_branches(final_tree.root)
    Phylo.write(final_tree, tree_file_path, "newick")
    print(f"Newick Tree saved to {tree_file_path}")
    # # 使用邻接法生成树
    # edges, init_params = neighbor_joining_topology(distance_matrix, labels)
    # print(init_params)
    # params_opt = optimize_branch_lengths(edges, n_leaves, distance_matrix, init_params)
    # print(">>> 优化后的分支长度：", params_opt)
    # newick_tree = build_newick_from_edges(edges, params_opt, labels)

    # 将结果写入文件
    # with open(tree_file_path, "w") as f:
    #     f.write(newick_tree)
    # print(f"Newick Tree saved to {tree_file_path}")

def max_branch_length(clade):
    """递归计算子树的最大深度"""
    if not clade.clades:  # 叶子
        return clade.branch_length or 0.0
    return max((c.branch_length or 0.0) + max_branch_length(c) for c in clade.clades)

def rearrange_zero_branches(clade, parent=None, tol=1e-8):
    """检测并重排 branch_length≈0 的内部节点"""
    # 递归子节点
    for child in list(clade.clades):
        rearrange_zero_branches(child, parent=clade, tol=tol)

    # 检查当前 clade 是否为零边
    if parent and clade.branch_length is not None and abs(clade.branch_length) < tol:
        if len(clade.clades) == 2 and len(parent.clades) == 2:
            # 找兄弟节点
            sibling = [c for c in parent.clades if c is not clade][0]
            c1, c2 = clade.clades

            # 计算三支的深度
            scores = [(max_branch_length(c1), c1),
                      (max_branch_length(c2), c2),
                      (max_branch_length(sibling), sibling)]
            scores.sort(key=lambda x: x[0], reverse=True)

            # 重新组合
            new_inner = Phylo.BaseTree.Clade(branch_length=None)
            new_inner.clades = [scores[0][1], scores[1][1]]

            new_parent = Phylo.BaseTree.Clade(branch_length=None)
            new_parent.clades = [new_inner, scores[2][1]]

            # 替换 parent 的子节点，保持顺序
            new_clades = []
            for c in parent.clades:
                if c is clade or c is sibling:
                    if new_parent not in new_clades:
                        new_clades.append(new_parent)
                else:
                    new_clades.append(c)
            parent.clades = new_clades


def rearrange_zero_branches_bfs(tree, tol=1e-8):
    """广度优先遍历树，修复 branch_length≈0 的拓扑"""
    queue = deque([(tree.root, None)])  # (clade, parent)

    while queue:
        clade, parent = queue.popleft()

        # 将子节点放入队列
        for child in clade.clades:
            queue.append((child, clade))

        # 检查当前 clade 是否为零边
        if parent and clade.branch_length is not None and abs(clade.branch_length) < tol:
            if len(clade.clades) == 2 and len(parent.clades) == 2:
                # 找兄弟
                sibling = [c for c in parent.clades if c is not clade][0]
                c1, c2 = clade.clades

                # 计算三支的深度
                scores = [(max_branch_length(c1), c1),
                          (max_branch_length(c2), c2),
                          (max_branch_length(sibling), sibling)]
                scores.sort(key=lambda x: x[0], reverse=True)

                # 重新组合
                new_inner = Phylo.BaseTree.Clade(branch_length=None)
                new_inner.clades = [scores[0][1], scores[1][1]]

                new_parent = Phylo.BaseTree.Clade(branch_length=None)
                new_parent.clades = [new_inner, scores[2][1]]

                # 替换 parent.clades，保持顺序
                new_clades = []
                for c in parent.clades:
                    if c is clade or c is sibling:
                        if new_parent not in new_clades:
                            new_clades.append(new_parent)
                    else:
                        new_clades.append(c)
                parent.clades = new_clades

def build_adjacency(edges, params):
    """
    根据 (edges, params) 构建邻接表：
      edges: [(u,v), ...]
      params: [w0, w1, ...] 与 edges 一一对应
    返回：adj = { node: [(nei, weight), ...], ... }
    """
    adj = {}
    for (u, v), w in zip(edges, params):
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))
    return adj


def tree_pairwise_distances(edges, params, n_leaves):
    """
    计算树上前 n_leaves 个叶节点之间的距离矩阵。
    - edges: 边列表
    - params: 边权重
    - n_leaves: 叶节点数量（编号 0..n_leaves-1）
    返回：D_tree (n_leaves x n_leaves) 的对称矩阵
    """
    adj = build_adjacency(edges, params)
    D_tree = np.zeros((n_leaves, n_leaves), float)

    # 对每个叶 i，做一次 BFS 求最短距离
    for i in range(n_leaves):
        dist = {i: 0.0}
        queue = deque([i])
        while queue:
            u = queue.popleft()
            for v, w in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + w
                    queue.append(v)
        # 填入距离矩阵
        for j in range(n_leaves):
            D_tree[i, j] = dist.get(j, np.inf)
    return D_tree


def lsq_branch_lengths(params, edges, n_leaves, D_input):
    """
    目标函数：给定分支长度 params，计算误差平方和
      f(params) = sum_{i<j}( d_tree(i,j) - D_input[i,j] )^2
    """
    D_tree = tree_pairwise_distances(edges, params, n_leaves)
    # 只累加 i<j 部分
    diff = D_tree - D_input
    # 也可以权重不同对，或加上正则化
    return np.sum(np.triu(diff**2, k=1))


def optimize_branch_lengths(edges, n_leaves, D_input, initial_params=None):
    """
    对固定拓扑(edges)和给定距离矩阵 D_input，
    使用最小二乘法优化分支长度。
    """
    m = len(edges)
    if initial_params is None:
        # 初始猜测：把所有边长设为 D_input 的平均值
        avg_dist = np.mean(D_input[np.triu_indices(n_leaves, k=1)])
        initial_params = np.ones(m) * avg_dist / (n_leaves - 1)
        print(initial_params)

    # 非负约束下界 0
    bounds = [(0.01, None)] * m

    # 使用 BFGS 进行无约束优化
    result = minimize(
        lsq_branch_lengths,
        initial_params,
        args=(edges, n_leaves, D_input),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True, 'maxiter': 2000}
    )

    if not result.success:
        print("WARNING: 优化未收敛：", result.message)
    return result.x  # 优化后的分支长度 params_opt


def build_newick_from_edges(edges, lengths, labels):
    """
    根据边列表 edges、对应的分支长度 lengths 以及叶节点标签 labels，生成 Newick 格式字符串。

    edges: List[Tuple[int,int]]
    lengths: List[float]
    labels: List[str] 叶节点标签，索引对应叶节点 ID

    返回: Newick 树字符串，以分号结尾
    """
    # 构建邻接表
    adj = {}
    for (u, v), w in zip(edges, lengths):
        # w = max(0.0, w)
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))

    # 选取根节点：最后一个 internal node（最大 ID）
    root = max(adj.keys())

    def dfs(node, parent):
        children = []
        for nbr, w in adj[node]:
            if nbr == parent:
                continue
            sub = dfs(nbr, node)
            children.append(f"{sub}:{w:.6f}")
        if not children:
            # 叶节点
            return labels[node]
        return f"({','.join(children)})"

    newick = dfs(root, None) + ';'
    return newick

def reestimate_branch_lengths_nnls(newick, D, labels):
    """
    newick: tree topology string (leaves named exactly as labels)
    D:   numpy array shape (n_leaves, n_leaves) 原始距离矩阵（对称，diag=0）
    labels: list of leaf names matching rows/cols of D in same order

    返回：新的 Biopython Tree（分支长度为非负最优解）
    """
    tree = Phylo.read(StringIO(newick), "newick")

    # 所有 clade（包括内部节点）列成列表，给每条边分配一个索引
    clades = [c for c in tree.find_clades(order="level")]  # 任意顺序，但固定
    # 只对有父边（即除 root 外）的 clade 建变量（每个 clade 的 branch_length）
    clade_vars = [c for c in clades if c is not tree.root]
    idx_map = {c: idx for idx, c in enumerate(clade_vars)}
    m = len(clade_vars)

    # 列出所有叶对 (i<j) 作为方程条目
    n = len(labels)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
    p = len(pairs)

    # 构造矩阵 A (p x m) 和观测向量 d (p,)
    A = np.zeros((p, m), float)
    d = np.zeros(p, float)
    for row_idx, (i, j) in enumerate(pairs):
        li = labels[i]
        lj = labels[j]
        # 通过 Biopython 获取叶节点对象
        t1 = next(tree.find_clades(name=li))
        t2 = next(tree.find_clades(name=lj))
        # get_path 返回两叶之间的 clade 列表（不包括起点还是包括两端？Biopython 返回 path from root to target; but Tree.get_path(a,b) available?)
        # 我们使用 tree.get_path(target) 从 root 到 target，再结合两条路径求对称差
        path1 = tree.get_path(t1)  # root->t1 的列表（含 t1，不含 root? 包含 root? 视 Biopython 版本）
        path2 = tree.get_path(t2)
        # 为了得到两叶之间的边集合，取对称差（path1 ∪ path2 - 2*(common prefix)）
        # 找公共前缀长度
        k = 0
        minlen = min(len(path1), len(path2))
        while k < minlen and path1[k] is path2[k]:
            k += 1
        # nodes on path between t1 and t2 are path1[k:] + path2[k:]
        between = path1[k:] + path2[k:]
        # 对于每个 clade in between（这些 clade 对应的 parent-edge 在这些 clade 上）
        for cl in between:
            if cl is tree.root:
                continue
            if cl in idx_map:
                A[row_idx, idx_map[cl]] = 1.0
        # 观测距离来自 D
        d[row_idx] = D[i, j]

    # 使用 NNLS 求解非负的边长
    x, rnorm = nnls(A, d)

    # 将求得的 x 写回树的 branch_length（clade.branch_length）
    for cl, idx in idx_map.items():
        cl.branch_length = float(x[idx])

    return tree, rnorm

# 示例：调用函数生成树
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tree from FASTA file.")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    parser.add_argument("-p", "--pdb_path", required=True, help="Path to the PDB files (leave blank to use ESMFold)")
    parser.add_argument("-o", "--output_tree", required=True, help="Output tree file (Newick format)")

    args = parser.parse_args()

    generate_tree_from_fasta(args.input, args.pdb_path, args.output_tree)


