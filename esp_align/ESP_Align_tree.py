import numpy as np
from collections import deque
from scipy.optimize import minimize, nnls
from Bio import Phylo
from io import StringIO
import extract_emd
from ESP_Align import compute_similarity_score

def extract_emd_model(seq, Model, Model_tokenizer):
    """Extract embedding for a single sequence using ESM2."""
    seq_1 = seq[1]
    name1 = seq[0]
    emb1 = extract_emd.get_embs_ESM2(Model, Model_tokenizer, [seq_1], 1)[0]
    return [seq_1, name1, emb1]

def extract_embeddings_and_generate_distance_matrix(seqs_path, pdb_path):
    """Extract embeddings for all sequences and compute distance matrix."""
    sequences = extract_emd.load_fasta(seqs_path)
    ESM2, batch_converter = extract_emd.ESM2_initialize()
    seqs = [extract_emd_model(seq, ESM2, batch_converter) for seq in sequences]

    num_seqs = len(seqs)
    similarity_matrix = np.zeros((num_seqs, num_seqs))
    for i in range(num_seqs):
        for j in range(i + 1, num_seqs):
            sim = compute_similarity_score(
                seqs[i], seqs[j], pdb_path,
                pearson_weight=0.8, Helix=-5.0, Strand=-3.0, Coil=-1.0, gap_ext=0.0
            )[0]['score_global']
            similarity_matrix[i, j] = similarity_matrix[j, i] = sim

    distance_matrix = 1 - similarity_matrix
    return distance_matrix, [seq[0] for seq in sequences], num_seqs

def neighbor_joining_topology_new(D, labels):
    """Construct NJ tree topology (edges and initial branch lengths) from distance matrix."""
    D = np.array(D, float)
    n_leaves = len(D)
    node_ids = list(range(n_leaves))
    next_node_id = n_leaves
    edges, lengths = [], []

    while len(node_ids) > 2:
        n = len(node_ids)
        Q = np.zeros((n, n), float)
        row_sums = D.sum(axis=1)
        for i in range(n):
            for j in range(i+1, n):
                Q[i, j] = Q[j, i] = (n-2)*D[i,j] - row_sums[i] - row_sums[j]

        i, j = np.unravel_index(np.argmin(Q + np.diag([np.inf]*n)), Q.shape)
        dist_ij = D[i, j]
        Li = 0.5*dist_ij + (row_sums[i]-row_sums[j])/(2*(n-2))
        Lj = dist_ij - Li
        u, v = node_ids[i], node_ids[j]
        edges.extend([(u, next_node_id), (v, next_node_id)])
        lengths.extend([Li, Lj])

        new_dist = [(D[i,k]+D[j,k]-dist_ij)/2 for k in range(n) if k not in (i,j)]
        D = np.delete(D, [i,j], axis=0)
        D = np.delete(D, [i,j], axis=1)
        new_row = np.array(new_dist, float).reshape(1,-1)
        D = np.hstack((D, new_row.T))
        D = np.vstack((D, np.append(new_row, 0)))
        for idx in sorted((i,j), reverse=True): node_ids.pop(idx)
        node_ids.append(next_node_id)
        next_node_id += 1

    u, v = node_ids
    w = D[0,1]/2
    edges.extend([(u, next_node_id), (v, next_node_id)])
    lengths.extend([w, w])
    return edges, lengths

def build_adjacency(edges, params):
    """Build adjacency list from edges and branch lengths."""
    adj = {}
    for (u,v), w in zip(edges, params):
        adj.setdefault(u, []).append((v,w))
        adj.setdefault(v, []).append((u,w))
    return adj

def tree_pairwise_distances(edges, params, n_leaves):
    """Compute leaf-to-leaf distance matrix from tree."""
    adj = build_adjacency(edges, params)
    D_tree = np.zeros((n_leaves, n_leaves), float)
    for i in range(n_leaves):
        dist = {i: 0.0}
        queue = deque([i])
        while queue:
            u = queue.popleft()
            for v,w in adj[u]:
                if v not in dist:
                    dist[v] = dist[u]+w
                    queue.append(v)
        for j in range(n_leaves):
            D_tree[i,j] = dist.get(j,np.inf)
    return D_tree

def lsq_branch_lengths(params, edges, n_leaves, D_input):
    """Least squares objective function for branch lengths."""
    D_tree = tree_pairwise_distances(edges, params, n_leaves)
    diff = D_tree - D_input
    return np.sum(np.triu(diff**2, k=1))

def optimize_branch_lengths(edges, n_leaves, D_input, initial_params=None):
    """Optimize branch lengths using L-BFGS-B with non-negative bounds."""
    m = len(edges)
    if initial_params is None:
        avg_dist = np.mean(D_input[np.triu_indices(n_leaves, k=1)])
        initial_params = np.ones(m) * avg_dist / (n_leaves - 1)
    bounds = [(0.01,None)]*m
    result = minimize(lsq_branch_lengths, initial_params, args=(edges,n_leaves,D_input),
                      method='L-BFGS-B', bounds=bounds,
                      options={'disp': True,'maxiter':2000})
    if not result.success: print("WARNING: Optimization did not converge:", result.message)
    return result.x

def build_newick_from_edges(edges, lengths, labels):
    """Generate Newick string from tree edges and branch lengths."""
    adj = {}
    for (u,v), w in zip(edges,lengths):
        adj.setdefault(u, []).append((v,w))
        adj.setdefault(v, []).append((u,w))
    root = max(adj.keys())
    def dfs(node, parent):
        children = []
        for nbr,w in adj[node]:
            if nbr==parent: continue
            sub = dfs(nbr,node)
            children.append(f"{sub}:{w:.6f}")
        if not children: return labels[node]
        return f"({','.join(children)})"
    return dfs(root,None)+';'

def reestimate_branch_lengths_nnls(newick, D, labels):
    """Reestimate non-negative branch lengths using NNLS."""
    tree = Phylo.read(StringIO(newick),"newick")
    clades = [c for c in tree.find_clades(order="level")]
    clade_vars = [c for c in clades if c is not tree.root]
    idx_map = {c: idx for idx,c in enumerate(clade_vars)}
    m = len(clade_vars)
    n = len(labels)
    pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    A = np.zeros((len(pairs),m), float)
    d = np.zeros(len(pairs), float)
    for row_idx,(i,j) in enumerate(pairs):
        t1 = next(tree.find_clades(name=labels[i]))
        t2 = next(tree.find_clades(name=labels[j]))
        path1 = tree.get_path(t1)
        path2 = tree.get_path(t2)
        k = 0
        minlen = min(len(path1),len(path2))
        while k<minlen and path1[k] is path2[k]: k+=1
        between = path1[k:] + path2[k:]
        for cl in between:
            if cl is tree.root: continue
            if cl in idx_map: A[row_idx,idx_map[cl]] = 1.0
        d[row_idx] = D[i,j]
    x,rnorm = nnls(A,d)
    for cl,idx in idx_map.items(): cl.branch_length = float(x[idx])
    return tree,rnorm

def rearrange_zero_branches(clade, parent=None, tol=1e-8):
    """Detect and rearrange internal branches with near-zero length."""
    for child in list(clade.clades):
        rearrange_zero_branches(child, parent=clade, tol=tol)
    if parent and clade.branch_length is not None and abs(clade.branch_length)<tol:
        if len(clade.clades)==2 and len(parent.clades)==2:
            sibling = [c for c in parent.clades if c is not clade][0]
            c1,c2 = clade.clades
            scores = sorted([(max_branch_length(c1),c1),(max_branch_length(c2),c2),(max_branch_length(sibling),sibling)],
                            key=lambda x:x[0], reverse=True)
            new_inner = Phylo.BaseTree.Clade(branch_length=None)
            new_inner.clades = [scores[0][1],scores[1][1]]
            new_parent = Phylo.BaseTree.Clade(branch_length=None)
            new_parent.clades = [new_inner,scores[2][1]]
            new_clades = []
            for c in parent.clades:
                if c is clade or c is sibling:
                    if new_parent not in new_clades: new_clades.append(new_parent)
                else: new_clades.append(c)
            parent.clades = new_clades

def max_branch_length(clade):
    """Recursively compute max depth of subtree."""
    if not clade.clades: return clade.branch_length or 0.0
    return max((c.branch_length or 0.0) + max_branch_length(c) for c in clade.clades)

def generate_tree_from_fasta(seqs_path, pdb_path, tree_file_path):
    """Generate phylogenetic tree from FASTA sequences."""
    distance_matrix, labels, n_leaves = extract_embeddings_and_generate_distance_matrix(seqs_path, pdb_path)
    edges, init_params = neighbor_joining_topology_new(distance_matrix, labels)
    newick_tree = build_newick_from_edges(edges, init_params, labels)
    final_tree, rnorm = reestimate_branch_lengths_nnls(newick_tree, distance_matrix, labels)
    rearrange_zero_branches(final_tree.root)
    Phylo.write(final_tree, tree_file_path, "newick")
    print(f"Newick Tree saved to {tree_file_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate tree from FASTA file.")
    parser.add_argument("-i","--input", required=True, help="Input FASTA file")
    parser.add_argument("-p", "--pdb_path", default=None, help="Path to PDB files (leave blank to use ESMFold)")
    parser.add_argument("-o","--output_tree", required=True, help="Output Newick tree file")
    args = parser.parse_args()
    generate_tree_from_fasta(args.input, args.pdb_path, args.output_tree)
