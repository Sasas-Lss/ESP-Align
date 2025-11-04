import os
import csv
import random
from tqdm import tqdm
from itertools import combinations
from Bio import SeqIO, AlignIO
import numpy as np
import torch
import esm
from numba import njit
import concurrent.futures

def compute_cosine_similarity_matrix(emb1: torch.Tensor, emb2: torch.Tensor) -> np.ndarray:
    emb1 = emb1.cpu().numpy()
    emb2 = emb2.cpu().numpy()

    norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)
    sim_matrix = np.dot(emb1, emb2.T) / (norm1 @ norm2.T + 1e-8)  # shape: (m, n)
    return sim_matrix

@njit
def affine_dp_numba(sim_matrix, gap_open, gap_extend):
    m, n = sim_matrix.shape
    M = np.zeros((m + 1, n + 1))
    L = np.zeros_like(M)
    U = np.zeros_like(M)

    M[0, 1:] = gap_open + gap_extend * np.arange(0, n)
    M[1:, 0] = gap_open + gap_extend * np.arange(0, m)
    L[1:, 0] = M[1:, 0] + gap_open
    U[0, 1:] = M[0, 1:] + gap_open

    tracer = np.zeros((m + 1, n + 1, 3), dtype=np.uint8)  # [diag, up, left]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            l1 = M[i, j - 1] + gap_open
            l2 = L[i, j - 1] + gap_extend
            L[i, j] = max(l1, l2)

            u1 = M[i - 1, j] + gap_open
            u2 = U[i - 1, j] + gap_extend
            U[i, j] = max(u1, u2)

            match = M[i - 1, j - 1] + sim_matrix[i - 1, j - 1]
            M[i, j] = max(match, L[i, j], U[i, j])

            if M[i, j] == match:
                tracer[i, j, 0] = 1  # diag
            elif M[i, j] == U[i, j]:
                tracer[i, j, 1] = 1  # up
            else:
                tracer[i, j, 2] = 1  # left

    return M, L, U, tracer



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

def ESM2_initialize():
  print("ESM2 Initialize : ")

  ESM2, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
  batch_converter = alphabet.get_batch_converter()
  ESM2.eval()

  return ESM2, batch_converter

def get_embs_ESM2(ESM2, batch_converter, sequences, n):
  sequences = sequences[:n]
  data = [("" , sequences[0])]

  batch_labels, batch_strs, batch_tokens = batch_converter(data)

  # Extract per-residue representations
  with torch.no_grad():
      results = ESM2(batch_tokens, repr_layers=[33], return_contacts= False)
  token_representations = results["representations"][33]

  final_embs = []
  for i in range(len(token_representations)):
    final_embs.append(token_representations[i][1:-1])

  return final_embs


def get_alignments(prot1, prot2, gap_penalty = 0, gap_extension_penalty = 0 ,
                   scoring = "" , alignment_type = "Global-regular" , Model = "" , Model_Tokenizer = ""):

    score , alignment= affine_global_dp(prot1, prot2, gap_penalty, gap_extension_penalty
                                                    ,scoring = scoring , Model = Model, Model_tokenizer = Model_Tokenizer)
    return alignment[0][0], alignment[0][1], score


def affine_global_dp(seq_1, seq_2, g_open, g_ext, scoring = "ESM2", Model = None, Model_tokenizer = None):

    #核心代码区域1
    m = len(seq_1); n = len(seq_2)
    M = np.zeros([m + 1, n + 1])
    M[0, 1:] = g_open + g_ext * np.arange(0, n, 1)
    M[1:, 0] = g_open + g_ext * np.arange(0, m, 1)
    L = np.copy(M); U = np.copy(M)
    L[1:,0] = L[1:,0]+g_open; U[0,1:] = U[0,1:]+g_open

    #fill up
    tracer = np.zeros([np.shape(M)[0], np.shape(M)[1], 7])

    if scoring == "ESM2":
      emb1 = get_embs_ESM2(Model, Model_tokenizer, [seq_1], 1)[0].cpu().numpy()
      emb2 = get_embs_ESM2(Model, Model_tokenizer, [seq_2], 1)[0].cpu().numpy()
      cos = torch.nn.CosineSimilarity(dim=0)


    #核心代码区域2
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            l_arr = np.array([M[i , j - 1] + g_open, L[i,j - 1] + g_ext])
            L[i,j] = np.max(l_arr)
            l_where = l_arr==np.max(l_arr)
            u_arr = np.array([M[i - 1,j] + g_open, U[i - 1, j] + g_ext])
            U[i,j] = np.max(u_arr)
            u_where = u_arr == np.max(u_arr)

            sim = cos(torch.tensor(emb1[i - 1] , dtype = torch.float32)
            , torch.tensor(emb2[j - 1] , dtype = torch.float32)).item()
            m_arr = np.array([M[i - 1, j - 1] + sim , U[i, j] , L[i, j]])
            M[i, j] = np.max(m_arr)
            m_where = m_arr == np.max(m_arr)
            idx = np.hstack([m_where, u_where, l_where])
            tracer[i, j, idx] = 1

    alignment = []
    alignment.append(traceback_g(tracer, seq_1, seq_2, affine=True))
    alignment = list(set(map(tuple, alignment)))
    score = max(M[-1, -1], L[-1, -1], U[-1, -1])

    print(score)

    return score, alignment

def traceback_g(tracer, seq_1, seq_2, mat=None, local=False, affine=True):

    # get sequence lengths
    m = len(seq_1); n = len(seq_2)

    # convert to numpy arrays
    x = np.array(list(seq_1),dtype='object')
    y = np.array(list(seq_2),dtype='object')
    roadmap = 1

    if local is False: st = [m+1,n+1]
    else:
        if roadmap == 0: r = np.random.choice(range(np.size(np.where(mat == np.max(mat))[0])), 1)[0] #random maxima
        elif roadmap == 1: r = -1  # highroad
        elif roadmap == 2: r = 0  # lowroad
        st = [(np.where(mat == np.max(mat))[0][r])+1, (np.where(mat == np.max(mat))[1][r])+1]

        start_size = ((m-st[0])-(n-st[1])) # how many gaps and for which sequence
        start_gap = (['-']*abs(start_size))
        if start_size > 0:
            y = np.append(y, start_gap)
        elif start_size < 0:
            x = np.append(x, start_gap)

    st_lv = 0
    while ((st[0]>1) & (st[1]>1)):
        B = np.zeros([2,2]) #define 2x2 box which specifies which way to move
        if affine is True:
            Tr = np.zeros([7]) #define a 7x1 Tr array (will store arrows at each step)
        else:
            Tr = np.zeros([3]) #define a 3x1 Tr array (will store arrows at each step)

        if affine is False:
            Tr[0] = np.copy(tracer[st[0]-1,st[1]-1,0])
            Tr[1] = np.copy(tracer[st[0]-1,st[1]-1,1])
            Tr[2] = np.copy(tracer[st[0]-1,st[1]-1,2])
        else:
            Tr[0] = np.copy(tracer[st[0]-1,st[1]-1,0])
            Tr[1] = np.copy(tracer[st[0]-1,st[1]-1,1])
            Tr[2] = np.copy(tracer[st[0]-1,st[1]-1,2])
            Tr[3] = np.copy(tracer[st[0]-1,st[1]-1,3])
            Tr[4] = np.copy(tracer[st[0]-1,st[1]-1,4])
            Tr[5] = np.copy(tracer[st[0]-1,st[1]-1,5])
            Tr[6] = np.copy(tracer[st[0]-1,st[1]-1,6])

        if affine is True: levels = [[2,0,1],[4,3],[6,5]]
        else: levels = [[2,0,1]]
        for l in levels:
            if np.sum(Tr[l])>1:
                choose = np.where(Tr[l]==1)[0]
                Tr[l] = 0
                if roadmap == 0: r = np.random.choice(choose,1)[0] #random turning
                elif roadmap == 1: r = choose[-1] #highroad
                elif roadmap == 2: r = choose[0] #lowroad
                Tr[l[r]] = 1

        #level up-down
        if ((Tr[0]==1) & (st_lv==0)): #diagonal
            B[0,0] = 1

        if ((Tr[1]==1) & (st_lv==0)):
            if affine is True: st_lv = 1 #level up
            else:
                B[0,1] = 1

        if ((Tr[2]==1) & (st_lv==0)):
            if affine is True: st_lv = 2 #level down
            else:
                B[1,0] = 1

        #affine gaps allow for level shifts
        if affine is True:
            if ((Tr[4]==1) & (st_lv==1)): #move up
                B[0,1] = 1

            if ((Tr[3]==1) & (st_lv==1)): #move up back to main
                st_lv = 0
                B[0,1] = 1

            if ((Tr[6]==1) & (st_lv==2)): #move left
                B[1,0] = 1

            if ((Tr[5]==1) & (st_lv==2)): #move left back to main
                st_lv = 0
                B[1,0] = 1

        #movements
        if B[0,1]==1: #upward
            y = np.insert(y,st[1]-1,'-') #add a gap
            st[0] = st[0]-1

        if B[1,0]==1: #leftward
            x = np.insert(x,st[0]-1,'-') #add a gap
            st[1] = st[1]-1

        if B[0,0]==1: #diagonal
            st[1] = st[1]-1
            st[0] = st[0]-1

    end_size = (np.size(x)-np.size(y)) #how many gaps and for which sequence
    end_gap = (['-']*abs(end_size))
    if end_size>0:
        y=np.insert(y,0,end_gap)
    elif end_size<0:
        x=np.insert(x,0,end_gap)

    #check no overlapping gaps
    x = np.where(((x=='-')&(y=='-')),None,x)
    y = np.where((x==None),'',y)
    x = np.where((x==None),'',x)

    return np.sum(x),np.sum(y)

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

# def process_all_pairs_esm2(tfa_dir, fasta_dir, output_csv_path,
#                             gap_open, gap_extend, model=None, tokenizer=None):
#     total_f1 = []
#     tfa_files = [f for f in os.listdir(tfa_dir) if f.endswith(".tfa")]
#     tfa_files.sort()
#
#     if not os.path.exists(output_csv_path):
#         with open(output_csv_path, 'w', newline='') as f_csv:
#             writer = csv.writer(f_csv)
#             writer.writerow(["TFA_File", "Avg_F1_Score"])
#
#     for tfa_file in tqdm(tfa_files, desc="Processing all TFA files"):
#         tfa_path = os.path.join(tfa_dir, tfa_file)
#         seq_records = load_fasta_nor(tfa_path)
#         if len(seq_records) < 2:
#             continue
#
#         pairs = list(combinations(seq_records, 2))
#         random.seed(17)
#         random.shuffle(pairs)
#         chosen_pairs = pairs[:100]
#
#         for (id1, seq1), (id2, seq2) in tqdm(chosen_pairs, desc=f"Processing {tfa_file}", leave=False):
#             ref_fasta = os.path.join(fasta_dir, tfa_file.replace('.tfa', '.fasta'))
#
#             try:
#                 pred1, pred2, _ = get_alignments(
#                     seq1, seq2,
#                     gap_penalty=gap_open,
#                     gap_extension_penalty=gap_extend,
#                     scoring="ESM2",
#                     Model=model,
#                     Model_Tokenizer=tokenizer
#                 )
#
#                 ref1, ref2 = get_reference_alignment(ref_fasta, id1, id2)
#                 f1 = get_match_labels_by_alignment_pairs(pred1, pred2, ref1, ref2)
#                 print(f"Pair {id1}-{id2} in {tfa_file}. F1 score: {f1:.4f}")
#                 total_f1.append(f1)
#             except Exception as e:
#                 print(f"❌ Failed on pair {id1}-{id2} in {tfa_file}: {e}")
#
#         avg_f1 = sum(total_f1) / len(total_f1) if total_f1 else 0
#         print(f"✅ {tfa_file} Avg F1: {avg_f1:.4f}")
#         with open(output_csv_path, 'a', newline='') as f_csv:
#             writer = csv.writer(f_csv)
#             writer.writerow([tfa_file, f"{avg_f1:.4f}"])
#
#     return sum(total_f1) / len(total_f1) if total_f1 else 0

def load_precomputed_embedding(seq1, seq2, id1, id2, tfa_file, embedding_dir):
    sub_dir = os.path.join(embedding_dir, tfa_file.replace('.tfa', ''))
    emb1_path = os.path.join(sub_dir, f"{id1}.pt")
    emb2_path = os.path.join(sub_dir, f"{id2}.pt")

    if not os.path.exists(emb1_path) or not os.path.exists(emb2_path):
        raise FileNotFoundError(f"Missing embedding for {id1} or {id2} in {sub_dir}")

    emb1 = torch.load(emb1_path)
    emb2 = torch.load(emb2_path)
    return [seq1, emb1], [seq2, emb2]

def process_all_pairs_esm2_cached(tfa_dir, fasta_dir, embedding_dir, output_csv_path,
                                   gap_open, gap_extend):
    total_f1, total_precision, total_recall = [], [], []
    tfa_files = [f for f in os.listdir(tfa_dir) if f.endswith(".tfa")]
    tfa_files.sort()

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
                emb1_pack, emb2_pack = load_precomputed_embedding(
                    seq1, seq2, id1, id2, tfa_file, embedding_dir
                )
                emb1 = emb1_pack[1]
                emb2 = emb2_pack[1]

                # sim_matrix = compute_cosine_similarity_matrix(emb1, emb2)
                # M, L, U, tracer = affine_dp_numba(sim_matrix, gap_open, gap_extend)
                #
                # alignment = [traceback_g(tracer, seq1, seq2, affine=False)]
                # alignment = list(set(map(tuple, alignment)))
                #
                # pred1, pred2 = alignment[0]

                pred1, pred2, _ = affine_global_alignment_escore_and_traceback(seq1, seq2, emb1, emb2, g_open=gap_open,
                                                                               g_ext=gap_extend)
                ref1, ref2 = get_reference_alignment(ref_fasta, id1, id2)
                f1_score, precision_score, recall_score = get_match_labels_by_alignment_pairs(pred1, pred2, ref1, ref2)
                if f1_score is not None:
                    print(f"Pair {id1}-{id2} in {tfa_file}. F1 score: {f1_score:.4f}")
                    # ##
                    f1_scores_this_file.append(f1_score)
                    precision_this_file.append(precision_score)
                    recall_this_file.append(recall_score)
                else:
                    print(f"Pair {id1}-{id2} in {tfa_file} skipped.")

            except Exception as e:
                print(f"❌ Failed on pair {id1}-{id2} in {tfa_file}: {e}")

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

def affine_global_alignment_escore_and_traceback(seq1, seq2, emb1, emb2, g_open=-0.25, g_ext=-0.01,):
    # 1) initialization
    m = len(seq1);
    n = len(seq2)
    # 3 working matrices:
    # A --> best alignment of seq1[1-i] and seq2[1-j] that aligns seq1[i] and seq2[j]
    # B --> best alignment of seq1[1-i] and seq2[1-j] that aligns gap with seq2[j] (gap in seq1)
    # C --> best alignment of seq1[1-i] and seq2[1-j] that aligns seq1[i] with gap (gap in seq2)
    A = np.zeros([m + 1, n + 1])
    B = np.zeros([m + 1, n + 1])
    C = np.zeros([m + 1, n + 1])

    # 3 traceback matrices
    A_tb = np.zeros([m + 1, n + 1])
    B_tb = np.zeros([m + 1, n + 1])
    C_tb = np.zeros([m + 1, n + 1])

    # initializing matrices
    A[0, 0] = 0  # needed for recursion
    A[1:, 0] = -np.inf
    A[0, 1:] = -np.inf

    # changes from local to global found here:
    B[0:, 0] = -np.inf  # -inf along left
    B[0, 1:] = g_open + g_ext * np.arange(0, n, 1)  # along top

    C[0, 0:] = -np.inf  # -inf along top
    C[1:, 0] = g_open + g_ext * np.arange(0, m, 1)  # along left

    # needed for computing embeddings
    # Model = model
    # Model_tokenizer = tokenizer

    # get Ankh embedded sequences
    # emb1 = get_embs_Ankh(seq1, model, tokenizer).cpu().numpy()
    # emb2 = get_embs_Ankh(seq2, model, tokenizer).cpu().numpy()

    # emb1 = get_embs_ESM2(Model, Model_tokenizer, [seq1], 1)[0].cpu().numpy()
    # emb2 = get_embs_ESM2(Model, Model_tokenizer, [seq2], 1)[0].cpu().numpy()
    cos = torch.nn.CosineSimilarity(dim=0)

    # 2) DP
    for i in range(1, m + 1):  # length of first sequence
        for j in range(1, n + 1):  # length of second sequence

            sim = cos(torch.tensor(emb1[i - 1], dtype=torch.float32),
                      torch.tensor(emb2[j - 1], dtype=torch.float32)).item()
            match_score = sim  # no shift down for global

            # filling up A
            A[i, j] = max(A[i - 1, j - 1] + match_score, B[i - 1, j - 1] + match_score,
                          C[i - 1, j - 1] + match_score)  # change from local
            # store traceback info
            max_index = np.argmax([A[i - 1, j - 1] + match_score, B[i - 1, j - 1] + match_score,
                                   C[i - 1, j - 1] + match_score])  # change from local
            A_tb[i, j] = max_index  # 0 = from A, 1 = from B, 2 = from C

            # filling up B (gap in seq1)
            B[i, j] = max((A[i, j - 1] + g_open + g_ext), (B[i, j - 1] + g_ext),
                          (C[i, j - 1] + g_open + g_ext))  # change from local
            # store traceback info
            max_index = np.argmax(
                [A[i, j - 1] + g_open + g_ext, B[i, j - 1] + g_ext, C[i, j - 1] + g_open + g_ext])  # change from local
            B_tb[i, j] = max_index  # 0 = from A, 1 = from B, 2 = from C

            # filling up C (gap in seq2)
            C[i, j] = max((A[i - 1, j] + g_open + g_ext), (B[i - 1, j] + g_open + g_ext), (C[i - 1, j] + g_ext))
            # store traceback info
            max_index = np.argmax([A[i - 1, j] + g_open + g_ext, B[i - 1, j] + g_open + g_ext, C[i - 1, j] + g_ext])
            C_tb[i, j] = max_index  # 0 = from A, 1 = from B, 2 = from C

    # 3) traceback
    # find max cell across all 3 matrices that is the bottom right corner
    A_val = A[m, n]
    B_val = B[m, n]
    C_val = C[m, n]

    max_score = max(A_val, B_val, C_val)

    if A_val == max_score:
        tb_matrix = A_tb
    elif B_val == max_score:
        tb_matrix = B_tb
    elif C_val == max_score:
        tb_matrix = C_tb

    aligned_seq1 = []
    aligned_seq2 = []

    while (i > 0) or (j > 0):

        # currently in matrix A
        if tb_matrix is A_tb:

            if tb_matrix[i, j] == 0:  # from A

                if (i - 1) >= 0:
                    aligned_seq1.append(seq1[i - 1])
                    i -= 1
                else:
                    aligned_seq1.append('-')

                if (j - 1) >= 0:
                    aligned_seq2.append(seq2[j - 1])
                    j -= 1
                else:
                    aligned_seq2.append('-')

                tb_matrix = A_tb

            elif tb_matrix[i, j] == 1:  # from B

                if (i - 1) >= 0:
                    aligned_seq1.append(seq1[i - 1])
                    i -= 1
                else:
                    aligned_seq1.append('-')

                if (j - 1) >= 0:
                    aligned_seq2.append(seq2[j - 1])
                    j -= 1
                else:
                    aligned_seq2.append('-')

                tb_matrix = B_tb

            elif tb_matrix[i, j] == 2:  # from C

                if (i - 1) >= 0:
                    aligned_seq1.append(seq1[i - 1])
                    i -= 1
                else:
                    aligned_seq1.append('-')

                if (j - 1) >= 0:
                    aligned_seq2.append(seq2[j - 1])
                    j -= 1
                else:
                    aligned_seq2.append('-')

                tb_matrix = C_tb

        # currently in matrix B (gap in seq1)
        elif tb_matrix is B_tb:

            if tb_matrix[i, j] == 0:
                aligned_seq1.append('-')
                aligned_seq2.append(seq2[j - 1])
                j -= 1
                tb_matrix = A_tb

            elif tb_matrix[i, j] == 1:
                aligned_seq1.append('-')
                aligned_seq2.append(seq2[j - 1])
                j -= 1
                tb_matrix = B_tb

            elif tb_matrix[i, j] == 2:
                aligned_seq1.append('-')
                aligned_seq2.append(seq2[j - 1])
                j -= 1
                tb_matrix = C_tb

        # currently in matrix C (gap in seq2)
        elif tb_matrix is C_tb:

            if tb_matrix[i, j] == 0:
                aligned_seq1.append(seq1[i - 1])
                aligned_seq2.append('-')
                i -= 1
                tb_matrix = A_tb

            elif tb_matrix[i, j] == 1:
                aligned_seq1.append(seq1[i - 1])
                aligned_seq2.append('-')
                i -= 1
                tb_matrix = B_tb

            elif tb_matrix[i, j] == 2:
                aligned_seq1.append(seq1[i - 1])
                aligned_seq2.append('-')
                i -= 1
                tb_matrix = C_tb

    # reverse the sequences
    aligned_seq1 = ''.join(reversed(aligned_seq1))
    aligned_seq2 = ''.join(reversed(aligned_seq2))

    # remove leading '-' if both sequences start with '-'
    if aligned_seq1.startswith('-') and aligned_seq2.startswith('-'):
        aligned_seq1 = aligned_seq1[1:]
        aligned_seq2 = aligned_seq2[1:]

    return aligned_seq1, aligned_seq2, max_score


CDD_sets = {
    "CDD_Tier1": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier1/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/E_score/Tire1.csv"
    },
    "CDD_Tier2": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier2/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/E_score/Tire2.csv"
    },
    "CDD_Tier3": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier3/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/E_score/Tire3.csv"
        },
    "CDD_Tier4": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_tfa",
        "pdb_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_pdb",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/CDD_Tier4/CDD_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/E_score/Tire4.csv"
        },
}

test_sets = {
    # "CDD": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD/CDD_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD/CDD_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD/CDD_emb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/E_score/CDD.csv"
    # },
    # "RV11": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV11_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV11_12_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/Embeddings",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/E_score/RV11_new.csv"
    # },
    # "RV12": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV12_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/RV11_12_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV11_12/Embeddings",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/E_score/RV12_new.csv"
    # },
    # "RV911": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV911/RV911_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV911/RV911_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV911/RV911_emb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/E_score/RV911_new.csv"
    # },
    # "RV912": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV912/RV912_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV912/RV912_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV912/RV912_emb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/E_score/RV912_new.csv"
    # },
    # "RV913": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/RV911_13/RV913/RV913_emb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/Bench_result/E_score/RV913_new.csv"
    # },
    # "CDD": {
    #     "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_25/CDD_25_tfa",
    #     "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_25/CDD_25_ref",
    #     "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_25/CDD_25_emb",
    #     "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/E_score/CDD_25.csv"
    # },
    "CDD_fix": {
        "tfa_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_fix/CDD_fix_tfa",
        "fasta_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_fix/CDD_fix_ref",
        "embedding_dir": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/CDD_fix/CDD_fix_emb",
        "output": "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Final_Bench/E_score/CDD_fix.csv"
    }
}

def run_one_set(set_name, paths, output_file):
    print(f"\n>>> Running test set: {set_name}")
    try:
        avg_f1, avg_precision, avg_recall = process_all_pairs_esm2_cached(
            tfa_dir=paths["tfa_dir"],
            fasta_dir=paths["fasta_dir"],
            embedding_dir=paths["embedding_dir"],
            output_csv_path=paths["output"],
            gap_open=-0.25,
            gap_extend=-0.01,
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
    output_file = "/public/home/lss/Protein_embedding/PLEA/ESP-Align/Test/Final_CDD/RESULT/E_score_result_precious.txt"
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(run_one_set, set_name, paths, output_file)
            for set_name, paths in CDD_sets.items()
        ]

        for future in concurrent.futures.as_completed(futures):
            _ = future.result()  # 触发异常抛出，确保不会静默失败

    print(f"\n✅ All test sets complete. Results written to {output_file}")
