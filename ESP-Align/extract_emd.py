from Bio import SeqIO
import torch
import esm
import re
import numpy as np
from scipy.spatial.distance import pdist, squareform

def get_embs_ProtT5(model, tokenizer, sequences, n, device=None):
    """
    取前 n 条序列，返回它们的残基级 embedding 列表（Tensor 形式）
    每个序列的维度为 (L, D=1024)
    """
    if device is None:
        device = next(model.parameters()).device

    sequences = sequences[:n]
    processed_seqs = []
    for seq in sequences:
        seq = re.sub(r"[UZOB]", "X", seq.upper())
        processed_seqs.append(" ".join(list(seq)))  # 空格分隔氨基酸

    # 编码为 tokens
    inputs = tokenizer(processed_seqs, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        reps = outputs.last_hidden_state  # (batch_size, L_padded, 1024)

    # 去除 [EOS] token（最后一个位置）
    final_embs = []
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        emb = reps[i, :seq_len, :].cpu()  # (L, 1024)
        final_embs.append(emb)

    return final_embs

def ESM2_initialize(device=None):
    """
    返回：ESM2 模型（已搬到 device）和 batch_converter
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"ESM2 initialize on {device}")

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    print(f"ESM2 initialize completed on {device}")
    return model, batch_converter


def get_embs_ESM2(model, batch_converter, sequences, n, device=None):
    """
    取前 n 条序列，返回它们的第 33 层残基级 embedding 列表（Tensor 形式）
    """
    if device is None:
        device = next(model.parameters()).device

    sequences = sequences[:n]
    data = [("", sequences[0])]
    labels, strs, tokens = batch_converter(data)

    # 把 tokens 也搬到 GPU
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[33], return_contacts=False)
    reps = results["representations"][33]  # (batch, L+2, D)

    # 取掉 cls/eos，返回 (L, D)
    # 这个GPU这点值得注意
    final_embs = [reps[i, 1:-1].cpu() for i in range(reps.size(0))]
    return final_embs


def load_fasta(path):
    fasta_sequences = SeqIO.parse(open(path), 'fasta')
    sequences = []

    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        sequences.append((name, sequence.upper()))

    return sequences

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

def extract_emd(seqs_path):
    seqs = load_fasta_nor(seqs_path)
    if len(seqs) < 2:
        raise ValueError(f"Not enough sequences in {seqs_path}")
    seq_1 = seqs[0][1]
    seq_2 = seqs[1][1]

    name1 = seqs[0][0]
    name2 = seqs[1][0]

    Model, Model_tokenizer = ESM2_initialize()

    emb1 = get_embs_ESM2(Model, Model_tokenizer, [seq_1], 1)[0]
    emb2 = get_embs_ESM2(Model, Model_tokenizer, [seq_2], 1)[0]


    return [seq_1, name1, emb1], [seq_2, name2, emb2]


def extract_emd_model(seqs_path, Model, Model_tokenizer):
    seqs = load_fasta(seqs_path)
    seq_1 = seqs[0][1]
    seq_2 = seqs[1][1]

    name1 = seqs[0][0]
    name2 = seqs[1][0]

    # Model, Model_tokenizer = ESM2_initialize()

    emb1 = get_embs_ESM2(Model, Model_tokenizer, [seq_1], 1)[0]
    emb2 = get_embs_ESM2(Model, Model_tokenizer, [seq_2], 1)[0]

    return [seq_1, name1, emb1], [seq_2, name2, emb2]
