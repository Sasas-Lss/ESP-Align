from Bio import SeqIO
import torch
import esm
import re
import numpy as np
from scipy.spatial.distance import pdist, squareform


def get_embs_ProtT5(model, tokenizer, sequences, n, device=None):
    """
    Generate ProtT5 embeddings for the first n sequences.
    Each sequence embedding has shape (L, 1024).
    """
    if device is None:
        device = next(model.parameters()).device

    sequences = sequences[:n]
    processed = [" ".join(re.sub(r"[UZOB]", "X", seq.upper())) for seq in sequences]

    inputs = tokenizer(processed, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        reps = outputs.last_hidden_state  # (batch_size, L_padded, 1024)

    final_embs = []
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        emb = reps[i, :seq_len, :].cpu()
        final_embs.append(emb)

    return final_embs


def ESM2_initialize(device=None):
    """
    Initialize the ESM-2 model and return the model and batch converter.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Initializing ESM-2 on {device}")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    print("ESM-2 initialization completed.")
    return model, batch_converter


def get_embs_ESM2(model, batch_converter, sequences, n, device=None):
    """
    Generate ESM-2 embeddings for the first n sequences (layer 33).
    Returns a list of tensors (L, D).
    """
    if device is None:
        device = next(model.parameters()).device

    sequences = sequences[:n]
    data = [("", sequences[0])]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[33], return_contacts=False)
    reps = results["representations"][33]  # (batch, L+2, D)

    final_embs = [reps[i, 1:-1].cpu() for i in range(reps.size(0))]
    return final_embs


def load_fasta(path):
    """
    Load all sequences from a FASTA file.
    Returns a list of (name, sequence) tuples.
    """
    fasta_sequences = SeqIO.parse(open(path), "fasta")
    return [(fasta.id, str(fasta.seq).upper()) for fasta in fasta_sequences]


def load_fasta_nor(path):
    """
    Load and normalize FASTA sequences, removing those with illegal amino acids.
    Returns a list of valid (name, sequence) tuples.
    """
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    sequences = []

    for record in SeqIO.parse(path, "fasta"):
        name = record.id
        sequence = str(record.seq).upper()
        invalid_chars = set(sequence) - valid_aas
        if invalid_chars:
            print(f"[WARNING] Illegal characters in sequence '{name}': {''.join(invalid_chars)}")
            continue
        sequences.append((name, sequence))

    return sequences


def extract_emd(seqs_path):
    """
    Extract embeddings for the first two sequences in a FASTA file using ESM-2.
    Returns ([seq1, name1, emb1], [seq2, name2, emb2]).
    """
    seqs = load_fasta_nor(seqs_path)
    if len(seqs) < 2:
        raise ValueError(f"Not enough sequences in {seqs_path}")

    (name1, seq1), (name2, seq2) = seqs[:2]
    model, tokenizer = ESM2_initialize()

    emb1 = get_embs_ESM2(model, tokenizer, [seq1], 1)[0]
    emb2 = get_embs_ESM2(model, tokenizer, [seq2], 1)[0]

    return [seq1, name1, emb1], [seq2, name2, emb2]


def extract_emd_model(seqs_path, model, tokenizer):
    """
    Extract embeddings for the first two sequences in a FASTA file
    using a preloaded ESM-2 model.
    """
    seqs = load_fasta(seqs_path)
    (name1, seq1), (name2, seq2) = seqs[:2]

    emb1 = get_embs_ESM2(model, tokenizer, [seq1], 1)[0]
    emb2 = get_embs_ESM2(model, tokenizer, [seq2], 1)[0]

    return [seq1, name1, emb1], [seq2, name2, emb2]
