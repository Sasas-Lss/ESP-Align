# ESP-Align🧬

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Sasas-Lss/ESP-Align/blob/main/LICENSE)


**ESP-Align** is a structure-aware protein sequence alignment tool that integrates **ESM-2 embeddings** with predicted secondary structure to achieve robust alignment performance, even in low sequence identity scenarios. It also supports construction of **similarity-derived phylogenetic trees** for intuitive visualization of protein relationships.

---

## 📌 Features

- Structure-aware global sequence alignment using a modified Needleman-Wunsch algorithm.
- Combines **embedding-based similarity** (from ESM-2) and **structural information** for accurate residue alignment.
- Handles long insertions and low sequence identity (“twilight zone”, 20–35%).
- Generates **normalized similarity scores** for quantitative protein comparison.
- Supports **similarity-derived phylogenetic tree** construction.
- Batch processing for multiple sequences.

---

## 🏗️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ESP-Align.git
cd ESP-Align
````

Install dependencies (Python 3.7+ recommended):

```bash
pip install -r requirements.txt
```

---

### 💻 Usage

### 1. Pairwise alignment using ESP-Align or other methods

```bash
python esp_align/ESP_Align.py \
    --seqs_path test_sequences.fasta \
    --pdb_path pdb_files/ \
    --result_dir results/alignments/
```

**Arguments:**

* `--seqs_path`: Path to the input FASTA file containing sequences.
* `--pdb_path`: Path to the PDB files (leave blank to use ESMFold predictions).
* `--result_dir`: Directory to save the alignment results.


**Example: Pairwise alignment of UDG-like domains**


```bash
python esp_align/ESP_Align.py \
    --seqs_path examples/Case_Studies/UDG-like/UDG_TDG.fasta \
    --result_dir results/Case_Studies/UDG-like/
```
---

### 2. Generating similarity-derived phylogenetic trees

```bash
python esp_align/ESP_Align_tree.py \
    -i test_sequences.fasta \
    -p pdb_files/ \
    -o results/phylogenetic_tree.nwk
```

**Arguments:**

* `-i / --input`: Path to the input FASTA file containing protein sequences.
* `-p / --pdb_path`: Path to the PDB files. Leave blank to automatically use ESMFold for structure prediction.
* `-o / --output_tree`: Path to save the output phylogenetic tree in Newick format.

**Example:**

```bash
python esp_align/ESP_Align_tree.py \
    -i data/test_sequences.fasta \
    -p "" \
    -o results/p53_family_tree.nwk
```

> In this example, no PDB files are provided, so ESP‑Align will use ESMFold predictions to generate the similarity-derived tree.

---

## 🧪 Benchmarking

ESP-Align has been evaluated on:

* **BAliBASE**: classical benchmark for multiple sequence alignment
* **CDD datasets**: conserved domain alignments

ESP-Align outperforms sequence-, structure-, and embedding-based methods across diverse sequence identity tiers.

---

## 🧩 Case Studies

* **UDG-like domain**: Accurate alignment of critical residues and secondary-structure elements.
* **β-trefoil MIR domain**: Correctly handles long insertions and structural variations.

---

## 📂 Repository Structure

```
ESP-Align/
├── README.md
├── LICENSE
├── requirements.txt
├── esp_align/        # Source code modules
├── scripts/          # Command-line scripts
├── examples/         # Example data
├── Benchmarks/       # Benchmark datasets (subset)
└── Bench_results/    # Benchmark outputs
```

---

## 📄 Citation

If you use ESP-Align in your research, please cite:

Liu S., Lei T., Li Y., Zhao W., Chen Y., He H., Zhang J., Chen J., Zeng H

ESP-Align: a structure-aware sequence alignment method integrating ESM-2 embeddings and secondary structure. 
Briefings in Bioinformatics, **submitted**, 2025.

## ⚖️ License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📌 Acknowledgements

* **ESM-2**: Facebook AI Research protein language model
* **PyTorch**, **Biopython**, **ESMFold**
* Benchmark datasets: **BAliBASE**, **CDD**
