# ESP-Align
A novel, structure-aware sequence alignment tool that integrates sequence embeddings with structural context, providing both methodological innovation and broad applicability to protein sequence comparison, function inference, and evolutionary analysis. 

好的，我帮你写一个 **科研工具投稿用的 README.md 示例**，针对 ESP-Align，结构清晰、简洁且符合 GitHub 规范，同时适合投稿论文时附带仓库。

---

````markdown
# ESP-Align

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

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

Optional: install as a package:

```bash
python setup.py install
```

---

## 💻 Usage

### 1. Pairwise alignment

```bash
python scripts/run_alignment.py --input data/test_sequences.fasta --method ESP-Align --output results/alignments/
```

* `--input`: Input FASTA file
* `--method`: Alignment method (`ESP-Align`, `E-score`, `NW-BLOSUM`)
* `--output`: Output directory for alignments

### 2. Construct similarity-derived phylogenetic trees

```bash
python scripts/build_tree.py --input results/alignments/ --output results/trees/
```

* Generates distance matrix and phylogenetic tree (Newick format).
* Supports visualization of protein clusters.

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

Refer to `docs/figures` for example alignment visualizations.

---

## 📂 Repository Structure

```
ESP-Align/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── esp_align/        # Source code modules
├── scripts/          # Command-line scripts
├── data/             # Example input sequences
├── benchmarks/       # Benchmark datasets (subset)
├── results/          # Example outputs
├── docs/             # Documentation and figures
└── tests/            # Unit tests
```

---

## 📄 Citation

If you use ESP-Align in your research, please cite:

> Liu S., Lei T., Li Y., Zhao W., Chen Y., He H., Zhang J., Chen J.
> ESP-Align: a structure-aware sequence alignment method integrating ESM-2 embeddings and secondary structure. *Briefings in Bioinformatics*, 2025.

---

## ⚖️ License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📌 Acknowledgements

* **ESM-2**: Facebook AI Research protein language model
* **PyTorch**, **Biopython**, **ESMFold**
* Benchmark datasets: **BAliBASE**, **CDD**

```

---

如果你愿意，我可以帮你再写一个 **简化版的快速上手指南**，用于 GitHub 首页显示，并直接配合论文 Figure 1 的流程图，这样一眼就能看懂工具功能。  

你希望我写吗？
```
