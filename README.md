# ESP-Align🧬

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Sasas-Lss/ESP-Align/blob/main/LICENSE)


**ESP-Align** is a structure-aware protein sequence alignment tool that integrates **protein language model embeddings** with  **protein structure information** to achieve robust alignment performance, even in low sequence identity scenarios. It also supports construction of **similarity-derived phylogenetic trees** for intuitive visualization of protein relationships.

---

## 📌 Features

- Structure-aware global sequence alignment using a modified Needleman-Wunsch algorithm.
- Combines **embedding-based similarity** and **structural information** for accurate residue alignment.
- Handles long insertions and low sequence identity (“twilight zone”, 20–35%).
- Generates **normalized similarity scores** for quantitative protein comparison.
- Supports **similarity-derived phylogenetic tree** construction.
- Batch processing for multiple sequences.

---

### 🔧 External Dependencies

**ESP-Align** relies on two external tools for **embedding extraction** and **structure information analysis**:


#### 🧠 ESM (Evolutionary Scale Modeling)

Used for **protein embedding extraction** and **structure prediction** (via **ESM-2** and **ESMFold**).
For detailed installation and model setup instructions, please refer to the official repository:
👉 [https://github.com/facebookresearch/esm/tree/main](https://github.com/facebookresearch/esm/tree/main)

After installation, verify that ESMFold is correctly configured and globally accessible by running:

```bash
esm-fold -h
```
If the command displays the help information, it indicates that ESMFold can be executed from any working directory.

#### 🧩 Stride

Used for **secondary structure assignment** from PDB files.

Download and compile following the official instructions:

```bash
wget http://webclu.bio.wzw.tum.de/stride/stride.tar.gz
mkdir stride
tar -xzf stride.tar.gz -C ./stride
cd stride
make
```

Ensure `stride` is available in your system `PATH`:

```bash
echo 'export PATH=$PATH:'$(pwd) >> ~/.bashrc
source ~/.bashrc
```
After installation, confirm that Stride is globally available by running:

```bash
stride -h
```
If the command outputs the usage information, Stride has been correctly added to your system PATH.

> ⚠️ **ESP-Align** automatically calls **ESM-2/ESMFold** for structure prediction when no PDB files are provided, and **Stride** for extracting secondary structure features.

---

## 🏗️ Installation

Clone the repository:

```bash
git clone https://github.com/Sasas-Lss/ESP-Align.git
cd ESP-Align
````

Install dependencies (Python 3.7+ recommended):

```bash
pip install -r requirements.txt
```

---

### 💻 Usage

### 1. Pairwise alignment using ESP-Align

Use the following command format to run pairwise alignment with **ESP-Align**:

```bash
python esp_align/ESP_Align.py \
    -i <input_fasta> \
    -p <pdb_directory> \
    -o <output_directory>
```

**Arguments:**

* `-i / --input`: Path to the input FASTA file containing two protein sequences to be aligned.
* `-p / --pdb_path`: Path to the PDB files.

  
  > **Note:** Each PDB file name **must match the corresponding sequence ID** in the FASTA file (e.g., >seq1 → `seq1.pdb`).
  > Leave this argument blank to automatically use **ESMFold** for structure prediction.
* `-o / --output`: Directory to save the alignment results.
* `-h / --help`: Show the full list of available options and exit.

**Optional parameters (with default values):**

* `--pearson_weight`: Weight for Pearson similarity (default: `0.8`).
* `--Helix`: Score adjustment for helix structure (default: `-5.0`).
* `--Strand`: Score adjustment for strand structure (default: `-3.0`).
* `--Coil`: Score adjustment for coil structure (default: `-1.0`).
* `--gap_ext`: Gap extension penalty (default: `0.0`).

---

**Example: Pairwise alignment of UDG-like domains**

```bash
python esp_align/ESP_Align.py \
    -i ./examples/Case_Studies/UDG-like/UDG-like.fasta \
    -o ./results/Case_Studies/UDG-like/ \
    --pearson_weight 0.8 \
    --Helix -5.0 
```

> In this example, ESP-Align performs a structure-aware pairwise alignment using user-defined weights for embedding and structural components.

---

### 2. Generating similarity-derived phylogenetic trees
Use the following command to generate a phylogenetic tree based on **structure-aware sequence similarity**:

```bash
python esp_align/ESP_Align_tree.py \
    -i <input_fasta> \
    -p <pdb_directory> \
    -o <output_tree_file>
```

**Arguments:**

* `-i / --input`: Path to the input FASTA file containing protein sequences used for tree construction.
* `-p / --pdb_path`: Path to the PDB files. Leave blank to automatically use ESMFold for structure prediction.

  > **Note:** File names should be consistent with sequence IDs in the FASTA file (e.g., `proteinA.pdb` for `proteinA` in FASTA).
  > Leave blank to automatically use **ESMFold** for structure prediction.

* `-o / --output_tree`: Path to save the output phylogenetic tree in Newick format.
* `-h / --help`: Show the full list of available options and exit.

**Example:**

```bash
python esp_align/ESP_Align_tree.py \
    -i ./examples/Tree_Construction/20_protein_sequences.fasta \
    -p ./examples/Tree_Construction/Structure \
    -o ./results/similarity_tree.nwk
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

ESP-Align: a structure-aware sequence alignment method integrating protein language model embeddings and structure information.(Submitted).

## ⚖️ License

This project is licensed under the **Apache License**. See [LICENSE](https://github.com/Sasas-Lss/ESP-Align/blob/main/LICENSE) for details.

---

## 📌 Acknowledgements

* **ESM-2**: Facebook AI Research protein language model
* **PyTorch**, **Biopython**, **ESMFold**
* Benchmark datasets: **BAliBASE**, **CDD**
