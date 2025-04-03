# EvoLSTM: Sequence-to-Sequence LSTM-Based Evolution Simulator

[![Paper](https://img.shields.io/badge/Paper-Bioinformatics-brightgreen)](https://academic.oup.com/bioinformatics/article/36/Supplement_1/i353/5870475)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)

## Overview

EvoLSTM is a sophisticated deep learning framework that simulates DNA sequence evolution using Long Short-Term Memory (LSTM) networks. This sequence-to-sequence model captures complex mutational patterns and context dependencies to provide realistic evolutionary simulations. EvoLSTM requires an external Nvidia GPU for training and simulation due to the computational demands of the LSTM architecture.

For a comprehensive understanding of the methodology and results, please refer to our publication in Bioinformatics: [EvoLSTM: Context-dependent models for sequence evolution using LSTM neural networks](https://academic.oup.com/bioinformatics/article/36/Supplement_1/i353/5870475)

## Requirements

- NVIDIA GPU (required for training and simulation)
- Python 3.6+
- TensorFlow 2.0+
- Additional dependencies listed in `requirements.txt`

## Getting Started

### 1. Cloning the Repository

```bash
git clone https://github.com/DongjoonLim/EvoLSTM.git
cd EvoLSTM
```

### 2. Setting Up Directories

Create the necessary directories for storing data, preprocessed files, models, and simulation outputs:

```bash
mkdir data
mkdir prepData
mkdir models
mkdir simulation
```

### 3. Downloading Training Data

The training data consists of sequence alignment files and phylogenetic tree information:

- **Sequence Alignment MAF Files**: Download from [McGill University Repository](http://repo.cs.mcgill.ca/PUB/blanchem/Boreoeutherian/) and place them in the `data` directory.

- **Phylogenetic Tree Structure**: Access the tree structure and species nomenclature from [UCSC Genome Browser](http://hgdownload.cse.ucsc.edu/goldenpath/hg38/multiz100way/hg38.100way.nh).

**Note:** Ancestral sequences are labeled with a prefix `_` followed by the first characters of the descendant species. For example, the most recent common ancestor of hg38 (human) and pantro4 (chimpanzee) is labeled as `_HP`.

### 4. Installing Dependencies

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Preprocessing Sequences

Generate meta-nucleotide sequences for training:

```bash
python3 prep_insert2.py <chromosome> <ancName> <desName>
```

**Parameters:**
- `chromosome`: The chromosome number
- `ancName`: The name of the ancestral sequence
- `desName`: The name of the descendant sequence

**Example:** To preprocess human chromosome 2 from the most recent common ancestor of hg38 and pantro4 evolving to hg38:

```bash
python3 prep_insert2.py 2 _HP hg38
```

### 2. Training EvoLSTM

Train the EvoLSTM model with preprocessed sequences:

```bash
python3 insert2Train_general.py <ancName> <desName> <train_size> <seq_length>
```

**Parameters:**
- `ancName`: The name of the ancestral sequence
- `desName`: The name of the descendant sequence
- `train_size`: The length of the training sequence (recommended starting point: 100,000)
- `seq_length`: The context length of the sequence (recommended: 15)

**Example:**

```bash
python3 insert2Train_general.py _HP hg38 100000 15
```

### 3. Simulating Sequence Evolution

Simulate sequence evolution with the trained model:

```bash
python3 simulate.py <ancName> <desName> <sample_size> <gpu_index> <chromosome>
```

**Parameters:**
- `ancName`: The name of the ancestral sequence
- `desName`: The name of the descendant sequence
- `sample_size`: Desired input sequence length
- `gpu_index`: GPU card index (use `nvidia-smi` to find available GPUs; set to 0 if only one GPU is available)
- `chromosome`: Chromosome number for the simulation

**Example:** To simulate the first 100,000 base pairs of the `_HP` sequence in chromosome 2:

```bash
python3 simulate.py _HP hg38 100000 0 2
```

### 4. Reading Simulation Output

The simulation output will be saved as `simulated_{ancName}_{desName}_{chromosome}.npy`. To read this file:

```python
import numpy as np
simulation_data = np.load('simulated__HP_hg38_2.npy')
```

## Citation

If you use EvoLSTM in your research, please cite:

```bibtex
@article{10.1093/bioinformatics/btaa440,
    author = {Lim, Dongjoon and Kılıç, Ayşe and Liò, Pietro and Won, Kyoung-Jae},
    title = "{EvoLSTM: Context-dependent models for sequence evolution using LSTM neural networks}",
    journal = {Bioinformatics},
    volume = {36},
    number = {Supplement_1},
    pages = {i353-i361},
    year = {2020},
    doi = {10.1093/bioinformatics/btaa440}
}
```

## Contact

For questions, issues, or contributions, please open an issue on the [GitHub repository](https://github.com/DongjoonLim/EvoLSTM/issues).
