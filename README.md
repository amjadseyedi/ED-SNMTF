# ICASSP 2026

This repository contains code and experiments accompanying our **accepted ICASSP 2026 paper** on  
**Encoder–Decoder Symmetric Nonnegative Matrix Tri-Factorization (ED-SNMTF)** for graph clustering.

## Overview
We introduce an encoder–decoder formulation of symmetric nonnegative matrix tri-factorization (SNMTF) that enforces consistency between graph reconstruction and latent recovery. This coupling yields stable embeddings and well-separated clusters without explicit orthogonality constraints or post-processing.

## Requirements
- Python 3.x  
- PyTorch  
- NumPy  
- SciPy  
- scikit-learn  
- Matplotlib

## Usage
Run the main script after placing the dataset in the expected format:
```bash
python ED-SNMTF.py
```

## Citation
If you use this code, please cite our ICASSP 2026 paper:

```bibtex
@inproceedings{seyedi2026edsnmtf,
  title     = {Encoder--Decoder Symmetric Nonnegative Matrix Tri-Factorization for Graph Clustering},
  author    = {Seyedi, Amjad and Gillis, Nicolas},
  booktitle = {Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026}
}
```

