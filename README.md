# Week 2 — Learning Quantum State Tomography (Track 2)

This repository trains a neural network to perform single-qubit quantum state tomography.

## Approach
- Measurement model: Pauli projective measurements (X, Y, Z bases)
- Input features: 6 measurement probabilities
- Model: MLP (6 → 64 → 64 → 4)
- Output parameterization: Cholesky lower-triangular matrix L
- Density matrix reconstruction:
  \[
  \rho = \frac{LL^\dagger}{\mathrm{Tr}(LL^\dagger)}
  \]

## How to run
```bash
pip install -r requirements.txt
python src/data_gen.py --train_n 10000 --test_n 2000 --shots 256 --depolarize 0.01
python -m src.train
python -m src.eval
