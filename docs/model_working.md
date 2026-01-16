# Model Working

## Inputs
Each training example consists of 6 features based on Pauli measurement probabilities:

x = [p(0|X), p(1|X), p(0|Y), p(1|Y), p(0|Z), p(1|Z)]

Shot noise and depolarizing noise can optionally be added.

## Model
We use a small MLP:

- Input: 6
- Hidden: 64 (ReLU)
- Hidden: 64 (ReLU)
- Output: 4 real values

## Output parameterization (physical density matrix)
The model predicts parameters of a lower-triangular matrix:

L = [[a, 0],
     [c + i d, b]]

The density matrix is built as:

rho = (L L†) / Tr(L L†)

This guarantees rho is always:
- Hermitian
- Positive semidefinite
- Trace 1

## Training Loss
The training loss is Frobenius distance:

L = ||rho_pred - rho_true||_F^2

## Results
Mean Fidelity: 0.9960  
Mean Trace Distance: 0.0370  
Latency: 0.00271 ms/sample
