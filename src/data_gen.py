import numpy as np
from pathlib import Path
import argparse

def random_pure_state():
    v = (np.random.randn(2) + 1j*np.random.randn(2))
    v = v / np.linalg.norm(v)
    rho = np.outer(v, v.conj())
    return rho

def depolarize(rho, p):
    I = np.eye(2, dtype=complex)
    return (1 - p) * rho + p * I / 2

def measure_pauli_probs(rho):
    P0z = np.array([[1,0],[0,0]], dtype=complex)
    P1z = np.array([[0,0],[0,1]], dtype=complex)

    plus = (1/np.sqrt(2))*np.array([1,1], dtype=complex)
    minus = (1/np.sqrt(2))*np.array([1,-1], dtype=complex)
    P0x = np.outer(plus, plus.conj())
    P1x = np.outer(minus, minus.conj())

    i_state = (1/np.sqrt(2))*np.array([1,1j], dtype=complex)
    minus_i = (1/np.sqrt(2))*np.array([1,-1j], dtype=complex)
    P0y = np.outer(i_state, i_state.conj())
    P1y = np.outer(minus_i, minus_i.conj())

    p0z = float(np.real(np.trace(P0z @ rho)))
    p1z = float(np.real(np.trace(P1z @ rho)))
    p0x = float(np.real(np.trace(P0x @ rho)))
    p1x = float(np.real(np.trace(P1x @ rho)))
    p0y = float(np.real(np.trace(P0y @ rho)))
    p1y = float(np.real(np.trace(P1y @ rho)))

    return np.array([p0x,p1x,p0y,p1y,p0z,p1z], dtype=np.float32)

def generate_dataset(n_samples, shots=0, depolarize_p=0.0, outpath="data/train.npz", seed=42):
    np.random.seed(seed)
    X = np.zeros((n_samples,6), dtype=np.float32)
    Y = np.zeros((n_samples,2,2), dtype=np.complex128)

    for i in range(n_samples):
        rho = random_pure_state()

        if depolarize_p > 0:
            rho = depolarize(rho, depolarize_p)

        probs = measure_pauli_probs(rho)

        if shots and shots > 0:
            noisy = []
            for b in range(3):
                p0 = probs[2*b]; p1 = probs[2*b+1]
                counts = np.random.multinomial(shots, [p0, p1])
                p0_hat = counts[0]/shots; p1_hat = counts[1]/shots
                noisy.extend([p0_hat,p1_hat])
            probs = np.array(noisy, dtype=np.float32)

        X[i] = probs
        Y[i] = rho

        if (i+1) % 1000 == 0:
            print(f"generated {i+1}/{n_samples}")

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    np.savez(outpath, X=X, Y=Y)
    print(f"Saved {outpath} with {n_samples} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_n", type=int, default=10000)
    parser.add_argument("--test_n", type=int, default=2000)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--depolarize", type=float, default=0.01)
    args = parser.parse_args()

    generate_dataset(args.train_n, shots=args.shots, depolarize_p=args.depolarize,
                     outpath="data/train.npz", seed=1)

    generate_dataset(args.test_n, shots=args.shots, depolarize_p=args.depolarize,
                     outpath="data/test.npz", seed=2)
