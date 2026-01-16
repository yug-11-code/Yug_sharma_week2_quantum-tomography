import numpy as np
import torch
from time import perf_counter
from scipy.linalg import sqrtm
from src.model import TomographyMLP

def fidelity(rho, sigma):
    
    sqrt_rho = sqrtm(rho)
    inter = sqrtm(sqrt_rho @ sigma @ sqrt_rho)
    val = np.real_if_close(np.trace(inter))
    F = float(np.real(val)**2)
    return max(0.0, min(1.0, F))

def trace_distance(rho, sigma):
    
    delta = rho - sigma
    eigs = np.linalg.eigvals(delta)
    return 0.5 * float(np.sum(np.abs(np.real_if_close(eigs))))

def evaluate(test_path="data/test.npz", model_path="outputs/model.pt", latency_samples=1000):
    d = np.load(test_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.complex128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TomographyMLP(hidden=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    X_t = torch.from_numpy(X).to(device)

    
    M = min(latency_samples, len(X))
    t0 = perf_counter()
    with torch.no_grad():
        _ = model(X_t[:M])
    t1 = perf_counter()
    latency_ms = (t1 - t0) / M * 1000.0

   
    with torch.no_grad():
        rho_pred = model(X_t).cpu().numpy().astype(np.complex128)

    fidelities = []
    distances = []

    for i in range(len(X)):
        fidelities.append(fidelity(rho_pred[i], Y[i]))
        distances.append(trace_distance(rho_pred[i], Y[i]))

    mean_fid = float(np.mean(fidelities))
    mean_tr  = float(np.mean(distances))

    print("Evaluation Results")
    print("Mean Fidelity       :", mean_fid)
    print("Mean Trace Distance :", mean_tr)
    print("Latency (ms/sample) :", latency_ms)

    # save results
    np.savez("outputs/eval_summary.npz",
             fidelities=np.array(fidelities),
             trace_distances=np.array(distances),
             mean_fidelity=mean_fid,
             mean_trace_distance=mean_tr,
             latency_ms=latency_ms)

    print("Saved -> outputs/eval_summary.npz")

if __name__ == "__main__":
    evaluate()
