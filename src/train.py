import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from src.model import TomographyMLP

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = torch.from_numpy(d["X"].astype(np.float32))           # (N,6)
    Y = torch.from_numpy(d["Y"].astype(np.complex64))         # (N,2,2)
    return X, Y

def train_model(
    train_path="data/train.npz",
    epochs=20,
    batch_size=128,
    lr=2e-3,
    out_path="outputs/model.pt"
):
    Path("outputs").mkdir(exist_ok=True)

    X, Y = load_npz(train_path)
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TomographyMLP(hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for xb, yb in tqdm(dl, desc=f"Epoch {ep}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)

            # Frobenius loss: ||pred - true||^2
            loss = torch.mean(torch.sum(torch.abs(pred - yb)**2, dim=(1,2)))

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(f"Epoch {ep}: mean loss = {np.mean(losses):.6f}")

    torch.save(model.state_dict(), out_path)
    print("Model saved to:", out_path)

if __name__ == "__main__":
    train_model()
