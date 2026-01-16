import torch
import torch.nn as nn

class TomographyMLP(nn.Module):
    """
    Input:  6 measurement probabilities [p0X,p1X,p0Y,p1Y,p0Z,p1Z]
    Output: complex density matrix rho (2x2), built via Cholesky L -> rho = LL† / Tr(LL†)
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4)  # outputs: t0, t1, cre, cim
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.net(x)

        # Force diagonal terms positive (important for stability)
        t0 = self.softplus(out[:, 0]) + 1e-6
        t1 = self.softplus(out[:, 1]) + 1e-6
        cre = out[:, 2]
        cim = out[:, 3]

        # Build L (lower-triangular complex)
        real = torch.zeros((x.size(0), 2, 2), device=x.device)
        imag = torch.zeros_like(real)

        real[:, 0, 0] = t0
        real[:, 1, 1] = t1
        real[:, 1, 0] = cre
        imag[:, 1, 0] = cim

        L = real + 1j * imag

        # rho = LL† / Tr(LL†)
        M = L @ torch.conj(L).transpose(-2, -1)
        tr = (M[:, 0, 0].real + M[:, 1, 1].real).view(-1, 1, 1)
        rho = M / tr

        return rho
