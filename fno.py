"""Vanilla Fourier Neural Operators (Li et al., 2021), 1D and 2D."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- 1D
class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat)
        )

    def forward(self, x):                                # (B, in_ch, N)
        B, _, N = x.shape
        x_ft = torch.fft.rfft(x)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.weight.shape[1], x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out_ft, n=N)


class FNO1d(nn.Module):
    def __init__(self, modes=16, width=64, in_ch=2, out_ch=1, n_layers=4):
        super().__init__()
        self.fc0 = nn.Linear(in_ch, width)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.w = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_ch)

    def forward(self, x):                                # (B, N, in_ch)
        x = self.fc0(x).permute(0, 2, 1)
        for s, w in zip(self.spectral, self.w):
            x = F.gelu(s(x) + w(x))
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)                               # (B, N, out_ch)


# --------------------------------------------------------------------------- 2D
class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, m1, m2, dtype=torch.cfloat))

    def forward(self, x):                                # (B, in_ch, H, W)
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)                        # (B, in_ch, H, W//2+1)
        m1 = min(self.m1, H // 2)
        m2 = min(self.m2, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.w1.shape[1], H, x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=32, in_ch=3, out_ch=1,
                 n_layers=4):
        super().__init__()
        self.fc0 = nn.Linear(in_ch, width)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)])
        self.w = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_ch)

    def forward(self, x):                                # (B, H, W, in_ch)
        x = self.fc0(x).permute(0, 3, 1, 2)              # (B, width, H, W)
        for s, w in zip(self.spectral, self.w):
            x = F.gelu(s(x) + w(x))
        x = x.permute(0, 2, 3, 1)                        # (B, H, W, width)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)                               # (B, H, W, out_ch)


def rel_l2(pred, true):
    """Mean relative L2 error, flattening everything but the batch axis."""
    B = pred.shape[0]
    p, t = pred.reshape(B, -1), true.reshape(B, -1)
    return (torch.linalg.norm(p - t, dim=1) /
            torch.linalg.norm(t, dim=1)).mean()
