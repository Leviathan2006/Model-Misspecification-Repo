"""Train and evaluate a vanilla FNO on one of the four benchmark problems.

Usage:
    python run_fno.py --problem diffusion_reaction
    python run_fno.py --problem burgers --epochs 500
    python run_fno.py --problem cavity_flow
    python run_fno.py --problem hyperelastic

Each problem learns the paper's true solution operator from data:
    diffusion_reaction : v(x)   -> u(x)              (1D, FNO1d)
    burgers            : u0(x)  -> u(x, t)           (2D, FNO2d)
    cavity_flow        : Re     -> (u_x, u_y)(x, y)  (2D, FNO2d)
    hyperelastic       : eps    -> (u_x, u_y)(x, y)  (2D, FNO2d)

Reports the mean relative L2 error on the held-out test set.
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from datasets import burgers as bg
from datasets import cavity_flow as cf
from datasets import diffusion_reaction as dr
from datasets import hyperelastic as he
from fno import FNO1d, FNO2d, rel_l2


def _grid(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def prep_diffusion_reaction(data_dir):
    x, Xtr, Ytr, Xte, Yte = dr.get_dataset(data_dir)
    mu, sd = Xtr.mean(), Xtr.std()
    xg = _grid(x)

    def build(X):
        B, N = X.shape
        g = np.broadcast_to(xg, (B, N))
        return np.stack([(X - mu) / sd, g], axis=-1)      # (B, N, 2)

    model = FNO1d(modes=16, width=64, in_ch=2, out_ch=1)
    return build(Xtr), Ytr[..., None], build(Xte), Yte[..., None], model


def prep_burgers(data_dir):
    t, x, u0tr, fldtr, u0te, fldte = bg.get_dataset(data_dir)
    mu, sd = u0tr.mean(), u0tr.std()
    T, Xg = np.meshgrid(_grid(t), _grid(x), indexing="ij")   # (Nt, Nx)

    def build(u0):
        B, Nt, Nx = u0.shape[0], len(t), len(x)
        u0f = np.broadcast_to(((u0 - mu) / sd)[:, None, :], (B, Nt, Nx))
        Tf = np.broadcast_to(T, (B, Nt, Nx))
        Xf = np.broadcast_to(Xg, (B, Nt, Nx))
        return np.stack([u0f, Tf, Xf], axis=-1)            # (B, Nt, Nx, 3)

    model = FNO2d(modes1=12, modes2=12, width=32, in_ch=3, out_ch=1)
    return build(u0tr), fldtr[..., None], build(u0te), fldte[..., None], model


def _prep_scalar_to_field(coords, Xtr, Ytr, Xte, Yte, out_ch):
    (x, y) = coords
    mu, sd = Xtr.mean(), Xtr.std()
    Yg, Xg = np.meshgrid(_grid(y), _grid(x), indexing="ij")  # (Ny, Nx)

    def build(s):
        B, Ny, Nx = s.shape[0], len(y), len(x)
        sf = np.broadcast_to(((s - mu) / sd)[:, None, None], (B, Ny, Nx))
        Yf = np.broadcast_to(Yg, (B, Ny, Nx))
        Xf = np.broadcast_to(Xg, (B, Ny, Nx))
        return np.stack([sf, Yf, Xf], axis=-1)             # (B, Ny, Nx, 3)

    model = FNO2d(modes1=12, modes2=12, width=32, in_ch=3, out_ch=out_ch)
    return build(Xtr), Ytr, build(Xte), Yte, model


def prep_cavity_flow(data_dir):
    coords, Retr, utr, Rete, ute = cf.get_dataset(data_dir)
    return _prep_scalar_to_field(coords, Retr, utr, Rete, ute, out_ch=2)


def prep_hyperelastic(data_dir):
    coords, etr, utr, ete, ute = he.get_dataset(data_dir)
    return _prep_scalar_to_field(coords, etr, utr, ete, ute, out_ch=2)


PREP = {
    "diffusion_reaction": prep_diffusion_reaction,
    "burgers": prep_burgers,
    "cavity_flow": prep_cavity_flow,
    "hyperelastic": prep_hyperelastic,
}


def evaluate(model, inp, Y, device, bs=50):
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(inp), bs):
            xb, yb = inp[i:i + bs].to(device), Y[i:i + bs].to(device)
            tot += rel_l2(model(xb), yb).item() * len(xb)
    return tot / len(inp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problem", choices=list(PREP), default="diffusion_reaction")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inp_tr, Ytr, inp_te, Yte, model = PREP[a.problem](a.data_dir)
    inp_tr = torch.tensor(np.ascontiguousarray(inp_tr), dtype=torch.float32)
    Ytr = torch.tensor(np.ascontiguousarray(Ytr), dtype=torch.float32)
    inp_te = torch.tensor(np.ascontiguousarray(inp_te), dtype=torch.float32)
    Yte = torch.tensor(np.ascontiguousarray(Yte), dtype=torch.float32)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loader = DataLoader(TensorDataset(inp_tr, Ytr), batch_size=a.batch,
                        shuffle=True)

    n_params = sum(q.numel() for q in model.parameters())
    print(f"[{a.problem}] device={device}  params={n_params:,}  "
          f"train={len(Ytr)}  test={len(Yte)}  in={tuple(inp_tr.shape[1:])}")

    for epoch in range(a.epochs):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = rel_l2(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        sched.step()
        if (epoch + 1) % max(1, a.epochs // 10) == 0 or epoch == 0:
            model.eval()
            te = evaluate(model, inp_te, Yte, device)
            print(f"  epoch {epoch+1:4d}  train relL2 {tot/len(Ytr):.4e}  "
                  f"test relL2 {te:.4e}")

    model.eval()
    test_err = evaluate(model, inp_te, Yte, device)
    print(f"[{a.problem}] FINAL test relative L2 = {test_err:.4e}")
    os.makedirs("results", exist_ok=True)
    with open("results/summary.txt", "a") as f:
        f.write(f"{a.problem}\tepochs={a.epochs}\ttest_relL2={test_err:.6e}\n")
    return test_err


if __name__ == "__main__":
    main()
