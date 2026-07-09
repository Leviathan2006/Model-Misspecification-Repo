"""Physics-guided correction under model misspecification, on an FNO backbone.

Implements the method of Ma, Boullé, Yang, Wu & Guo (2026), arXiv:2606.03469,
for the 1D diffusion-reaction benchmark, replacing their physics-informed
DeepONet with a physics-informed FNO (PINO-style residual on the grid).

True model      :  D u_xx - k_r(u) u = v,   k_r(u) = 0.5 exp(-u),   D = 0.1
Misspecified N0 :  D u_xx - k_r_const = v   (reaction term collapsed to a const)

Networks
    prior      G_theta : v(x)            -> u_prior(x)
    correction G_psi   : [v, u_prior]    -> forcing-space correction c(x)

Losses (paper Eqs.)
    L_data    = || G_theta(v)(y_obs) - u(y_obs) ||^2          (N_u sparse obs)
    L_physics = || N0[G_theta(v)] + G_psi(...) - v ||^2       (collocation)
    L         = L_physics + lambda_d * L_data

Test prediction: the paper writes N#[v] ~ G_theta + G_psi. Because G_psi is
trained in forcing space, this is subtle, so we report BOTH the prior-only error
and the prior+correction error.

Three modes reproduce the paper's headline comparison:
    known         : train the prior with the TRUE physics residual (reference)
    misspecified  : train the prior with the wrong N0 (no correction) -> blows up
    corrected     : prior + correction with the N0 residual  -> recovered

Notes on the FNO adaptation: the FNO output lives on the fixed spatial grid, so
the differential operator is applied by finite differences and the collocation
set is the interior grid (the paper's 1000 mesh-free collocation points are a
DeepONet detail). Observations are a random N_u-subset of the grid.
"""
import argparse
import os

import numpy as np
import torch

from fno import FNO1d, rel_l2

D = 0.1
K_R_CONST = 0.5           # constant used by the misspecified operator N0
LAMBDA_D = 1.0            # weight on the data loss
N_U = 100                # number of sparse solution observations


def second_derivative(u, dx):
    """Central finite-difference u_xx on a uniform grid. u: (B, N)."""
    uxx = torch.zeros_like(u)
    uxx[:, 1:-1] = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / dx**2
    return uxx


def N_true(u, dx):
    return D * second_derivative(u, dx) - 0.5 * torch.exp(-u) * u


def N_mis(u, dx):
    return D * second_derivative(u, dx) - K_R_CONST


def make_input(field, grid):
    """Stack [field, grid] -> (B, N, 2) for a 1-channel input field."""
    B, N = field.shape
    g = grid.view(1, N).expand(B, N)
    return torch.stack([field, g], dim=-1)


def train_mode(mode, v_tr, u_tr, v_te, u_te, x, epochs, lr, device, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    N = x.shape[0]
    dx = float(x[1] - x[0])
    gridt = torch.tensor((x - x.min()) / (x.max() - x.min()),
                         dtype=torch.float32, device=device)
    xg = torch.tensor(x, dtype=torch.float32, device=device)          # physical x

    v_tr = torch.tensor(v_tr, dtype=torch.float32, device=device)
    u_tr = torch.tensor(u_tr, dtype=torch.float32, device=device)
    v_te = torch.tensor(v_te, dtype=torch.float32, device=device)
    u_te = torch.tensor(u_te, dtype=torch.float32, device=device)

    # normalise the input source with train statistics
    mu, sd = v_tr.mean(), v_tr.std()
    vin_tr, vin_te = (v_tr - mu) / sd, (v_te - mu) / sd

    interior = torch.arange(1, N - 1, device=device)                  # collocation
    rng = np.random.default_rng(seed)
    obs = torch.tensor(np.sort(rng.choice(N, size=min(N_U, N), replace=False)),
                       device=device)

    prior = FNO1d(modes=16, width=64, in_ch=2, out_ch=1).to(device)
    corr = FNO1d(modes=16, width=64, in_ch=3, out_ch=1).to(device)
    params = list(prior.parameters())
    if mode == "corrected":
        params += list(corr.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def forward(vin, vraw):
        up = prior(make_input(vin, gridt)).squeeze(-1)                # (B, N)
        c = torch.zeros_like(up)
        if mode == "corrected":
            ci = torch.stack([vin, up.detach(), gridt.expand_as(up)], dim=-1)
            c = corr(ci).squeeze(-1)
        # misspecified/corrected residual uses N0; known uses N_true
        if mode == "known":
            res = N_true(up, dx) - vraw
        else:
            res = N_mis(up, dx) + c - vraw
        return up, c, res

    for epoch in range(epochs):
        prior.train(); corr.train()
        opt.zero_grad()
        up, c, res = forward(vin_tr, v_tr)
        L_phys = (res[:, interior] ** 2).mean()
        L_data = ((up[:, obs] - u_tr[:, obs]) ** 2).mean()
        loss = L_phys + LAMBDA_D * L_data
        loss.backward()
        opt.step(); sched.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"  [{mode}] epoch {epoch+1:5d}  L_phys {L_phys.item():.3e}  "
                  f"L_data {L_data.item():.3e}")

    prior.eval(); corr.eval()
    with torch.no_grad():
        up, c, _ = forward(vin_te, v_te)
        err_prior = rel_l2(up, u_te).item()
        err_full = rel_l2(up + c, u_te).item() if mode == "corrected" else err_prior
    return err_prior, err_full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["known", "misspecified", "corrected", "all"],
                   default="all")
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data_dir", type=str, default="data")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(os.path.join(a.data_dir, "diffusion_reaction.npz"))
    x = d["x"]
    args = (d["v_train"], d["u_train"], d["v_test"], d["u_test"], x)
    print(f"diffusion-reaction physics-guided correction (FNO) on {device}; "
          f"train={d['v_train'].shape[0]}")

    modes = ["known", "misspecified", "corrected"] if a.mode == "all" else [a.mode]
    results = {}
    for m in modes:
        ep, ef = train_mode(m, *args, epochs=a.epochs, lr=a.lr, device=device)
        results[m] = (ep, ef)

    print("\n==== relative L2 on test (diffusion-reaction) ====")
    for m in modes:
        ep, ef = results[m]
        if m == "corrected":
            print(f"  {m:13s} prior-only {ep:.4e}  [recovered solution]   "
                  f"prior+correction {ef:.4e}  [literal Eq.(c)]")
        else:
            print(f"  {m:13s} {ep:.4e}")
    print("\nReadout: for 'corrected', PRIOR-ONLY is the recovered solution "
          "(G_psi is a forcing-space residual correction, so adding it to the\n"
          "solution as in the literal Eq.(c) is dimensionally off and is reported "
          "only as a diagnostic).")
    print("(paper DeepONet reference: misspecified ~1.6e0, corrected ~1.85e-3, "
          "known ~9.0e-4)")


if __name__ == "__main__":
    main()
