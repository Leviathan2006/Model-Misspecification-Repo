"""Our method 3: Denoising-diffusion model of the correction (missing physics).

The paper's correction G_psi is a single deterministic field. But the correction
has a well-defined target -- the physics-implied residual correction

    c*(x) = v(x) - N0[G_theta](x)   ( = the missing physics DeltaN[u], which here
                                       is analytically  k_const - 0.5 e^{-u} u )

so we can train a genuine conditional DIFFUSION model to sample it. We condition
the denoiser on the SPARSE observations (not the full input), which is what makes
the correction genuinely under-determined -> a real distribution over the model
error. Sampling K corrections gives:

  * a MEAN that should recover the true missing physics DeltaN[u], and
  * a STD that is a calibrated map of *where* the recovered model error is
    uncertain -- checked against the analytically known true discrepancy.

The solution itself is still read off the deterministic prior G_theta (as the
paper actually does; the correction is a forcing-space term). The diffusion adds
distributional model-error UQ on top -- the trustworthy-operator-learning payoff.

Results -> results/method_diffusion.txt.  (Exploratory v1.)
"""
import argparse
import datetime
import math
import os
import time

import numpy as np
import torch

import datasets.diffusion_reaction as dr
from deeponet import MLP, DeepONet, enable_fast_math, trunk_derivatives
from run_correction import (FULL, K_R_CONST, LAMBDA_D, QUICK, _hard_bc,
                            build_data, rel_l2)

T_DIFF = 50
D_COND = 32


def cosine_abar(T, device, s=0.008):
    ts = torch.arange(T + 1, device=device) / T
    f = torch.cos((ts + s) / (1 + s) * math.pi / 2) ** 2
    abar = (f / f[0])[1:].clamp(1e-5, 0.9999)          # (T,)
    return abar


class Denoiser(torch.nn.Module):
    """x0-prediction denoiser for the correction field, conditioned on sparse obs.
    Input per (sample, point): [x, c_t, t/T, cond]. Output: predicted c*."""

    def __init__(self, n_obs, width=128):
        super().__init__()
        self.enc = MLP([n_obs, 64, D_COND])
        self.net = MLP([3 + D_COND, width, width, width, 1])

    def forward(self, x, c_t, t_norm, cond):
        # x:(Q,) c_t:(M,Q) t_norm:(M,) cond:(M,D_COND)
        M, Q = c_t.shape
        xg = x.view(1, Q, 1).expand(M, Q, 1)
        ct = c_t.unsqueeze(-1)
        tg = t_norm.view(M, 1, 1).expand(M, Q, 1)
        cg = cond.view(M, 1, D_COND).expand(M, Q, D_COND)
        return self.net(torch.cat([xg, ct, tg, cg], -1)).squeeze(-1)   # (M,Q)


@torch.no_grad()
def sample(den, cond, x_query, abar, n_samples, device):
    """DDPM ancestral sampling (x0-param). Returns (K, M, Q) correction samples."""
    abar_prev = torch.cat([torch.ones(1, device=device), abar[:-1]])
    M, Q = cond.shape[0], x_query.shape[0]
    out = []
    for _ in range(n_samples):
        c = torch.randn(M, Q, device=device)
        for i in reversed(range(len(abar))):
            t_norm = torch.full((M,), (i + 1) / len(abar), device=device)
            x0 = den(x_query, c, t_norm, cond)
            at, ap, ab = abar[i], abar_prev[i], abar[i]
            alpha = at / ap
            beta = 1 - alpha
            mean = (torch.sqrt(ap) * beta / (1 - ab)) * x0 \
                + (torch.sqrt(alpha) * (1 - ap) / (1 - ab)) * c
            var = beta * (1 - ap) / (1 - ab)
            c = mean + (torch.sqrt(var) * torch.randn_like(c) if i > 0 else 0.0)
        out.append(c)
    return torch.stack(out, 0)


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    enable_fast_math()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(QUICK if a.quick else FULL)
    if a.epochs is not None:
        cfg["epochs"] = a.epochs

    d = build_data(cfg, device)
    abar = cosine_abar(T_DIFF, device)

    torch.manual_seed(0)
    prior = DeepONet(cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    corr = DeepONet(2 * cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    den = Denoiser(cfg["N_u"]).to(device)
    params = list(prior.parameters()) + list(corr.parameters()) + list(den.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    gc, gcx, gcxx = _hard_bc(d["y_coll"])
    go, _, _ = _hard_bc(d["x_o"])
    gs, _, _ = _hard_bc(d["x_s"])

    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        opt.zero_grad(set_to_none=True)
        t, t_x, t_xx = trunk_derivatives(prior.trunk, d["y_coll"], order=2)
        bp = prior.branch_out(d["vs_tr"])
        f, fx, fxx = bp @ t.T + prior.bias, bp @ t_x.T, bp @ t_xx.T
        u_coll = gc * f
        u_xx = gcxx * f + 2.0 * gcx * fx + gc * fxx
        u_obs = go * (bp @ prior.trunk_out(d["x_o"]).T + prior.bias)
        u_sens = gs * (bp @ prior.trunk_out(d["x_s"]).T + prior.bias)
        cin = torch.cat([d["vs_tr"], u_sens.detach()], dim=1)
        c_det = corr.branch_out(cin) @ corr.trunk_out(d["y_coll"]).T + corr.bias
        res = dr.D * u_xx - K_R_CONST + c_det - d["vc_tr"]
        L_sol = (res ** 2).mean() + LAMBDA_D * ((u_obs - d["uo_tr"]) ** 2).mean()

        # diffusion over the physics-implied correction, conditioned on sparse obs
        c_star = (d["vc_tr"] - (dr.D * u_xx - K_R_CONST)).detach()      # (M,Q)
        cond = den.enc(d["uo_tr"])                                     # (M,D_COND)
        M = c_star.shape[0]
        ti = torch.randint(0, T_DIFF, (M,), device=device)
        ab = abar[ti].unsqueeze(1)
        eps = torch.randn_like(c_star)
        c_t = torch.sqrt(ab) * c_star + torch.sqrt(1 - ab) * eps
        x0_hat = den(d["y_coll"].squeeze(-1), c_t, (ti + 1).float() / T_DIFF, cond)
        L_diff = ((x0_hat - c_star) ** 2).mean()

        (L_sol + L_diff).backward(inputs=params)
        opt.step()
        sched.step()
        if (epoch + 1) % max(1, cfg["epochs"] // 10) == 0 or epoch == 0:
            print(f"  epoch {epoch+1:6d}  L_sol {L_sol.item():.3e}  "
                  f"L_diff {L_diff.item():.3e}")

    # ---- evaluation on test set ----
    prior.eval(); corr.eval(); den.eval()
    with torch.no_grad():
        bp = prior.branch_out(d["vs_te"])
        gt, _, _ = _hard_bc(d["x_t"])
        u_pred = gt * (bp @ prior.trunk_out(d["x_t"]).T + prior.bias)
        err_sol = rel_l2(u_pred, d["ut_te"])
        # true missing physics DeltaN[u] on the test grid
        delta_true = K_R_CONST - 0.5 * torch.exp(-d["ut_te"]) * d["ut_te"]
        # need obs of the test solutions to condition the diffusion
        go_t, _, _ = _hard_bc(d["x_o"])
        uo_te = go_t * (bp @ prior.trunk_out(d["x_o"]).T + prior.bias)
        cond_te = den.enc(uo_te)
        samp = sample(den, cond_te, d["x_t"].squeeze(-1), abar, a.n_samples, device)
        mean_c, std_c = samp.mean(0), samp.std(0)                     # (Mte, 201)

    err_delta = rel_l2(mean_c, delta_true)          # did diffusion recover DeltaN?
    calib = pearson(std_c.flatten(), (mean_c - delta_true).abs().flatten())
    dt = time.time() - t0

    os.makedirs("results", exist_ok=True)
    path = "results/method_diffusion.txt"
    with open(path, "w") as fo:
        fo.write("Method 3 -- Denoising-Diffusion Correction / model-error UQ "
                 "(diffusion-reaction)\n")
        fo.write(datetime.datetime.now().isoformat(timespec="seconds") + "\n")
        fo.write(f"device={device}  epochs={cfg['epochs']}  T={T_DIFF}  "
                 f"samples={a.n_samples}  M={cfg['M_train']}  sensors={cfg['n_sensors']}\n\n")
        fo.write(f"solution relL2 (prior G_theta)          : {err_sol:.4e}\n")
        fo.write(f"recovered missing-physics relL2         : {err_delta:.4e}    "
                 "(diffusion mean vs true DeltaN[u])\n")
        fo.write(f"mean model-error uncertainty (std)      : {std_c.mean().item():.4e}\n")
        fo.write(f"CALIBRATION corr(std, |mean-true DeltaN|): {calib:.3f}    "
                 "(>0 means uncertainty flags recovery error)\n")
        fo.write(f"wall time                               : {dt:.0f}s\n")
    print(f"\nwrote {path}  | sol {err_sol:.4e}  missing-physics {err_delta:.4e}  "
          f"calib {calib:.3f}")


if __name__ == "__main__":
    main()
