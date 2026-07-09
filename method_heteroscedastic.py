"""Our method 2: Heteroscedastic physics-guided correction under noisy observations.

The paper's correction is deterministic and its data term is a plain least-squares
fit -- it implicitly assumes clean observations. Real misspecified settings have
NOISY, sparse data. We give the prior a variance head s(x) and replace the data
term with a Gaussian negative log-likelihood

    L_data = mean[ 0.5 e^{-s} (u_pred - u_obs)^2 + 0.5 s ]        (at obs points)

so the model (a) down-weights noisy observations instead of overfitting them, and
(b) outputs a calibrated per-point aleatoric-uncertainty map exp(s/2).

We train, on the SAME noisy data, both the deterministic correction (MSE data
term) and our heteroscedastic one, and report their clean-solution error plus the
calibration of the heteroscedastic uncertainty.

Results are written to results/method_heteroscedastic.txt.
"""
import argparse
import datetime
import os
import time

import numpy as np
import torch

import datasets.diffusion_reaction as dr
from deeponet import DeepONet, enable_fast_math, trunk_derivatives
from run_correction import (FULL, K_R_CONST, LAMBDA_D, QUICK, _hard_bc,
                            build_data, rel_l2)


def fit(d, cfg, device, noisy_obs, hetero, seed=0):
    """Train a corrected model on noisy_obs. Returns (u_pred, std_pred).
    std_pred is the aleatoric std map if hetero else None."""
    torch.manual_seed(seed)
    prior = DeepONet(cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    corr = DeepONet(2 * cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    unc = DeepONet(cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    params = list(prior.parameters()) + list(corr.parameters())
    if hetero:
        params += list(unc.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    gc, gcx, gcxx = _hard_bc(d["y_coll"])
    go, _, _ = _hard_bc(d["x_o"])
    gs, _, _ = _hard_bc(d["x_s"])

    for _ in range(cfg["epochs"]):
        opt.zero_grad(set_to_none=True)
        t, t_x, t_xx = trunk_derivatives(prior.trunk, d["y_coll"], order=2)
        bp = prior.branch_out(d["vs_tr"])
        f, fx, fxx = bp @ t.T + prior.bias, bp @ t_x.T, bp @ t_xx.T
        u_coll = gc * f
        u_xx = gcxx * f + 2.0 * gcx * fx + gc * fxx
        u_obs = go * (bp @ prior.trunk_out(d["x_o"]).T + prior.bias)
        u_sens = gs * (bp @ prior.trunk_out(d["x_s"]).T + prior.bias)
        cin = torch.cat([d["vs_tr"], u_sens.detach()], dim=1)
        c_coll = corr.branch_out(cin) @ corr.trunk_out(d["y_coll"]).T + corr.bias
        res = dr.D * u_xx - K_R_CONST + c_coll - d["vc_tr"]

        if hetero:
            s_obs = unc.branch_out(d["vs_tr"]) @ unc.trunk_out(d["x_o"]).T + unc.bias
            data = (0.5 * torch.exp(-s_obs) * (u_obs - noisy_obs) ** 2 + 0.5 * s_obs).mean()
        else:
            data = ((u_obs - noisy_obs) ** 2).mean()

        loss = (res ** 2).mean() + LAMBDA_D * data
        loss.backward(inputs=params)
        opt.step()
        sched.step()

    with torch.no_grad():
        bp = prior.branch_out(d["vs_te"])
        gt, _, _ = _hard_bc(d["x_t"])
        u_pred = gt * (bp @ prior.trunk_out(d["x_t"]).T + prior.bias)
        std_pred = None
        if hetero:
            s_te = unc.branch_out(d["vs_te"]) @ unc.trunk_out(d["x_t"]).T + unc.bias
            std_pred = torch.exp(0.5 * s_te)
    return u_pred, std_pred


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.15,
                    help="obs noise std as a fraction of the observation std")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    enable_fast_math()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(QUICK if a.quick else FULL)
    if a.epochs is not None:
        cfg["epochs"] = a.epochs

    d = build_data(cfg, device)
    u_true = d["ut_te"]
    torch.manual_seed(0)
    sigma = a.noise * d["uo_tr"].std()
    noisy_obs = d["uo_tr"] + sigma * torch.randn_like(d["uo_tr"])

    t0 = time.time()
    u_det, _ = fit(d, cfg, device, noisy_obs, hetero=False)
    u_het, std = fit(d, cfg, device, noisy_obs, hetero=True)
    err_det = rel_l2(u_det, u_true)
    err_het = rel_l2(u_het, u_true)
    calib = pearson(std.flatten(), (u_het - u_true).abs().flatten())
    dt = time.time() - t0

    os.makedirs("results", exist_ok=True)
    path = "results/method_heteroscedastic.txt"
    with open(path, "w") as fo:
        fo.write("Method 2 -- Heteroscedastic Correction under Noisy Observations "
                 "(diffusion-reaction)\n")
        fo.write(datetime.datetime.now().isoformat(timespec="seconds") + "\n")
        fo.write(f"device={device}  epochs={cfg['epochs']}  noise={a.noise} "
                 f"(sigma={float(sigma):.4e})  M={cfg['M_train']}  "
                 f"sensors={cfg['n_sensors']}\n\n")
        fo.write(f"deterministic correction (MSE) relL2 : {err_det:.4e}\n")
        fo.write(f"heteroscedastic correction    relL2 : {err_het:.4e}    "
                 "(lower = more robust to noise)\n")
        fo.write(f"mean predicted aleatoric std        : {std.mean().item():.4e}\n")
        fo.write(f"calibration corr(std,|err|)         : {calib:.3f}    "
                 "(>0 means uncertainty tracks error)\n")
        fo.write(f"wall time                           : {dt:.0f}s\n")
    print(f"\nwrote {path}  | det {err_det:.4e}  hetero {err_het:.4e}  calib {calib:.3f}")


if __name__ == "__main__":
    main()
