"""Our method 1: Deep-ensemble physics-guided correction (a trustworthy extension).

The paper's physics-guided correction is a single deterministic model. We train K
corrected PI-DeepONets from different initialisations. Then:
  * the ENSEMBLE MEAN is a better point estimate than any single model, and
  * the ENSEMBLE STD is an epistemic-uncertainty map -- an estimate of *where*
    the recovered solution is least trustworthy under misspecification.

We also report a calibration score: the correlation between the predicted std and
the actual error. A positive correlation means the uncertainty is meaningful.

Results are written to results/method_ensemble.txt (not just stdout).
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


def fit_corrected(d, cfg, device, seed):
    """Train one corrected model; return its prior-only test prediction (Mte,201)."""
    torch.manual_seed(seed)
    prior = DeepONet(cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    corr = DeepONet(2 * cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    params = list(prior.parameters()) + list(corr.parameters())
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
        loss = (res ** 2).mean() + LAMBDA_D * ((u_obs - d["uo_tr"]) ** 2).mean()
        loss.backward(inputs=params)
        opt.step()
        sched.step()

    with torch.no_grad():
        bp = prior.branch_out(d["vs_te"])
        gt, _, _ = _hard_bc(d["x_t"])
        return gt * (bp @ prior.trunk_out(d["x_t"]).T + prior.bias)


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ensemble", type=int, default=5)
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
    t0 = time.time()
    preds, member_err = [], []
    for k in range(a.n_ensemble):
        up = fit_corrected(d, cfg, device, seed=k)
        preds.append(up)
        member_err.append(rel_l2(up, u_true))
        print(f"  member {k}  prior-only relL2 {member_err[-1]:.4e}")

    P = torch.stack(preds, 0)                       # (K, Mte, 201)
    mean, std = P.mean(0), P.std(0)
    err_mean = rel_l2(mean, u_true)
    calib = pearson(std.flatten(), (mean - u_true).abs().flatten())
    dt = time.time() - t0

    os.makedirs("results", exist_ok=True)
    path = "results/method_ensemble.txt"
    with open(path, "w") as fo:
        fo.write("Method 1 -- Deep-Ensemble Physics-Guided Correction "
                 "(diffusion-reaction)\n")
        fo.write(datetime.datetime.now().isoformat(timespec="seconds") + "\n")
        fo.write(f"device={device}  ensemble={a.n_ensemble}  epochs={cfg['epochs']}  "
                 f"M={cfg['M_train']}  sensors={cfg['n_sensors']}  N_f={cfg['N_f']}\n\n")
        fo.write(f"per-member prior-only relL2 : {[round(e, 5) for e in member_err]}\n")
        fo.write(f"mean of member relL2        : {np.mean(member_err):.4e}\n")
        fo.write(f"ENSEMBLE-MEAN relL2         : {err_mean:.4e}    "
                 f"(single-model baseline ~3.0e-2)\n")
        fo.write(f"mean epistemic std          : {std.mean().item():.4e}\n")
        fo.write(f"calibration corr(std,|err|) : {calib:.3f}    "
                 "(>0 means uncertainty tracks error)\n")
        fo.write(f"wall time                   : {dt:.0f}s\n")
    print(f"\nwrote {path}  | ensemble-mean relL2 {err_mean:.4e}  calib {calib:.3f}")


if __name__ == "__main__":
    main()
