"""Our method 2: Heteroscedastic physics-guided correction under STRUCTURED noise.

The paper's correction is deterministic with a plain least-squares data term -- it
implicitly trusts every observation equally. Real misspecified settings have
noisy, sparse data whose reliability varies across the domain. We give the prior a
variance head s(x) and use a Gaussian negative log-likelihood data term

    L_data = mean[ 0.5 e^{-s(x)} (u_pred - u_obs)^2 + 0.5 s(x) ]     (at obs points)

so the model learns *which observations to trust* and outputs a calibrated
aleatoric-uncertainty map exp(s/2).

To test this properly we inject SPATIALLY-VARYING observation noise: the std ramps
across the domain (nearly clean near x=-1, noisy near x=+1). We then check two
things against the deterministic correction trained on the same noisy data:
  1. robustness -- clean-solution relative L2, and
  2. calibration -- does the predicted std recover the true spatial noise map?
     (correlation between the learned std at obs points and the true noise std).

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
    """Train a corrected model on noisy_obs. Returns (u_pred_test, std_at_obs).
    std_at_obs is (M, N_u) predicted aleatoric std if hetero else None."""
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
        std_obs = None
        if hetero:                                 # obs belong to the train samples
            s_obs = unc.branch_out(d["vs_tr"]) @ unc.trunk_out(d["x_o"]).T + unc.bias
            std_obs = torch.exp(0.5 * s_obs)
    return u_pred, std_obs


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.3,
                    help="peak obs-noise std as a fraction of the observation std")
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

    # STRUCTURED noise: std ramps from ~0.1x (near x=-1) to 1.0x (near x=+1)
    torch.manual_seed(0)
    xo = d["x_o"].squeeze(-1)                              # (N_u,) in [-1,1]
    ramp = 0.1 + 0.9 * (xo + 1.0) / 2.0
    sigma_o = a.noise * d["uo_tr"].std() * ramp           # (N_u,) per-location std
    noisy_obs = d["uo_tr"] + sigma_o[None, :] * torch.randn_like(d["uo_tr"])

    t0 = time.time()
    u_det, _ = fit(d, cfg, device, noisy_obs, hetero=False)
    u_het, std_obs = fit(d, cfg, device, noisy_obs, hetero=True)
    err_det = rel_l2(u_det, u_true)
    err_het = rel_l2(u_het, u_true)
    # calibration: does the learned std recover the true spatial noise map?
    pred_std_map = std_obs.mean(0)                        # (N_u,) avg over samples
    calib_noise = pearson(pred_std_map, sigma_o)
    dt = time.time() - t0

    os.makedirs("results", exist_ok=True)
    path = "results/method_heteroscedastic.txt"
    with open(path, "w") as fo:
        fo.write("Method 2 -- Heteroscedastic Correction under STRUCTURED Noise "
                 "(diffusion-reaction)\n")
        fo.write(datetime.datetime.now().isoformat(timespec="seconds") + "\n")
        fo.write(f"device={device}  epochs={cfg['epochs']}  peak_noise={a.noise}  "
                 f"M={cfg['M_train']}  sensors={cfg['n_sensors']}\n")
        fo.write("noise std ramps ~0.1x -> 1.0x across x in [-1,1] "
                 f"(true sigma range {float(sigma_o.min()):.4e}..{float(sigma_o.max()):.4e})\n\n")
        fo.write(f"deterministic correction (MSE) relL2 : {err_det:.4e}\n")
        fo.write(f"heteroscedastic correction    relL2 : {err_het:.4e}    "
                 "(lower = more robust)\n")
        fo.write(f"mean predicted aleatoric std        : {pred_std_map.mean().item():.4e}\n")
        fo.write(f"CALIBRATION corr(pred std, true noise map): {calib_noise:.3f}    "
                 "(->1 means it recovered where the noise is)\n")
        fo.write(f"wall time                           : {dt:.0f}s\n")
    print(f"\nwrote {path}  | det {err_det:.4e}  hetero {err_het:.4e}  "
          f"noise-map calib {calib_noise:.3f}")


if __name__ == "__main__":
    main()
