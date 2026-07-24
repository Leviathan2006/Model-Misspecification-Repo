"""Method One (two-stage prior correction) on the 1d diffusion-reaction benchmark.

Stage 1  train the prior G_theta alone on the physics residual of the KNOWN-but-
         misspecified operator N0, then FREEZE it:
             L_phys = || N0[G_theta(v)](y_f) - v(y_f) ||^2
Stage 2  train the correction G_phi on the DATA loss only, with G_theta frozen, so
         backprop updates G_phi alone. The prediction is the sum:
             u_pred = G_theta(v) + G_phi(v, G_theta)
             L_data = || u_pred(y_obs) - u(y_obs) ||^2

True operator      N[u]  = D u_xx - 0.5 e^{-u} u        (used to make the data / 'known')
Misspecified N0    N0[u] = D u_xx - k_r_const           (k_r_const = 0.5)

The hard Dirichlet factor g(x) = 1 - x^2 multiplies BOTH networks, so u(+-1)=0
holds for the prior and for the corrected field by construction.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets.diffusion_reaction as dr                        # noqa: E402
from deeponet import DeepONet, enable_fast_math, trunk_derivatives  # noqa: E402

K_R_CONST = 0.5


def rel_l2(pred, true):
    return (torch.linalg.norm(pred - true, dim=1) /
            torch.linalg.norm(true, dim=1)).mean().item()


def _hard_bc(x):
    g = (1.0 - x ** 2).T
    gx = (-2.0 * x).T
    return g, gx, -2.0


def build_data(cfg, device, seed=0):
    x, v_tr, u_tr, v_te, u_te = dr.get_dataset(
        cfg["data_dir"], cfg["M_train"], cfg["M_test"], cfg["n_x"])
    rng = np.random.default_rng(seed)
    N = x.size

    s_idx = np.linspace(0, N - 1, cfg["n_sensors"]).astype(int)
    c_idx = rng.choice(np.arange(1, N - 1), cfg["N_f"], replace=cfg["N_f"] > N - 2)
    o_idx = rng.choice(N, cfg["N_u"], replace=cfg["N_u"] > N)

    mu, sd = v_tr[:, s_idx].mean(), v_tr[:, s_idx].std()

    def T(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)

    return {
        "vs_tr": T((v_tr[:, s_idx] - mu) / sd), "vs_te": T((v_te[:, s_idx] - mu) / sd),
        "vc_tr": T(v_tr[:, c_idx]), "uo_tr": T(u_tr[:, o_idx]),
        "y_coll": T(x[c_idx, None]), "x_o": T(x[o_idx, None]),
        "x_s": T(x[s_idx, None]), "x_t": T(x[:, None]), "u_te": T(u_te),
    }


def _prior_fields(prior, bp, y, g, gx, gxx):
    t, t_x, t_xx = trunk_derivatives(prior.trunk, y, order=2)
    f = prior.combine(bp, t) + prior.bias[0]
    u = g * f
    u_xx = gxx * f + 2.0 * gx * prior.combine(bp, t_x) + g * prior.combine(bp, t_xx)
    return u, u_xx


def _value(net, bp, y, g):
    return g * (net.combine(bp, net.trunk_out(y)) + net.bias[0])


# --- stage 1: physics-only prior -------------------------------------------- #
def train_prior(operator, d, cfg, device, seed=0):
    torch.manual_seed(seed)
    prior = DeepONet(cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    opt = torch.optim.Adam(prior.parameters(), lr=cfg["lr"], betas=cfg["betas"])
    gc, gcx, gcxx = _hard_bc(d["y_coll"])
    for epoch in range(cfg["epochs_prior"]):
        opt.zero_grad(set_to_none=True)
        bp = prior.branch_out(d["vs_tr"])
        u, u_xx = _prior_fields(prior, bp, d["y_coll"], gc, gcx, gcxx)
        if operator == "true":
            res = dr.D * u_xx - 0.5 * torch.exp(-u) * u - d["vc_tr"]
        else:
            res = dr.D * u_xx - K_R_CONST - d["vc_tr"]
        loss = (res ** 2).mean()
        loss.backward()
        opt.step()
        if (epoch + 1) % max(1, cfg["epochs_prior"] // 5) == 0 or epoch == 0:
            print(f"  [prior:{operator}] epoch {epoch+1:6d}  L_phys {loss.item():.3e}")
    prior.eval()
    return prior


# --- stage 2: data-only correction on the frozen prior ---------------------- #
def train_correction(prior, d, cfg, device, seed=0):
    torch.manual_seed(seed)
    corr = DeepONet(2 * cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    for pth in prior.parameters():
        pth.requires_grad_(False)
    opt = torch.optim.Adam(corr.parameters(), lr=cfg["lr"], betas=cfg["betas"])
    go, _, _ = _hard_bc(d["x_o"])
    gs, _, _ = _hard_bc(d["x_s"])
    with torch.no_grad():                        # frozen prior -> constants
        bp = prior.branch_out(d["vs_tr"])
        u_prior_obs = _value(prior, bp, d["x_o"], go)
        cin = torch.cat([d["vs_tr"], _value(prior, bp, d["x_s"], gs)], dim=1)
    for epoch in range(cfg["epochs_corr"]):
        opt.zero_grad(set_to_none=True)
        phi = go * (corr.combine(corr.branch_out(cin), corr.trunk_out(d["x_o"]))
                    + corr.bias[0])
        loss = ((u_prior_obs + phi - d["uo_tr"]) ** 2).mean()
        loss.backward()
        opt.step()
        if (epoch + 1) % max(1, cfg["epochs_corr"] // 5) == 0 or epoch == 0:
            print(f"  [corr] epoch {epoch+1:6d}  L_data {loss.item():.3e}")
    corr.eval()
    return corr


def evaluate(prior, corr, d):
    with torch.no_grad():
        gt, _, _ = _hard_bc(d["x_t"])
        gs, _, _ = _hard_bc(d["x_s"])
        bp = prior.branch_out(d["vs_te"])
        u_prior = _value(prior, bp, d["x_t"], gt)
        err_prior = rel_l2(u_prior, d["u_te"])
        err_full = err_prior
        if corr is not None:
            cin = torch.cat([d["vs_te"], _value(prior, bp, d["x_s"], gs)], dim=1)
            phi = gt * (corr.combine(corr.branch_out(cin), corr.trunk_out(d["x_t"]))
                        + corr.bias[0])
            err_full = rel_l2(u_prior + phi, d["u_te"])
    return err_prior, err_full


BETAS = (0.999, 0.999)
FULL = dict(M_train=200, M_test=50, n_x=201, n_sensors=101, N_f=200, N_u=100,
            p=100, width=64, depth=4, lr=1e-3, betas=BETAS,
            epochs_prior=20000, epochs_corr=10000, data_dir=None)
QUICK = dict(M_train=200, M_test=50, n_x=201, n_sensors=41, N_f=64, N_u=40,
             p=100, width=64, depth=4, lr=1e-3, betas=BETAS,
             epochs_prior=200, epochs_corr=200, data_dir=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["known", "misspecified", "corrected", "all"],
                    default="all")
    ap.add_argument("--epochs_prior", type=int, default=None)
    ap.add_argument("--epochs_corr", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    enable_fast_math()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(QUICK if a.quick else FULL)
    cfg["data_dir"] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    for k in ("epochs_prior", "epochs_corr"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    print(f"Method One (two-stage) diffusion-reaction on {device} | "
          f"M={cfg['M_train']} sensors={cfg['n_sensors']} N_f={cfg['N_f']} "
          f"N_u={cfg['N_u']} epochs={cfg['epochs_prior']}+{cfg['epochs_corr']}")

    d = build_data(cfg, device)
    modes = ["known", "misspecified", "corrected"] if a.mode == "all" else [a.mode]
    results, prior_mis = {}, None
    for m in modes:
        t0 = time.time()
        if m == "known":
            results[m] = (*evaluate(train_prior("true", d, cfg, device), None, d),
                          time.time() - t0)
        elif m == "misspecified":
            prior_mis = prior_mis or train_prior("mis", d, cfg, device)
            results[m] = (*evaluate(prior_mis, None, d), time.time() - t0)
        else:
            prior_mis = prior_mis or train_prior("mis", d, cfg, device)
            corr = train_correction(prior_mis, d, cfg, device)
            results[m] = (*evaluate(prior_mis, corr, d), time.time() - t0)

    print("\n==== relative L2 on test (diffusion-reaction, two-stage) ====")
    for m in modes:
        ep, ef, dt = results[m]
        if m == "corrected":
            print(f"  {m:13s} prior-only {ep:.4e}   prior+corr {ef:.4e}  ({dt:.0f}s)")
        else:
            print(f"  {m:13s} {ep:.4e}  ({dt:.0f}s)")


if __name__ == "__main__":
    main()
