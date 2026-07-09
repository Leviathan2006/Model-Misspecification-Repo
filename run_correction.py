"""Physics-guided correction under model misspecification, on a PI-DeepONet
backbone -- i.e. the paper's actual method (Ma, Boulle, Yang, Wu & Guo, 2026,
arXiv:2606.03469), reproduced for the 1D diffusion-reaction benchmark.

    u(x) solves   D u_xx - k_r(u) u = v,   k_r(u)=0.5 e^{-u},  D=0.1,  x in [-1,1]
    misspecified N0:   D u_xx - k_r_const

Networks
    prior      G_theta : v(sensors)              -> u(y)
    correction G_psi   : [v(sensors), u_prior(sensors)] -> c(y)   (forcing space)

Losses (paper Eqs.)
    L_data = || G_theta(v)(y_obs) - u(y_obs) ||^2          (N_u sparse obs)
    L_phys = || N0[G_theta(v)](y_f) + G_psi(y_f) - v(y_f) ||^2  (N_f collocation)
    L      = L_phys + lambda_d L_data + lambda_bc L_bc

Derivatives (u_xx) are taken by exact autodiff of the trunk basis (see
deeponet.trunk_derivatives) -- the mesh-free DeepONet residual the paper uses.
Collocation/observation/sensor points are sampled continuously and the manufactured
forcing v is evaluated analytically at any of them.

Three modes reproduce the paper's headline table:
    known         true physics residual (reference floor)
    misspecified  wrong N0, no correction -> error blows up
    corrected     prior + forcing-space correction -> recovered
"""
import argparse
import time

import numpy as np
import torch

import datasets.diffusion_reaction as dr
from deeponet import DeepONet, enable_fast_math, trunk_derivatives

K_R_CONST = 0.5
LAMBDA_D = 1.0
LAMBDA_BC = 1.0


def eval_field(w, x):
    """Analytic forcing v and solution u at points x for coefficients w."""
    u, uxx = dr.manufactured(x, w)
    v = dr.D * uxx - 0.5 * np.exp(-u) * u
    return v, u


def rel_l2(pred, true):
    return (torch.linalg.norm(pred - true, dim=1) /
            torch.linalg.norm(true, dim=1)).mean().item()


def build_data(cfg, device, seed=0):
    rng = np.random.default_rng(seed)
    nm = 2 * dr.N_MODES
    w_tr = rng.uniform(0, 1, (cfg["M_train"], nm))
    w_te = rng.uniform(0, 1, (cfg["M_test"], nm))
    x_s = np.linspace(-1, 1, cfg["n_sensors"])
    x_o = rng.uniform(-1, 1, cfg["N_u"])
    x_c = rng.uniform(-1, 1, cfg["N_f"])
    x_t = np.linspace(-1, 1, 201)
    x_bc = np.array([-1.0, 1.0])

    vs_tr = eval_field(w_tr, x_s)[0]
    vs_te = eval_field(w_te, x_s)[0]
    uo_tr = eval_field(w_tr, x_o)[1]
    vc_tr = eval_field(w_tr, x_c)[0]
    ut_te = eval_field(w_te, x_t)[1]
    mu, sd = vs_tr.mean(), vs_tr.std()

    def T(a, grad=False):
        t = torch.tensor(a, dtype=torch.float32, device=device)
        return t.requires_grad_(True) if grad else t

    return {
        "vs_tr": T((vs_tr - mu) / sd), "vs_te": T((vs_te - mu) / sd),
        "uo_tr": T(uo_tr), "vc_tr": T(vc_tr), "ut_te": T(ut_te),
        "y_coll": T(x_c[:, None]),   # jvp supplies tangents; no leaf-grad needed
        "x_s": T(x_s[:, None]), "x_o": T(x_o[:, None]),
        "x_t": T(x_t[:, None]), "x_bc": T(x_bc[:, None]),
    }


def train_mode(mode, d, cfg, device, seed=0):
    torch.manual_seed(seed)
    prior = DeepONet(cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    corr = DeepONet(2 * cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)

    params = list(prior.parameters())
    if mode == "corrected":
        params += list(corr.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    for epoch in range(cfg["epochs"]):
        opt.zero_grad(set_to_none=True)
        t, _, t_xx = trunk_derivatives(prior.trunk, d["y_coll"], order=2)   # (Q,p)
        bp = prior.branch_out(d["vs_tr"])                                   # (M,p)
        u_coll = bp @ t.T + prior.bias                                      # (M,Q)
        u_xx = bp @ t_xx.T
        u_obs = bp @ prior.trunk_out(d["x_o"]).T + prior.bias
        u_bc = bp @ prior.trunk_out(d["x_bc"]).T + prior.bias

        if mode == "known":
            res = dr.D * u_xx - 0.5 * torch.exp(-u_coll) * u_coll - d["vc_tr"]
        elif mode == "misspecified":
            res = dr.D * u_xx - K_R_CONST - d["vc_tr"]
        else:  # corrected
            u_sens = bp @ prior.trunk_out(d["x_s"]).T + prior.bias          # (M, n_sensors)
            cin = torch.cat([d["vs_tr"], u_sens.detach()], dim=1)
            c_coll = corr.branch_out(cin) @ corr.trunk_out(d["y_coll"]).T + corr.bias
            res = dr.D * u_xx - K_R_CONST + c_coll - d["vc_tr"]

        loss = (res ** 2).mean() + LAMBDA_D * ((u_obs - d["uo_tr"]) ** 2).mean() \
            + LAMBDA_BC * (u_bc ** 2).mean()
        loss.backward(inputs=params)
        opt.step()
        sched.step()
        if (epoch + 1) % max(1, cfg["epochs"] // 10) == 0 or epoch == 0:
            print(f"  [{mode}] epoch {epoch+1:6d}  loss {loss.item():.3e}")

    prior.eval()
    corr.eval()
    with torch.no_grad():
        bp = prior.branch_out(d["vs_te"])
        u_pred = bp @ prior.trunk_out(d["x_t"]).T + prior.bias
        err_prior = rel_l2(u_pred, d["ut_te"])
        err_full = err_prior
        if mode == "corrected":
            u_sens = bp @ prior.trunk_out(d["x_s"]).T + prior.bias
            cin = torch.cat([d["vs_te"], u_sens], dim=1)
            c_pred = corr.branch_out(cin) @ corr.trunk_out(d["x_t"]).T + corr.bias
            err_full = rel_l2(u_pred + c_pred, d["ut_te"])
    return err_prior, err_full


FULL = dict(M_train=1000, M_test=100, n_sensors=101, N_u=100, N_f=1000,
            p=100, width=64, depth=4, lr=1e-3, epochs=50000)
QUICK = dict(M_train=64, M_test=32, n_sensors=41, N_u=40, N_f=128,
             p=100, width=64, depth=4, lr=1e-3, epochs=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["known", "misspecified", "corrected", "all"],
                    default="all")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    enable_fast_math()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(QUICK if a.quick else FULL)
    if a.epochs is not None:
        cfg["epochs"] = a.epochs
    print(f"PI-DeepONet diffusion-reaction on {device} | "
          f"M={cfg['M_train']} sensors={cfg['n_sensors']} N_f={cfg['N_f']} "
          f"epochs={cfg['epochs']}")

    d = build_data(cfg, device)
    modes = ["known", "misspecified", "corrected"] if a.mode == "all" else [a.mode]
    results = {}
    for m in modes:
        t0 = time.time()
        results[m] = (*train_mode(m, d, cfg, device), time.time() - t0)

    print("\n==== relative L2 on test (diffusion-reaction, PI-DeepONet) ====")
    for m in modes:
        ep, ef, dt = results[m]
        if m == "corrected":
            print(f"  {m:13s} prior-only {ep:.4e}  [recovered]   "
                  f"prior+corr {ef:.4e}  ({dt:.0f}s)")
        else:
            print(f"  {m:13s} {ep:.4e}  ({dt:.0f}s)")
    print("(paper reference: misspecified ~1.6e0, corrected ~1.85e-3, known ~9.0e-4)")


if __name__ == "__main__":
    main()
