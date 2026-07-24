import argparse
import time

import numpy as np
import torch

import datasets.diffusion_reaction as dr
from deeponet import DeepONet, enable_fast_math, trunk_derivatives

K_R_CONST = 0.5


def eval_field(w, x):
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

    vs_tr = eval_field(w_tr, x_s)[0]
    vs_te = eval_field(w_te, x_s)[0]
    uo_tr = eval_field(w_tr, x_o)[1]
    vc_tr = eval_field(w_tr, x_c)[0]
    ut_te = eval_field(w_te, x_t)[1]
    mu, sd = vs_tr.mean(), vs_tr.std()

    def T(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    return {
        "vs_tr": T((vs_tr - mu) / sd), "vs_te": T((vs_te - mu) / sd),
        "uo_tr": T(uo_tr), "vc_tr": T(vc_tr), "ut_te": T(ut_te),
        "y_coll": T(x_c[:, None]),
        "x_s": T(x_s[:, None]), "x_o": T(x_o[:, None]), "x_t": T(x_t[:, None]),
    }


def _hard_bc(x):
    g = (1.0 - x ** 2).T
    gx = (-2.0 * x).T
    return g, gx, -2.0


def _prior_solution(prior, bp, y, g):
    return g * (bp @ prior.trunk_out(y).T + prior.bias)


# ---- stage 1: train the prior alone on the physics residual ---------------
def train_prior(operator, d, cfg, device, seed=0):
    torch.manual_seed(seed)
    prior = DeepONet(cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    opt = torch.optim.Adam(prior.parameters(), lr=cfg["lr"])
    epochs = cfg["epochs_prior"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gc, gcx, gcxx = _hard_bc(d["y_coll"])

    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        t, t_x, t_xx = trunk_derivatives(prior.trunk, d["y_coll"], order=2)
        bp = prior.branch_out(d["vs_tr"])
        f = bp @ t.T + prior.bias
        f_x, f_xx = bp @ t_x.T, bp @ t_xx.T
        u = gc * f
        u_xx = gcxx * f + 2.0 * gcx * f_x + gc * f_xx
        if operator == "true":
            res = dr.D * u_xx - 0.5 * torch.exp(-u) * u - d["vc_tr"]
        else:                                            # misspecified N0
            res = dr.D * u_xx - K_R_CONST - d["vc_tr"]
        loss = (res ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"  [prior:{operator}] epoch {epoch+1:6d}  L_phys {loss.item():.3e}")

    prior.eval()
    return prior


# ---- stage 2: freeze the prior, fit a solution-space correction to data ----
def train_correction(prior, d, cfg, device, seed=0):
    torch.manual_seed(seed)
    corr = DeepONet(2 * cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    for p in prior.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(corr.parameters(), lr=cfg["lr"])
    epochs = cfg["epochs_corr"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    go, _, _ = _hard_bc(d["x_o"])
    gs, _, _ = _hard_bc(d["x_s"])
    with torch.no_grad():                                # frozen prior -> constants
        bp = prior.branch_out(d["vs_tr"])
        u_prior_obs = _prior_solution(prior, bp, d["x_o"], go)
        u_prior_sens = _prior_solution(prior, bp, d["x_s"], gs)
        cin = torch.cat([d["vs_tr"], u_prior_sens], dim=1)

    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        phi = go * (corr.branch_out(cin) @ corr.trunk_out(d["x_o"]).T + corr.bias)
        u_pred = u_prior_obs + phi                       # solution-space correction
        loss = ((u_pred - d["uo_tr"]) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"  [corr] epoch {epoch+1:6d}  L_data {loss.item():.3e}")

    corr.eval()
    return corr


def evaluate(prior, corr, d):
    with torch.no_grad():
        gt, _, _ = _hard_bc(d["x_t"])
        gs, _, _ = _hard_bc(d["x_s"])
        bp = prior.branch_out(d["vs_te"])
        u_prior = _prior_solution(prior, bp, d["x_t"], gt)
        err_prior = rel_l2(u_prior, d["ut_te"])
        err_full = err_prior
        if corr is not None:
            u_sens = _prior_solution(prior, bp, d["x_s"], gs)
            cin = torch.cat([d["vs_te"], u_sens], dim=1)
            phi = gt * (corr.branch_out(cin) @ corr.trunk_out(d["x_t"]).T + corr.bias)
            err_full = rel_l2(u_prior + phi, d["ut_te"])
    return err_prior, err_full


FULL = dict(M_train=1000, M_test=100, n_sensors=101, N_u=100, N_f=1000,
            p=100, width=64, depth=4, lr=1e-3,
            epochs_prior=50000, epochs_corr=20000)
QUICK = dict(M_train=64, M_test=32, n_sensors=41, N_u=40, N_f=128,
             p=100, width=64, depth=4, lr=1e-3,
             epochs_prior=100, epochs_corr=100)


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
    if a.epochs_prior is not None:
        cfg["epochs_prior"] = a.epochs_prior
    if a.epochs_corr is not None:
        cfg["epochs_corr"] = a.epochs_corr
    print(f"two-stage PI-DeepONet diffusion-reaction on {device} | "
          f"M={cfg['M_train']} sensors={cfg['n_sensors']} N_f={cfg['N_f']} "
          f"N_u={cfg['N_u']} epochs={cfg['epochs_prior']}+{cfg['epochs_corr']}")

    d = build_data(cfg, device)
    modes = ["known", "misspecified", "corrected"] if a.mode == "all" else [a.mode]
    results = {}

    prior_mis = None                                     # shared by misspecified/corrected
    for m in modes:
        t0 = time.time()
        if m == "known":
            prior = train_prior("true", d, cfg, device)
            results[m] = (*evaluate(prior, None, d), time.time() - t0)
        elif m == "misspecified":
            prior_mis = prior_mis or train_prior("mis", d, cfg, device)
            results[m] = (*evaluate(prior_mis, None, d), time.time() - t0)
        else:                                            # corrected
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
    print("(paper reference: misspecified ~1.6e0, corrected ~1.85e-3, known ~9.0e-4)")


if __name__ == "__main__":
    main()
