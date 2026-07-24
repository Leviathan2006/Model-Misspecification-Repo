"""Method One (two-stage prior correction) on the 1d Burgers benchmark.

    true system   N[u] = u_t + u u_x - nu u_xx = 0,  nu = 0.01,  periodic in x,
                  u(x,0) = v(x)   (the input function is the initial condition)

Stage 1  train the prior G_theta alone on physics + structural constraints of the
         KNOWN-but-misspecified operator N0, then FREEZE it:
             L1 = || N0[G_theta] ||^2 + lam_ic || G_theta(.,0) - v ||^2
                                      + lam_bc L_periodic
Stage 2  train the correction G_phi on the DATA loss only (interior solution
         observations), G_theta frozen:
             u_pred = G_theta(v) + G_phi(v, G_theta)
             L2 = || u_pred(y_obs) - u(y_obs) ||^2

Three misspecified priors N0 (paper Sec. 3.2):
    A  extra cubic  w_t + w w_x + eps w^3 - nu w_xx  (eps=10)
    B  advection dropped   w_t - nu w_xx
    C  diffusion dropped   w_t + w w_x

NOTE: per the two-stage design, stage 2 fits the correction to data only; the
periodic BC / initial condition are enforced on the prior in stage 1, not on the
correction. The reported field for 'corrected' is G_theta + G_phi.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets.burgers as bg                                   # noqa: E402
from deeponet import DeepONet, enable_fast_math, trunk_jet      # noqa: E402

NU = bg.NU
EPS_A = 10.0
LAM_BC, LAM_IC = 1.0, 50.0
BETAS = (0.999, 0.999)
CASES = ("A", "B", "C")


def N_true(w, w_t, w_x, w_xx):
    return w_t + w * w_x - NU * w_xx


def N0(case, w, w_t, w_x, w_xx):
    if case == "A":
        return w_t + w * w_x + EPS_A * w ** 3 - NU * w_xx
    if case == "B":
        return w_t - NU * w_xx
    if case == "C":
        return w_t + w * w_x
    raise ValueError(case)


def rel_l2(pred, true):
    p = pred.reshape(pred.shape[0], -1)
    t = true.reshape(true.shape[0], -1)
    return (torch.linalg.norm(p - t, dim=1) /
            torch.linalg.norm(t, dim=1)).mean().item()


def build_data(cfg, device, seed=0):
    t_g, x_g, u0_tr, fld_tr, u0_te, fld_te = bg.get_dataset(
        cfg["data_dir"], cfg["M_train"], cfg["M_test"], cfg["n_x"])
    rng = np.random.default_rng(seed)

    s_idx = np.linspace(0, x_g.size - 1, cfg["n_sensors"]).astype(int)
    x_s = x_g[s_idx]
    vs_tr, vs_te = u0_tr[:, s_idx], u0_te[:, s_idx]

    xc = np.linspace(0.0, 1.0, cfg["n_coll"], endpoint=False)
    tc = np.linspace(0.0, 1.0, cfg["n_coll"] + 1)[1:]
    Xc, Tc = np.meshgrid(xc, tc, indexing="ij")
    y_coll = np.stack([Xc.ravel(), Tc.ravel()], axis=1)

    xg2 = np.linspace(0.0, 1.0, cfg["n_cgrid"], endpoint=False)
    tg2 = np.linspace(0.0, 1.0, cfg["n_cgrid"])
    Xg2, Tg2 = np.meshgrid(xg2, tg2, indexing="ij")
    y_cgrid = np.stack([Xg2.ravel(), Tg2.ravel()], axis=1)

    oi = rng.integers(0, t_g.size, cfg["N_u"])
    oj = rng.integers(0, x_g.size, cfg["N_u"])
    y_obs = np.stack([x_g[oj], t_g[oi]], axis=1)
    u_obs_tr = fld_tr[:, oi, oj]

    y_ic = np.stack([x_s, np.zeros_like(x_s)], axis=1)
    t_bc = np.linspace(0.0, 1.0, cfg["n_bc"])
    y_bc0 = np.stack([np.zeros_like(t_bc), t_bc], axis=1)
    y_bc1 = np.stack([np.ones_like(t_bc), t_bc], axis=1)

    Xt, Tt = np.meshgrid(x_g, t_g, indexing="ij")
    y_test = np.stack([Xt.ravel(), Tt.ravel()], axis=1)
    tr = lambda a: np.ascontiguousarray(a.transpose(0, 2, 1)).reshape(a.shape[0], -1)

    mu, sd = vs_tr.mean(), vs_tr.std()

    def T(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)

    return {
        "vs_tr": T((vs_tr - mu) / sd), "vs_te": T((vs_te - mu) / sd),
        "v_ic_tr": T(vs_tr), "u_obs_tr": T(u_obs_tr),
        "y_coll": T(y_coll), "y_cgrid": T(y_cgrid), "y_obs": T(y_obs),
        "y_ic": T(y_ic), "y_bc0": T(y_bc0), "y_bc1": T(y_bc1),
        "y_test": T(y_test), "u_test": T(tr(fld_te)),
    }


def _fields(net, b, y):
    t, d1, d2 = trunk_jet(net.trunk, y, [(0, 0), (1, None)])
    w = net.combine(b, t) + net.bias[0]
    return w, net.combine(b, d1[1]), net.combine(b, d1[0]), net.combine(b, d2[(0, 0)])


def _value(net, b, y):
    return net.combine(b, net.trunk_out(y)) + net.bias[0]


def _corr_branch_in(prior, bp, vs, y_cgrid):
    return torch.cat([vs, _value(prior, bp, y_cgrid)], dim=1)


# --- stage 1: physics-only prior -------------------------------------------- #
def train_prior(operator, case, d, cfg, device, seed=0):
    torch.manual_seed(seed)
    prior = DeepONet(cfg["n_sensors"], 2, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    opt = torch.optim.Adam(prior.parameters(), lr=cfg["lr"], betas=cfg["betas"])
    M, bs = d["vs_tr"].shape[0], cfg["batch"]
    g = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(cfg["epochs_prior"]):
        idx = torch.randperm(M, generator=g)[:bs].to(device)
        vs, v_ic = d["vs_tr"][idx], d["v_ic_tr"][idx]
        opt.zero_grad(set_to_none=True)
        bp = prior.branch_out(vs)
        u, u_t, u_x, u_xx = _fields(prior, bp, d["y_coll"])
        res = (N_true(u, u_t, u_x, u_xx) if operator == "true"
               else N0(case, u, u_t, u_x, u_xx))
        l_ic = ((_value(prior, bp, d["y_ic"]) - v_ic) ** 2).mean()
        b0, _, b0x, _ = _fields(prior, bp, d["y_bc0"])
        b1, _, b1x, _ = _fields(prior, bp, d["y_bc1"])
        l_bc = ((b0 - b1) ** 2).mean() + ((b0x - b1x) ** 2).mean()
        loss = (res ** 2).mean() + LAM_IC * l_ic + LAM_BC * l_bc
        loss.backward()
        opt.step()
        if (epoch + 1) % max(1, cfg["epochs_prior"] // 5) == 0 or epoch == 0:
            print(f"  [prior:{operator} {case}] epoch {epoch+1:6d}  loss {loss.item():.3e}")
    prior.eval()
    return prior


# --- stage 2: data-only correction on the frozen prior ---------------------- #
def train_correction(prior, d, cfg, device, seed=0):
    torch.manual_seed(seed)
    m = cfg["n_sensors"]
    corr = DeepONet(m + cfg["n_cgrid"] ** 2, 2, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    for pth in prior.parameters():
        pth.requires_grad_(False)
    opt = torch.optim.Adam(corr.parameters(), lr=cfg["lr"], betas=cfg["betas"])
    M, bs = d["vs_tr"].shape[0], cfg["batch"]
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    for epoch in range(cfg["epochs_corr"]):
        idx = torch.randperm(M, generator=g)[:bs].to(device)
        vs, u_obs = d["vs_tr"][idx], d["u_obs_tr"][idx]
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            bp = prior.branch_out(vs)
            u_prior_obs = _value(prior, bp, d["y_obs"])
            cin = _corr_branch_in(prior, bp, vs, d["y_cgrid"])
        phi = _value(corr, corr.branch_out(cin), d["y_obs"])
        loss = ((u_prior_obs + phi - u_obs) ** 2).mean()
        loss.backward()
        opt.step()
        if (epoch + 1) % max(1, cfg["epochs_corr"] // 5) == 0 or epoch == 0:
            print(f"  [corr] epoch {epoch+1:6d}  L_data {loss.item():.3e}")
    corr.eval()
    return corr


def evaluate(prior, corr, d, cfg):
    outs = []
    n = d["vs_te"].shape[0]
    for i in range(0, n, cfg["eval_batch"]):
        vs = d["vs_te"][i:i + cfg["eval_batch"]]
        with torch.no_grad():
            bp = prior.branch_out(vs)
            u = _value(prior, bp, d["y_test"])
            if corr is not None:
                cin = _corr_branch_in(prior, bp, vs, d["y_cgrid"])
                u = u + _value(corr, corr.branch_out(cin), d["y_test"])
        outs.append(u)
    return rel_l2(torch.cat(outs), d["u_test"])


FULL = dict(M_train=200, M_test=50, n_x=201, n_sensors=101, n_coll=41, n_cgrid=11,
            N_u=200, n_bc=40, p=100, width=128, depth=4, lr=1e-4, betas=BETAS,
            epochs_prior=40000, epochs_corr=20000, batch=50, eval_batch=25, data_dir=None)
QUICK = dict(M_train=200, M_test=50, n_x=201, n_sensors=101, n_coll=21, n_cgrid=9,
             N_u=200, n_bc=20, p=100, width=128, depth=4, lr=1e-4, betas=BETAS,
             epochs_prior=200, epochs_corr=200, batch=32, eval_batch=10, data_dir=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["A", "B", "C", "all"], default="all")
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
    print(f"Method One (two-stage) 1d Burgers on {device} | N_p={cfg['M_train']} "
          f"sensors={cfg['n_sensors']} coll={cfg['n_coll']}x{cfg['n_coll']} "
          f"N_u={cfg['N_u']} epochs={cfg['epochs_prior']}+{cfg['epochs_corr']}")

    d = build_data(cfg, device)
    cases = list(CASES) if a.case == "all" else [a.case]
    modes = ["known", "misspecified", "corrected"] if a.mode == "all" else [a.mode]

    rows = []
    if "known" in modes:                         # case-independent reference floor
        t0 = time.time()
        r = evaluate(train_prior("true", "A", d, cfg, device), None, d, cfg)
        rows.append(("-", "known", r, time.time() - t0))
    for case in cases:
        prior_mis = None
        for mode in [m for m in modes if m != "known"]:
            t0 = time.time()
            prior_mis = prior_mis or train_prior("mis", case, d, cfg, device)
            corr = train_correction(prior_mis, d, cfg, device) if mode == "corrected" else None
            rows.append((case, mode, evaluate(prior_mis, corr, d, cfg), time.time() - t0))

    print(f"\n==== relative L2 of u on {cfg['M_test']} test samples ====")
    print(f"{'case':<6}{'model':<14}{'u':>13}{'u (%)':>10}     time")
    for case, mode, r, dt in rows:
        print(f"{case:<6}{mode:<14}{r:>13.4e}{100*r:>10.2f}   {dt:.0f}s")


if __name__ == "__main__":
    main()
