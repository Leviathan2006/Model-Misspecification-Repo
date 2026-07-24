"""Method One (two-stage prior correction) on the 2d hyperelastic beam.

    true system   -div P(u) = f,  neo-Hookean first Piola-Kirchhoff stress P
    misspecified  linear elasticity:  div sigma(u) + f = 0

Operator  G : eps |-> u = (u_x, u_y), eps the right-boundary compression.

Stage 1  train the prior G_theta alone on physics + boundary + energy of the
         KNOWN-but-misspecified (linear-elastic) operator N0, then FREEZE it:
             L1 = || N0[G_theta] + f ||^2 + lam_bc L_bc + lam_e Pi(G_theta)
Stage 2  train the correction G_phi on the DATA loss only (interior displacement
         observations), G_theta frozen:
             u_pred = G_theta + G_phi,   L2 = || u_pred(y_obs) - u(y_obs) ||^2

Nondimensionalisation follows the paper: stresses, body force and material
parameters divided by E; coordinates NOT rescaled (y in [0,0.1], x in [0,1]).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets.hyperelastic as he                             # noqa: E402
from deeponet import DeepONet, enable_fast_math, trunk_jet     # noqa: E402

MU = he.MU / he.E
LAM = he.LAM / he.E
BODY = he.BODY / he.E
LX, LY = 1.0, 0.1
LAM_BC, LAM_E = 1.0, 100.0
BETAS = (0.999, 0.999)


def _assemble(gx, gy, hxx, hyy, hxy):
    gradu = torch.stack([gx, gy], dim=-1)
    row0 = torch.stack([hxx, hxy], dim=-1)
    row1 = torch.stack([hxy, hyy], dim=-1)
    H = torch.stack([row0, row1], dim=-2)
    return gradu, H


def div_P(gradu, H):
    I2 = torch.eye(2, dtype=gradu.dtype, device=gradu.device)
    F = I2 + gradu
    detF = F[..., 0, 0] * F[..., 1, 1] - F[..., 0, 1] * F[..., 1, 0]
    detF = torch.clamp(detF, min=1e-6)
    Finv = torch.stack([
        torch.stack([F[..., 1, 1], -F[..., 0, 1]], dim=-1),
        torch.stack([-F[..., 1, 0], F[..., 0, 0]], dim=-1),
    ], dim=-2) / detF[..., None, None]
    lnJ = torch.log(detF)
    lap = torch.stack([H[..., 0, 0, 0] + H[..., 0, 1, 1],
                       H[..., 1, 0, 0] + H[..., 1, 1, 1]], dim=-1)
    t1 = MU * lap
    t2 = ((MU - LAM * lnJ)[..., None] *
          torch.einsum("...Jk,...Li,...kLJ->...i", Finv, Finv, H))
    t3 = LAM * torch.einsum("...Ji,...Lk,...kLJ->...i", Finv, Finv, H)
    return t1 + t2 + t3


def div_sigma(H):
    div_u_x = H[..., 0, 0, 0] + H[..., 1, 0, 1]
    div_u_y = H[..., 0, 1, 0] + H[..., 1, 1, 1]
    grad_div = torch.stack([div_u_x, div_u_y], dim=-1)
    lap = torch.stack([H[..., 0, 0, 0] + H[..., 0, 1, 1],
                       H[..., 1, 0, 0] + H[..., 1, 1, 1]], dim=-1)
    return (LAM + MU) * grad_div + MU * lap


def strain_energy(gradu):
    I2 = torch.eye(2, dtype=gradu.dtype, device=gradu.device)
    F = I2 + gradu
    detF = torch.clamp(F[..., 0, 0] * F[..., 1, 1] - F[..., 0, 1] * F[..., 1, 0],
                       min=1e-6)
    lnJ = torch.log(detF)
    trFtF = (F ** 2).sum(dim=(-2, -1))
    return 0.5 * MU * (trFtF - 2.0 - 2.0 * lnJ) + 0.5 * LAM * lnJ ** 2


def rel_l2(pred, true):
    p = pred.reshape(pred.shape[0], -1)
    t = true.reshape(true.shape[0], -1)
    return (torch.linalg.norm(p - t, dim=1) /
            torch.linalg.norm(t, dim=1)).mean().item()


def build_data(cfg, device, seed=0):
    (x_g, y_g), eps_tr, u_tr, eps_te, u_te = he.get_dataset(
        cfg["data_dir"], cfg["M_train"], cfg["M_test"], cfg["nx"], cfg["ny"])
    rng = np.random.default_rng(seed)
    ny, nx = u_tr.shape[1], u_tr.shape[2]

    vs_tr = np.repeat(-eps_tr[:, None], cfg["n_bin"], axis=1)
    vs_te = np.repeat(-eps_te[:, None], cfg["n_bin"], axis=1)

    def grid(nxq, nyq):
        X, Y = np.meshgrid(np.linspace(0, LX, nxq), np.linspace(0, LY, nyq),
                           indexing="ij")
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    y_coll = grid(cfg["n_cx"], cfg["n_cy"])
    y_cgrid = grid(cfg["n_gx"], cfg["n_gy"])

    ii = rng.integers(1, ny - 1, cfg["N_u"])
    jj = rng.integers(1, nx - 1, cfg["N_u"])
    y_obs = np.stack([x_g[jj], y_g[ii]], axis=1)
    u_obs_tr = u_tr[:, ii, jj, :]

    bj = np.linspace(0, nx - 1, cfg["n_btb"]).astype(int)
    bi = np.linspace(0, ny - 1, cfg["n_blr"]).astype(int)
    b_ij = ([(0, j) for j in bj] + [(ny - 1, j) for j in bj] +
            [(i, 0) for i in bi] + [(i, nx - 1) for i in bi])
    bi_a = np.array([q[0] for q in b_ij])
    bj_a = np.array([q[1] for q in b_ij])
    y_bc = np.stack([x_g[bj_a], y_g[bi_a]], axis=1)
    u_bc_tr = u_tr[:, bi_a, bj_a, :]

    Xt, Yt = np.meshgrid(x_g, y_g, indexing="ij")
    y_test = np.stack([Xt.ravel(), Yt.ravel()], axis=1)
    u_test = np.ascontiguousarray(u_te.transpose(0, 2, 1, 3)).reshape(
        u_te.shape[0], -1, 2)

    sd = np.abs(vs_tr).std() + 1e-12
    wx = np.ones(cfg["n_cx"]); wx[0] = wx[-1] = 0.5
    wy = np.ones(cfg["n_cy"]); wy[0] = wy[-1] = 0.5
    w = np.outer(wx, wy).ravel() * (LX / (cfg["n_cx"] - 1)) * (LY / (cfg["n_cy"] - 1))

    def T(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)

    return {
        "vs_tr": T(vs_tr / sd), "vs_te": T(vs_te / sd),
        "u_obs_tr": T(u_obs_tr), "u_bc_tr": T(u_bc_tr),
        "y_coll": T(y_coll), "y_cgrid": T(y_cgrid), "y_obs": T(y_obs),
        "y_bc": T(y_bc), "y_test": T(y_test), "u_test": T(u_test), "quad": T(w),
    }


def _jet(net, b, y):
    t, d1, d2 = trunk_jet(net.trunk, y, [(0, 0), (1, 1), (0, 1)])
    val = net.combine(b, t) + net.bias
    gx, gy = net.combine(b, d1[0]), net.combine(b, d1[1])
    hxx, hyy = net.combine(b, d2[(0, 0)]), net.combine(b, d2[(1, 1)])
    hxy = net.combine(b, d2[(0, 1)])
    return val, _assemble(gx, gy, hxx, hyy, hxy)


def _value(net, b, y):
    return net.combine(b, net.trunk_out(y)) + net.bias


def _corr_branch_in(prior, bp, vs, y_cgrid):
    return torch.cat([vs, _value(prior, bp, y_cgrid).flatten(1)], dim=1)


# --- stage 1: physics-only prior -------------------------------------------- #
def train_prior(operator, d, cfg, device, seed=0):
    torch.manual_seed(seed)
    prior = DeepONet(cfg["n_bin"], 2, cfg["p"], cfg["width"], cfg["depth"], n_out=2).to(device)
    opt = torch.optim.Adam(prior.parameters(), lr=cfg["lr"], betas=cfg["betas"])
    M, bs = d["vs_tr"].shape[0], cfg["batch"]
    g = torch.Generator(device="cpu").manual_seed(seed)
    f_body = torch.tensor(BODY, dtype=torch.float32, device=device)
    for epoch in range(cfg["epochs_prior"]):
        idx = torch.randperm(M, generator=g)[:bs].to(device)
        vs = d["vs_tr"][idx]
        opt.zero_grad(set_to_none=True)
        bp = prior.branch_out(vs)
        u, (gradu, H) = _jet(prior, bp, d["y_coll"])
        res = (div_P(gradu, H) if operator == "true" else div_sigma(H)) + f_body
        l_bc = ((_value(prior, bp, d["y_bc"]) - d["u_bc_tr"][idx]) ** 2).mean()
        W = strain_energy(gradu)
        l_e = ((W - (u * f_body).sum(-1)) * d["quad"]).sum(-1).mean()
        loss = (res ** 2).mean() + LAM_BC * l_bc + LAM_E * l_e
        loss.backward()
        opt.step()
        if (epoch + 1) % max(1, cfg["epochs_prior"] // 5) == 0 or epoch == 0:
            print(f"  [prior:{operator}] epoch {epoch+1:6d}  loss {loss.item():.3e}")
    prior.eval()
    return prior


# --- stage 2: data-only correction on the frozen prior ---------------------- #
def train_correction(prior, d, cfg, device, seed=0):
    torch.manual_seed(seed)
    corr = DeepONet(cfg["n_bin"] + 2 * cfg["n_gx"] * cfg["n_gy"], 2, cfg["p"],
                    cfg["width"], cfg["depth"], n_out=2).to(device)
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
    preds = []
    for i in range(0, d["vs_te"].shape[0], cfg["eval_batch"]):
        vs = d["vs_te"][i:i + cfg["eval_batch"]]
        with torch.no_grad():
            bp = prior.branch_out(vs)
            u = _value(prior, bp, d["y_test"])
            if corr is not None:
                cin = _corr_branch_in(prior, bp, vs, d["y_cgrid"])
                u = u + _value(corr, corr.branch_out(cin), d["y_test"])
        preds.append(u)
    u = torch.cat(preds)
    ref = d["u_test"]
    mag = lambda a: torch.linalg.norm(a, dim=-1)
    return {"ux": rel_l2(u[..., 0], ref[..., 0]),
            "uy": rel_l2(u[..., 1], ref[..., 1]),
            "u": rel_l2(mag(u), mag(ref))}


FULL = dict(M_train=20, M_test=10, nx=60, ny=12, n_bin=21, n_cx=61, n_cy=13,
            n_gx=31, n_gy=13, N_u=200, n_btb=61, n_blr=13, p=100, width=256,
            depth=4, lr=1e-3, betas=BETAS, epochs_prior=40000, epochs_corr=20000,
            batch=8, eval_batch=5, data_dir=None)
QUICK = dict(M_train=20, M_test=10, nx=60, ny=12, n_bin=21, n_cx=41, n_cy=11,
             n_gx=21, n_gy=11, N_u=100, n_btb=41, n_blr=11, p=100, width=128,
             depth=4, lr=1e-3, betas=BETAS, epochs_prior=200, epochs_corr=200,
             batch=8, eval_batch=5, data_dir=None)


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
    print(f"Method One (two-stage) 2d hyperelastic on {device} | N_p={cfg['M_train']} "
          f"coll={cfg['n_cx']}x{cfg['n_cy']} N_u={cfg['N_u']} "
          f"epochs={cfg['epochs_prior']}+{cfg['epochs_corr']}")

    d = build_data(cfg, device)
    modes = ["known", "misspecified", "corrected"] if a.mode == "all" else [a.mode]
    res, prior_mis = {}, None
    for m in modes:
        t0 = time.time()
        if m == "known":
            res[m] = (evaluate(train_prior("true", d, cfg, device), None, d, cfg),
                      time.time() - t0)
        elif m == "misspecified":
            prior_mis = prior_mis or train_prior("mis", d, cfg, device)
            res[m] = (evaluate(prior_mis, None, d, cfg), time.time() - t0)
        else:
            prior_mis = prior_mis or train_prior("mis", d, cfg, device)
            corr = train_correction(prior_mis, d, cfg, device)
            res[m] = (evaluate(prior_mis, corr, d, cfg), time.time() - t0)

    print(f"\n==== relative L2 on {cfg['M_test']} test samples ====")
    print(f"{'model':<15}{'u_x':>14}{'u_y':>14}{'|u|':>14}     time")
    for m in modes:
        r, dt = res[m]
        print(f"{m:<15}{r['ux']:>14.4e}{r['uy']:>14.4e}{r['u']:>14.4e}   {dt:.0f}s")


if __name__ == "__main__":
    main()
