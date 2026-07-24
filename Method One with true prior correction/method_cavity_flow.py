"""Method One (two-stage prior correction) on the 2d lid-driven cavity.

    operator      G_theta : Re |-> u = (u_x, u_y)   (velocity)
    pressure op   G_xi    : Re |-> P
    misspecified  Newtonian (constant-viscosity) momentum (paper Eq. 6):
                      (u . grad) u + grad P - (1/Re) laplacian(u) = 0,  div u = 0
    true          power-law (shear-thinning) viscosity -- only enters the data.

Stage 1  train the velocity prior G_theta and pressure operator G_xi on ONLY what
         the scientist's Newtonian simulator knows -- the momentum + continuity
         residual, the a-priori KNOWN velocity BCs (lid profile + no-slip, built
         analytically) and a pressure-gauge anchor. NO real velocity or pressure
         data enters G_theta; pressure is solved from the physics, its free
         additive constant pinned by the gauge (mean P = 0, gauge-invariant for
         the momentum coupling). Then FREEZE both:
             L1 = ||mom||^2 + ||div u||^2 + lam_bc L_bc(known) + lam_g (mean P)^2
Stage 2  train the vector velocity correction G_phi on interior velocity DATA
         only, G_theta frozen:
             u_pred = G_theta + G_phi,  L2 = || u_pred(y_obs) - u(y_obs) ||^2

Departure from the paper: the paper supervises pressure on a dense 81x81 grid to
stabilise it. That is real data, so under the two-stage philosophy (G_theta uses
only simulator physics) it is dropped in favour of the gauge anchor -- pressure is
solved, not observed.

CAVEAT: unlike the other three scripts, this residual has no verified reference in
the repo -- it is derived here from paper Sec. 3.3. The loss weights lam_bc/lam_g
are not printed in the paper and are set to reasonable defaults. Treat cavity as
the highest-risk of the four and validate the physics term before trusting it.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets.cavity_flow as cf                              # noqa: E402
from deeponet import DeepONet, enable_fast_math, trunk_jet     # noqa: E402

LAM_BC, LAM_GAUGE, LAM_CONT = 100.0, 1.0, 1.0
BETAS = (0.999, 0.999)


def rel_l2(pred, true):
    p = pred.reshape(pred.shape[0], -1)
    t = true.reshape(true.shape[0], -1)
    return (torch.linalg.norm(p - t, dim=1) /
            torch.linalg.norm(t, dim=1)).mean().item()


def build_data(cfg, device, seed=0):
    (x_g, y_g), Re_tr, u_tr, p_tr, Re_te, u_te, p_te = cf.get_dataset(
        cfg["data_dir"], cfg["M_train"], cfg["M_test"], cfg["N"])
    rng = np.random.default_rng(seed)
    N = x_g.size
    mu, sd = Re_tr.mean(), Re_tr.std() + 1e-12

    def gridpts(n):
        X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    y_coll = gridpts(cfg["n_c"])
    y_cgrid = gridpts(cfg["n_g"])

    # KNOWN velocity boundary conditions (identical for every sample, independent
    # of Re): lid profile on the top wall, no-slip elsewhere -- built analytically,
    # never read from the solution data
    top = [(N - 1, j) for j in range(N)]
    bot = [(0, j) for j in range(N)]
    lef = [(i, 0) for i in range(N)]
    rig = [(i, N - 1) for i in range(N)]
    bij = top + bot + lef + rig
    bi = np.array([q[0] for q in bij]); bj = np.array([q[1] for q in bij])
    y_bc = np.stack([x_g[bj], y_g[bi]], axis=1)
    u_bc = np.zeros((y_bc.shape[0], 2))
    is_top = y_bc[:, 1] > 1.0 - 1e-6
    u_bc[is_top, 0] = 1.0 - np.cosh(10.0 * (y_bc[is_top, 0] - 0.5)) / np.cosh(5.0)

    # interior velocity observations -- the only real data, used in stage 2
    ii = rng.integers(1, N - 1, cfg["N_u"]); jj = rng.integers(1, N - 1, cfg["N_u"])
    y_obs = np.stack([x_g[jj], y_g[ii]], axis=1)
    u_obs = u_tr[:, ii, jj, :]

    Xt, Yt = np.meshgrid(x_g, y_g, indexing="ij")
    y_test = np.stack([Xt.ravel(), Yt.ravel()], axis=1)
    u_test = np.ascontiguousarray(u_te.transpose(0, 2, 1, 3)).reshape(u_te.shape[0], -1, 2)

    def T(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)

    return {
        "vs_tr": T(((Re_tr - mu) / sd)[:, None]), "vs_te": T(((Re_te - mu) / sd)[:, None]),
        "Re_tr": T(Re_tr[:, None]),
        "u_bc": T(u_bc), "u_obs": T(u_obs), "u_test": T(u_test),
        "y_coll": T(y_coll), "y_cgrid": T(y_cgrid), "y_bc": T(y_bc),
        "y_obs": T(y_obs), "y_test": T(y_test),
    }


def _vjet(net, b, y):
    t, d1, d2 = trunk_jet(net.trunk, y, [(0, 0), (1, 1)])
    u = net.combine(b, t) + net.bias                            # (B,Q,2)
    return (u, net.combine(b, d1[0]), net.combine(b, d1[1]),
            net.combine(b, d2[(0, 0)]), net.combine(b, d2[(1, 1)]))


def _pgrad(net, b, y):
    t, d1, _ = trunk_jet(net.trunk, y, [(0, None), (1, None)])
    P = net.combine(b, t) + net.bias[0]                         # (B,Q)
    return P, net.combine(b, d1[0]), net.combine(b, d1[1])


def _vvalue(net, b, y):
    return net.combine(b, net.trunk_out(y)) + net.bias          # (B,Q,2)


def momentum(u, ux, uy, uxx, uyy, Px, Py, inv_Re):
    adv = u[..., 0:1] * ux + u[..., 1:2] * uy                   # (u.grad)u
    lap = uxx + uyy
    gradP = torch.stack([Px, Py], dim=-1)
    return adv + gradP - inv_Re * lap                           # (B,Q,2)


# --- stage 1: physics-only velocity prior + pressure operator --------------- #
def train_prior(d, cfg, device, seed=0):
    torch.manual_seed(seed)
    vel = DeepONet(1, 2, cfg["p"], cfg["width"], cfg["depth"], n_out=2).to(device)
    pre = DeepONet(1, 2, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    params = list(vel.parameters()) + list(pre.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"], betas=cfg["betas"])
    M, bs = d["vs_tr"].shape[0], cfg["batch"]
    g = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(cfg["epochs_prior"]):
        idx = torch.randperm(M, generator=g)[:bs].to(device)
        vs, Re = d["vs_tr"][idx], d["Re_tr"][idx]
        opt.zero_grad(set_to_none=True)
        bv, bp = vel.branch_out(vs), pre.branch_out(vs)
        u, ux, uy, uxx, uyy = _vjet(vel, bv, d["y_coll"])
        P, Px, Py = _pgrad(pre, bp, d["y_coll"])
        inv_Re = (1.0 / Re).view(-1, 1, 1)
        res = momentum(u, ux, uy, uxx, uyy, Px, Py, inv_Re)
        cont = ux[..., 0] + uy[..., 1]
        l_bc = ((_vvalue(vel, bv, d["y_bc"]) - d["u_bc"]) ** 2).mean()   # known BC
        l_gauge = (P.mean(dim=1) ** 2).mean()          # pin pressure gauge (no data)
        loss = ((res ** 2).mean() + LAM_CONT * (cont ** 2).mean()
                + LAM_BC * l_bc + LAM_GAUGE * l_gauge)
        loss.backward()
        opt.step()
        if (epoch + 1) % max(1, cfg["epochs_prior"] // 5) == 0 or epoch == 0:
            print(f"  [prior] epoch {epoch+1:6d}  loss {loss.item():.3e} "
                  f"(bc {l_bc.item():.2e} gauge {l_gauge.item():.2e})")
    vel.eval(); pre.eval()
    return vel


# --- stage 2: data-only velocity correction on the frozen prior ------------- #
def train_correction(vel, d, cfg, device, seed=0):
    torch.manual_seed(seed)
    corr = DeepONet(1 + 2 * cfg["n_g"] ** 2, 2, cfg["p"], cfg["width"],
                    cfg["depth"], n_out=2).to(device)
    for pth in vel.parameters():
        pth.requires_grad_(False)
    opt = torch.optim.Adam(corr.parameters(), lr=cfg["lr"], betas=cfg["betas"])
    M, bs = d["vs_tr"].shape[0], cfg["batch"]
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    for epoch in range(cfg["epochs_corr"]):
        idx = torch.randperm(M, generator=g)[:bs].to(device)
        vs, u_obs = d["vs_tr"][idx], d["u_obs"][idx]
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            bv = vel.branch_out(vs)
            u_prior_obs = _vvalue(vel, bv, d["y_obs"])
            cin = torch.cat([vs, _vvalue(vel, bv, d["y_cgrid"]).flatten(1)], dim=1)
        phi = _vvalue(corr, corr.branch_out(cin), d["y_obs"])
        loss = ((u_prior_obs + phi - u_obs) ** 2).mean()
        loss.backward()
        opt.step()
        if (epoch + 1) % max(1, cfg["epochs_corr"] // 5) == 0 or epoch == 0:
            print(f"  [corr] epoch {epoch+1:6d}  L_data {loss.item():.3e}")
    corr.eval()
    return corr


def evaluate(vel, corr, d, cfg):
    preds = []
    for i in range(0, d["vs_te"].shape[0], cfg["eval_batch"]):
        vs = d["vs_te"][i:i + cfg["eval_batch"]]
        with torch.no_grad():
            bv = vel.branch_out(vs)
            u = _vvalue(vel, bv, d["y_test"])
            if corr is not None:
                cin = torch.cat([vs, _vvalue(vel, bv, d["y_cgrid"]).flatten(1)], dim=1)
                u = u + _vvalue(corr, corr.branch_out(cin), d["y_test"])
        preds.append(u)
    u = torch.cat(preds)
    ref = d["u_test"]
    mag = lambda a: torch.linalg.norm(a, dim=-1)
    return {"ux": rel_l2(u[..., 0], ref[..., 0]),
            "uy": rel_l2(u[..., 1], ref[..., 1]),
            "u": rel_l2(mag(u), mag(ref))}


FULL = dict(M_train=20, M_test=10, N=61, n_c=41, n_g=21, N_u=250,
            p=100, width=128, depth=4, lr=1e-3, betas=BETAS,
            epochs_prior=40000, epochs_corr=20000, batch=8, eval_batch=5, data_dir=None)
QUICK = dict(M_train=20, M_test=10, N=61, n_c=21, n_g=11, N_u=200,
             p=100, width=128, depth=4, lr=1e-3, betas=BETAS,
             epochs_prior=200, epochs_corr=200, batch=8, eval_batch=5, data_dir=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["misspecified", "corrected", "all"],
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
    print(f"Method One (two-stage) 2d cavity on {device} | N_p={cfg['M_train']} "
          f"grid={cfg['N']}x{cfg['N']} coll={cfg['n_c']}x{cfg['n_c']} "
          f"N_u={cfg['N_u']} epochs={cfg['epochs_prior']}+{cfg['epochs_corr']}")

    d = build_data(cfg, device)
    modes = ["misspecified", "corrected"] if a.mode == "all" else [a.mode]
    vel = train_prior(d, cfg, device)            # stage 1 shared by both rows
    res = {}
    for m in modes:
        t0 = time.time()
        corr = train_correction(vel, d, cfg, device) if m == "corrected" else None
        res[m] = (evaluate(vel, corr, d, cfg), time.time() - t0)

    print(f"\n==== relative L2 on {cfg['M_test']} test samples ====")
    print(f"{'model':<15}{'u_x':>14}{'u_y':>14}{'|u|':>14}     time")
    for m in modes:
        r, dt = res[m]
        print(f"{m:<15}{r['ux']:>14.4e}{r['uy']:>14.4e}{r['u']:>14.4e}   {dt:.0f}s")


if __name__ == "__main__":
    main()
