"""Physics-guided operator correction on the 2d hyperelastic beam (paper Sec. 3.4).

    true system      -div P(u) = f,   f = (0, -1000)^T,   Omega = [0,1] x [0,0.1]
                     P = mu F + (-mu + lambda ln J) F^-T,  F = I + grad u,  J = det F
                     E = 1e6, nu = 0.3   ->   mu = E/(2(1+nu)), lambda = E nu/((1+nu)(1-2nu))
    misspecified     linear elasticity:  div sigma(u) + f = 0,
                     sigma = lambda tr(eps) I + 2 mu eps,  eps = (grad u + grad u^T)/2

The learned operator is G : v |-> u = (u_x, u_y), where v is the compressive
displacement applied on the right boundary, u|Gamma_R = (-eps, 0) with eps in
[0, 0.2]. The left edge is clamped and the top/bottom edges are traction-free.

Serial architecture (paper Sec. 3.4)
    prior       G_theta : v(21 boundary points)                 -> u(x,y)
    correction  G_psi   : [v(21), u_theta(51 x 21 grid, both components)] -> c(x,y)

Loss (paper Eq. 7), with the same modification as run_correction.py -- N0 applied
to *both* of the first two terms, i.e. to their sum:

    L_phys   = || N0[ G_theta(v) + G_psi(v,u_theta) ] + f ||^2      (101 x 21 grid)
    L_energy = Pi(u_theta),  Pi(u) = int W(F) dOmega - int f . u dOmega
    L = L_phys + lambda_bc L_bc + lambda_u L_u + lambda_e L_energy
    lambda_e = 100,  lambda_u = 100000,  lambda_bc = 1

Traction-free top/bottom means the Neumann term of Pi drops, per the paper.

NONDIMENSIONALISATION. Following Sec. 3.4, stresses, the body force and the
material parameters are divided by E; the spatial coordinates are deliberately
NOT rescaled (the paper is explicit about this), so the trunk sees y in [0, 0.1]
against x in [0, 1].

Modes reproduce the four rows of the paper's Table 5:
    known         residual uses the true neo-Hookean operator
    misspecified  residual uses linear elasticity, no correction
    data          standard DeepONet: data-consistency loss only, no physics
    corrected     N0 applied to prior + correction
"""
import argparse
import time

import numpy as np
import torch

import datasets.hyperelastic as he
from deeponet import DeepONet, enable_fast_math, trunk_jet

# nondimensionalised material parameters and body force (divided by E)
MU = he.MU / he.E                       # 1 / (2(1+nu))
LAM = he.LAM / he.E                     # nu / ((1+nu)(1-2nu))
BODY = he.BODY / he.E                   # (0, -1e-3)
LX, LY = 1.0, 0.1

LAM_BC, LAM_U, LAM_E = 1.0, 100000.0, 100.0
BETAS = (0.999, 0.999)


# --------------------------------------------------------------------------- #
# physics
# --------------------------------------------------------------------------- #
def _assemble(gx, gy, hxx, hyy, hxy):
    """Pack the component-wise derivatives into grad u and the Hessian.

    gradu[..., i, J] = du_i/dX_J ;  H[..., k, L, J] = d^2 u_k / dX_L dX_J."""
    gradu = torch.stack([gx, gy], dim=-1)                       # (B,Q,2,2)
    row0 = torch.stack([hxx, hxy], dim=-1)                      # d/dX_0 d/dX_{0,1}
    row1 = torch.stack([hxy, hyy], dim=-1)
    H = torch.stack([row0, row1], dim=-2)                       # (B,Q,2,2,2)
    return gradu, H


def div_P(gradu, H):
    """Divergence of the neo-Hookean first Piola-Kirchhoff stress.

    (div P)_i = dP_iJ/dX_J = A_iJkL d^2 u_k / dX_L dX_J with the material tangent
    A_iJkL = mu d_ik d_JL + (mu - lambda lnJ) Finv_Jk Finv_Li + lambda Finv_Ji Finv_Lk
    -- the same tangent the FEM generator assembles, so the two agree by
    construction."""
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
    """Divergence of the linear-elastic (misspecified) stress:
    (div sigma)_i = (lambda + mu) d_i(div u) + mu laplacian(u_i)."""
    div_u_x = H[..., 0, 0, 0] + H[..., 1, 0, 1]     # d/dX_0 (u_0,0 + u_1,1)
    div_u_y = H[..., 0, 1, 0] + H[..., 1, 1, 1]     # d/dX_1 (u_0,0 + u_1,1)
    grad_div = torch.stack([div_u_x, div_u_y], dim=-1)
    lap = torch.stack([H[..., 0, 0, 0] + H[..., 0, 1, 1],
                       H[..., 1, 0, 0] + H[..., 1, 1, 1]], dim=-1)
    return (LAM + MU) * grad_div + MU * lap


def strain_energy(gradu):
    """W(F) = mu/2 (tr(F^T F) - 2 - 2 ln J) + lambda/2 (ln J)^2."""
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


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_data(cfg, device, seed=0):
    (x_g, y_g), eps_tr, u_tr, eps_te, u_te = he.get_dataset(
        cfg["data_dir"], cfg["M_train"], cfg["M_test"], cfg["nx"], cfg["ny"])
    rng = np.random.default_rng(seed)
    ny, nx = u_tr.shape[1], u_tr.shape[2]

    # branch input: the right-boundary displacement sampled at 21 points
    vs_tr = np.repeat(-eps_tr[:, None], cfg["n_bin"], axis=1)
    vs_te = np.repeat(-eps_te[:, None], cfg["n_bin"], axis=1)

    def grid(nxq, nyq):
        gx = np.linspace(0.0, LX, nxq)
        gy = np.linspace(0.0, LY, nyq)
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    y_coll = grid(cfg["n_cx"], cfg["n_cy"])          # 101 x 21 collocation
    y_cgrid = grid(cfg["n_gx"], cfg["n_gy"])         # 51 x 21 correction branch

    # observations: 200 interior points, plus boundary points
    # (101 top and bottom, 21 on each of left and right)
    ii = rng.integers(1, ny - 1, cfg["N_u"])
    jj = rng.integers(1, nx - 1, cfg["N_u"])
    y_obs = np.stack([x_g[jj], y_g[ii]], axis=1)
    u_obs_tr = u_tr[:, ii, jj, :]

    bj = np.linspace(0, nx - 1, cfg["n_btb"]).astype(int)
    bi = np.linspace(0, ny - 1, cfg["n_blr"]).astype(int)
    b_ij = ([(0, j) for j in bj] + [(ny - 1, j) for j in bj] +
            [(i, 0) for i in bi] + [(i, nx - 1) for i in bi])
    bi_a = np.array([p[0] for p in b_ij])
    bj_a = np.array([p[1] for p in b_ij])
    y_bc = np.stack([x_g[bj_a], y_g[bi_a]], axis=1)
    u_bc_tr = u_tr[:, bi_a, bj_a, :]

    # full evaluation grid = the stored node grid
    Xt, Yt = np.meshgrid(x_g, y_g, indexing="ij")
    y_test = np.stack([Xt.ravel(), Yt.ravel()], axis=1)
    u_test = np.ascontiguousarray(u_te.transpose(0, 2, 1, 3)).reshape(
        u_te.shape[0], -1, 2)

    sd = np.abs(vs_tr).std() + 1e-12

    def T(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)

    # trapezoid weights for the energy integral over the collocation grid
    wx = np.ones(cfg["n_cx"]); wx[0] = wx[-1] = 0.5
    wy = np.ones(cfg["n_cy"]); wy[0] = wy[-1] = 0.5
    w = np.outer(wx, wy).ravel()
    w = w * (LX / (cfg["n_cx"] - 1)) * (LY / (cfg["n_cy"] - 1))

    return {
        "vs_tr": T(vs_tr / sd), "vs_te": T(vs_te / sd),
        "u_obs_tr": T(u_obs_tr), "u_bc_tr": T(u_bc_tr),
        "y_coll": T(y_coll), "y_cgrid": T(y_cgrid), "y_obs": T(y_obs),
        "y_bc": T(y_bc), "y_test": T(y_test), "u_test": T(u_test),
        "quad": T(w), "eps_te": eps_te,
    }


# --------------------------------------------------------------------------- #
# network evaluation
# --------------------------------------------------------------------------- #
def _jet(net, b, y):
    """Value, both first partials and all second partials of a 2-component
    DeepONet on (x, y) query points."""
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


# --------------------------------------------------------------------------- #
# training / evaluation
# --------------------------------------------------------------------------- #
def train_mode(mode, d, cfg, device, seed=0, verbose=True):
    torch.manual_seed(seed)
    prior = DeepONet(cfg["n_bin"], 2, cfg["p"], cfg["width"], cfg["depth"],
                     n_out=2).to(device)
    corr = DeepONet(cfg["n_bin"] + 2 * cfg["n_gx"] * cfg["n_gy"], 2, cfg["p"],
                    cfg["width"], cfg["depth"], n_out=2).to(device)

    params = list(prior.parameters())
    if mode == "corrected":
        params += list(corr.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"], betas=cfg["betas"])

    M, bs = d["vs_tr"].shape[0], cfg["batch"]
    g = torch.Generator(device="cpu").manual_seed(seed)
    f_body = torch.tensor(BODY, dtype=torch.float32, device=device)

    for epoch in range(cfg["epochs"]):
        idx = torch.randperm(M, generator=g)[:bs].to(device)
        vs = d["vs_tr"][idx]
        opt.zero_grad(set_to_none=True)
        bp = prior.branch_out(vs)

        l_u = ((_value(prior, bp, d["y_obs"]) - d["u_obs_tr"][idx]) ** 2).mean()
        l_bc = ((_value(prior, bp, d["y_bc"]) - d["u_bc_tr"][idx]) ** 2).mean()

        if mode == "data":                       # standard DeepONet, no physics
            loss = LAM_U * l_u + LAM_BC * l_bc
            l_phys = l_e = torch.zeros((), device=device)
        else:
            u, (gradu, H) = _jet(prior, bp, d["y_coll"])
            if mode == "known":
                res = div_P(gradu, H) + f_body
            elif mode == "misspecified":
                res = div_sigma(H) + f_body
            else:                                # corrected: N0 on the sum
                bc = corr.branch_out(_corr_branch_in(prior, bp, vs, d["y_cgrid"]))
                _, (gradc, Hc) = _jet(corr, bc, d["y_coll"])
                res = div_sigma(H + Hc) + f_body
            l_phys = (res ** 2).mean()
            # total potential energy of the prior prediction (paper Eq. 7)
            W = strain_energy(gradu)
            l_e = ((W - (u * f_body).sum(-1)) * d["quad"]).sum(-1).mean()
            loss = l_phys + LAM_BC * l_bc + LAM_U * l_u + LAM_E * l_e

        loss.backward(inputs=params)
        opt.step()
        if verbose and ((epoch + 1) % max(1, cfg["epochs"] // 10) == 0 or epoch == 0):
            print(f"  [{mode}] epoch {epoch+1:6d}  loss {loss.item():.3e} "
                  f"(phys {l_phys.item():.2e} u {l_u.item():.2e} "
                  f"bc {l_bc.item():.2e} energy {l_e.item():.2e})")

    prior.eval()
    corr.eval()
    return evaluate(mode, prior, corr, d, cfg)


def evaluate(mode, prior, corr, d, cfg):
    """Relative L2 of u_x, u_y and |u| on the test set (paper Table 5).

    The reported field is G_theta: the data terms anchor it to the observations
    while G_psi absorbs the model-form discrepancy inside the residual."""
    preds = []
    for i in range(0, d["vs_te"].shape[0], cfg["eval_batch"]):
        vs = d["vs_te"][i:i + cfg["eval_batch"]]
        with torch.no_grad():
            bp = prior.branch_out(vs)
            preds.append(_value(prior, bp, d["y_test"]))
    u = torch.cat(preds)
    ref = d["u_test"]
    mag = lambda a: torch.linalg.norm(a, dim=-1)
    return {"ux": rel_l2(u[..., 0], ref[..., 0]),
            "uy": rel_l2(u[..., 1], ref[..., 1]),
            "u": rel_l2(mag(u), mag(ref))}


# --------------------------------------------------------------------------- #
# configs / driver
# --------------------------------------------------------------------------- #
FULL = dict(M_train=200, M_test=100, nx=100, ny=20, n_bin=21,
            n_cx=101, n_cy=21, n_gx=51, n_gy=21, N_u=200, n_btb=101, n_blr=21,
            p=100, width=256, depth=4, lr=1e-3, epochs=100000, batch=32,
            eval_batch=25, betas=BETAS, data_dir="data")
QUICK = dict(M_train=20, M_test=10, nx=60, ny=12, n_bin=21,
             n_cx=41, n_cy=11, n_gx=21, n_gy=11, N_u=100, n_btb=41, n_blr=11,
             p=100, width=128, depth=4, lr=1e-3, epochs=100, batch=8,
             eval_batch=5, betas=BETAS, data_dir="data/quick")  # scratch cache

PAPER_TABLE5 = {"known": (0.68e-3, 0.61e-3, 0.46e-3),
                "misspecified": (25.50e-3, 11.00e-3, 11.51e-3),
                "data": (6.45e-3, 5.83e-3, 4.19e-3),
                "corrected": (3.75e-3, 0.97e-3, 2.06e-3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["known", "misspecified", "data",
                                       "corrected", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--beta1", type=float, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    enable_fast_math()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(QUICK if a.quick else FULL)
    if a.epochs is not None:
        cfg["epochs"] = a.epochs
    if a.batch is not None:
        cfg["batch"] = a.batch
    if a.beta1 is not None:
        cfg["betas"] = (a.beta1, cfg["betas"][1])

    print(f"serial PI-DeepONet operator correction (2d hyperelastic) on {device}")
    print(f"  N_p={cfg['M_train']} collocation={cfg['n_cx']}x{cfg['n_cy']} "
          f"N_u={cfg['N_u']} width={cfg['width']} lr={cfg['lr']} "
          f"epochs={cfg['epochs']} batch={cfg['batch']}")
    print(f"  lambda_e={LAM_E:g} lambda_u={LAM_U:g} lambda_bc={LAM_BC:g}")
    print("  physics loss: || N0[ G_theta + G_psi ] + f ||^2  "
          "(N0 applied to both terms)\n")

    d = build_data(cfg, device)
    modes = (["known", "misspecified", "data", "corrected"]
             if a.mode == "all" else [a.mode])
    res = {}
    for m in modes:
        t0 = time.time()
        res[m] = (train_mode(m, d, cfg, device), time.time() - t0)

    print(f"\n==== relative L2 on {cfg['M_test']} test samples ====")
    print(f"{'model':<15}{'u_x':>14}{'u_y':>14}{'|u|':>14}     time")
    for m in modes:
        r, dt = res[m]
        print(f"{m:<15}{r['ux']:>14.4e}{r['uy']:>14.4e}{r['u']:>14.4e}   {dt:.0f}s")

    print("\npaper Table 5 (relative L2):")
    for m in modes:
        px, py, pu = PAPER_TABLE5[m]
        print(f"  {m:<15}{px:>12.2e}{py:>14.2e}{pu:>14.2e}")


if __name__ == "__main__":
    main()
