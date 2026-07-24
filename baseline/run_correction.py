"""Physics-guided operator correction under model misspecification, realised as a
serial PI-DeepONet -- the method of Ma, Boulle, Yang, Wu & Guo (2026),
arXiv:2606.03469 -- on the 1d diffusion-reaction benchmark of their Sec. 3.1.

True system (used only to manufacture the data):

    N[u] = D u_xx - k_r(u) u = v,   k_r(u) = 0.5 e^{-u},  D = 0.1,  x in [-1,1]
    u(-1) = u(1) = 0

Misspecified prior operator (the physics the model is actually trained with):

    N0[w] = D w_xx - k_r_const                              (k_r_const = 0.5)

Serial architecture (paper Sec. 2.2)
    prior       G_theta : v(x_1..x_m)                        -> u_theta(y)
    correction  G_psi   : [v(x_1..x_m), u_theta(y_c^1..y_c^n)] -> c(y)

Losses.  The data-consistency term is the paper's Eq. (2),

    L_data = || G_theta(v)(y_obs) - u(y_obs) ||^2 ,

and the physics term is the paper's Eq. (1) WITH THE REQUESTED MODIFICATION:
N0 is applied to *both* of the first two terms, i.e. to their sum, rather than to
G_theta alone,

    paper  Eq. (1): L_phys = || N0[G_theta(v)(y_f)] + G_psi(v,u_theta)(y_f)  - v(y_f) ||^2
    here (modified): L_phys = || N0[G_theta(v)(y_f)  + G_psi(v,u_theta)(y_f)] - v(y_f) ||^2

    L = L_phys + lambda_d L_data

N0 therefore acts on the *corrected field* u_theta + c, which makes the correction
a solution-space object -- consistent with G_psi : V x U -> U in the paper's
Sec. 2.1 -- instead of a forcing-space term added straight into the residual.
Written out for this problem the enforced equation is

    D (u_theta + c)_xx - k_r_const = v
    <=>  D u_theta_xx + Phi = v,    Phi := D c_xx - k_r_const = N0[G_psi],

so the *learned reaction term* recovered from the correction network is exactly
N0[G_psi], and it is this quantity that is compared against the paper's correction
target phi = -k_r(u) u.  The reported solution stays G_theta: the data term anchors
G_theta to the observations while G_psi absorbs the model-form discrepancy inside
the residual, so the misspecified physics no longer biases G_theta.

Derivatives are exact: a DeepONet is sum_k branch_k(v) trunk_k(y), so d^2/dy^2
acts on the shared trunk basis only and is obtained with forward-mode autodiff
(see deeponet.trunk_derivatives) -- mesh-free and O(1) passes.

Three training modes reproduce the paper's headline comparison (their Table 2):
    known         residual uses the true operator N          (reference floor)
    misspecified  residual uses N0, no correction            (error blows up)
    corrected     residual uses N0[G_theta] + N0[G_psi]      (recovered)
"""
import argparse
import time

import numpy as np
import torch

import datasets.diffusion_reaction as dr
from deeponet import DeepONet, enable_fast_math, trunk_derivatives

K_R_CONST = 0.5        # misspecified (concentration-independent) reaction rate
LAMBDA_D = 1.0         # lambda_d in L = L_phys + lambda_d L_data
BETAS = (0.999, 0.999)  # Adam betas, paper Sec. 3


# --------------------------------------------------------------------------- #
# physics
# --------------------------------------------------------------------------- #
def N0(w, w_xx):
    """Misspecified prior operator  N0[w] = D w_xx - k_r_const."""
    return dr.D * w_xx - K_R_CONST


def N_true(u, u_xx):
    """True operator  N[u] = D u_xx - k_r(u) u,  k_r(u) = 0.5 e^{-u}."""
    return dr.D * u_xx - 0.5 * torch.exp(-u) * u


def eval_field(w, x):
    """Analytic forcing v and solution u at points x for coefficients w."""
    u, uxx = dr.manufactured(x, w)
    v = dr.D * uxx - 0.5 * np.exp(-u) * u
    return v, u


def rel_l2(pred, true):
    return (torch.linalg.norm(pred - true, dim=1) /
            torch.linalg.norm(true, dim=1)).mean().item()


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_data(cfg, device, seed=0):
    """Paper's Sec. 3.1 setup: N_p training samples, 101 sensors on [-1,1],
    N_u solution observations and N_f collocation points drawn uniformly from
    [-1,1], test set of 100 samples on 201 uniform locations."""
    rng = np.random.default_rng(seed)          # training samples / points
    rng_te = np.random.default_rng(12345)      # test set: fixed across runs
    nm = 2 * dr.N_MODES
    w_tr = rng.uniform(0, 1, (cfg["M_train"], nm))
    w_te = rng_te.uniform(0, 1, (cfg["M_test"], nm))
    x_s = np.linspace(-1, 1, cfg["n_sensors"])
    x_o = rng.uniform(-1, 1, cfg["N_u"])
    x_c = rng.uniform(-1, 1, cfg["N_f"])
    x_t = np.linspace(-1, 1, 201)

    vs_tr = eval_field(w_tr, x_s)[0]
    vs_te = eval_field(w_te, x_s)[0]
    uo_tr = eval_field(w_tr, x_o)[1]
    vc_tr = eval_field(w_tr, x_c)[0]
    vt_te, ut_te = eval_field(w_te, x_t)
    phi_te = -0.5 * np.exp(-ut_te) * ut_te     # correction target, paper Sec. 3.1
    mu, sd = vs_tr.mean(), vs_tr.std()

    def T(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    return {
        "vs_tr": T((vs_tr - mu) / sd), "vs_te": T((vs_te - mu) / sd),
        "uo_tr": T(uo_tr), "vc_tr": T(vc_tr),
        "ut_te": T(ut_te), "vt_te": T(vt_te), "phi_te": T(phi_te),
        "y_coll": T(x_c[:, None]),   # jvp supplies tangents; no leaf-grad needed
        "x_s": T(x_s[:, None]), "x_o": T(x_o[:, None]),
        "x_t": T(x_t[:, None]), "x_bc": T(np.array([[-1.0], [1.0]])),
    }


def _hard_bc(x):
    """Boundary factor g(x)=1-x^2 and its derivatives, so u=g*f obeys u(+-1)=0
    exactly. Returns g, g' as (1,N) rows; g'' is the constant -2."""
    g = (1.0 - x ** 2).T
    gx = (-2.0 * x).T
    return g, gx, -2.0


# --------------------------------------------------------------------------- #
# network evaluation
# --------------------------------------------------------------------------- #
def _prior_fields(prior, bp, y, g, gx, gxx):
    """u_theta = g(y) f(y) (hard Dirichlet BC) and its second derivative."""
    t, t_x, t_xx = trunk_derivatives(prior.trunk, y, order=2)
    f = bp @ t.T + prior.bias
    u = g * f
    u_xx = gxx * f + 2.0 * gx * (bp @ t_x.T) + g * (bp @ t_xx.T)
    return u, u_xx


def _prior_value(prior, bp, y, g):
    return g * (bp @ prior.trunk_out(y).T + prior.bias)


def _corr_fields(corr, bc, y):
    """Correction c(y) and c_xx(y); no boundary factor is imposed on c."""
    t, _, t_xx = trunk_derivatives(corr.trunk, y, order=2)
    return bc @ t.T + corr.bias, bc @ t_xx.T


def _corr_branch_in(corr_v, prior, bp, x_s, gs):
    """Branch input of G_psi: the sampled input function together with the prior
    prediction sampled at the same locations (paper Sec. 2.2)."""
    return torch.cat([corr_v, _prior_value(prior, bp, x_s, gs)], dim=1)


# --------------------------------------------------------------------------- #
# training / evaluation
# --------------------------------------------------------------------------- #
def train_mode(mode, d, cfg, device, seed=0, verbose=True):
    torch.manual_seed(seed)
    prior = DeepONet(cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    corr = DeepONet(2 * cfg["n_sensors"], 1, cfg["p"], cfg["width"], cfg["depth"]).to(device)

    params = list(prior.parameters())
    if mode == "corrected":
        params += list(corr.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"], betas=cfg["betas"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
             if cfg["cosine"] else None)

    gc, gcx, gcxx = _hard_bc(d["y_coll"])          # collocation points
    go, _, _ = _hard_bc(d["x_o"])                  # observation points
    gs, _, _ = _hard_bc(d["x_s"])                  # sensor points

    for epoch in range(cfg["epochs"]):
        opt.zero_grad(set_to_none=True)
        bp = prior.branch_out(d["vs_tr"])                              # (M, p)
        u_coll, u_xx = _prior_fields(prior, bp, d["y_coll"], gc, gcx, gcxx)
        u_obs = _prior_value(prior, bp, d["x_o"], go)

        if mode == "known":
            res = N_true(u_coll, u_xx) - d["vc_tr"]
        elif mode == "misspecified":
            res = N0(u_coll, u_xx) - d["vc_tr"]
        else:  # corrected -- N0 applied to BOTH G_theta and G_psi, i.e. to their sum
            bc = corr.branch_out(_corr_branch_in(d["vs_tr"], prior, bp, d["x_s"], gs))
            c, c_xx = _corr_fields(corr, bc, d["y_coll"])
            res = N0(u_coll + c, u_xx + c_xx) - d["vc_tr"]

        loss_phys = (res ** 2).mean()
        loss_data = ((u_obs - d["uo_tr"]) ** 2).mean()
        loss = loss_phys + cfg["lambda_d"] * loss_data
        loss.backward(inputs=params)
        opt.step()
        if sched is not None:
            sched.step()
        if verbose and ((epoch + 1) % max(1, cfg["epochs"] // 10) == 0 or epoch == 0):
            print(f"  [{mode}] epoch {epoch+1:6d}  loss {loss.item():.3e} "
                  f"(phys {loss_phys.item():.3e}  data {loss_data.item():.3e})")

    prior.eval()
    corr.eval()
    return evaluate(mode, prior, corr, d)


def evaluate(mode, prior, corr, d):
    """Relative L2 errors of the solution u, the source term v, and the reaction
    (correction) term phi on the test set -- the three columns of the paper's
    Table 2."""
    gt, gtx, gtxx = _hard_bc(d["x_t"])
    gs, _, _ = _hard_bc(d["x_s"])
    with torch.enable_grad():
        bp = prior.branch_out(d["vs_te"])
        u, u_xx = _prior_fields(prior, bp, d["x_t"], gt, gtx, gtxx)
        if mode == "known":
            phi = -0.5 * torch.exp(-u) * u
        elif mode == "misspecified":
            phi = torch.full_like(u, -K_R_CONST)
        else:
            bc = corr.branch_out(_corr_branch_in(d["vs_te"], prior, bp, d["x_s"], gs))
            c, c_xx = _corr_fields(corr, bc, d["x_t"])
            phi = N0(c, c_xx)                      # = D c_xx - k_r_const
        v_pred = dr.D * u_xx + phi                 # source implied by the model
    u, phi, v_pred = u.detach(), phi.detach(), v_pred.detach()
    return {"u": rel_l2(u, d["ut_te"]),
            "v": rel_l2(v_pred, d["vt_te"]),
            "phi": rel_l2(phi, d["phi_te"])}


# --------------------------------------------------------------------------- #
# configs / driver
# --------------------------------------------------------------------------- #
# paper Table 1 (diffusion-reaction) + Sec. 3.1 sample counts
FULL = dict(M_train=1000, M_test=100, n_sensors=101, N_u=100, N_f=1000,
            p=100, width=64, depth=4, lr=1e-3, epochs=100000,
            betas=BETAS, lambda_d=LAMBDA_D, cosine=False)
QUICK = dict(M_train=64, M_test=32, n_sensors=41, N_u=40, N_f=128,
             p=100, width=64, depth=4, lr=1e-3, epochs=100,
             betas=BETAS, lambda_d=LAMBDA_D, cosine=False)

PAPER_TABLE2 = {"known": (0.90e-3, 0.58e-3, 0.81e-3),
                "misspecified": (1614.20e-3, 109386.97e-3, 1209.27e-3),
                "corrected": (1.85e-3, 1.10e-3, 32.10e-3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["known", "misspecified", "corrected", "all"],
                    default="all")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lambda_d", type=float, default=None)
    ap.add_argument("--beta1", type=float, default=None,
                    help="Adam beta1 (paper uses 0.999)")
    ap.add_argument("--n_runs", type=int, default=1,
                    help="independent runs; the paper reports mean +- std over 5")
    ap.add_argument("--cosine", action="store_true",
                    help="cosine-anneal the learning rate (off = paper's constant lr)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    enable_fast_math()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(QUICK if a.quick else FULL)
    if a.epochs is not None:
        cfg["epochs"] = a.epochs
    if a.lambda_d is not None:
        cfg["lambda_d"] = a.lambda_d
    if a.beta1 is not None:
        cfg["betas"] = (a.beta1, cfg["betas"][1])
    cfg["cosine"] = a.cosine

    print(f"serial PI-DeepONet operator correction (diffusion-reaction) on {device}")
    print(f"  N_p={cfg['M_train']} sensors={cfg['n_sensors']} N_u={cfg['N_u']} "
          f"N_f={cfg['N_f']} p={cfg['p']} width={cfg['width']} depth={cfg['depth']} "
          f"lr={cfg['lr']} epochs={cfg['epochs']} runs={a.n_runs}")
    print("  physics loss: || N0[ G_theta + G_psi ] - v ||^2  "
          "(N0 applied to both terms)\n")

    modes = ["known", "misspecified", "corrected"] if a.mode == "all" else [a.mode]
    results = {m: [] for m in modes}
    times = {m: 0.0 for m in modes}
    for run in range(a.n_runs):
        d = build_data(cfg, device, seed=run)
        for m in modes:
            t0 = time.time()
            results[m].append(train_mode(m, d, cfg, device, seed=run))
            times[m] += time.time() - t0

    print(f"\n==== relative L2 on {cfg['M_test']} test samples "
          f"({a.n_runs} run{'s' if a.n_runs > 1 else ''}) ====")
    print(f"{'model':<14}{'u':>22}{'v':>22}{'phi':>22}     time")
    for m in modes:
        cells = []
        for key in ("u", "v", "phi"):
            vals = np.array([r[key] for r in results[m]])
            cells.append(f"{vals.mean():.3e}" + (f" +- {vals.std():.1e}"
                                                 if a.n_runs > 1 else ""))
        print(f"{m:<14}" + "".join(f"{c:>22}" for c in cells) +
              f"   {times[m]:.0f}s")

    print("\npaper Table 2 (relative L2, u / v / phi):")
    for m in modes:
        pu, pv, pp = PAPER_TABLE2[m]
        print(f"  {m:<14}{pu:>10.2e}{pv:>12.2e}{pp:>12.2e}")


if __name__ == "__main__":
    main()
