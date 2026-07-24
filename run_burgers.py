"""Physics-guided operator correction on the 1d Burgers benchmark (paper Sec. 3.2).

    true system   N[u] = u_t + u u_x - nu u_xx = 0,  nu = 0.01
                  (x, t) in (0,1) x (0,1],  periodic in x,  u(x,0) = v(x)

The learned operator is G : v |-> u(x,t) over the whole space-time domain, and
three misspecified prior operators N0 are studied:

    Case A  extra cubic term      N0[w] = w_t + w w_x + eps w^3 - nu w_xx  (eps=10)
    Case B  advection omitted     N0[w] = w_t - nu w_xx
    Case C  diffusion omitted     N0[w] = w_t + w w_x

Serial architecture (paper Sec. 2.2)
    prior       G_theta : v(x_1..x_101)                       -> u_theta(x,t)
    correction  G_psi   : [v(x_1..x_101), u_theta(51 x 51 grid)] -> c(x,t)

Physics loss, with the same modification as run_correction.py -- N0 applied to
*both* of the first two terms, i.e. to their sum (here the source term is zero,
since for Burgers the input function v is the initial condition, not a forcing):

    L_phys = || N0[ G_theta(v)(y_f) + G_psi(v,u_theta)(y_f) ] ||^2

    L = L_phys + lambda_bc L_bc + lambda_ic L_ic + lambda_u L_u
    lambda_bc = 1,  lambda_ic = lambda_u = 50            (paper Sec. 3.2)

CORRECTION TARGET. The paper defines phi as the term the misspecified operator
gets wrong, phi = N[u] - N0[u] at the reference solution, giving -eps u^3, u u_x
and -nu u_xx for cases A, B, C. (The paper prints "+nu u_xx" for Case C, which
is a sign slip relative to its own convention in A and B -- the correction has to
*restore* the diffusion N0 dropped, and N - N0 = -nu u_xx.) The model's
counterpart is the contribution the correction makes to the residual,
phi_pred = N0[u_theta + c] - N0[u_theta].

AMPLITUDE CAVEAT. The initial-condition amplitude in datasets/burgers.grf cannot
be reconciled with the paper's Fig. 7 (Cases A/B want ~8-10x the printed GRF,
Case C wants ~0.6x). The default here is the printed formula; --u0_scale is
passed straight through. Case A/B/C errors therefore should NOT be expected to
match Table 3 until that is resolved -- see the caveat in datasets/burgers.py.
"""
import argparse
import time

import numpy as np
import torch

import datasets.burgers as bg
from deeponet import DeepONet, enable_fast_math, trunk_jet

NU = bg.NU                       # 0.01
EPS_A = 10.0                     # Case A cubic coefficient
LAM_BC, LAM_IC, LAM_U = 1.0, 50.0, 50.0
BETAS = (0.999, 0.999)           # paper Sec. 3

CASES = ("A", "B", "C")


# --------------------------------------------------------------------------- #
# physics
# --------------------------------------------------------------------------- #
def N_true(w, w_t, w_x, w_xx):
    return w_t + w * w_x - NU * w_xx


def N0(case, w, w_t, w_x, w_xx):
    """Misspecified prior operator for Case A / B / C."""
    if case == "A":
        return w_t + w * w_x + EPS_A * w ** 3 - NU * w_xx
    if case == "B":
        return w_t - NU * w_xx
    if case == "C":
        return w_t + w * w_x
    raise ValueError(case)


def phi_reference(case, u, u_x, u_xx):
    """phi = N[u] - N0[u] at the reference solution."""
    if case == "A":
        return -EPS_A * u ** 3
    if case == "B":
        return u * u_x
    if case == "C":
        return -NU * u_xx
    raise ValueError(case)


def rel_l2(pred, true):
    """Mean over samples of ||pred - true||_2 / ||true||_2, flattening the rest."""
    p = pred.reshape(pred.shape[0], -1)
    t = true.reshape(true.shape[0], -1)
    return (torch.linalg.norm(p - t, dim=1) /
            torch.linalg.norm(t, dim=1)).mean().item()


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_data(cfg, device, seed=0):
    """Paper Sec. 3.2: 2000/100 samples on a 101(t) x 201(x) grid, 101 equally
    spaced sensors for v, a 101 x 101 collocation grid, N_u solution
    observations, 101 initial-condition points and 100 temporal points for the
    periodic boundary constraint."""
    t_g, x_g, u0_tr, fld_tr, u0_te, fld_te = bg.get_dataset(
        cfg["data_dir"], cfg["M_train"], cfg["M_test"], cfg["n_x"],
        u0_scale=cfg["u0_scale"])
    rng = np.random.default_rng(seed)

    s_step = (x_g.size - 1) // (cfg["n_sensors"] - 1) if cfg["n_sensors"] > 1 else 1
    s_idx = np.arange(0, x_g.size, max(1, x_g.size // cfg["n_sensors"]))[:cfg["n_sensors"]]
    x_s = x_g[s_idx]                                        # 101 sensor locations
    vs_tr, vs_te = u0_tr[:, s_idx], u0_te[:, s_idx]

    # collocation: uniform 101 x 101 over [0,1) x (0,1]
    nf = cfg["n_coll"]
    xc = np.linspace(0.0, 1.0, nf, endpoint=False)
    tc = np.linspace(0.0, 1.0, nf + 1)[1:]
    Xc, Tc = np.meshgrid(xc, tc, indexing="ij")
    y_coll = np.stack([Xc.ravel(), Tc.ravel()], axis=1)

    # correction-branch sampling grid (fixed 51 x 51, paper Sec. 3.2)
    xg2 = np.linspace(0.0, 1.0, cfg["n_cgrid"], endpoint=False)
    tg2 = np.linspace(0.0, 1.0, cfg["n_cgrid"])
    Xg2, Tg2 = np.meshgrid(xg2, tg2, indexing="ij")
    y_cgrid = np.stack([Xg2.ravel(), Tg2.ravel()], axis=1)

    # N_u solution observations at random points of the stored data grid
    oi = rng.integers(0, t_g.size, cfg["N_u"])
    oj = rng.integers(0, x_g.size, cfg["N_u"])
    y_obs = np.stack([x_g[oj], t_g[oi]], axis=1)
    u_obs_tr = fld_tr[:, oi, oj]

    # initial condition and periodic boundary points
    y_ic = np.stack([x_s, np.zeros_like(x_s)], axis=1)
    t_bc = np.linspace(0.0, 1.0, cfg["n_bc"])
    y_bc0 = np.stack([np.zeros_like(t_bc), t_bc], axis=1)
    y_bc1 = np.stack([np.ones_like(t_bc), t_bc], axis=1)

    # full test grid + reference phi, computed spectrally from the stored field
    Xt, Tt = np.meshgrid(x_g, t_g, indexing="ij")
    y_test = np.stack([Xt.ravel(), Tt.ravel()], axis=1)
    k = 2 * np.pi * np.arange(x_g.size // 2 + 1)
    fh = np.fft.rfft(fld_te, axis=2)
    ux_te = np.fft.irfft(1j * k * fh, n=x_g.size, axis=2)
    uxx_te = np.fft.irfft(-k ** 2 * fh, n=x_g.size, axis=2)
    # stored field is (n, n_t, n_x); the test grid is ordered (x, t)
    tr = lambda a: np.ascontiguousarray(a.transpose(0, 2, 1)).reshape(a.shape[0], -1)

    mu, sd = vs_tr.mean(), vs_tr.std()

    def T(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)

    return {
        "vs_tr": T((vs_tr - mu) / sd), "vs_te": T((vs_te - mu) / sd),
        "v_ic_tr": T(vs_tr), "v_ic_te": T(vs_te),
        "u_obs_tr": T(u_obs_tr),
        "y_coll": T(y_coll), "y_cgrid": T(y_cgrid), "y_obs": T(y_obs),
        "y_ic": T(y_ic), "y_bc0": T(y_bc0), "y_bc1": T(y_bc1),
        "y_test": T(y_test),
        "u_test": T(tr(fld_te)), "ux_test": T(tr(ux_te)), "uxx_test": T(tr(uxx_te)),
        "grid": (x_g, t_g),
    }


# --------------------------------------------------------------------------- #
# network evaluation
# --------------------------------------------------------------------------- #
def _fields(net, b, y):
    """Value and the derivatives the Burgers operators need: w, w_t, w_x, w_xx.

    Query columns are (x, t), so coordinate 0 is x and coordinate 1 is t. One
    nested jvp on (0,0) gives the basis, d/dx and d^2/dx^2; a second on
    (1, None) gives d/dt."""
    t, d1, d2 = trunk_jet(net.trunk, y, [(0, 0), (1, None)])
    w = net.combine(b, t) + net.bias[0]
    return w, net.combine(b, d1[1]), net.combine(b, d1[0]), net.combine(b, d2[(0, 0)])


def _value(net, b, y):
    return net.combine(b, net.trunk_out(y)) + net.bias[0]


def _corr_branch_in(prior, bp, vs, y_cgrid):
    """G_psi branch input: v at the sensors, plus the prior prediction sampled on
    the fixed 51 x 51 spatio-temporal grid (paper Sec. 3.2)."""
    return torch.cat([vs, _value(prior, bp, y_cgrid)], dim=1)


# --------------------------------------------------------------------------- #
# training / evaluation
# --------------------------------------------------------------------------- #
def train_case(mode, case, d, cfg, device, seed=0, verbose=True):
    torch.manual_seed(seed)
    m = cfg["n_sensors"]
    prior = DeepONet(m, 2, cfg["p"], cfg["width"], cfg["depth"]).to(device)
    corr = DeepONet(m + cfg["n_cgrid"] ** 2, 2, cfg["p"], cfg["width"],
                    cfg["depth"]).to(device)

    params = list(prior.parameters())
    if mode == "corrected":
        params += list(corr.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"], betas=cfg["betas"])

    M, bs = d["vs_tr"].shape[0], cfg["batch"]
    g = torch.Generator(device="cpu").manual_seed(seed)

    for epoch in range(cfg["epochs"]):
        idx = torch.randperm(M, generator=g)[:bs].to(device)
        vs, v_ic, u_obs = d["vs_tr"][idx], d["v_ic_tr"][idx], d["u_obs_tr"][idx]
        opt.zero_grad(set_to_none=True)

        bp = prior.branch_out(vs)
        u, u_t, u_x, u_xx = _fields(prior, bp, d["y_coll"])

        if mode == "known":
            res = N_true(u, u_t, u_x, u_xx)
        elif mode == "misspecified":
            res = N0(case, u, u_t, u_x, u_xx)
        else:  # corrected -- N0 applied to the sum G_theta + G_psi
            bc = corr.branch_out(_corr_branch_in(prior, bp, vs, d["y_cgrid"]))
            c, c_t, c_x, c_xx = _fields(corr, bc, d["y_coll"])
            res = N0(case, u + c, u_t + c_t, u_x + c_x, u_xx + c_xx)

        l_phys = (res ** 2).mean()
        l_ic = ((_value(prior, bp, d["y_ic"]) - v_ic) ** 2).mean()
        l_u = ((_value(prior, bp, d["y_obs"]) - u_obs) ** 2).mean()
        # periodic boundary: match the field and its x-derivative at x=0 and x=1
        b0, _, b0x, _ = _fields(prior, bp, d["y_bc0"])
        b1, _, b1x, _ = _fields(prior, bp, d["y_bc1"])
        l_bc = ((b0 - b1) ** 2).mean() + ((b0x - b1x) ** 2).mean()

        loss = l_phys + LAM_BC * l_bc + LAM_IC * l_ic + LAM_U * l_u
        loss.backward(inputs=params)
        opt.step()
        if verbose and ((epoch + 1) % max(1, cfg["epochs"] // 10) == 0 or epoch == 0):
            print(f"  [{mode} {case}] epoch {epoch+1:7d}  loss {loss.item():.3e} "
                  f"(phys {l_phys.item():.2e} ic {l_ic.item():.2e} "
                  f"u {l_u.item():.2e} bc {l_bc.item():.2e})")

    prior.eval()
    corr.eval()
    return evaluate(mode, case, prior, corr, d, cfg)


def evaluate(mode, case, prior, corr, d, cfg):
    """Relative L2 of u and of the correction term phi on the test set."""
    outs_u, outs_phi = [], []
    n = d["vs_te"].shape[0]
    for i in range(0, n, cfg["eval_batch"]):
        vs = d["vs_te"][i:i + cfg["eval_batch"]]
        with torch.enable_grad():
            bp = prior.branch_out(vs)
            u, u_t, u_x, u_xx = _fields(prior, bp, d["y_test"])
            if mode == "corrected":
                bc = corr.branch_out(_corr_branch_in(prior, bp, vs, d["y_cgrid"]))
                c, c_t, c_x, c_xx = _fields(corr, bc, d["y_test"])
                phi = (N0(case, u + c, u_t + c_t, u_x + c_x, u_xx + c_xx)
                       - N0(case, u, u_t, u_x, u_xx))
            else:
                # no correction network: the residual the model leaves behind
                phi = -N0(case, u, u_t, u_x, u_xx)
        outs_u.append(u.detach())
        outs_phi.append(phi.detach())
    u = torch.cat(outs_u)
    phi = torch.cat(outs_phi)
    phi_ref = phi_reference(case, d["u_test"], d["ux_test"], d["uxx_test"])
    return {"u": rel_l2(u, d["u_test"]), "phi": rel_l2(phi, phi_ref)}


# --------------------------------------------------------------------------- #
# configs / driver
# --------------------------------------------------------------------------- #
FULL = dict(M_train=2000, M_test=100, n_x=201, n_sensors=101, n_coll=101,
            n_cgrid=51, N_u=1000, n_bc=100, p=100, width=128, depth=4,
            lr=1e-4, epochs=400000, batch=50, eval_batch=25, betas=BETAS,
            u0_scale=1.0, data_dir="data")
QUICK = dict(M_train=64, M_test=16, n_x=201, n_sensors=101, n_coll=21,
             n_cgrid=11, N_u=200, n_bc=20, p=100, width=128, depth=4,
             lr=1e-4, epochs=100, batch=16, eval_batch=8, betas=BETAS,
             u0_scale=1.0, data_dir="data/quick")  # never clobber the real cache

PAPER_TABLE3 = {  # u error (%), phi relative L2
    ("A", "misspecified"): (14.23, 0.67), ("A", "corrected"): (0.99, 0.25),
    ("B", "misspecified"): (7.98, 1.47), ("B", "corrected"): (1.01, 0.23),
    ("C", "misspecified"): (6.64, 1.13), ("C", "corrected"): (0.95, 0.19),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--mode", choices=["known", "misspecified", "corrected", "all"],
                    default="all")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--beta1", type=float, default=None)
    ap.add_argument("--u0_scale", type=float, default=None,
                    help="initial-condition amplitude (see datasets/burgers.grf)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    enable_fast_math()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(QUICK if a.quick else FULL)
    for key, val in (("epochs", a.epochs), ("batch", a.batch),
                     ("u0_scale", a.u0_scale)):
        if val is not None:
            cfg[key] = val
    if a.beta1 is not None:
        cfg["betas"] = (a.beta1, cfg["betas"][1])

    print(f"serial PI-DeepONet operator correction (1d Burgers) on {device}")
    print(f"  N_p={cfg['M_train']} sensors={cfg['n_sensors']} "
          f"collocation={cfg['n_coll']}x{cfg['n_coll']} N_u={cfg['N_u']} "
          f"width={cfg['width']} lr={cfg['lr']} epochs={cfg['epochs']} "
          f"batch={cfg['batch']} u0_scale={cfg['u0_scale']}")
    print("  physics loss: || N0[ G_theta + G_psi ] ||^2  (N0 applied to both terms)\n")

    d = build_data(cfg, device)
    cases = list(CASES) if a.case == "all" else [a.case]
    modes = (["known", "misspecified", "corrected"] if a.mode == "all" else [a.mode])

    rows = []
    if "known" in modes:                       # case-independent reference floor
        t0 = time.time()
        r = train_case("known", "A", d, cfg, device)
        rows.append(("-", "known", r, time.time() - t0))
    for case in cases:
        for mode in [m for m in modes if m != "known"]:
            t0 = time.time()
            r = train_case(mode, case, d, cfg, device)
            rows.append((case, mode, r, time.time() - t0))

    print(f"\n==== relative L2 on {cfg['M_test']} test samples ====")
    print(f"{'case':<6}{'model':<14}{'u':>13}{'u (%)':>10}{'phi':>13}     time")
    for case, mode, r, dt in rows:
        print(f"{case:<6}{mode:<14}{r['u']:>13.4e}{100*r['u']:>10.2f}"
              f"{r['phi']:>13.4e}   {dt:.0f}s")

    print("\npaper Table 3 (u error %, phi relative L2) -- see the amplitude "
          "caveat above:")
    for (case, mode), (pu, pp) in PAPER_TABLE3.items():
        if case in cases and mode in modes:
            print(f"  {case} {mode:<14}{pu:>8.2f}%{pp:>10.2f}")
    print("  known model            0.79%         -")


if __name__ == "__main__":
    main()
