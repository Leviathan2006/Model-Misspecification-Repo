"""1D viscous Burgers benchmark.

Source: Ma, Boulle, Yang, Wu & Guo (2026), "Physics-guided correction for
operator learning under model misspecification", arXiv:2606.03469, Sec. 4.2.

TRUE model used to generate the data:

    u_t + u u_x = nu u_xx,   nu = 0.01,   (x, t) in (0, 1) x (0, 1],   periodic in x

Initial conditions are drawn from the Gaussian random field

    u0 ~ N(0, 25^2 (-Laplacian + 5^2 I)^{-4})

and the PDE is integrated with a Fourier pseudo-spectral discretisation in space
and an explicit forward-Euler scheme in time (paper's stated solver). The stored
fields live on a 101 (time) x 201 (space) grid.

The learned operator here maps  u0(x) -> u(x, T=1)  (the standard vanilla-FNO
Burgers formulation). The full space-time field is available via --save_full.

Note on "misspecification": the paper studies three misspecified variants
(extra cubic term, advection dropped, diffusion dropped). Those only change the
DeepONet physics *loss*; the underlying data always come from the true viscous
Burgers equation, which is what this script generates. A data-driven FNO learns
that true operator directly.
"""
import argparse
import os

import numpy as np

from ._cache import load_cache

NU = 0.01


def grf(n, N, sigma=25.0, tau=5.0, gamma=4.0, rng=None, scale=1.0):
    """Sample n periodic Gaussian random fields on N points.

    Covariance sigma^2 (-Laplacian + tau^2 I)^{-gamma}, i.e. the Karhunen-Loeve
    expansion u(x) = sum_k sigma ((2 pi k)^2 + tau^2)^{-gamma/2} (a_k cos + b_k sin)
    on [0, 1] with periodic boundary conditions. Real fields via rfft.

    AMPLITUDE CAVEAT -- UNRESOLVED. With the paper's printed parameters
    (sigma=25, tau=5, gamma=4) this gives std(u0) ~ 0.012 and |u| <~ 0.036 over
    the whole solution field. The paper's Fig. 7 colourbars for the Case-A/B/C
    correction targets are inconsistent with that, and inconsistent with each
    other under any single rescaling of u0:

        target            paper   scale needed   (scaling in u)
        A: eps u^3 (eps=10)  0.48     ~10          cubic
        B: u u_x             1.22      ~8          quadratic
        C: nu u_xx           0.34      ~0.6        linear

    A and B agree on an initial condition roughly 8-10x larger than the formula
    yields, while C -- the only one linear in u -- wants one slightly *smaller*.
    So the mismatch is not a pure amplitude factor: the paper's fields must also
    carry different spectral content (C is sensitive to curvature, A and B to
    amplitude). The initial-condition distribution therefore cannot be recovered
    from the text as printed, and Cases A-C cannot all be reproduced at once
    until that is pinned down.

    The KL sampling below is the correct discretisation of the stated
    covariance, so it stays the default (scale=1.0) and `scale` is exposed as an
    explicit knob rather than silently fudging the covariance to fit one figure.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    kmax = N // 2 + 1
    k = np.arange(kmax)
    lam = (2.0 * np.pi * k) ** 2
    std = sigma * (lam + tau**2) ** (-gamma / 2.0)
    std[0] = 0.0                                         # zero mean
    re = rng.standard_normal((n, kmax))
    im = rng.standard_normal((n, kmax))
    coeff = std[None, :] * (re + 1j * im)
    coeff[:, 0] = 0.0
    if N % 2 == 0:
        coeff[:, -1] = coeff[:, -1].real
    return np.fft.irfft(coeff, n=N, axis=1) * N * scale


def solve_burgers(u0, nu=NU, T=1.0, dt=2.5e-4, n_save=101):
    """Integrate viscous Burgers for a batch of initial conditions.

    Returns field of shape (n, n_save, N). Pseudo-spectral space, forward Euler
    time, with 2/3-rule dealiasing on the quadratic nonlinearity.
    """
    n, N = u0.shape
    kmax = N // 2 + 1
    k = 2.0 * np.pi * np.arange(kmax)
    ik = 1j * k
    k2 = k**2
    mask = np.ones(kmax)
    mask[int(kmax * 2 / 3):] = 0.0                       # dealiasing

    nt = int(round(T / dt))
    save_every = nt // (n_save - 1)

    u = u0.copy()
    out = np.empty((n, n_save, N))
    out[:, 0] = u
    s = 1
    for step in range(1, nt + 1):
        uh = np.fft.rfft(u, axis=1)
        ux = np.fft.irfft(ik[None, :] * uh, n=N, axis=1)
        uxx = np.fft.irfft(-k2[None, :] * uh, n=N, axis=1)
        nh = np.fft.rfft(u * ux, axis=1) * mask[None, :]
        nonlin = np.fft.irfft(nh, n=N, axis=1)
        u = u + dt * (-nonlin + nu * uxx)
        if step % save_every == 0 and s < n_save:
            out[:, s] = u
            s += 1
    return out


def generate(n_samples, n_x=201, seed=0, u0_scale=1.0):
    """Return (x, u0, uT, field). field is the full (n, 101, n_x) space-time."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_x, endpoint=False)
    u0 = grf(n_samples, n_x, rng=rng, scale=u0_scale)
    field = solve_burgers(u0)
    if not np.all(np.isfinite(field)):
        raise RuntimeError("burgers: solver produced non-finite values -- "
                           "forward Euler went unstable, reduce dt or u0_scale")
    return x, field[:, 0, :].copy(), field[:, -1, :].copy(), field


def get_dataset(data_dir="data", n_train=2000, n_test=100, n_x=201, seed=0,
                u0_scale=1.0):
    """Load cached data or generate + cache it. Returns
    (t, x, u0_train, field_train, u0_test, field_test), where the learned
    operator maps the initial condition u0(x) to the full field u(x, t)."""
    path = os.path.join(data_dir, "burgers.npz")
    d = load_cache(path, n_train, n_test, [("field_train", 2, n_x)])
    if d is not None:
        return (d["t"], d["x"], d["u0_train"], d["field_train"],
                d["u0_test"], d["field_test"])
    x, u0_tr, _, fld_tr = generate(n_train, n_x, seed, u0_scale)
    _, u0_te, _, fld_te = generate(n_test, n_x, seed + 1, u0_scale)
    t = np.linspace(0.0, 1.0, fld_tr.shape[1])
    os.makedirs(data_dir, exist_ok=True)
    np.savez(path, t=t, x=x, u0_train=u0_tr, field_train=fld_tr,
             u0_test=u0_te, field_test=fld_te)
    return t, x, u0_tr, fld_tr, u0_te, fld_te


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--n_x", type=int, default=201)
    p.add_argument("--u0_scale", type=float, default=1.0,
                   help="initial-condition amplitude; 1.0 = the paper's printed "
                        "GRF. Its Fig. 7 implies 8-10 for Cases A/B but ~0.6 "
                        "for Case C -- see the caveat in grf()")
    p.add_argument("--data_dir", type=str, default="data")
    a = p.parse_args()
    t, x, u0_tr, fld_tr, u0_te, fld_te = get_dataset(
        a.data_dir, a.n_train, a.n_test, a.n_x, u0_scale=a.u0_scale)
    print(f"burgers: train u0 {u0_tr.shape}, field {fld_tr.shape}; "
          f"test field {fld_te.shape}")
    print(f"  |u0| range [{u0_tr.min():.3f}, {u0_tr.max():.3f}], "
          f"|u| range [{fld_tr.min():.3f}, {fld_tr.max():.3f}], "
          f"any NaN: {np.isnan(fld_tr).any()}")
