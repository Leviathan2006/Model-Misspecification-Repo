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

NU = 0.01


def grf(n, N, sigma=25.0, tau=5.0, gamma=4.0, rng=None):
    """Sample n periodic Gaussian random fields on N points.

    Covariance sigma^2 (-Laplacian + tau^2 I)^{-gamma}. Real fields via rfft.
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
    return np.fft.irfft(coeff, n=N, axis=1) * N


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


def generate(n_samples, n_x=201, seed=0):
    """Return (x, u0, uT, field). field is the full (n, 101, n_x) space-time."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_x, endpoint=False)
    u0 = grf(n_samples, n_x, rng=rng)
    field = solve_burgers(u0)
    return x, field[:, 0, :].copy(), field[:, -1, :].copy(), field


def get_dataset(data_dir="data", n_train=2000, n_test=100, n_x=201, seed=0):
    """Load cached data or generate + cache it. Returns
    (t, x, u0_train, field_train, u0_test, field_test), where the learned
    operator maps the initial condition u0(x) to the full field u(x, t)."""
    path = os.path.join(data_dir, "burgers.npz")
    if os.path.exists(path):
        d = np.load(path)
        return (d["t"], d["x"], d["u0_train"], d["field_train"],
                d["u0_test"], d["field_test"])
    x, u0_tr, _, fld_tr = generate(n_train, n_x, seed)
    _, u0_te, _, fld_te = generate(n_test, n_x, seed + 1)
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
    p.add_argument("--data_dir", type=str, default="data")
    a = p.parse_args()
    t, x, u0_tr, fld_tr, u0_te, fld_te = get_dataset(
        a.data_dir, a.n_train, a.n_test, a.n_x)
    print(f"burgers: train u0 {u0_tr.shape}, field {fld_tr.shape}; "
          f"test field {fld_te.shape}")
    print(f"  |u0| range [{u0_tr.min():.3f}, {u0_tr.max():.3f}], "
          f"|u| range [{fld_tr.min():.3f}, {fld_tr.max():.3f}], "
          f"any NaN: {np.isnan(fld_tr).any()}")
