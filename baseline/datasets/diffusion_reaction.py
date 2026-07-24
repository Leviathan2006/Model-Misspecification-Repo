"""1D steady diffusion-reaction benchmark.

Source: Ma, Boulle, Yang, Wu & Guo (2026), "Physics-guided correction for
operator learning under model misspecification", arXiv:2606.03469, Sec. 4.1.

TRUE model used to generate the data:

    D u_xx - k_r(u) * u = v,   k_r(u) = 0.5 * exp(-u),   D = 0.1,   x in [-1, 1]
    u(-1) = u(1) = 0

Data are built with the method of manufactured solutions: sample a smooth u that
already satisfies the boundary conditions, then DEFINE the source term v by
applying the true operator. No PDE solver is required. The learned operator maps
    v  -->  u.

Manufactured solution (paper Eq.):

    u(x) = (x^2 - 1)/10 * sum_{i=1..5} [ w_{2i-1} sin(i pi x) + w_{2i} cos(i pi x) ]
    w_j ~ U[0, 1]                       (10 coefficients per sample)

The prefactor (x^2 - 1)/10 vanishes at x = +-1, so the Dirichlet BCs hold for
every sample by construction.

Note on "misspecification": the paper's misspecified variant (which drops the
-k_r u term) only enters their physics-informed DeepONet *loss*. A data-driven
FNO never sees a governing equation, so it simply learns the true v -> u map.
This script therefore produces the faithful (v, u) pairs; misspecification is a
training-time concept handled elsewhere, not in the data.
"""
import argparse
import os

import numpy as np

from ._cache import load_cache

D = 0.1
N_MODES = 5


def manufactured(x, w):
    """Return (u, u_xx) for coefficients w.

    x: (N,) grid.       w: (M, 2*N_MODES) uniform coefficients.
    """
    i = np.arange(1, N_MODES + 1)                       # (5,)
    ipix = np.pi * np.outer(x, i)                       # (N, 5)
    sin, cos = np.sin(ipix), np.cos(ipix)               # (N, 5)
    a = w[:, 0::2][:, None, :]                           # (M,1,5) sin coeffs
    b = w[:, 1::2][:, None, :]                           # (M,1,5) cos coeffs
    sin, cos = sin[None], cos[None]                      # (1,N,5)
    iw = i * np.pi                                       # angular wavenumbers

    # S and its x-derivatives, summed over the 5 Fourier modes (broadcasting)
    S = (a * sin + b * cos).sum(-1)                      # (M, N)
    Sp = (a * iw * cos - b * iw * sin).sum(-1)
    Spp = -((a * iw**2 * sin + b * iw**2 * cos).sum(-1))

    g = (x**2 - 1.0) / 10.0
    gp = x / 5.0
    gpp = np.full_like(x, 1.0 / 5.0)

    u = g * S
    uxx = gpp * S + 2.0 * gp * Sp + g * Spp
    return u, uxx


def generate(n_samples, n_x=201, seed=0):
    """Return (x, v, u) with v, u of shape (n_samples, n_x)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n_x)
    w = rng.uniform(0.0, 1.0, size=(n_samples, 2 * N_MODES))
    u, uxx = manufactured(x, w)
    kr = 0.5 * np.exp(-u)
    v = D * uxx - kr * u                                 # true operator applied
    return x, v, u


def get_dataset(data_dir="data", n_train=1000, n_test=100, n_x=201, seed=0):
    """Load cached data or generate + cache it. Returns (x, Xtr, Ytr, Xte, Yte)."""
    path = os.path.join(data_dir, "diffusion_reaction.npz")
    d = load_cache(path, n_train, n_test, [("v_train", 1, n_x)])
    if d is not None:
        return d["x"], d["v_train"], d["u_train"], d["v_test"], d["u_test"]
    x, v_tr, u_tr = generate(n_train, n_x, seed)
    _, v_te, u_te = generate(n_test, n_x, seed + 1)
    os.makedirs(data_dir, exist_ok=True)
    np.savez(path, x=x, v_train=v_tr, u_train=u_tr, v_test=v_te, u_test=u_te)
    return x, v_tr, u_tr, v_te, u_te


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--n_x", type=int, default=201)
    p.add_argument("--data_dir", type=str, default="data")
    a = p.parse_args()
    x, v_tr, u_tr, v_te, u_te = get_dataset(a.data_dir, a.n_train, a.n_test, a.n_x)
    print(f"diffusion_reaction: train v/u {v_tr.shape}, test v/u {v_te.shape}")
    print(f"  |v| range [{v_tr.min():.3f}, {v_tr.max():.3f}], "
          f"|u| range [{u_tr.min():.3f}, {u_tr.max():.3f}]")
