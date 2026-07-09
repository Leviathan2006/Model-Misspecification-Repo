"""2D lid-driven cavity, non-Newtonian (power-law) fluid.

Source: Ma, Boulle, Yang, Wu & Guo (2026), "Physics-guided correction for
operator learning under model misspecification", arXiv:2606.03469, Sec. 4.3.

Learned operator (paper):  G : Re  ->  velocity field (u_x, u_y).

TRUE model used to generate the data (steady incompressible flow):
    power-law viscosity   nu = |S|^{n-1},  n = 1.5     (S = strain-rate tensor)
    Reynolds number       Re ~ U[100, 200]
    domain                Omega = [0, 1] x [0, 1],  grid 101 x 101
    lid (top) velocity    v(x) = 1 - cosh(10 (x - 1/2)) / cosh(5)
    other walls           no-slip
    solver                Lattice-Boltzmann (the paper's stated solver)

SOLVER: self-contained D2Q9 BGK Lattice-Boltzmann with a LOCAL, shear-dependent
relaxation time implementing the power-law rheology. The whole batch of Reynolds
numbers is advanced simultaneously (vectorised over a leading sample axis), so it
maps cleanly onto a GPU on Kaggle.

Non-dimensionalisation (documented, since the paper does not print the constant):
we fix a small lattice lid speed U0 and set the consistency so that
    nu0 = U0 * (N - 1) / Re           (reference kinematic viscosity)
    nu(x) = clip( nu0 * (|S| / S0)^{n-1} ),   S0 = U0 / (N - 1)
    tau(x) = 3 nu(x) + 1/2.
This reproduces the qualitative physics and the operator Re -> u; the absolute
Reynolds constant may differ from the paper by an O(1) factor.

STATUS: smoke-tested at coarse resolution / low iteration count. Full 101x101
runs to the paper's 1e-6 convergence are intended for Kaggle.
"""
import argparse
import os

import numpy as np

N_POWER = 1.5

# D2Q9 lattice (y points up)
_C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
               [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int64)
_W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
_CS2 = 1.0 / 3.0


def lid_profile(N):
    x = np.linspace(0.0, 1.0, N)
    return 1.0 - np.cosh(10.0 * (x - 0.5)) / np.cosh(5.0)


def _equilibrium(rho, ux, uy):
    """Return feq of shape (S, 9, Ny, Nx)."""
    cu = _C[:, 0][None, :, None, None] * ux[:, None] + \
        _C[:, 1][None, :, None, None] * uy[:, None]
    usq = ux**2 + uy**2
    feq = _W[None, :, None, None] * rho[:, None] * (
        1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usq[:, None]
    )
    return feq


def _strain_magnitude(fneq, rho, tau):
    """|S| = sqrt(2 S_ab S_ab), S_ab = -(3 / (2 rho tau)) sum_i c_ia c_ib fneq_i."""
    Sxx = np.einsum("i,siyx->syx", _C[:, 0] * _C[:, 0], fneq)
    Syy = np.einsum("i,siyx->syx", _C[:, 1] * _C[:, 1], fneq)
    Sxy = np.einsum("i,siyx->syx", _C[:, 0] * _C[:, 1], fneq)
    pref = -3.0 / (2.0 * rho * tau)
    Sxx, Syy, Sxy = pref * Sxx, pref * Syy, pref * Sxy
    return np.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2))


def solve_cavity(Re, N=101, U0=0.1, max_iter=60000, tol=1e-6, check_every=200):
    """Solve the batch of cavity flows. Re: (S,). Returns u of shape (S, N, N, 2)."""
    Re = np.atleast_1d(np.asarray(Re, dtype=np.float64))
    S = Re.shape[0]
    nu0 = (U0 * (N - 1) / Re).reshape(S, 1, 1)
    S0 = U0 / (N - 1)
    nu_min, nu_max = 1e-4, 0.2

    u_lid = lid_profile(N)                                  # (N,)
    rho = np.ones((S, N, N))
    ux = np.zeros((S, N, N))
    uy = np.zeros((S, N, N))
    ux[:, -1, :] = U0 * u_lid[None, :]                      # top row = lid
    f = _equilibrium(rho, ux, uy)
    tau = np.full((S, N, N), 0.6)

    prev = None
    for it in range(max_iter):
        rho = f.sum(axis=1)
        ux = np.einsum("i,siyx->syx", _C[:, 0], f) / rho
        uy = np.einsum("i,siyx->syx", _C[:, 1], f) / rho
        ux[:, -1, :] = U0 * u_lid[None, :]                  # enforce lid velocity
        uy[:, -1, :] = 0.0

        feq = _equilibrium(rho, ux, uy)

        # power-law rheology: local viscosity -> local relaxation time
        Smag = _strain_magnitude(f - feq, rho, tau)
        nu = np.clip(nu0 * (Smag / S0 + 1e-8) ** (N_POWER - 1.0), nu_min, nu_max)
        tau = 3.0 * nu + 0.5

        f = f - (f - feq) / tau[:, None]                    # BGK collision

        for i in range(9):                                  # streaming
            f[:, i] = np.roll(f[:, i], shift=(_C[i, 1], _C[i, 0]), axis=(1, 2))

        f = _apply_boundaries(f, rho, u_lid, U0)

        if (it + 1) % check_every == 0:
            speed = np.sqrt(ux**2 + uy**2)
            if prev is not None:
                num = np.abs(speed - prev).sum(axis=(1, 2))
                den = np.abs(speed).sum(axis=(1, 2)) + 1e-12
                if np.max(num / den) < tol:
                    break
            prev = speed

    out = np.stack([ux, uy], axis=-1)                       # (S, N, N, 2)
    return out


def _apply_boundaries(f, rho, u_lid, U0):
    """Half-way bounce-back on the three static walls and the moving lid."""
    rho_w = rho[:, -1, :]
    # bottom wall (y=0): incoming dirs 2,5,6
    f[:, 2, 0, :] = f[:, 4, 0, :]
    f[:, 5, 0, :] = f[:, 7, 0, :]
    f[:, 6, 0, :] = f[:, 8, 0, :]
    # left wall (x=0): incoming dirs 1,5,8
    f[:, 1, :, 0] = f[:, 3, :, 0]
    f[:, 5, :, 0] = f[:, 7, :, 0]
    f[:, 8, :, 0] = f[:, 6, :, 0]
    # right wall (x=Nx-1): incoming dirs 3,6,7
    f[:, 3, :, -1] = f[:, 1, :, -1]
    f[:, 6, :, -1] = f[:, 8, :, -1]
    f[:, 7, :, -1] = f[:, 5, :, -1]
    # top moving lid (y=Ny-1): incoming dirs 4,7,8 with momentum correction
    uw = U0 * u_lid[None, :]
    f[:, 4, -1, :] = f[:, 2, -1, :]
    f[:, 7, -1, :] = f[:, 5, -1, :] + 6.0 * _W[7] * rho_w * uw   # c7.u = -uw
    f[:, 8, -1, :] = f[:, 6, -1, :] - 6.0 * _W[8] * rho_w * uw   # c8.u = +uw
    return f


def generate(n_samples, N=101, seed=0, max_iter=60000, tol=1e-6, chunk=100):
    """Return (x, y, Re, u) with u of shape (n_samples, N, N, 2).

    Samples are solved in chunks so the vectorised LBM state stays within memory
    (a full 1000-sample batch at 101x101 would need many GB)."""
    rng = np.random.default_rng(seed)
    Re = rng.uniform(100.0, 200.0, size=n_samples)
    parts = []
    for i in range(0, n_samples, chunk):
        parts.append(solve_cavity(Re[i:i + chunk], N=N,
                                  max_iter=max_iter, tol=tol))
    u = np.concatenate(parts, axis=0)
    grid = np.linspace(0.0, 1.0, N)
    return grid, grid, Re, u


def get_dataset(data_dir="data", n_train=1000, n_test=100, N=101, seed=0,
                max_iter=60000, tol=1e-6, chunk=100):
    """Load cached data or generate + cache it.
    Returns (coords, Re_train, u_train, Re_test, u_test)."""
    path = os.path.join(data_dir, "cavity_flow.npz")
    if os.path.exists(path):
        d = np.load(path)
        return ((d["x"], d["y"]), d["Re_train"], d["u_train"],
                d["Re_test"], d["u_test"])
    x, y, Re_tr, u_tr = generate(n_train, N, seed, max_iter, tol, chunk)
    _, _, Re_te, u_te = generate(n_test, N, seed + 1, max_iter, tol, chunk)
    os.makedirs(data_dir, exist_ok=True)
    np.savez(path, x=x, y=y, Re_train=Re_tr, u_train=u_tr,
             Re_test=Re_te, u_test=u_te)
    return (x, y), Re_tr, u_tr, Re_te, u_te


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--N", type=int, default=101)
    p.add_argument("--max_iter", type=int, default=60000)
    p.add_argument("--data_dir", type=str, default="data")
    a = p.parse_args()
    (x, y), Re_tr, u_tr, Re_te, u_te = get_dataset(
        a.data_dir, a.n_train, a.n_test, a.N, max_iter=a.max_iter)
    print(f"cavity_flow: train u {u_tr.shape}, test u {u_te.shape}")
    spd = np.sqrt((u_tr**2).sum(-1))
    print(f"  Re in [{Re_tr.min():.1f}, {Re_tr.max():.1f}], "
          f"speed max {spd.max():.4f}, any NaN: {np.isnan(u_tr).any()}")
