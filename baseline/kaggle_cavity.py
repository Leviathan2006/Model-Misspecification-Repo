# 2d cavity-flow dataset (paper Sec. 3.3) -- fully self-contained.
# Paste into a Kaggle cell and run. Only needs numpy. Writes to /kaggle/working/.
# Runtime: HOURS (D2Q9 Lattice-Boltzmann, up to 5e5 iterations per sample).
import numpy as np, time

N_POWER = 1.5
_C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
               [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int64)
_W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
_CS2 = 1.0 / 3.0

def lid_profile(N):
    x = np.linspace(0.0, 1.0, N)
    return 1.0 - np.cosh(10.0 * (x - 0.5)) / np.cosh(5.0)

def _equilibrium(rho, ux, uy):
    cu = _C[:, 0][None, :, None, None] * ux[:, None] + \
        _C[:, 1][None, :, None, None] * uy[:, None]
    usq = ux**2 + uy**2
    return _W[None, :, None, None] * rho[:, None] * (
        1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usq[:, None])

def _strain_magnitude(fneq, rho, tau):
    Sxx = np.einsum("i,siyx->syx", _C[:, 0] * _C[:, 0], fneq)
    Syy = np.einsum("i,siyx->syx", _C[:, 1] * _C[:, 1], fneq)
    Sxy = np.einsum("i,siyx->syx", _C[:, 0] * _C[:, 1], fneq)
    pref = -3.0 / (2.0 * rho * tau)
    Sxx, Syy, Sxy = pref * Sxx, pref * Syy, pref * Sxy
    return np.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2))

def _apply_boundaries(f, rho, u_lid, U0):
    rho_w = rho[:, -1, :]
    f[:, 2, 0, :] = f[:, 4, 0, :]
    f[:, 5, 0, :] = f[:, 7, 0, :]
    f[:, 6, 0, :] = f[:, 8, 0, :]
    f[:, 1, :, 0] = f[:, 3, :, 0]
    f[:, 5, :, 0] = f[:, 7, :, 0]
    f[:, 8, :, 0] = f[:, 6, :, 0]
    f[:, 3, :, -1] = f[:, 1, :, -1]
    f[:, 6, :, -1] = f[:, 8, :, -1]
    f[:, 7, :, -1] = f[:, 5, :, -1]
    uw = U0 * u_lid[None, :]
    f[:, 4, -1, :] = f[:, 2, -1, :]
    f[:, 7, -1, :] = f[:, 5, -1, :] + 6.0 * _W[7] * rho_w * uw
    f[:, 8, -1, :] = f[:, 6, -1, :] - 6.0 * _W[8] * rho_w * uw
    return f

def solve_cavity(Re, N=101, U0=0.1, max_iter=500000, tol=1e-6, check_every=500):
    Re = np.atleast_1d(np.asarray(Re, dtype=np.float64))
    S = Re.shape[0]
    nu0 = (U0 * (N - 1) / Re).reshape(S, 1, 1)
    S0 = U0 / (N - 1)
    nu_min, nu_max = 1e-4, 0.2
    u_lid = lid_profile(N)
    rho = np.ones((S, N, N))
    ux = np.zeros((S, N, N)); uy = np.zeros((S, N, N))
    ux[:, -1, :] = U0 * u_lid[None, :]
    f = _equilibrium(rho, ux, uy)
    tau = np.full((S, N, N), 0.6)
    prev = None; E = np.full(S, np.inf)
    for it in range(max_iter):
        rho = f.sum(axis=1)
        ux = np.einsum("i,siyx->syx", _C[:, 0], f) / rho
        uy = np.einsum("i,siyx->syx", _C[:, 1], f) / rho
        ux[:, -1, :] = U0 * u_lid[None, :]; uy[:, -1, :] = 0.0
        feq = _equilibrium(rho, ux, uy)
        Smag = _strain_magnitude(f - feq, rho, tau)
        nu = np.clip(nu0 * (Smag / S0 + 1e-8) ** (N_POWER - 1.0), nu_min, nu_max)
        tau = 3.0 * nu + 0.5
        f = f - (f - feq) / tau[:, None]
        for i in range(9):
            f[:, i] = np.roll(f[:, i], shift=(_C[i, 1], _C[i, 0]), axis=(1, 2))
        f = _apply_boundaries(f, rho, u_lid, U0)
        if (it + 1) % check_every == 0:
            vel = np.stack([ux, uy], axis=-1)
            if prev is not None:
                num = np.abs(vel - prev).sum(axis=(1, 2, 3))
                den = np.abs(vel).sum(axis=(1, 2, 3)) + 1e-12
                E = num / den
                if np.max(E) < tol:
                    break
            prev = vel
    out = np.stack([ux, uy], axis=-1)
    p = _CS2 * rho
    p = p - p.mean(axis=(1, 2), keepdims=True)
    out[:, 0, :, :] = 0.0; out[:, :, 0, :] = 0.0; out[:, :, -1, :] = 0.0
    out[:, -1, :, 0] = U0 * u_lid[None, :]; out[:, -1, :, 1] = 0.0
    if not np.all(E < tol):
        print(f"WARNING: {int((E >= tol).sum())}/{S} sample(s) not converged "
              f"(worst E={np.max(E):.2e})")
    return out, p

def generate(n_samples, N=101, seed=0, chunk=50):
    rng = np.random.default_rng(seed)
    Re = rng.uniform(100.0, 200.0, size=n_samples)
    us, ps = [], []
    for i in range(0, n_samples, chunk):
        u_c, p_c = solve_cavity(Re[i:i + chunk], N=N)
        us.append(u_c); ps.append(p_c)
        print(f"  cavity {min(i+chunk, n_samples)}/{n_samples}", flush=True)
    grid = np.linspace(0.0, 1.0, N)
    return grid, grid, Re, np.concatenate(us, 0), np.concatenate(ps, 0)

t0 = time.time()
x, y, Re_tr, u_tr, p_tr = generate(1000, N=101, seed=0, chunk=50)
_, _, Re_te, u_te, p_te = generate(100, N=101, seed=1, chunk=50)
np.savez("/kaggle/working/cavity_flow.npz", x=x, y=y,
         Re_train=Re_tr, u_train=u_tr, p_train=p_tr,
         Re_test=Re_te, u_test=u_te, p_test=p_te)
print(f"cavity_flow: u_train {u_tr.shape} p_train {p_tr.shape}  {time.time()-t0:.0f}s")
