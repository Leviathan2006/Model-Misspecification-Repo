# 1d Burgers dataset (paper Sec. 3.2) -- self-contained. Only needs numpy.
import numpy as np, time

NU = 0.01

def grf(n, N, sigma=25.0, tau=5.0, gamma=4.0, rng=None, scale=1.0):
    if rng is None:
        rng = np.random.default_rng(0)
    kmax = N // 2 + 1
    k = np.arange(kmax)
    lam = (2.0 * np.pi * k) ** 2
    std = sigma * (lam + tau**2) ** (-gamma / 2.0)
    std[0] = 0.0
    re = rng.standard_normal((n, kmax))
    im = rng.standard_normal((n, kmax))
    coeff = std[None, :] * (re + 1j * im)
    coeff[:, 0] = 0.0
    if N % 2 == 0:
        coeff[:, -1] = coeff[:, -1].real
    return np.fft.irfft(coeff, n=N, axis=1) * N * scale

def solve_burgers(u0, nu=NU, T=1.0, dt=2.5e-4, n_save=101):
    n, N = u0.shape
    kmax = N // 2 + 1
    k = 2.0 * np.pi * np.arange(kmax)
    ik = 1j * k
    k2 = k**2
    mask = np.ones(kmax)
    mask[int(kmax * 2 / 3):] = 0.0
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
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_x, endpoint=False)
    u0 = grf(n_samples, n_x, rng=rng, scale=u0_scale)
    field = solve_burgers(u0)
    if not np.all(np.isfinite(field)):
        raise RuntimeError("solver went unstable -- reduce dt or u0_scale")
    return x, field[:, 0, :].copy(), field[:, -1, :].copy(), field

t0 = time.time()
x, u0_tr, _, fld_tr = generate(2000, 201, seed=0, u0_scale=1.0)
_, u0_te, _, fld_te = generate(100, 201, seed=1, u0_scale=1.0)
t = np.linspace(0.0, 1.0, fld_tr.shape[1])
np.savez("/kaggle/working/burgers.npz", t=t, x=x,
         u0_train=u0_tr, field_train=fld_tr, u0_test=u0_te, field_test=fld_te)
print(f"burgers: field_train {fld_tr.shape} field_test {fld_te.shape}  {time.time()-t0:.0f}s")
