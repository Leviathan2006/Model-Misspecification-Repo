# Diffusion-reaction dataset (paper Sec. 3.1) -- fully self-contained.
# Paste into a Kaggle cell and run. Only needs numpy. Writes to /kaggle/working/.
import numpy as np, time

D, N_MODES = 0.1, 5

def manufactured(x, w):
    i = np.arange(1, N_MODES + 1)
    ipix = np.pi * np.outer(x, i)
    sin, cos = np.sin(ipix), np.cos(ipix)
    a = w[:, 0::2][:, None, :]
    b = w[:, 1::2][:, None, :]
    sin, cos = sin[None], cos[None]
    iw = i * np.pi
    S = (a * sin + b * cos).sum(-1)
    Sp = (a * iw * cos - b * iw * sin).sum(-1)
    Spp = -((a * iw**2 * sin + b * iw**2 * cos).sum(-1))
    g = (x**2 - 1.0) / 10.0
    gp = x / 5.0
    gpp = np.full_like(x, 1.0 / 5.0)
    u = g * S
    uxx = gpp * S + 2.0 * gp * Sp + g * Spp
    return u, uxx

def generate(n_samples, n_x=201, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n_x)
    w = rng.uniform(0.0, 1.0, size=(n_samples, 2 * N_MODES))
    u, uxx = manufactured(x, w)
    v = D * uxx - 0.5 * np.exp(-u) * u
    return x, v, u

t0 = time.time()
x, v_tr, u_tr = generate(1000, 201, seed=0)
_, v_te, u_te = generate(100, 201, seed=1)
np.savez("/kaggle/working/diffusion_reaction.npz",
         x=x, v_train=v_tr, u_train=u_tr, v_test=v_te, u_test=u_te)
print(f"diffusion_reaction: train {v_tr.shape} test {v_te.shape}  {time.time()-t0:.0f}s")
