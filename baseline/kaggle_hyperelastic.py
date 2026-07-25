# 2d hyperelastic-beam dataset (paper Sec. 3.4) -- fully self-contained.
# Paste into a Kaggle cell and run. Needs numpy + scipy (both preinstalled on
# Kaggle). Writes to /kaggle/working/. Runtime: HOURS (nonlinear FEM, Newton
# with load continuation, one sample at a time). Progress every 10 samples.
import numpy as np, time
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

E, NU = 1.0e6, 0.3
MU = E / (2.0 * (1.0 + NU))
LAM = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
BODY = np.array([0.0, -1000.0])
_XI_A = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
_G = 1.0 / np.sqrt(3.0)
_GP = np.array([[-_G, -_G], [_G, -_G], [_G, _G], [-_G, _G]])
_GW = np.ones(4)

def _shape():
    xi, eta = _GP[:, 0][:, None], _GP[:, 1][:, None]
    xa, ea = _XI_A[:, 0][None, :], _XI_A[:, 1][None, :]
    N = 0.25 * (1 + xi * xa) * (1 + eta * ea)
    dNdxi = np.stack([0.25 * xa * (1 + eta * ea),
                      0.25 * ea * (1 + xi * xa)], axis=-1)
    return N, dNdxi

def _mesh(nx, ny):
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 0.1, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    nid = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    elems = []
    for j in range(ny):
        for i in range(nx):
            elems.append([nid[j, i], nid[j, i + 1], nid[j + 1, i + 1], nid[j + 1, i]])
    return coords, np.array(elems), nid

def _det_inv_2x2(A):
    a, b = A[..., 0, 0], A[..., 0, 1]
    c, d = A[..., 1, 0], A[..., 1, 1]
    det = a * d - b * c
    inv = np.empty_like(A)
    inv[..., 0, 0], inv[..., 0, 1] = d, -b
    inv[..., 1, 0], inv[..., 1, 1] = -c, a
    return det, inv / det[..., None, None]

def _element_terms(coords, elems, u):
    ne = elems.shape[0]
    N, dNdxi = _shape()
    Xe = coords[elems]
    ue = u.reshape(-1, 2)[elems]
    Jref = np.einsum("eai,gaJ->egiJ", Xe, dNdxi)
    detJ, invJ = _det_inv_2x2(Jref)
    gradN = np.einsum("gaK,egKJ->egaJ", dNdxi, invJ)
    dV = detJ * _GW[None, :]
    gradu = np.einsum("eai,egaJ->egiJ", ue, gradN)
    I2 = np.eye(2)
    F = I2[None, None] + gradu
    detF, Finv = _det_inv_2x2(F)
    lnJ = np.log(np.clip(detF, 1e-9, None))
    FinvT = np.swapaxes(Finv, -1, -2)
    P = MU * (F - FinvT) + LAM * lnJ[..., None, None] * FinvT
    c2 = (MU - LAM * lnJ)
    term1 = MU * np.einsum("ik,JL->iJkL", I2, I2)
    term2 = np.einsum("eg,egJk,egLi->egiJkL", c2, Finv, Finv)
    term3 = LAM * np.einsum("egJi,egLk->egiJkL", Finv, Finv)
    A = term1[None, None] + term2 + term3
    fint = np.einsum("egaJ,egiJ,eg->eai", gradN, P, dV)
    T = np.einsum("egaJ,egiJkL->egaikL", gradN, A)
    Ke = np.einsum("egaikL,egbL,eg->eaibk", T, gradN, dV)
    edof = np.empty((ne, 8), dtype=np.int64)
    edof[:, 0::2] = 2 * elems
    edof[:, 1::2] = 2 * elems + 1
    ndof = coords.shape[0] * 2
    Fint = np.zeros(ndof)
    np.add.at(Fint, edof.ravel(), fint.reshape(ne, 8).ravel())
    Ke = Ke.reshape(ne, 8, 8)
    rows = np.repeat(edof[:, :, None], 8, axis=2).ravel()
    cols = np.repeat(edof[:, None, :], 8, axis=1).ravel()
    K = coo_matrix((Ke.ravel(), (rows, cols)), shape=(ndof, ndof)).tocsr()
    return Fint, K

def _body_force(coords, elems):
    ne = elems.shape[0]
    N, dNdxi = _shape()
    Xe = coords[elems]
    Jref = np.einsum("eai,gaJ->egiJ", Xe, dNdxi)
    detJ, _ = _det_inv_2x2(Jref)
    dV = detJ * _GW[None, :]
    fext = np.einsum("ga,eg,i->eai", N, dV, BODY)
    edof = np.empty((ne, 8), dtype=np.int64)
    edof[:, 0::2] = 2 * elems
    edof[:, 1::2] = 2 * elems + 1
    Fext = np.zeros(coords.shape[0] * 2)
    np.add.at(Fext, edof.ravel(), fext.reshape(ne, 8).ravel())
    return Fext

def _min_detF(coords, elems, u):
    _, dNdxi = _shape()
    Xe, ue = coords[elems], u.reshape(-1, 2)[elems]
    Jref = np.einsum("eai,gaJ->egiJ", Xe, dNdxi)
    _, invJ = _det_inv_2x2(Jref)
    gradN = np.einsum("gaK,egKJ->egaJ", dNdxi, invJ)
    F = np.eye(2)[None, None] + np.einsum("eai,egaJ->egiJ", ue, gradN)
    detF, _ = _det_inv_2x2(F)
    return detF.min()

def _newton(coords, elems, u, Fext, free, scale, tol, max_it):
    Fint, K = _element_terms(coords, elems, u)
    R = (Fint - Fext)[free]
    rn = np.linalg.norm(R)
    for _ in range(max_it):
        if rn < tol * scale:
            return True
        try:
            du = spsolve(K[free][:, free].tocsc(), -R)
        except Exception:
            return False
        if not np.all(np.isfinite(du)):
            return False
        alpha, accepted = 1.0, False
        for _ in range(40):
            u_try = u.copy()
            u_try[free] += alpha * du
            if _min_detF(coords, elems, u_try) > 1e-8:
                Fint_t, K_t = _element_terms(coords, elems, u_try)
                R_t = (Fint_t - Fext)[free]
                rn_t = np.linalg.norm(R_t)
                if np.isfinite(rn_t) and rn_t < (1.0 - 1e-4 * alpha) * rn:
                    u[free] = u_try[free]
                    R, rn, K = R_t, rn_t, K_t
                    accepted = True
                    break
            alpha *= 0.5
        if not accepted:
            return False
    return rn < tol * scale

def solve_beam(eps, nx=100, ny=20, n_steps=25, newton_tol=1e-8, max_newton=40,
               min_frac=1e-4, strict=True):
    coords, elems, nid = _mesh(nx, ny)
    ndof = coords.shape[0] * 2
    Fext = _body_force(coords, elems)
    left = nid[:, 0].ravel()
    right = nid[:, -1].ravel()
    fixed = np.concatenate([2 * left, 2 * left + 1, 2 * right, 2 * right + 1])
    free = np.setdiff1d(np.arange(ndof), fixed)
    scale = 1.0 + np.linalg.norm(Fext[free])
    u = np.zeros(ndof)
    done = 0.0
    dfrac = 1.0 / n_steps
    while done < 1.0 - 1e-12:
        frac = min(done + dfrac, 1.0)
        u_bak = u.copy()
        u[2 * right] = -eps * frac
        u[2 * right + 1] = 0.0
        if _newton(coords, elems, u, Fext, free, scale, newton_tol, max_newton):
            done = frac
            dfrac = min(dfrac * 1.5, 1.0 / n_steps)
        else:
            u = u_bak
            dfrac *= 0.5
            if dfrac < min_frac:
                msg = f"Newton failed at eps={eps:.4f}, load fraction {done:.4f}"
                if strict:
                    raise RuntimeError(msg)
                print("WARNING: " + msg)
                break
    return u.reshape(-1, 2)[nid]

def generate(n_samples, nx=100, ny=20, n_steps=25, seed=0):
    rng = np.random.default_rng(seed)
    eps = rng.uniform(0.0, 0.2, size=n_samples)
    fields = []
    for i, e in enumerate(eps):
        fields.append(solve_beam(e, nx, ny, n_steps))
        if (i + 1) % 10 == 0:
            print(f"  hyperelastic {i+1}/{n_samples}", flush=True)
    fields = np.stack(fields, axis=0)
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 0.1, ny + 1)
    return x, y, eps, fields

# paper Sec. 3.4: 300 compression levels split 200 train / 100 test
t0 = time.time()
x, y, eps, u = generate(300, nx=100, ny=20, n_steps=25, seed=0)
eps_tr, u_tr = eps[:200], u[:200]
eps_te, u_te = eps[200:], u[200:]
np.savez("/kaggle/working/hyperelastic.npz", x=x, y=y,
         eps_train=eps_tr, u_train=u_tr, eps_test=eps_te, u_test=u_te)
print(f"hyperelastic: u_train {u_tr.shape} u_test {u_te.shape}  {time.time()-t0:.0f}s")
