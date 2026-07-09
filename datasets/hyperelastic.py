"""2D hyperelastic slender beam under compression.

Source: Ma, Boulle, Yang, Wu & Guo (2026), "Physics-guided correction for
operator learning under model misspecification", arXiv:2606.03469, Sec. 4.4.

Learned operator (paper):  G : eps  ->  displacement field (u_x, u_y).

TRUE model used to generate the data (compressible neo-Hookean, plane strain):
    W(F) = mu/2 (tr(F^T F) - 2 - 2 ln J) + lambda/2 (ln J)^2,   J = det F
    E = 1e6,  nu = 0.3,  mu = E/(2(1+nu)),  lambda = E nu/((1+nu)(1-2nu))
    body force   f = (0, -1000)^T
    domain       Omega = [0, 1] x [0, 0.1]   (slender beam), grid 101 x 21 nodes
    left edge    clamped, u = 0
    right edge   compressive displacement  u = (-eps, 0),  eps in [0, 0.2]
    top/bottom   traction-free
    solver       nonlinear FEM (Newton) with load continuation

SOLVER: self-contained bilinear (Q1) quadrilateral finite elements, 2x2 Gauss
quadrature, total-Lagrangian neo-Hookean model with analytic first Piola-
Kirchhoff stress P and material tangent A, assembled with scipy.sparse and solved
by Newton-Raphson with load stepping (continuation from eps=0). No FEM library
required.

STATUS: smoke-tested on a coarse mesh with small strain; the full 101x21 mesh and
eps up to 0.2 are intended for Kaggle.
"""
import argparse
import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

E, NU = 1.0e6, 0.3
MU = E / (2.0 * (1.0 + NU))
LAM = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
BODY = np.array([0.0, -1000.0])

# reference element: nodes (-1,-1), (1,-1), (1,1), (-1,1); 2x2 Gauss
_XI_A = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
_G = 1.0 / np.sqrt(3.0)
_GP = np.array([[-_G, -_G], [_G, -_G], [_G, _G], [-_G, _G]])
_GW = np.ones(4)


def _shape():
    """Return N (gp,4) and dNdxi (gp,4,2)."""
    xi, eta = _GP[:, 0][:, None], _GP[:, 1][:, None]       # (gp,1)
    xa, ea = _XI_A[:, 0][None, :], _XI_A[:, 1][None, :]    # (1,4)
    N = 0.25 * (1 + xi * xa) * (1 + eta * ea)              # (gp,4)
    dNdxi = np.stack([0.25 * xa * (1 + eta * ea),
                      0.25 * ea * (1 + xi * xa)], axis=-1)  # (gp,4,2)
    return N, dNdxi


def _mesh(nx, ny):
    """Structured quad mesh of [0,1]x[0,0.1]. Returns coords, elems, node grid."""
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 0.1, ny + 1)
    X, Y = np.meshgrid(xs, ys)                             # (ny+1, nx+1)
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    nid = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    elems = []
    for j in range(ny):
        for i in range(nx):
            elems.append([nid[j, i], nid[j, i + 1],
                          nid[j + 1, i + 1], nid[j + 1, i]])
    return coords, np.array(elems), nid


def _det_inv_2x2(A):
    """Batched 2x2 determinant and inverse. A: (..., 2, 2)."""
    a, b = A[..., 0, 0], A[..., 0, 1]
    c, d = A[..., 1, 0], A[..., 1, 1]
    det = a * d - b * c
    inv = np.empty_like(A)
    inv[..., 0, 0], inv[..., 0, 1] = d, -b
    inv[..., 1, 0], inv[..., 1, 1] = -c, a
    return det, inv / det[..., None, None]


def _element_terms(coords, elems, u):
    """Assemble global internal force and tangent for displacement u (ndof,)."""
    ne = elems.shape[0]
    N, dNdxi = _shape()
    Xe = coords[elems]                                     # (ne,4,2)
    ue = u.reshape(-1, 2)[elems]                           # (ne,4,2)

    # reference Jacobian, physical shape gradients, volume element
    Jref = np.einsum("eai,gaJ->egiJ", Xe, dNdxi)           # (ne,gp,2,2)
    detJ, invJ = _det_inv_2x2(Jref)
    gradN = np.einsum("gaK,egKJ->egaJ", dNdxi, invJ)       # (ne,gp,4,2)
    dV = detJ * _GW[None, :]                               # (ne,gp)

    # deformation gradient F = I + du/dX
    gradu = np.einsum("eai,egaJ->egiJ", ue, gradN)         # (ne,gp,2,2)
    I2 = np.eye(2)
    F = I2[None, None] + gradu
    detF, Finv = _det_inv_2x2(F)
    lnJ = np.log(np.clip(detF, 1e-9, None))
    FinvT = np.swapaxes(Finv, -1, -2)

    # first Piola-Kirchhoff stress
    P = MU * (F - FinvT) + LAM * lnJ[..., None, None] * FinvT

    # material tangent A_{iJkL}
    c2 = (MU - LAM * lnJ)
    term1 = MU * np.einsum("ik,JL->iJkL", I2, I2)
    term2 = np.einsum("eg,egJk,egLi->egiJkL", c2, Finv, Finv)
    term3 = LAM * np.einsum("egJi,egLk->egiJkL", Finv, Finv)
    A = term1[None, None] + term2 + term3                  # (ne,gp,2,2,2,2)

    # element internal force (ne,4,2) and tangent (ne,4,2,4,2)
    fint = np.einsum("egaJ,egiJ,eg->eai", gradN, P, dV)
    T = np.einsum("egaJ,egiJkL->egaikL", gradN, A)
    Ke = np.einsum("egaikL,egbL,eg->eaibk", T, gradN, dV)

    # scatter to global
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
    """Constant external body-force vector (ndof,)."""
    ne = elems.shape[0]
    N, dNdxi = _shape()
    Xe = coords[elems]
    Jref = np.einsum("eai,gaJ->egiJ", Xe, dNdxi)
    detJ, _ = _det_inv_2x2(Jref)
    dV = detJ * _GW[None, :]
    fext = np.einsum("ga,eg,i->eai", N, dV, BODY)          # (ne,4,2)
    edof = np.empty((ne, 8), dtype=np.int64)
    edof[:, 0::2] = 2 * elems
    edof[:, 1::2] = 2 * elems + 1
    Fext = np.zeros(coords.shape[0] * 2)
    np.add.at(Fext, edof.ravel(), fext.reshape(ne, 8).ravel())
    return Fext


def solve_beam(eps, nx=100, ny=20, n_steps=10, newton_tol=1e-8, max_newton=25):
    """Solve one beam at compression eps. Returns u field (ny+1, nx+1, 2)."""
    coords, elems, nid = _mesh(nx, ny)
    ndof = coords.shape[0] * 2
    Fext = _body_force(coords, elems)

    left = nid[:, 0].ravel()
    right = nid[:, -1].ravel()
    fixed = np.concatenate([2 * left, 2 * left + 1, 2 * right, 2 * right + 1])
    free = np.setdiff1d(np.arange(ndof), fixed)

    u = np.zeros(ndof)
    for step in range(1, n_steps + 1):
        eps_s = eps * step / n_steps
        u[2 * right] = -eps_s                              # prescribed u_x on right
        u[2 * right + 1] = 0.0
        for _ in range(max_newton):
            Fint, K = _element_terms(coords, elems, u)
            R = Fint - Fext
            Rf = R[free]
            if np.linalg.norm(Rf) < newton_tol * (1 + np.linalg.norm(Fext[free])):
                break
            Kff = K[free][:, free]
            du = spsolve(Kff.tocsc(), -Rf)
            u[free] += du
    return u.reshape(-1, 2)[nid]                           # (ny+1, nx+1, 2)


def generate(n_samples, nx=100, ny=20, n_steps=10, seed=0):
    """Return (x, y, eps, u) with u of shape (n_samples, ny+1, nx+1, 2)."""
    rng = np.random.default_rng(seed)
    eps = rng.uniform(0.0, 0.2, size=n_samples)
    fields = np.stack([solve_beam(e, nx, ny, n_steps) for e in eps], axis=0)
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 0.1, ny + 1)
    return x, y, eps, fields


def get_dataset(data_dir="data", n_train=200, n_test=100, nx=100, ny=20,
                n_steps=10, seed=0):
    """Load cached data or generate + cache it.
    Returns (coords, eps_train, u_train, eps_test, u_test)."""
    path = os.path.join(data_dir, "hyperelastic.npz")
    if os.path.exists(path):
        d = np.load(path)
        return ((d["x"], d["y"]), d["eps_train"], d["u_train"],
                d["eps_test"], d["u_test"])
    x, y, eps_tr, u_tr = generate(n_train, nx, ny, n_steps, seed)
    _, _, eps_te, u_te = generate(n_test, nx, ny, n_steps, seed + 1)
    os.makedirs(data_dir, exist_ok=True)
    np.savez(path, x=x, y=y, eps_train=eps_tr, u_train=u_tr,
             eps_test=eps_te, u_test=u_te)
    return (x, y), eps_tr, u_tr, eps_te, u_te


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=200)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--nx", type=int, default=100)
    p.add_argument("--ny", type=int, default=20)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--data_dir", type=str, default="data")
    a = p.parse_args()
    (x, y), eps_tr, u_tr, eps_te, u_te = get_dataset(
        a.data_dir, a.n_train, a.n_test, a.nx, a.ny, a.n_steps)
    print(f"hyperelastic: train u {u_tr.shape}, test u {u_te.shape}")
    print(f"  eps in [{eps_tr.min():.3f}, {eps_tr.max():.3f}], "
          f"|u_x| max {np.abs(u_tr[...,0]).max():.4e}, any NaN: {np.isnan(u_tr).any()}")
