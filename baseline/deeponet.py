"""DeepONet with exact autodiff derivatives for physics-informed training.

This is the backbone the paper actually uses (Ma, Boulle, Yang, Wu & Guo, 2026,
arXiv:2606.03469). A DeepONet represents the solution as

    u(v)(y) = sum_k  branch_k(v) * trunk_k(y)  +  bias

so any derivative in the query variable y acts only on the trunk:
    d^n u / dy^n = sum_k branch_k(v) * d^n trunk_k / dy^n .
We therefore differentiate the (shared) trunk basis once per step with batched
autograd and combine it with the branch coefficients -- exact, mesh-free, and far
cheaper than differentiating u per sample.
"""
import torch
import torch.nn as nn
from torch.func import jvp


class MLP(nn.Module):
    def __init__(self, sizes, act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    """Unstacked DeepONet. branch_in = m sensors (+ extra channels for a serial
    correction network). trunk_in = query dimension (1 for x, 2 for (x,t) etc.).

    n_out > 1 gives a vector-valued operator (cavity velocity, hyperelastic
    displacement): branch and trunk both emit n_out * p coefficients, which are
    contracted per component, i.e. u_c(y) = sum_k a_{ck}(v) b_{ck}(y) + bias_c.
    """

    def __init__(self, branch_in, trunk_in=1, p=100, width=64, depth=4, n_out=1):
        super().__init__()
        hidden = [width] * (depth - 1)
        self.branch = MLP([branch_in] + hidden + [p * n_out])
        self.trunk = MLP([trunk_in] + hidden + [p * n_out])
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.p = p
        self.n_out = n_out

    def branch_out(self, v):
        return self.branch(v)                              # (B, p*n_out)

    def trunk_out(self, y):
        return self.trunk(y)                               # (Q, p*n_out)

    def combine(self, b, t):
        """Contract branch (B, p*n_out) with trunk (Q, p*n_out).

        Returns (B, Q) when n_out == 1, else (B, Q, n_out). `t` may carry a
        derivative of the trunk basis, in which case the bias is not added."""
        if self.n_out == 1:
            return b @ t.T
        B, Q = b.shape[0], t.shape[0]
        b = b.view(B, self.n_out, self.p)
        t = t.view(Q, self.n_out, self.p)
        return torch.einsum("bcp,qcp->bqc", b, t)

    def forward(self, v, y):
        out = self.combine(self.branch_out(v), self.trunk_out(y))
        return out + self.bias if self.n_out > 1 else out + self.bias[0]


def trunk_derivatives(trunk, y, order=2):
    """Return the trunk basis and its y-derivatives at query points y (1D input).

    Uses FORWARD-mode autodiff (jvp): since the trunk input is one-dimensional, a
    single jvp yields the derivative of all p basis functions at once -- O(1)
    passes instead of O(p). Nested jvp gives the second derivative. The outputs
    stay connected to the trunk parameters, so loss.backward() trains normally.

    Returns t (Q,p) and, up to `order`, first/second derivatives (Q,p).
    """
    ones = torch.ones_like(y)
    if order == 1:
        t, t_y = jvp(trunk, (y,), (ones,))
        return t, t_y
    (t, t_y), (_, t_yy) = jvp(lambda z: jvp(trunk, (z,), (ones,)), (y,), (ones,))
    return t, t_y, t_yy


def trunk_jet(trunk, y, pairs):
    """Trunk basis plus selected first/second partials, for multi-D query points.

    `trunk_derivatives` above takes a tangent of all ones, which is the right
    thing only when the trunk input is one-dimensional -- in 2D it would give the
    directional derivative along (1,1) rather than the partials. Here each
    derivative is taken with a one-hot tangent instead.

    pairs: iterable of (i, j) coordinate indices. One nested jvp per pair yields
    d/dy_i, d/dy_j and d^2/dy_i dy_j simultaneously, so e.g. [(0,0), (1,1)] gets
    the value, both first partials and both unmixed second partials in two
    passes. Pass (i, None) for a first derivative only.

    Returns (t, first, second) where first[i] is dt/dy_i (Q, p*n_out) and
    second[(i,j)] is d^2 t / dy_i dy_j.
    """
    first, second = {}, {}
    t = None
    for i, j in pairs:
        e_i = torch.zeros_like(y)
        e_i[:, i] = 1.0
        if j is None:
            t, t_i = jvp(trunk, (y,), (e_i,))
            first[i] = t_i
            continue
        e_j = torch.zeros_like(y)
        e_j[:, j] = 1.0
        (t, t_i), (t_j, t_ij) = jvp(lambda z: jvp(trunk, (z,), (e_i,)), (y,), (e_j,))
        first[i], first[j] = t_i, t_j
        second[(i, j)] = t_ij
    return t, first, second


def enable_fast_math():
    """TF32 + cuDNN autotune -- real speedups on Ampere/Hopper/Blackwell."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
