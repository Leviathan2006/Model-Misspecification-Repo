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
    correction network). trunk_in = query dimension (1 for x, 2 for (x,t) etc.)."""

    def __init__(self, branch_in, trunk_in=1, p=100, width=64, depth=4):
        super().__init__()
        hidden = [width] * (depth - 1)
        self.branch = MLP([branch_in] + hidden + [p])
        self.trunk = MLP([trunk_in] + hidden + [p])
        self.bias = nn.Parameter(torch.zeros(1))
        self.p = p

    def branch_out(self, v):
        return self.branch(v)                              # (B, p)

    def trunk_out(self, y):
        return self.trunk(y)                               # (Q, p)

    def forward(self, v, y):
        return self.branch_out(v) @ self.trunk_out(y).T + self.bias   # (B, Q)


def trunk_derivatives(trunk, y, order=2):
    """Return the trunk basis and its y-derivatives at query points y.

    y: (Q, d) leaf tensor with requires_grad=True.
    Returns t (Q,p) and, for d==1, first/second derivatives (Q,p) up to `order`.
    Uses batched autograd so all p basis functions are differentiated in one call.
    """
    t = trunk(y)                                           # (Q, p)
    Q, p = t.shape
    eye = torch.eye(p, device=y.device)[:, None, :].expand(p, Q, p)

    def dydx(f):
        g = torch.autograd.grad(f, y, grad_outputs=eye, is_grads_batched=True,
                                create_graph=True, retain_graph=True)[0]
        return g.squeeze(-1).transpose(0, 1).contiguous()  # (Q, p)

    t_y = dydx(t)
    if order == 1:
        return t, t_y
    t_yy = dydx(t_y)
    return t, t_y, t_yy


def enable_fast_math():
    """TF32 + cuDNN autotune -- real speedups on Ampere/Hopper/Blackwell."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
