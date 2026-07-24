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
    ones = torch.ones_like(y)
    if order == 1:
        t, t_y = jvp(trunk, (y,), (ones,))
        return t, t_y
    (t, t_y), (_, t_yy) = jvp(lambda z: jvp(trunk, (z,), (ones,)), (y,), (ones,))
    return t, t_y, t_yy


def enable_fast_math():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
