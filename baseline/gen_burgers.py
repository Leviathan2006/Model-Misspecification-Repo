"""Generate the 1d Burgers dataset at the paper's size (Sec. 3.2).

    python gen_burgers.py

Writes data/burgers.npz (2000 train / 100 test, full 101x201 space-time field).
Fourier pseudo-spectral + forward Euler, batched over samples -- a few minutes.

Note: the initial-condition amplitude uses the paper's printed GRF (u0_scale=1).
See the caveat in datasets/burgers.grf -- this cannot be reconciled with Fig. 7.
"""
import time

import datasets.burgers as bg

if __name__ == "__main__":
    t0 = time.time()
    t, x, u0_tr, fld_tr, u0_te, fld_te = bg.get_dataset(
        "data", n_train=2000, n_test=100, n_x=201, u0_scale=1.0)
    print(f"burgers: field_train {fld_tr.shape}  field_test {fld_te.shape}")
    print(f"done in {time.time() - t0:.0f}s")
