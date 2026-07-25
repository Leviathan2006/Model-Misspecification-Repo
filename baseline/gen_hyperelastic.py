"""Generate the 2d hyperelastic-beam dataset at the paper's size (Sec. 3.4).

    python gen_hyperelastic.py

Writes data/hyperelastic.npz (200 train / 100 test, 21x101 node grid).
Nonlinear neo-Hookean FEM, Newton with load continuation, one sample at a
time -- HOURS. This is the other slow one; leave it running. Progress prints
every 10 samples.
"""
import time

import datasets.hyperelastic as he

if __name__ == "__main__":
    t0 = time.time()
    (x, y), eps_tr, u_tr, eps_te, u_te = he.get_dataset(
        "data", n_train=200, n_test=100, nx=100, ny=20, verbose=True)
    print(f"hyperelastic: u_train {u_tr.shape}  u_test {u_te.shape}")
    print(f"done in {time.time() - t0:.0f}s")
