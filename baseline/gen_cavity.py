"""Generate the 2d cavity-flow dataset at the paper's size (Sec. 3.3).

    python gen_cavity.py

Writes data/cavity_flow.npz (1000 train / 100 test, 101x101 grid, velocity +
pressure). D2Q9 Lattice-Boltzmann, up to 5e5 iterations per sample -- HOURS.
This is one of the two slow ones; leave it running.

Samples are solved in chunks (chunk=50) so the vectorised LBM state fits in RAM;
lower it if you hit a memory limit, raise it if you have headroom to go faster.
"""
import time

import datasets.cavity_flow as cf

if __name__ == "__main__":
    t0 = time.time()
    (x, y), Re_tr, u_tr, p_tr, Re_te, u_te, p_te = cf.get_dataset(
        "data", n_train=1000, n_test=100, N=101, chunk=50)
    print(f"cavity_flow: u_train {u_tr.shape}  p_train {p_tr.shape}")
    print(f"done in {time.time() - t0:.0f}s")
