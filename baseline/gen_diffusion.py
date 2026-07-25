"""Generate the diffusion-reaction dataset at the paper's size (Sec. 3.1).

    python gen_diffusion.py

Writes data/diffusion_reaction.npz (1000 train / 100 test, 201 points).
Manufactured solutions -- no PDE solver, finishes in seconds.
"""
import time

import datasets.diffusion_reaction as dr

if __name__ == "__main__":
    t0 = time.time()
    x, v_tr, u_tr, v_te, u_te = dr.get_dataset(
        "data", n_train=1000, n_test=100, n_x=201)
    print(f"diffusion_reaction: v_train {v_tr.shape}  v_test {v_te.shape}")
    print(f"done in {time.time() - t0:.0f}s")
