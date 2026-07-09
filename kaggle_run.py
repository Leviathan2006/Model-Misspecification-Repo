"""End-to-end Kaggle driver: generate all four datasets, train a vanilla FNO on
each, print a results table, and save sample-solution figures.

    python kaggle_run.py            # full-ish run (data gen is CPU/numpy bound)
    python kaggle_run.py --quick    # tiny sizes, ~a few minutes, to sanity-check

Notes:
  * The FNO training uses the GPU (torch.cuda); data generation is numpy/scipy
    on CPU, so the 2D solvers (cavity LBM, hyperelastic FEM) are the slow part.
  * Cavity/hyperelastic default to reduced sample counts so a run finishes in a
    sensible time. Bump them toward the paper sizes (1000 / 200) once you know
    the wall-clock on your Kaggle machine.
"""
import argparse
import os
import time

import numpy as np
import torch

from datasets import burgers as bg
from datasets import cavity_flow as cf
from datasets import diffusion_reaction as dr
from datasets import hyperelastic as he
from run_fno import train

PROBLEMS = ["diffusion_reaction", "burgers", "cavity_flow", "hyperelastic"]
BATCH = {"diffusion_reaction": 20, "burgers": 20,
         "cavity_flow": 10, "hyperelastic": 20}


def generate_all(quick):
    print("generating datasets ...", flush=True)
    t0 = time.time()
    if quick:
        dr.get_dataset("data", n_train=200, n_test=50)
        bg.get_dataset("data", n_train=200, n_test=50)
        cf.get_dataset("data", n_train=20, n_test=10, N=61,
                       max_iter=8000, chunk=10)
        he.get_dataset("data", n_train=20, n_test=10, nx=60, ny=12, n_steps=6)
    else:
        dr.get_dataset("data", n_train=1000, n_test=100)              # paper size
        bg.get_dataset("data", n_train=2000, n_test=100)              # paper size
        cf.get_dataset("data", n_train=200, n_test=50, N=101,         # reduced
                       max_iter=30000, chunk=50)
        he.get_dataset("data", n_train=200, n_test=100,               # paper size
                       nx=100, ny=20, n_steps=10)
    print(f"  data ready in {time.time() - t0:.0f}s", flush=True)


def save_figures():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                            # noqa: BLE001
        print("matplotlib unavailable, skipping figures:", e)
        return
    os.makedirs("results/figures", exist_ok=True)

    d = np.load("data/diffusion_reaction.npz")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(d["x"], d["v_train"][0], label="input v")
    ax.plot(d["x"], d["u_train"][0], label="solution u")
    ax.legend(); ax.set_title("diffusion-reaction sample"); fig.tight_layout()
    fig.savefig("results/figures/diffusion_reaction.png", dpi=120); plt.close(fig)

    d = np.load("data/burgers.npz")
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(d["field_train"][0], aspect="auto", origin="lower",
                   extent=[0, 1, 0, 1], cmap="RdBu_r")
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_title("Burgers u(x,t)")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig("results/figures/burgers.png", dpi=120); plt.close(fig)

    d = np.load("data/cavity_flow.npz")
    u = d["u_train"][0]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(np.sqrt((u ** 2).sum(-1)), origin="lower",
                   extent=[0, 1, 0, 1], cmap="viridis")
    ax.set_title(f"cavity speed (Re={d['Re_train'][0]:.0f})")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig("results/figures/cavity_flow.png", dpi=120); plt.close(fig)

    d = np.load("data/hyperelastic.npz")
    u = d["u_train"][0]
    fig, ax = plt.subplots(figsize=(6, 2))
    im = ax.imshow(np.sqrt((u ** 2).sum(-1)), origin="lower",
                   extent=[0, 1, 0, 0.1], aspect="auto", cmap="magma")
    ax.set_title(f"hyperelastic |u| (eps={d['eps_train'][0]:.3f})")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig("results/figures/hyperelastic.png", dpi=120); plt.close(fig)
    print("  figures saved to results/figures/", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny sizes for a smoke run")
    ap.add_argument("--epochs", type=int, default=500)
    a = ap.parse_args()

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    else:
        print("no GPU found — training on CPU", flush=True)

    generate_all(a.quick)
    epochs = 100 if a.quick else a.epochs

    results = {}
    for prob in PROBLEMS:
        t = time.time()
        err = train(prob, epochs=epochs, batch=BATCH[prob])
        results[prob] = (err, time.time() - t)

    save_figures()

    print("\n==== vanilla FNO baselines — test relative L2 ====")
    lines = ["| problem | test rel. L2 | train time |",
             "|---|---|---|"]
    for prob in PROBLEMS:
        err, dt = results[prob]
        print(f"  {prob:20s}  {err:.4e}   ({dt:.0f}s)")
        lines.append(f"| `{prob}` | {err:.4e} | {dt:.0f}s |")
    os.makedirs("results", exist_ok=True)
    with open("results/baselines.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote results/baselines.md")


if __name__ == "__main__":
    main()
