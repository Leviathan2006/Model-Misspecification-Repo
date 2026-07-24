"""Shared .npz cache handling for the four benchmark generators.

Every generator used to do

    if os.path.exists(path):
        return np.load(path)          # <- regardless of the requested sizes

so a cache written at demo sizes was silently returned when the caller asked for
the paper's sizes. The bundled caches are small (200/50, 20/10, ...) while the
paper uses 1000-2000 training samples on finer grids, which made that a quiet
correctness bug rather than a convenience. `load_cache` validates the cache
against what was actually requested and regenerates on any mismatch.
"""
import numpy as np


def load_cache(path, n_train, n_test, shapes=()):
    """Return the loaded npz if it matches the request, else None (-> regenerate).

    n_train/n_test: sample counts the caller asked for.
    shapes:         iterable of (key, axis, expected) extra dimension checks,
                    e.g. ("u_train", 1, 21) to require 21 nodes along axis 1.
    """
    import os
    if not os.path.exists(path):
        return None
    d = np.load(path)
    bad = []
    for key in d.files:
        if key.endswith("_train") and d[key].shape[0] != n_train:
            bad.append(f"n_train: cached {d[key].shape[0]} != requested {n_train}")
            break
    for key in d.files:
        if key.endswith("_test") and d[key].shape[0] != n_test:
            bad.append(f"n_test: cached {d[key].shape[0]} != requested {n_test}")
            break
    for key, axis, expected in shapes:
        if key in d.files and d[key].shape[axis] != expected:
            bad.append(f"{key} axis {axis}: cached {d[key].shape[axis]} "
                       f"!= requested {expected}")
    if bad:
        print(f"NOTE: {path} does not match the requested configuration "
              f"({'; '.join(bad)}) -- regenerating. Delete the file to silence "
              f"this, or pass the cached sizes.")
        return None
    return d
