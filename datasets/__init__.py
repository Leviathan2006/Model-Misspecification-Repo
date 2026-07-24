"""Faithful data generators for the benchmarks in arXiv:2606.03469.

All four carry their own solver: manufactured solutions (diffusion_reaction),
Fourier pseudo-spectral + forward Euler (burgers), D2Q9 lattice Boltzmann with
power-law rheology (cavity_flow), and Q1 neo-Hookean FEM with Newton line search
and load continuation (hyperelastic).

Each `get_dataset` validates the cached .npz against the requested sizes and grid
and regenerates on a mismatch -- the bundled caches are small demo files, not the
paper's sizes, so asking for the paper's configuration will trigger a rebuild.

Known gap: the burgers initial-condition amplitude cannot be reconciled with the
paper's Fig. 7 -- see the caveat in `burgers.grf`.
"""
from . import burgers, cavity_flow, diffusion_reaction, hyperelastic

__all__ = ["diffusion_reaction", "burgers", "cavity_flow", "hyperelastic"]
