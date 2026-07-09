"""Faithful data generators for the benchmarks in arXiv:2606.03469.

diffusion_reaction, burgers  -> fully implemented
cavity_flow, hyperelastic    -> spec-complete stubs (need LBM / FEM solvers)
"""
from . import burgers, cavity_flow, diffusion_reaction, hyperelastic

__all__ = ["diffusion_reaction", "burgers", "cavity_flow", "hyperelastic"]
