# Model-Misspecification-Repo

Reproductions and extensions of

> L. Ma, N. Boullé, Y.-S. Yang, H. Wu, L. Guo,
> *Physics-guided correction for operator learning under model misspecification*,
> [arXiv:2606.03469](https://arxiv.org/abs/2606.03469) (2026).

The paper corrects a **misspecified governing equation** by training a second
("correction") operator alongside the main solution operator, so a physics-informed
model can still be trained when the physics it's given is wrong.

## Layout

Different implementations of the method live in their own top-level folder, so
parallel work doesn't collide on the same files.

- **[`baseline/`](baseline/README.md)** — serial PI-DeepONet where the
  misspecified operator `N₀` is applied to the *sum* of the prior and correction
  networks, `N₀[G_θ + G_ψ]`. Implements the paper's method on three of its four
  benchmarks (diffusion-reaction, Burgers, hyperelastic beam) plus three
  uncertainty-aware variants (ensemble, heteroscedastic, diffusion). See its
  README for setup, data, and known gaps.

Everything each folder needs — code, dataset generators, cached data — is
self-contained inside it; there's no shared code between folders. If you're
looking for a specific result table from the paper, check the folder's README
for which of Tables 2–5 it reproduces.
