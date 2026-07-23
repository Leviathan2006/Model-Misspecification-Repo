# Physics-guided correction under model misspecification (PI-DeepONet)

Reproduction and extensions of

> L. Ma, N. Boullé, Y.-S. Yang, H. Wu, L. Guo,
> *Physics-guided correction for operator learning under model misspecification*,
> [arXiv:2606.03469](https://arxiv.org/abs/2606.03469) (2026).

The paper's method is a physics-informed DeepONet that corrects a **misspecified
governing equation** (e.g. collapsing a reaction term to a constant). This repo
reproduces that method on the 1D diffusion-reaction benchmark and adds several
uncertainty-aware variants.

## Benchmark suite

Faithful data generators for all four benchmarks from the paper are included. The
PI-DeepONet method and its variants currently run on `diffusion_reaction`; the
other three generators are available to extend the method to (their solvers are
the slow, GPU/Kaggle-oriented part).

| generator | operator | dim | solver used to make the data |
|---|---|---|---|
| `diffusion_reaction` | `v(x) → u(x)` | 1D | method of manufactured solutions (no solver) |
| `burgers` | `u₀(x) → u(x,t)` | 2D | Fourier pseudo-spectral + forward Euler, ν=0.01 |
| `cavity_flow` | `Re → (u_x,u_y)(x,y)` | 2D | D2Q9 Lattice-Boltzmann, power-law rheology (n=1.5) |
| `hyperelastic` | `ε → (u_x,u_y)(x,y)` | 2D | Q1 neo-Hookean FEM, Newton + load continuation |

Each generator is self-contained (`numpy`, plus `scipy.sparse` for the FEM) and
vectorised over the sample axis. Run a generator directly to cache its `.npz`,
e.g. `python datasets/burgers.py`.

## Diffusion-reaction benchmark (the reproduced one)

The learned operator is `v(x) → u(x)`, where `u` solves

```
D u_xx − k_r(u) u = v,   k_r(u) = 0.5 e^{−u},   D = 0.1,   x ∈ [−1, 1],   u(±1) = 0
```

Data are built by the method of manufactured solutions (sample a smooth `u`
obeying the BCs, then define `v` by applying the true operator — no PDE solver).
The misspecified operator drops the reaction nonlinearity to a constant:
`N₀ = D u_xx − k_r_const`.

## Layout

```
deeponet.py            DeepONet + exact forward-mode trunk derivatives (jvp)
datasets/
  diffusion_reaction.py   v → u, manufactured solutions
  burgers.py              u₀ → u(x,t), ν=0.01, GRF ICs, pseudo-spectral solver
  cavity_flow.py          Re → velocity, power-law n=1.5, Re∈[100,200], LBM
  hyperelastic.py         ε → displacement, neo-Hookean beam, ε∈[0,0.2], FEM
run_correction.py      the paper's method: PI-DeepONet, three-mode comparison
method_ensemble.py     variant: deep-ensemble correction (epistemic UQ)
method_heteroscedastic.py  variant: variance head + Gaussian NLL (aleatoric UQ)
method_diffusion.py    variant: conditional diffusion model of the correction
```

## The method — `run_correction.py`

- **prior** `G_θ : v(sensors) → u(y)`, **correction** `G_ψ : [v, u_prior] → c(y)` (forcing space)
- `L_data = ‖G_θ(v)(y_obs) − u(y_obs)‖²` (sparse observations)
- `L_phys = ‖N₀[G_θ(v)] + G_ψ − v‖²` (collocation)
- `u_xx` via exact autodiff of the trunk basis; hard BCs via `g(x) = 1 − x²`.

```bash
python run_correction.py --mode all            # full run
python run_correction.py --mode all --quick    # tiny sizes, sanity check
```

Three modes reproduce the paper's headline story:

- `known` (true physics residual) → reference floor
- `misspecified` (wrong `N₀`, no correction) → error blows up
- `corrected` (prior + forcing-space correction) → recovered

Paper reference numbers: misspecified ≈1.6, corrected ≈1.85×10⁻³, known ≈9.0×10⁻⁴.

## Variants (uncertainty-aware extensions)

The paper's correction is a single deterministic model with no uncertainty. Each
variant builds on the PI-DeepONet backbone and logs to `results/`.

- **`method_ensemble.py`** — K corrected models from different inits; ensemble mean
  improves the point estimate, ensemble std is an epistemic-uncertainty map
  (reports corr(std, |error|)).
- **`method_heteroscedastic.py`** — variance head + Gaussian-NLL data term under
  spatially-varying observation noise; gives robustness and a calibrated aleatoric
  std map, compared against the deterministic correction on the same noisy data.
- **`method_diffusion.py`** — conditional denoising-diffusion model of the
  correction field, conditioned on sparse observations; sampling gives a mean
  (the missing physics) and a calibrated model-error std. (Exploratory.)

```bash
python method_ensemble.py --n_ensemble 5
python method_heteroscedastic.py --noise 0.15
python method_diffusion.py
```

## Dependencies

`numpy` and `torch`. The dataset generator needs only `numpy`.
