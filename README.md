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

**Caches.** The bundled `data/*.npz` are small demo files, *not* the paper's
sizes. `get_dataset` validates the cache against the requested sample counts and
grid and regenerates on a mismatch (it used to return the cache regardless, which
silently gave you 20 samples when you asked for 1000). Paper sizes are 1000/100
(diffusion-reaction), 2000/100 (burgers), 1000/100 at 101×101 (cavity),
200/100 at 21×101 (hyperelastic); the last two are hours of CPU.

**Known gap — burgers amplitude.** The GRF as printed in the paper
(`N(0, 25²(−Δ+5²I)⁻⁴)`) yields `std(u₀) ≈ 0.012`, but the Fig. 7 colourbars imply
initial conditions 8–10× larger for Cases A/B and ~0.6× for Case C. No single
rescaling satisfies all three, so the paper's IC distribution can't be recovered
from the text. The generator implements the printed formula (the correct KL
discretisation) and exposes `--u0_scale`. See the caveat in `burgers.grf`.

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
deeponet.py            DeepONet (scalar/vector) + exact forward-mode trunk
                       derivatives: trunk_derivatives (1D) and trunk_jet (n-D)
datasets/
  diffusion_reaction.py   v → u, manufactured solutions
  burgers.py              u₀ → u(x,t), ν=0.01, GRF ICs, pseudo-spectral solver
  cavity_flow.py          Re → (velocity, pressure), power-law n=1.5, LBM
  hyperelastic.py         ε → displacement, neo-Hookean beam, ε∈[0,0.2], FEM
  _cache.py               size/grid-checked .npz cache loader (shared)
run_correction.py      the paper's method on diffusion-reaction (Table 2)
run_burgers.py         the paper's method on Burgers, Cases A/B/C (Table 3)
run_hyperelastic.py    the paper's method on the hyperelastic beam (Table 5)
method_ensemble.py     variant: deep-ensemble correction (epistemic UQ)
method_heteroscedastic.py  variant: variance head + Gaussian NLL (aleatoric UQ)
method_diffusion.py    variant: conditional diffusion model of the correction
```

The method is implemented for three of the four benchmarks. Cavity flow (Table 4)
has a verified data generator (velocity **and** pressure) but no method driver
yet — it needs the two-operator setup (`G_θ` for velocity plus a separate
pressure operator `G_ξ : Re → P`), which none of the other three share.

## The method — `run_correction.py`

Serial DeepONet realisation of the paper's Sec. 2.2, on the diffusion-reaction
benchmark of Sec. 3.1, with the paper's Table 1 settings (`p = 100`, depth 4,
width 64, `tanh`, Adam `β = (0.999, 0.999)`, `lr = 1e-3`, `10⁵` epochs,
`N_p = 1000`, `N_f = 1000`, `N_u = 100`, 101 sensors, 100 test samples on 201 points).

- **prior** `G_θ : v(sensors) → u_θ(y)`, **correction** `G_ψ : [v(sensors), u_θ(sensors)] → c(y)`
- `L_data = ‖G_θ(v)(y_obs) − u(y_obs)‖²` — paper Eq. (2), sparse observations
- `L_phys = ‖N₀[G_θ(v) + G_ψ(v,u_θ)] − v‖²` — paper Eq. (1) **with one modification:
  `N₀` is applied to both of the first two terms** (i.e. to their sum) rather than
  to `G_θ` alone
- `L = L_phys + λ_d L_data`; `u_xx` via exact autodiff of the trunk basis; hard BCs
  via `g(x) = 1 − x²`.

Because `N₀` now acts on the corrected field `u_θ + c`, the correction is a
solution-space object (matching `G_ψ : V × U → U` in Sec. 2.1) rather than a
forcing-space term added into the residual. For this problem the enforced equation
is `D(u_θ + c)_xx − k_r = v`, i.e. `D u_θ,xx + Φ = v` with `Φ = D c_xx − k_r = N₀[G_ψ]`,
so the learned reaction term recovered from the correction network is exactly
`N₀[G_ψ]`, compared against the paper's target `φ = −k_r(u) u`. The reported
solution is `G_θ`: the data term anchors it to the observations while `G_ψ` absorbs
the model-form discrepancy inside the residual.

```bash
python run_correction.py --mode all               # full run (paper settings)
python run_correction.py --mode all --quick       # tiny sizes, sanity check
python run_correction.py --mode all --n_runs 5    # paper's mean ± std over 5 runs
```

Three modes reproduce the paper's headline story (their Table 2, which reports
relative L₂ for `u`, `v` and `φ` — all three are printed):

- `known` (true residual `D u_xx − k_r(u)u − v`) → reference floor
- `misspecified` (wrong `N₀`, no correction) → error blows up
- `corrected` (`N₀` on prior + correction) → recovered

Paper reference numbers for `u`: misspecified ≈1.61, corrected ≈1.85×10⁻³, known ≈9.0×10⁻⁴.

## The method on the other benchmarks

Both use the same serial construction and the same modified physics loss
(`N₀` applied to the sum `G_θ + G_ψ`), on the shared vector-capable `DeepONet`
and `trunk_jet` (exact forward-mode partials for multi-dimensional query points).

**`run_burgers.py`** — 1D Burgers (Sec. 3.2, Table 3), operator `v = u₀ → u(x,t)`.
Three misspecifications: **A** extra cubic `εu³` (ε=10), **B** advection dropped,
**C** diffusion dropped. Loss weights `λ_bc = 1, λ_ic = λ_u = 50`; periodic BC on
value and `u_x`; 101 sensors, 51×51 correction grid, 101×101 collocation.

```bash
python run_burgers.py --case all --mode all           # full run
python run_burgers.py --case B --mode all --quick      # sanity check
```

**`run_hyperelastic.py`** — 2D neo-Hookean beam (Sec. 3.4, Table 5), operator
`ε → (u_x, u_y)`. Misspecified prior is linear elasticity. Adds the paper's
energy loss `L = L_phys + λ_bc L_bc + λ_u L_u + λ_e L_energy` with
`λ_e = 100, λ_u = 10⁵, λ_bc = 1`, and a fourth `data` mode (standard DeepONet,
no physics). The residual `div σ` uses the exact neo-Hookean material tangent —
the same one the FEM generator assembles, so they agree by construction.

```bash
python run_hyperelastic.py --mode all                  # full run
python run_hyperelastic.py --mode all --quick          # sanity check
```

> **Burgers caveat:** Cases A/B/C errors are *not* expected to match Table 3
> until the initial-condition amplitude is resolved (see the data section above).
> The method, losses and derivatives are correct; the input distribution can't be
> recovered from the paper's text. `--u0_scale` is passed through for experiments.

## Variants (uncertainty-aware extensions)

The paper's correction is a single deterministic model with no uncertainty. Each
variant builds on the PI-DeepONet backbone and logs to `results/`. These are *not*
part of the paper's method, and they still use the paper's original Eq. (1)
residual (`N₀[G_θ] + G_ψ − v`), not the modified one above.

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
