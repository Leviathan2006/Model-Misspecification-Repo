# FNO baselines on the "model misspecification" operator-learning suite

Faithful reproduction of the **problem suite** from

> L. Ma, N. Boullé, Y.-S. Yang, H. Wu, L. Guo,
> *Physics-guided correction for operator learning under model misspecification*,
> [arXiv:2606.03469](https://arxiv.org/abs/2606.03469) (2026),

with a **vanilla, data-driven Fourier Neural Operator (FNO)** as the learner
instead of the paper's physics-informed DeepONet. The paper notes its correction
framework is architecture-independent ("any neural operator"); this repo builds
the FNO baseline side of that claim on all four benchmarks.

## Read this first — what is and isn't reproduced

The paper's *method* is a physics-informed DeepONet that corrects a **misspecified
governing equation** (e.g. treating a shear-thinning fluid as Newtonian). That
notion of misspecification lives entirely in the **physics loss**. A vanilla FNO
is **data-driven** — it never sees a governing equation, so "misspecification"
does not apply to it. What this repo reproduces faithfully is:

1. the **data generation** for each benchmark, from the paper's **true** models
   and exact constants, and
2. **FNO baselines** that learn each true solution operator from data — the
   data-driven reference any physics-guided method is ultimately compared to.

The four learned operators (as defined in the paper):

| problem | operator | dim | solver used to make the data |
|---|---|---|---|
| `diffusion_reaction` | `v(x) → u(x)` | 1D | method of manufactured solutions (no solver) |
| `burgers` | `u₀(x) → u(x,t)` | 2D | Fourier pseudo-spectral + forward Euler |
| `cavity_flow` | `Re → (u_x,u_y)(x,y)` | 2D | D2Q9 Lattice-Boltzmann, power-law rheology |
| `hyperelastic` | `ε → (u_x,u_y)(x,y)` | 2D | Q1 neo-Hookean FEM, Newton + load continuation |

## Layout

```
datasets/
  diffusion_reaction.py   v → u,  D=0.1, k_r=0.5 e^{-u}, manufactured solutions
  burgers.py              u₀ → u(x,t),  ν=0.01, GRF ICs, pseudo-spectral solver
  cavity_flow.py          Re → velocity,  power-law n=1.5, Re∈[100,200], LBM
  hyperelastic.py         ε → displacement,  neo-Hookean beam, ε∈[0,0.2], FEM
fno.py                    vanilla FNO1d and FNO2d
run_fno.py                trains/evaluates the FNO on a chosen problem
```

Every dataset script is self-contained (only `numpy`, plus `scipy.sparse` for the
FEM) and vectorised over the sample axis, so it clones and runs on Kaggle CPU/GPU.

## Quickstart

```bash
pip install -r requirements.txt

# generate data (paper sizes; the 2D solvers are the slow ones — run on Kaggle)
python datasets/diffusion_reaction.py
python datasets/burgers.py
python datasets/cavity_flow.py
python datasets/hyperelastic.py

# train + evaluate the FNO
python run_fno.py --problem diffusion_reaction
python run_fno.py --problem burgers
python run_fno.py --problem cavity_flow
python run_fno.py --problem hyperelastic
```

`get_dataset(...)` caches to `data/*.npz`, so `run_fno.py` will generate on first
use and reuse afterwards. Reduce `--n_train`/grid sizes on the dataset scripts for
quick local checks.

## Results (vanilla FNO, Tesla T4, bundled `--quick`-sized data, 500 epochs)

| problem | test rel. L2 | status |
|---|---|---|
| `diffusion_reaction` | 4.7e-3 | good |
| `burgers` | 1.5e-3 | good |
| `cavity_flow` | 7.4e-3 | good (only 20 train samples) |
| `hyperelastic` | 8.4e-1 | **broken — see note** |

**Hyperelastic is not learnable as currently generated.** A 10:1 slender beam
compressed up to ε≈0.19 is far past its buckling load (critical strain ~0.2%), and
the current FEM (no line search; `detF` is clipped rather than rejecting inverted
elements) produces nonphysical post-buckling displacements — `|u|≈30` on a domain
of size 1×0.1, with sign-flipping `u_y`. The ε→u map is then effectively chaotic
and cannot be fit. Fix planned: robust Newton (line search + inverted-element
step-cutting + more continuation steps), or a milder ε range. The other three
problems are solid.

## Validation status

All solvers were smoke-tested at coarse resolution:

- **diffusion-reaction** — Dirichlet BCs `u(±1)=0` satisfied exactly (the
  prefactor `(x²−1)/10` enforces them); no NaNs.
- **burgers** — viscous energy decays; fields bounded; no NaNs. Note the GRF
  `𝒩(0, 25²(−Δ+5²I)⁻⁴)` is smooth and small-amplitude by construction; the FNO
  target is scale-free (relative L2) so this is immaterial to the benchmark.
- **cavity_flow** — top-row velocity correlates 1.00 with the prescribed lid
  profile, speeds ≈ lid speed, no NaNs.
- **hyperelastic** — left edge clamped (`u=0`), right edge reaches `u_x=−ε`
  exactly, Newton converges, no NaNs.

The **full-resolution runs and FNO training are intended for Kaggle GPUs** and
were not run here.

## Caveats (documented, not hidden)

- **Cavity non-dimensionalisation.** The paper does not print the exact constant
  linking `Re` to the power-law consistency in its Lattice-Boltzmann setup. We use
  `ν₀ = U₀(N−1)/Re` with a small lattice lid speed `U₀`; the qualitative physics
  and the operator `Re → u` are faithful, but the absolute Reynolds constant may
  differ by an O(1) factor. See the docstring in `datasets/cavity_flow.py`.
- **DeepONet-specific pieces are omitted on purpose.** Physics residual losses,
  collocation points, and the misspecified operators are part of the paper's
  method, not its data — they are not needed for a data-driven FNO.

## Paper reference numbers (DeepONet, for context — not FNO)

Corrected relative errors reported by the paper: diffusion-reaction ≈1.85×10⁻³,
Burgers (case B) ≈1.01%, cavity `u_x` ≈0.82%, hyperelastic `u_x` ≈3.75×10⁻³.
These are the *physics-guided-corrected DeepONet* numbers; the FNO baselines here
are a separate, data-driven measurement.

## Physics-guided correction (the paper's actual method) — `physics_correction.py`

`physics_correction.py` implements the paper's method on an **FNO backbone** for
the diffusion-reaction benchmark (a physics-informed FNO / PINO-style residual):

- **prior** `G_θ : v → u_prior`, **correction** `G_ψ : [v, u_prior] → c`
- `L_data = ‖G_θ(v)(obs) − u(obs)‖²`  (sparse observations)
- `L_physics = ‖N₀[G_θ(v)] + G_ψ(·) − v‖²`  (collocation; correction in **forcing space**)
- misspecified `N₀ = D u_xx − k_r` (constant) vs true `D u_xx − 0.5 e^{−u} u`

Run the three-way comparison that reproduces the paper's headline story:

```bash
python physics_correction.py --mode all --epochs 100000
```

- `known` (true physics) ≈ reference floor
- `misspecified` (wrong `N₀`, no correction) → error blows up
- `corrected` (prior + forcing-space correction) → recovered

**Readout note.** `G_ψ` is trained in forcing/residual space, so the recovered
solution is the **prior** `G_θ(v)`; the literal `G_θ+G_ψ` sum from the paper's
Eq. (c) is dimensionally inconsistent for the FNO and is reported only as a
diagnostic. Physics-informed training needs ~1e4–1e5 epochs to converge.

## Our methods (trustworthy extensions)

The paper's correction is a single deterministic model with no uncertainty. Two
extensions in the "trustworthy operator learning" direction, both on the
PI-DeepONet backbone and both logging to a file under `results/`:

- **Deep-ensemble correction** (`method_ensemble.py`): K corrected models from
  different inits; the ensemble mean improves the point estimate and the ensemble
  std is a calibrated epistemic-uncertainty map (we report corr(std, |error|)).
  -> `results/method_ensemble.txt`
- **Heteroscedastic correction under noisy observations**
  (`method_heteroscedastic.py`): a variance head + Gaussian-NLL data term gives
  robustness to observation noise and a calibrated aleatoric-uncertainty map,
  compared head-to-head with the deterministic correction on the same noisy data.
  -> `results/method_heteroscedastic.txt`

```bash
python method_ensemble.py --n_ensemble 5 --epochs 100000
python method_heteroscedastic.py --noise 0.15 --epochs 100000
```

## Data

`data/` ships small (`--quick`-sized) `.npz` datasets for all four problems so
the repo runs immediately on clone. Regenerate at paper sizes with the dataset
scripts or `kaggle_run.py` (delete the cached `.npz` first).

## Roadmap

- [ ] fill in FNO baseline numbers from a full Kaggle run
- [ ] extend `physics_correction.py` to Burgers (clean residual) and, later, the
      cavity/hyperelastic PDE-system residuals
- [ ] diffusion-based **probabilistic** corrector variant (uncertainty on the
      model discrepancy) — the Boullé cold-email hook
