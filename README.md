# TPM Stability Mapping & Neural Network Fast Prior

> **A three-phase pipeline for efficient ghost/gradient stability mapping of the
> Transitional Planck Mass (TPM) model in EFTCAMB, and for accelerating Cobaya
> MCMC chains via a neural network viability prior.**

---

## Table of Contents

1. [Scientific Context](#1-scientific-context)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Repository Structure](#3-repository-structure)
4. [Requirements](#4-requirements)
5. [Phase 1 — Sobol Stability Mapping](#5-phase-1--sobol-stability-mapping)
6. [Phase 2 — Neural Network Training](#6-phase-2--neural-network-training)
7. [Phase 3 — Fast Prior in Cobaya](#7-phase-3--fast-prior-in-cobaya)
8. [Visualisation](#8-visualisation)
9. [Mathematical Background](#9-mathematical-background)
10. [Performance Notes](#10-performance-notes)
11. [References](#11-references)

---

## 1. Scientific Context

The **Transitional Planck Mass (TPM)** model is a modified gravity theory
implemented within the [EFTCAMB](http://www.eftcamb.org) framework
(Hu et al. 2014; Raveri et al. 2014). It describes a smooth, Gaussian-shaped
transition of the effective Planck mass at early times — parameterised via the
EFT function $\Omega(a)$ — followed by a late-time kinetic braiding regime.
Its background evolution is governed by four phenomenological parameters:

| Symbol | Script name | Physical meaning | Prior range |
|--------|-------------|-----------------|-------------|
| $\log_{10} a_T$ | `Log_aT` | Scale factor at the centre of the transition | $[-7.5,\ -3.5]$ |
| $\sigma$ | `sig` | Width of the transition in e-folds | $[0.4,\ 3.0]$ |
| $\Omega_0$ | `M` | Amplitude of the Planck-mass shift | $[-0.15,\ 0.015]$ |
| $c_0$ | `c` | Late-time kinetic braiding coefficient | $[-0.1,\ 0.01]$ |

For typical values $\sigma \sim 1$, the $\Omega$ function evolves over a short
fraction of cosmic history. At early times (near the transition) the model
behaves like a pure $f(R)$ theory ($\alpha_M = 2\alpha_B$, $\alpha_K = 0$);
at late times it enters a kinetic gravity braiding regime
($\alpha_M = 0$, $\alpha_K \propto \alpha_B \propto c_0/H^2$).

EFTCAMB enforces **ghost** and **gradient stability** conditions on the scalar
perturbations at every step of an MCMC chain:

- **No-ghost:** the kinetic matrix of scalar perturbations must be positive
  definite, i.e. $\alpha_K + \tfrac{3}{2}\alpha_B^2 > 0$. For the TPM model
  this reduces to requiring $c_0 < 0$, which ensures $\alpha_K > 0$.
- **No-gradient instability:** the squared sound speed of scalar perturbations
  must be positive throughout the entire cosmic history,
  $c_s^2(a) > 0\ \forall\, a$.
  This depends non-trivially on all four parameters and cannot be reduced to a
  simple analytic inequality.

Because a single EFTCAMB stability evaluation takes $\mathcal{O}(10)$ s, and
a Cobaya MCMC chain proposes $\mathcal{O}(10^5)$ points — a significant
fraction of which lie in the unstable region — the stability check is a major
**computational bottleneck**. This repository addresses that bottleneck with a
three-phase pipeline.

---

## 2. Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — tpm_stability_map.py                                          │
│                                                                          │
│  Sobol QMC sampling in (Log_aT, sig, M, c) space                        │
│    ↓  parallel EFTCAMB evaluation (ghost + gradient stability flags)     │
│  K-NN boundary uncertainty score → adaptive Sobol refinement            │
│    ↓                                                                     │
│  tpm_stability_map.pkl   ·  {points: (N,4),  stable: (N,) bool}         │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────────┐
│  PHASE 2 — Stability_nn.py                                               │
│                                                                          │
│  StandardScaler normalisation + 80/20 train/val split                   │
│    ↓  MLP  4 → 64 → 64 → 32 → 1   (BCEWithLogitsLoss, Adam)            │
│  Best checkpoint saved on minimum validation loss                        │
│    ↓                                                                     │
│  tpm_stability_model.pt   (PyTorch weights)                              │
│  tpm_scaler.pkl           (fitted StandardScaler)                        │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────────┐
│  PHASE 3 — NN_fast_prior_TPM.py   (Cobaya Likelihood)                   │
│                                                                          │
│  At each MCMC step:                                                      │
│    θ  →  scale  →  forward pass  →  logit z                             │
│    z > 0  ⟹  logp = 0.0     (pass to EFTCAMB and likelihoods)          │
│    z ≤ 0  ⟹  logp = −∞     (reject immediately, skip EFTCAMB)          │
│                                                                          │
│  Cost: ~0.1 ms/step  vs  ~10 s for a full EFTCAMB call                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
.
├── tpm_stability_map.py      # Phase 1: Sobol sampling + EFTCAMB labelling
├── tpm_stability_viz.py      # Visualisation: scatter corner + heatmap corner
├── Stability_nn.py           # Phase 2: MLP training
├── NN_fast_prior_TPM.py      # Phase 3: Cobaya Likelihood wrapper
├── TPM.yaml                  # Reference Cobaya YAML (MCMC run configuration)
└── README.md
```

Produced artefacts (not tracked by git):

```
├── tpm_stability_map.pkl     # Labelled point cloud from Phase 1
├── tpm_stability_model.pt    # Trained MLP weights from Phase 2
└── tpm_scaler.pkl            # Fitted StandardScaler from Phase 2
```

---

## 4. Requirements

### Python packages

```
cobaya >= 3.3
camb / eftcamb          (your custom EFTCAMB-patched build)
torch >= 2.0
scikit-learn >= 1.3
scipy >= 1.11
numpy
matplotlib
pyyaml
```

### EFTCAMB

This pipeline requires the EFTCAMB-patched version of CAMB that includes the
TPM model, but it can be generalized to any model. Point Cobaya to your build via the `path` field
in the YAML:

```yaml
theory:
  camb:
    path: /path/to/your/eftcamb
```

---

## 5. Phase 1 — Sobol Stability Mapping

### Usage

```bash
# Smoke test — verifies the pipeline end-to-end (~5 min on 4 workers)
python tpm_stability_map.py --yaml TPM.yaml --base 64 --refine-iters 0 --workers 4

# Production run
python tpm_stability_map.py \
    --yaml TPM.yaml \
    --base 4096 \
    --refine-iters 3 \
    --refine-batch 1024 \
    --workers 36 \
    --output tpm_stability_map.pkl
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--yaml` | `TPM.yaml` | Path to the Cobaya YAML |
| `--base` | `4096` | Number of base Sobol points |
| `--refine-iters` | `3` | Number of boundary-refinement passes |
| `--refine-batch` | `1024` | Points added per refinement pass |
| `--workers` | `ncpu − 1` | Parallel worker processes |
| `--seed` | `42` | Sobol scramble seed |
| `--output` | `tpm_stability_map.pkl` | Output pickle path |
| `--serial` | flag | Disable multiprocessing (for debugging) |

### Thread control

Each worker runs one CAMB instance. Set the following before launching to
avoid OpenMP oversubscription ($N_w \times N_\text{OMP} \leq C_\text{cores}$):

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

---

## 6. Phase 2 — Neural Network Training

### Usage

```bash
python Stability_nn.py \
    --data tpm_stability_map.pkl \
    --epochs 1000 \
    --lr 0.001
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | (hardcoded path) | Path to the Phase 1 pickle |
| `--epochs` | `1000` | Training epochs |
| `--lr` | `0.001` | Adam learning rate |

### Output files

| File | Description |
|------|-------------|
| `tpm_stability_model.pt` | Best model weights (selected on validation loss) |
| `tpm_scaler.pkl` | Fitted `StandardScaler` — must be paired with the weights |

---

## 7. Phase 3 — Fast Prior in Cobaya

### YAML configuration

```yaml
likelihood:
  # Physical likelihoods (unchanged)
  planck_2018_highl_plik.TTTEEE_lite_native: null
  act_dr6_lenslike.ACTDR6LensLike:
    variant: actplanck_baseline
    lens_only: false
  planck_2018_lowl.TT: null
  planck_2018_lowl.EE: null
  bao.desi_dr2.desi_bao_lrg1: null
  sn.desdovekie: null

  # Neural network fast prior
  NN_fast_prior_TPM.NNFastPrior:
    python_path: /path/to/this/repo
    model_path:  /path/to/tpm_stability_model.pt
    scaler_path: /path/to/tpm_scaler.pkl
```

---

## 8. Visualisation

```bash
python tpm_stability_viz.py tpm_stability_map.pkl \
    --bins 25 \
    --min-count 3 \
    --outdir figs/
```

Produces two figures:

| File | Content |
|------|---------|
| `tpm_stability_scatter.png` | Corner scatter: 2-D projections of the labelled point cloud. Green = stable, red = unstable. Diagonal panels show 1-D histograms. |
| `tpm_stability_heatmap.png` | Corner heatmap: $\hat{p}(x_i, x_j)$, the fraction of stable points per 2-D bin, marginalised over the remaining two parameters. Red ($\hat{p}=0$) to green ($\hat{p}=1$). |

> **Note on sample size.** The heatmap bins each require at least `--min-count`
> points to display a colour (grey = masked). With the smoke-test sample of 64
> points and the default `--bins 25`, the expected number of points per 2-D
> cell is $64/625 \approx 0.1$, so almost all cells will appear grey.
> Use `--bins 4 --min-count 2` for a coarse but visible map from a small
> sample.

### Interpreting the heatmap

The value $\hat{p}(x_i, x_j)$ answers the question:

> *"If I fix $x_i$ and $x_j$ to values in this bin, what fraction of the
> remaining 2-D space $(x_k, x_l)$ yields a stable model?"*

$\hat{p} = 1$ (dark green): **always** stable, regardless of $x_k, x_l$.  
$\hat{p} = 0$ (red): **never** stable.  
$\hat{p} \approx 0.5$ (yellow): the projected stability boundary passes here.

A single panel cannot distinguish between a sharp boundary crossing and a
uniformly mixed region — the corner layout (all $\binom{4}{2} = 6$ pairs)
provides the additional context to resolve such degeneracies.

---

## 9. Mathematical Background

### 9.1 Why not a regular grid?

A grid of $N$ points per axis in $d = 4$ dimensions requires $N^d$ evaluations.
For $N = 20$ this is $20^4 = 160{,}000$ EFTCAMB calls. Halving the grid spacing
costs a factor $2^d = 16$. This scaling is prohibitive because the stability
boundary is a $(d-1)$-dimensional surface: most of the volume far from the
boundary is either uniformly stable or uniformly unstable and carries no useful
information about the boundary's location.

### 9.2 Sobol low-discrepancy sequences

A **Sobol sequence** is a deterministic, quasi-random sequence designed so that
every new point fills the largest empty region of the parameter space. The
quality of a point set is measured by its **star-discrepancy** $D_N^*$, which
quantifies how far the empirical distribution of $N$ points deviates from the
uniform distribution. For a Sobol sequence of $N$ points in $d$ dimensions:

$$D_N^* \sim \frac{(\log N)^d}{N}$$

compared to $D_N^* \sim N^{-1/2}$ for pseudo-random Monte Carlo and
$D_N^* \sim N^{-1/d}$ for a regular grid. 

![Computer](Sobol_uniformity_rate.png)

The practical consequence is that
the Sobol sequence achieves the same coverage uniformity as a grid but with
**exponentially fewer points** in high dimensions.

The sequence is constructed via base-2 arithmetic using a set of direction
numbers that ensure successive points are always placed in the least-covered
sub-interval. The scrambled variant (used here via `scipy.stats.qmc.Sobol`)
randomises the starting direction while preserving the low-discrepancy property,
which provides an unbiased estimator and enables error estimation via
independent scrambled replicates.

An important implementation detail: the balance property of a Sobol sequence is
strongest when $N$ is an exact power of 2. The script therefore rounds $N$ up
to the next power of 2 internally and then truncates, preserving the uniform
coverage guarantee while respecting the user's requested sample size.

### 9.3 K-NN boundary uncertainty and adaptive refinement

After the base Sobol sample is labelled, the script identifies the
**stability boundary** $\partial S$ without knowing its analytic form, using a
$k$-nearest-neighbour ($k$-NN) estimator.

**Step 1 — normalisation.** The four parameters live on very different scales.
Euclidean distances in raw parameter space are dominated by whichever parameter
has the largest range. To make the geometry meaningful, all coordinates are
first mapped to $[0, 1]^4$:

$$\tilde{\theta}^{(j)} = \frac{\theta^{(j)} - \theta^{(j)}_\min}
                               {\theta^{(j)}_\max - \theta^{(j)}_\min}$$

**Step 2 — local stable fraction.** For each point $\theta_i$, the $k$ nearest
neighbours in $\tilde{\theta}$-space are found (excluding $\theta_i$ itself).
The local stable fraction is:

$$\bar{f}_i = \frac{1}{k} \sum_{j \in \mathcal{N}_k(i)} y_j \in [0, 1]$$

where $y_j \in \{0, 1\}$ is the stability label.

**Step 3 — uncertainty score.** A scalar uncertainty is derived from
$\bar{f}_i$:

$$u_i = 1 - \left| 2\bar{f}_i - 1 \right| \in [0, 1]$$

This function peaks at $u_i = 1$ when $\bar{f}_i = 0.5$, i.e. when the
neighbourhood contains an equal number of stable and unstable points — the
hallmark of a point sitting exactly on $\partial S$. It reaches $u_i = 0$ when
$\bar{f}_i \in \{0, 1\}$, i.e. in a region that is homogeneously stable or
homogeneously unstable, where additional evaluations carry no boundary
information.

**Step 4 — bounding box refinement.** The top 10% of points by $u_i$ define
the current estimate of $\partial S$. A new Sobol sequence is drawn inside the
axis-aligned bounding box of these high-uncertainty points (plus a small padding
fraction), concentrating new evaluations near the boundary. This sub-box is also
clipped back to the physical prior ranges.

This adaptive procedure is iterated $K$ times (default $K = 3$), each time
updating the boundary estimate with the new labels and generating the next
refinement batch. The result is a point cloud whose density is highest near
$\partial S$ — exactly where the MLP classifier (Phase 2) has the hardest
classification task and needs the most training data.

### 9.4 How the stability check is invoked

For each parameter point, the script builds a minimal Cobaya model: the EFTCAMB
theory is configured exactly as in `TPM.yaml` (same flags: `EFTflag=3`,
`DesignerEFTmodel=3`, ghost and gradient stability enabled), but the physical
likelihoods are replaced by the no-op `one` likelihood which always returns 0.
The six ΛCDM parameters are pinned at the centres of their `ref` distributions:

| Parameter | Value |
|-----------|-------|
| `ombh2`  | 0.0224 |
| `omch2`  | 0.118  |
| `tau`    | 0.055  |
| `logA`   | 3.05   |
| `ns`     | 0.965  |
| `H0`     | 72.0   |

Calling `model.loglike(θ)` then triggers only the EFTCAMB background
initialisation — the step at which the stability conditions are evaluated.
If EFTCAMB declares the model stable, the call returns $0$; if it fails, it
returns $-\infty$. The binary label is simply `np.isfinite(loglike)`.

### 9.5 Parallelism

The $N$ stability evaluations are **embarrassingly parallel**: each depends only
on its own parameter values and produces no shared state. The script uses
`multiprocessing.Pool` with a **pool initialiser**: each worker process builds
its own Cobaya model once at startup (a ~10 s overhead) and then receives
parameter points one at a time, amortising the initialisation cost over hundreds
of evaluations.

The multiprocessing start method is set to `spawn` rather than `fork`. This is
necessary because `fork` would copy the parent process's memory, including any
OpenMP thread-pool state initialised by numpy or CAMB imports. An OpenMP runtime
that has been forked is in an undefined state and can deadlock. With `spawn`,
each worker starts a fresh Python interpreter and initialises OpenMP cleanly
under its own `OMP_NUM_THREADS` setting.

### 9.6 MLP architecture and training

The neural network is a **Multi-Layer Perceptron (MLP)** that maps the
4-dimensional normalised parameter vector to a single real number (the logit):

$$f_\theta: \mathbb{R}^4 \longrightarrow \mathbb{R}$$

The architecture consists of three hidden layers with ReLU activations:

```
Input: θ̃ ∈ ℝ⁴
  │
  ├─ Linear(4 → 64) + ReLU
  ├─ Linear(64 → 64) + ReLU
  ├─ Linear(64 → 32) + ReLU
  └─ Linear(32 → 1)              ← raw logit  z ∈ ℝ
```

**Why this depth?** The stability boundary $\partial S$ is a nonlinear
$(d-1) = 3$-dimensional surface embedded in $\mathbb{R}^4$. A network with a
single hidden layer can in principle approximate any continuous function
(universal approximation theorem), but requires exponentially many neurons. Two
to three hidden layers learn hierarchical features (local boundary curvature,
global topology) much more efficiently. The widths $64 \to 64 \to 32$ provide
sufficient capacity without overfitting on $\mathcal{O}(7000)$ training points.

**Why ReLU?** The ReLU activation $\max(0, x)$ is piecewise linear, which means
the network partitions $\mathbb{R}^4$ into polytopes (convex regions with linear
boundaries). The union of many such polytopes can approximate the curved
stability boundary $\partial S$ to arbitrary precision as the number of neurons
increases. ReLU also has no vanishing-gradient problem for positive inputs,
making training with Adam stable.

**Output and probability interpretation.** The final layer produces an
unconstrained scalar $z \in \mathbb{R}$, the **logit**. The probability of
stability is recovered by the sigmoid function:

$$P(\text{stable} \mid \theta) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

The sign of $z$ alone determines the classification:
- $z > 0 \Leftrightarrow P > 0.5$: predict stable.
- $z \leq 0 \Leftrightarrow P \leq 0.5$: predict unstable.

**Preprocessing.** Before training, the inputs are standardised with a
`StandardScaler`:

$$\tilde{\theta}^{(j)} = \frac{\theta^{(j)} - \mu^{(j)}}{\sigma^{(j)}}$$

where $\mu^{(j)}$ and $\sigma^{(j)}$ are the mean and standard deviation of
the $j$-th parameter computed on the training set only. This ensures all inputs
have zero mean and unit variance, preventing parameters with large ranges
(e.g. $\Omega_0$) from dominating the gradient updates of the first layer. The
fitted scaler is saved to `tpm_scaler.pkl` and **must be applied identically**
at inference time — the network weights are calibrated to normalised inputs and
are meaningless without it.

**Loss function.** The network is trained to minimise the
**binary cross-entropy with logits**:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N}
    \Bigl[ y_i \log \sigma(z_i) + (1-y_i) \log(1 - \sigma(z_i)) \Bigr]$$

This loss penalises confident wrong predictions exponentially: predicting
$z \to +\infty$ for a truly unstable point ($y = 0$) gives
$\mathcal{L} \to +\infty$. Using PyTorch's `BCEWithLogitsLoss` — which fuses
the sigmoid and the logarithm into a single numerically stable operation via
the log-sum-exp trick — avoids underflow when $|z|$ is large.

**Optimiser.** Adam (Adaptive Moment Estimation) is used with learning rate
$\alpha = 10^{-3}$. Adam maintains per-parameter running estimates of the
first moment $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ and second moment
$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ of the gradient, and applies
bias-corrected updates:

$$\Delta w = -\alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

This adapts the effective step size for each weight independently, making it
robust to the different curvatures of the loss along different parameter
directions — particularly useful when the stability boundary has very different
sharpness in the $\Omega_0$ - $c_0$ plane compared to the $\log_{10}a_T$ - $\sigma$
plane.

**Model selection.** The training set uses 80% of the labelled data, with the
remaining 20% held out as a validation set. Validation loss and accuracy are
computed every 100 epochs without gradient updates. The checkpoint with the
lowest validation loss is saved, rather than the last epoch, to prevent
overfitting.

### 9.7 The fast prior: logic and bias

`NNFastPrior` is registered with Cobaya as a `Likelihood`, making it part of
the posterior:

$$\log\mathcal{P}(\theta \mid d) = \log\mathcal{L}_\text{Planck}(\theta)+ \log\mathcal{L}_\text{BAO}(\theta) + \ldots+ \underbrace{\log p_\text{NN}(\theta)}_{\in \{0,\,-\infty\}}+ \log\pi(\theta)$$

The fast prior contributes $0$ if the network predicts stability and $-\infty$
otherwise. Crucially, within the stable region it leaves the posterior
**completely unmodified** — it acts purely as a projector onto the physically
admissible subspace, not as an informative prior.

The potential source of bias is at the **boundary**: the network has finite
accuracy, so some truly stable points near $\partial S$ may be classified as
unstable (false negatives) and rejected before EFTCAMB is called. The decision
threshold `z > 0` can be lowered (e.g. to `z > −1.0`, corresponding to
$P > 27\%$) to make the filter more conservative near the boundary at the cost
of passing more points to EFTCAMB. The correct threshold is a trade-off between
chain efficiency and boundary bias, and should be calibrated against a held-out
set of EFTCAMB evaluations near $\partial S$.

---

## 10. Performance Notes

| Configuration | Evaluations | Indicative wall time |
|---------------|-------------|---------------------|
| Smoke test (`--base 64`, 4 workers) | 64 | ~5 min |
| Production (`--base 4096` + $3 \times 1024$ refinement, 36 workers) | ~7200 | 4–12 h |
| NN inference (per MCMC step) | — | ~0.1 ms |
| EFTCAMB background (per MCMC step) | — | ~5–20 s |

The fast prior is most effective when the unstable region is large. If, for
example, 60% of proposed MCMC points fall in the unstable region, the fast
prior reduces EFTCAMB calls by ~60%, cutting MCMC wall time proportionally.

For background-only EFTCAMB calls (as in Phase 1), the parallelisable fraction
of the code is small (background solver is mostly serial, unlike the
perturbation and lensing modules). One OpenMP thread per worker
(`OMP_NUM_THREADS=1`) therefore maximises throughput by avoiding thread
synchronisation overhead. The recommended setup for a 40-core node is
`--workers 36` with `OMP_NUM_THREADS=1`, leaving 4 cores free for the OS and
the master Python process.

---

## 11. References

- Benevento, G. et al. 2022, *ApJ* **935**, 156 — TPM model: definition,
  EFT mapping, and first MCMC constraints.
- Kable, J. et al. 2023, *ApJ* **959**, 143 — TPM constraints with SPT data.
- Hu, B. et al. 2014 — EFTCAMB:
  [arXiv:1312.5742](https://arxiv.org/abs/1312.5742)
- Raveri, M. et al. 2014 — EFTCosmoMC:
  [arXiv:1405.1022](https://arxiv.org/abs/1405.1022)
- Frusciante, N. & Perenon, L. 2020, *Phys. Rep.* **857**, 1 — EFTCAMB review.
- Deffayet, C. et al. 2010, *JCAP* **10**, 026 — Kinetic gravity braiding.
- Joe, S. & Kuo, F. Y. 2008, *SIAM J. Sci. Comput.* — Sobol direction numbers.
- Kingma, D. P. & Ba, J. 2015 — Adam optimiser:
  [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

---

## Author

Lorenzo Baldazzi  
PhD student, Università degli Studi di Roma Tor Vergata  
2026
