# TPM Stability Mapping & Neural Network Fast Prior

> **A three-phase pipeline for efficient ghost/gradient stability mapping of the Transitional Planck Mass (TPM) model in EFTCAMB, and for accelerating Cobaya MCMC chains via a neural network viability prior.**

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

The **Transitional Planck Mass (TPM)** model is a modified gravity theory implemented within the [EFTCAMB](http://www.eftcamb.org) framework (Hu et al. 2014; Raveri et al. 2014). Its background evolution is governed by four phenomenological parameters:

| Symbol | Script name | Physical meaning | Prior range |
|--------|-------------|-----------------|-------------|
| $\log_{10} a_T$ | `Log_aT` | Scale factor of the Planck-mass transition | $[-7.5,\ -3.5]$ |
| $\sigma$ | `sig` | Width (in e-folds) of the transition | $[0.4,\ 3.0]$ |
| $\Omega_0$ | `M` | Amplitude of the Planck-mass shift | $[-0.15,\ 0.015]$ |
| $c_0$ | `c` | Late-time kinetic braiding parameter | $[-0.1,\ 0.01]$ |

EFTCAMB enforces **ghost** and **gradient stability** conditions on the scalar perturbations at every step of an MCMC chain. For the TPM model these translate into:

- **No-ghost:** $\alpha_K + \tfrac{3}{2}\alpha_B^2 > 0$, always satisfied for $c_0 < 0$.
- **No-gradient:** $c_s^2 > 0$, which depends non-trivially on all four parameters through the full cosmic history.

Because an EFTCAMB stability evaluation takes $\mathcal{O}(10)$ s and a Cobaya MCMC chain proposes $\mathcal{O}(10^5)$ points, a large fraction of which lie in the unstable region, the stability check is a **computational bottleneck**. This repository provides a solution in three phases.

---

## 2. Pipeline Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — tpm_stability_map.py                                        │
│                                                                        │
│  Sobol QMC sampling of (Log_aT, sig, M, c) space                      │
│        ↓   parallel EFTCAMB evaluation (ghost + gradient flags)        │
│  Adaptive refinement near the stability boundary                       │
│        ↓                                                               │
│  tpm_stability_map.pkl   {points: (N,4), stable: (N,) bool}           │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────────┐
│  PHASE 2 — Stability_nn.py                                             │
│                                                                        │
│  StandardScaler normalisation + 80/20 train/val split                 │
│        ↓   MLP  4 → 64 → 64 → 32 → 1   (BCEWithLogitsLoss, Adam)     │
│  Best checkpoint selected on validation loss                           │
│        ↓                                                               │
│  tpm_stability_model.pt  (PyTorch weights)                             │
│  tpm_scaler.pkl          (fitted StandardScaler)                       │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────────────┐
│  PHASE 3 — NN_fast_prior_TPM.py  (Cobaya Likelihood)                  │
│                                                                        │
│  At each MCMC step:                                                    │
│    scale θ → forward pass → logit z                                   │
│    if z > 0  →  logp = 0.0    (pass to EFTCAMB)                       │
│    if z ≤ 0  →  logp = -∞    (reject immediately, skip EFTCAMB)       │
│        ↓                                                               │
│  ~10^5× speedup per rejected unstable point                           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
.
├── tpm_stability_map.py      # Phase 1: Sobol sampling + EFTCAMB labelling
├── tpm_stability_viz.py      # Visualisation: scatter corner + heatmap corner
├── Stability_nn.py           # Phase 2: MLP training
├── NN_fast_prior_TPM.py      # Phase 3: Cobaya Likelihood wrapper
├── TPM.yaml                  # Reference Cobaya YAML (MCMC configuration)
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

Install the Python dependencies (excluding the EFTCAMB-patched CAMB):

```bash
pip install torch scikit-learn scipy numpy matplotlib pyyaml
```

### EFTCAMB

This pipeline requires the EFTCAMB-patched version of CAMB that includes the TPM model (`007p8_TPM.f90`). Point Cobaya to your build via the `path` field in the YAML:

```yaml
theory:
  camb:
    path: /path/to/your/eftcamb
```

---

## 5. Phase 1 — Sobol Stability Mapping

### What it does

Evaluates EFTCAMB ghost and gradient stability over the 4-D TPM parameter space using a Sobol low-discrepancy sequence, followed by adaptive boundary refinement.

The six ΛCDM parameters are fixed at the centres of their `ref` distributions in `TPM.yaml`:

| Parameter | Fixed value |
|-----------|------------|
| `ombh2`  | 0.0224 |
| `omch2`  | 0.118  |
| `tau`    | 0.055  |
| `logA`   | 3.05   |
| `ns`     | 0.965  |
| `H0`     | 72.0   |

### Usage

```bash
# Smoke test (~1–5 min, verifies the pipeline end-to-end)
python tpm_stability_map.py \
    --yaml TPM.yaml \
    --base 64 \
    --refine-iters 0 \
    --workers 4

# Production run (recommended settings)
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

### Parallelism and thread control

Each worker process runs one CAMB instance. To avoid OpenMP oversubscription, set:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

before launching the script. On a 40-core node, `--workers 36` (reserving 4 cores for the OS and master process) is a safe default.

### SLURM example

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python tpm_stability_map.py \
    --yaml TPM.yaml \
    --base 4096 \
    --refine-iters 3 \
    --workers 36
```

---

## 6. Phase 2 — Neural Network Training

### What it does

Trains a binary classifier (MLP) on the labelled point cloud from Phase 1. The trained network learns the boundary $\partial S$ that separates the stable region $S$ from the unstable one, and can predict stability for any new point in $\sim 0.1$ ms.

### Architecture

```
Input (4)  →  Linear(64)  →  ReLU
           →  Linear(64)  →  ReLU
           →  Linear(32)  →  ReLU
           →  Linear(1)              ← raw logit z ∈ ℝ
```

The probability of stability is recovered as $P(\text{stable} \mid \theta) = \sigma(z) = (1 + e^{-z})^{-1}$.

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
| `tpm_scaler.pkl` | Fitted `StandardScaler` — **must be used at inference** |

### Training diagnostics

The script prints validation loss and accuracy every 100 epochs. A good model typically reaches validation accuracy $> 95\%$ on a dataset of $\mathcal{O}(7000)$ points. If accuracy is lower, try increasing `--base` in Phase 1 (more training data) or `--epochs` here.

---

## 7. Phase 3 — Fast Prior in Cobaya

### What it does

`NN_fast_prior_TPM.py` defines a Cobaya `Likelihood` subclass that acts as a **fast viability gate**: it intercepts every proposed MCMC point, runs it through the neural network ($\sim 0.1$ ms), and rejects it with $\log p = -\infty$ if the network predicts instability — without ever calling EFTCAMB.

```python
def logp(self, **params_values):
    # normalise → forward pass → logit z
    if z > 0:
        return 0.0      # stable: pass to EFTCAMB and likelihoods
    else:
        return -np.inf  # unstable: reject immediately
```

### YAML configuration

Add the fast prior to your existing Cobaya YAML under `likelihood`:

```yaml
likelihood:
  # --- Physical likelihoods (unchanged) ---
  planck_2018_highl_plik.TTTEEE_lite_native: null
  act_dr6_lenslike.ACTDR6LensLike:
    variant: actplanck_baseline
    lens_only: false
  planck_2018_lowl.TT: null
  planck_2018_lowl.EE: null
  bao.desi_dr2.desi_bao_lrg1: null
  sn.desdovekie: null

  # --- Neural network fast prior (Phase 3) ---
  NN_fast_prior_TPM.NNFastPrior:
    python_path: /path/to/this/repo
    model_path:  /path/to/tpm_stability_model.pt
    scaler_path: /path/to/tpm_scaler.pkl
```

`python_path` tells Cobaya where to find `NN_fast_prior_TPM.py`. The other two paths point to the artefacts produced in Phase 2.

### Threshold tuning

The default threshold `z > 0` corresponds to $P(\text{stable}) > 50\%$. Near the stability boundary, the network is uncertain and may produce **false negatives** (stable points incorrectly rejected). To make the filter more tolerant:

```python
# In NN_fast_prior_TPM.py, logp():
if output.item() > -1.0:   # P(stable) > 27% — lets boundary cases through
    return 0.0
```

This trades a slight increase in EFTCAMB calls for a reduction in boundary bias. The optimal value depends on your network's calibration and on how sharp the physical stability boundary is.

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
| `tpm_stability_scatter.png` | Corner scatter: each panel shows a 2-D projection of the labelled points (green = stable, red = unstable). Diagonal panels show 1-D histograms. |
| `tpm_stability_heatmap.png` | Corner heatmap: each panel shows $\hat{p}(x_i, x_j)$, the fraction of stable points in each 2-D bin, **marginalised over the remaining two parameters**. Colour scale from red ($\hat{p}=0$) to green ($\hat{p}=1$). |

> **Note:** the heatmap requires a minimum of `min-count` points per bin to display a colour (grey = masked). With 64 points (smoke test) all 2-D panels will appear grey. Use `--bins 4 --min-count 2` to get a coarse but visible map from a small sample.

### Interpreting the heatmap

The value $\hat{p}(x_i, x_j)$ answers:

> *"If I fix parameters $x_i$ and $x_j$ in this bin, what fraction of the remaining parameter space is stable?"*

$\hat{p} = 1$ (dark green): the model is stable for **all** values of the other two parameters.  
$\hat{p} = 0$ (red): the model is **never** stable, regardless of the other two parameters.  
$\hat{p} \approx 0.5$ (yellow): the stability boundary passes through this region.

---

## 9. Mathematical Background

### Sobol discrepancy

A Sobol sequence of $N$ points in $d$ dimensions has star-discrepancy

$$D_N^* \sim \frac{(\log N)^d}{N}$$

compared to $D_N^* \sim N^{-1/2}$ for random Monte Carlo. This means that for the same $N$, Sobol points cover the parameter space more uniformly, requiring far fewer evaluations than a regular grid ($N^d$ evaluations) to achieve comparable coverage.

### Boundary uncertainty score

After labelling a set of points $\{\theta_i, y_i\}$, each point receives a score

$$u_i = 1 - \left| 2\bar{f}_i - 1 \right|, \qquad \bar{f}_i = \frac{1}{k}\sum_{j \in \mathcal{N}_k(i)} y_j$$

where $\mathcal{N}_k(i)$ are the $k$ nearest neighbours of $\theta_i$ in normalised coordinates. Points deep inside a stable or unstable region have $u_i \approx 0$; points on the boundary have $u_i \approx 1$.

### Binary cross-entropy loss

The MLP is trained to minimise

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log \sigma(z_i) + (1-y_i)\log(1-\sigma(z_i)) \right]$$

where $z_i$ is the raw logit output and $\sigma(z) = (1+e^{-z})^{-1}$. Using `BCEWithLogitsLoss` (which fuses the sigmoid and the log) avoids numerical underflow near $z \to \pm\infty$.

---

## 10. Performance Notes

| Configuration | Evaluations | Wall time (indicative) |
|---------------|-------------|----------------------|
| Smoke test (`--base 64`, 4 workers) | 64 | ~5 min |
| Production (`--base 4096`, 3 × 1024 refinement, 36 workers) | ~7200 | ~4–12 h |
| NN inference (per MCMC step) | — | ~0.1 ms |
| EFTCAMB (per MCMC step) | — | ~5–20 s |

The neural network fast prior is most effective when the unstable region is large. If, for example, 60% of proposed MCMC points fall in the unstable region, the fast prior reduces the number of EFTCAMB calls by approximately 60%, cutting total MCMC wall time accordingly.

### Thread configuration (single node)

For $N_w$ workers on a node with $C$ cores:

$$N_w \times N_\mathrm{OMP} \leq C$$

For background-only EFTCAMB calls (as in Phase 1), $N_\mathrm{OMP} = 1$ (i.e. `OMP_NUM_THREADS=1`) maximises throughput because the parallelisable fraction of the background solver is small. The recommended setup for a 40-core node is therefore `--workers 36` with `OMP_NUM_THREADS=1`.

---

## 11. References

- Benevento, G. et al. 2022, *ApJ* 935, 156 — TPM model definition and first constraints.
- Kable, J. et al. 2023, *ApJ* 959, 143 — TPM constraints with SPT data.
- Hu, B. et al. 2014 — EFTCAMB: [arXiv:1312.5742](https://arxiv.org/abs/1312.5742)
- Raveri, M. et al. 2014 — EFTCosmoMC: [arXiv:1405.1022](https://arxiv.org/abs/1405.1022)
- Frusciante, N. & Perenon, L. 2020, *Phys. Rep.* 857, 1 — EFTCAMB review.
- Torquato, S. & Harayama, F.H. 2004 — Sobol sequences and low-discrepancy sampling.
- Deffayet, C. et al. 2010, *JCAP* 10, 026 — Kinetic gravity braiding.

---

## Author

Lorenzo Baldazzi  
PhD student, Università di Bologna  
2026
