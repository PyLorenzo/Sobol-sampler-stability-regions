"""
Author: Lorenzo Baldazzi
Affiliation: University of Rome Tor Vergata
Date: 2026-05-18

Stability-region mapper using Sobol quasi-Monte Carlo sampling.

Goal
----
For a modified gravity model implemented in EFTCAMB, identify the subset of
the parameter space for which EFTCAMB declares the model "stable" under chosen
stability flags. Fixed parameters are pinned at reference values from the
user's Cobaya YAML.

Algorithm
---------
1. Base Sobol QMC sample of N0 points in the parameter space (defined by priors).
2. Parallel stability evaluation via Cobaya: model.loglike(theta) is finite
   if EFTCAMB initialises successfully (= stable for chosen flags).
3. K-NN based boundary-uncertainty score; new Sobol points drawn inside the
   bounding box of the top decile of uncertain points. Repeat K times.
4. Pickle the labelled point cloud for downstream visualisation.

Usage
-----
    # Create a configuration file from the template
    cp config_template.yaml my_config.yaml
    # Edit my_config.yaml with your paths and parameter ranges

    # Quick smoke test (~ 1 minute on 4 cores)
    python tpm_stability_map.py --config my_config.yaml --base 64 --refine-iters 0

    # Full run (tens of minutes to hours depending on cores and parameters)
    python tpm_stability_map.py --config my_config.yaml --base 4096 --refine-iters 3 --refine-batch 1024

    # Using a custom output directory
    python tpm_stability_map.py --config my_config.yaml --outdir ./results

The script writes a pickle file containing the labelled samples; pass that to
`tpm_stability_viz.py` to plot the maps.
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import pickle
import sys
import time
from typing import Dict, Tuple

import numpy as np
import yaml
from scipy.stats import qmc


# ----------------------------------------------------------------------
# 1. CONFIGURATION HELPERS
# ----------------------------------------------------------------------

def load_config(yaml_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: dict) -> None:
    """Validate that required keys exist in the configuration."""
    required_keys = ["cobaya_yaml", "tpm_ranges", "tpm_fixed"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")


def get_tpm_ranges_and_fixed(config: dict) -> Tuple[Dict, Dict]:
    """Extract parameter ranges and fixed parameters from config."""
    tpm_ranges = config.get("tpm_ranges", {})
    tpm_fixed = config.get("tpm_fixed", {})
    
    if not tpm_ranges:
        raise ValueError("tpm_ranges cannot be empty in configuration")
    if not tpm_fixed:
        raise ValueError("tpm_fixed cannot be empty in configuration")
    
    # Convert lists to tuples if needed
    tpm_ranges = {k: tuple(v) if isinstance(v, list) else v 
                  for k, v in tpm_ranges.items()}
    
    return tpm_ranges, tpm_fixed


# Store TPM parameters globally for worker processes
_TPM_RANGES = None
_TPM_FIXED = None
_MODEL = None
_YAML_PATH = None


# ----------------------------------------------------------------------
# 2. BUILD A MINIMAL COBAYA MODEL
# ----------------------------------------------------------------------

def build_cobaya_info(yaml_path: str, tpm_ranges: Dict, tpm_fixed: Dict) -> dict:
    """
    Construct a Cobaya `info` dict whose loglike only fires the EFTCAMB
    stability check.

    Steps
    -----
    - Copy the `theory` block (CAMB + all extra_args) from the user's YAML.
    - Silence EFTCAMB feedback if you want to (avoid console spam in parallel).
    - Drop the user's likelihoods / sampler / output; install the no-op `one`
      likelihood instead -- loglike == 0 if all theories succeed, -inf else.
    - Declare every parameter as `sampled` with uniform priors covering their ranges.
    - Preserve the YAML's derived-parameter definitions (e.g. As <- logA)
      so that Cobaya can still satisfy CAMB's input requirements.
    """
    with open(yaml_path) as f:
        base = yaml.safe_load(f)

    info: dict = {
        "theory":     copy.deepcopy(base["theory"]),
        "likelihood": {"one": None},
        "params":     {},
        "stop_at_error": False,
    }

    # Silence CAMB / EFTCAMB feedback inside the workers.
    if "camb" in info["theory"]:
        info["theory"]["camb"].setdefault("extra_args", {})["feedback_level"] = 3

    # ---- Fixed parameters (sampled with uniform priors).
    # A uniform prior over [v-eps, v+eps] is enough because loglike never
    # consults the prior; we just need Cobaya to accept these names as inputs.
    for name, val in tpm_fixed.items():
        eps = max(abs(val), 1.0)
        info["params"][name] = {
            "prior": {"min": val - eps, "max": val + eps},
            "latex": name,
        }

    # ---- Variable parameters (sampled over their physical ranges).
    for name, (lo, hi) in tpm_ranges.items():
        info["params"][name] = {
            "prior": {"min": lo, "max": hi},
            "latex": name,
        }

    # ---- Carry over derived params from the user's YAML (logA -> As, etc.).
    # We skip any param that we are already declaring as a control parameter
    # to avoid redefining sampled vars.
    for name, item in base["params"].items():
        if name in info["params"]:
            continue
        if isinstance(item, dict) and ("derived" in item or "value" in item):
            info["params"][name] = item

    return info


def make_model(yaml_path: str, tpm_ranges: Dict, tpm_fixed: Dict):
    """Lazy import of Cobaya so workers can pickle this module."""
    from cobaya.model import get_model
    return get_model(build_cobaya_info(yaml_path, tpm_ranges, tpm_fixed))


# ----------------------------------------------------------------------
# 3. STABILITY CHECK
# ----------------------------------------------------------------------

def is_stable(model, theta: Dict[str, float]) -> bool:
    """
    Return True iff the theory declares the parameter point stable.

    `theta` must contain values for every control parameter.
    """
    try:
        ll = model.loglike(theta,
                           make_finite=False,
                           cached=False,
                           return_derived=False)
    except Exception:
        # Any unexpected exception (NaN propagation, CAMB error, etc.) =>
        # treat the point as unstable. We err on the safe side.
        return False
    return np.isfinite(ll)


# ----------------------------------------------------------------------
# 4. SOBOL SAMPLER
# ----------------------------------------------------------------------

def sobol_points(n: int,
                 ranges: Dict[str, Tuple[float, float]],
                 seed: int = 0) -> np.ndarray:
    """
    Generate n Sobol points in the box `ranges`.

    Mathematical note
    -----------------
    scipy.qmc.Sobol delivers its strongest balance properties when n is a
    power of 2. We therefore round n UP to the next power of two and then
    truncate to n. This is a deliberate trade-off: a perfectly balanced 2^m
    sequence is preferable, but the caller's `n` is respected as an upper
    bound on the number of points returned.
    """
    d = len(ranges)
    m = int(np.ceil(np.log2(max(n, 2))))      # at least 2 points
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = sampler.random_base2(m=m)             # shape (2^m, d), in [0,1]^d
    u = u[:n]
    lo = np.array([r[0] for r in ranges.values()])
    hi = np.array([r[1] for r in ranges.values()])
    return lo + (hi - lo) * u                 # shape (n, d)


# ----------------------------------------------------------------------
# 5. PARALLEL EVALUATION DRIVER
# ----------------------------------------------------------------------
# Each worker process keeps its own Cobaya model alive (heavy to build).
# We use globals because mp.Pool initialisers can't return values.

def _worker_init(yaml_path: str, tpm_ranges: Dict, tpm_fixed: Dict):
    """Build one Cobaya model per worker, once."""
    global _MODEL, _YAML_PATH, _TPM_RANGES, _TPM_FIXED
    _YAML_PATH = yaml_path
    _TPM_RANGES = tpm_ranges
    _TPM_FIXED = tpm_fixed
    _MODEL = make_model(yaml_path, tpm_ranges, tpm_fixed)


def _worker_eval(theta_row_with_names) -> bool:
    """Evaluate stability for a single point."""
    names, row = theta_row_with_names
    theta = dict(zip(names, row))
    # Add fixed parameters (constant for all calls)
    theta.update(_TPM_FIXED)
    return is_stable(_MODEL, theta)


def evaluate_points(points: np.ndarray,
                    yaml_path: str,
                    tpm_ranges: Dict,
                    tpm_fixed: Dict,
                    n_workers: int,
                    chunksize: int = 4) -> np.ndarray:
    """
    Evaluate stability on every row of `points`.

    Returns
    -------
    labels: ndarray of bool, shape (n,)
    """
    names = tuple(tpm_ranges.keys())
    payload = [(names, row) for row in points]

    if n_workers <= 1:
        # Serial fallback (also useful for debugging tracebacks).
        _worker_init(yaml_path, tpm_ranges, tpm_fixed)
        labels = [bool(_worker_eval(p)) for p in payload]
        return np.array(labels, dtype=bool)

    # `maxtasksperchild` recycles workers periodically to prevent memory
    # creep from CAMB's Fortran heap.
    with mp.Pool(processes=n_workers,
                 initializer=_worker_init,
                 initargs=(yaml_path, tpm_ranges, tpm_fixed),
                 maxtasksperchild=200) as pool:
        labels = pool.map(_worker_eval, payload, chunksize=chunksize)

    return np.array(labels, dtype=bool)


# ----------------------------------------------------------------------
# 6. BOUNDARY-DRIVEN ADAPTIVE REFINEMENT
# ----------------------------------------------------------------------

def boundary_uncertainty(points: np.ndarray,
                         labels: np.ndarray,
                         n_neighbors: int = 8) -> np.ndarray:
    """
    Compute an uncertainty score in [0, 1] for each point.

    Mathematical definition
    -----------------------
    Let f_i be the fraction of stable points among the k nearest neighbours
    of point i (excluding itself), where distances are computed in
    coordinates normalised to [0, 1]^d. Then

            u_i = 1 - |2 f_i - 1|

    so u_i = 0 in a homogeneous neighbourhood (all stable or all unstable)
    and u_i = 1 when exactly half the neighbours are stable -- the surest
    sign of being on the stability boundary.
    """
    from sklearn.neighbors import NearestNeighbors

    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = np.maximum(hi - lo, 1e-15)
    Xn = (points - lo) / span                 # shape (n, d) in [0,1]^d

    k = min(n_neighbors + 1, len(points))     # +1 because self is included
    nn = NearestNeighbors(n_neighbors=k).fit(Xn)
    _, idx = nn.kneighbors(Xn)
    neigh = labels[idx[:, 1:]]                # drop self -> (n, k-1)
    frac_stable = neigh.mean(axis=1)
    return 1.0 - np.abs(2.0 * frac_stable - 1.0)


def refine_box(points: np.ndarray,
               labels: np.ndarray,
               n_new: int,
               ranges: Dict[str, Tuple[float, float]],
               seed: int,
               top_quantile: float = 0.9,
               padding_frac: float = 0.05) -> np.ndarray:
    """
    Draw `n_new` Sobol points inside the bounding box of the most uncertain
    fraction of the labelled set.

    Steps
    -----
    1. Score each labelled point with `boundary_uncertainty`.
    2. Keep the points whose score is in the top (1 - top_quantile) quantile.
    3. Build the axis-aligned bounding box of those points.
    4. Expand it by padding_frac * extent on each side.
    5. Clip back to the physical ranges to stay inside the prior box.
    6. Sobol-sample inside the resulting sub-box.
    """
    score = boundary_uncertainty(points, labels)
    thr = np.quantile(score, top_quantile)
    sel = score >= thr
    if sel.sum() < 4:
        # not enough boundary points yet -- fall back to global Sobol
        return sobol_points(n_new, ranges, seed=seed)

    bpts = points[sel]
    lo = bpts.min(axis=0)
    hi = bpts.max(axis=0)
    pad = padding_frac * np.maximum(hi - lo, 1e-12)
    lo, hi = lo - pad, hi + pad

    rlo = np.array([r[0] for r in ranges.values()])
    rhi = np.array([r[1] for r in ranges.values()])
    lo = np.maximum(lo, rlo)
    hi = np.minimum(hi, rhi)

    local = {k: (lo[i], hi[i]) for i, k in enumerate(ranges)}
    return sobol_points(n_new, local, seed=seed)


# ----------------------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="Path to configuration YAML file with parameter ranges and paths")
    ap.add_argument("--yaml", default=None,
                    help="Override Cobaya YAML path from config (optional)")
    ap.add_argument("--base", type=int, default=4096,
                    help="Base Sobol sample size (default: %(default)s)")
    ap.add_argument("--refine-iters", type=int, default=3,
                    help="Number of boundary-refinement passes "
                         "(default: %(default)s; set 0 to disable)")
    ap.add_argument("--refine-batch", type=int, default=1024,
                    help="Points added per refinement pass "
                         "(default: %(default)s)")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="Number of parallel workers (default: ncpu-1)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Sobol seed (default: %(default)s)")
    ap.add_argument("--outdir", default=".",
                    help="Output directory for results (default: %(default)s)")
    ap.add_argument("--serial", action="store_true",
                    help="Run serially (useful for debugging)")
    args = ap.parse_args()

    # Load and validate configuration
    try:
        config = load_config(args.config)
        validate_config(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Get parameter ranges and fixed values from config
    tpm_ranges, tpm_fixed = get_tpm_ranges_and_fixed(config)
    
    # Get Cobaya YAML path (can be overridden by command line)
    cobaya_yaml = args.yaml if args.yaml else config["cobaya_yaml"]
    
    if not os.path.exists(cobaya_yaml):
        print(f"ERROR: Cobaya YAML not found: {cobaya_yaml}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    n_workers = 1 if args.serial else args.workers
    output_file = os.path.join(args.outdir, "stability_map.pkl")

    print("=" * 64)
    print(" Stability Region Mapper")
    print("=" * 64)
    print(f"   Config file     : {args.config}")
    print(f"   Cobaya YAML     : {cobaya_yaml}")
    print(f"   Workers         : {n_workers}")
    print(f"   Base Sobol      : {args.base}")
    print(f"   Refine passes   : {args.refine_iters} x {args.refine_batch}")
    print(f"   Output dir      : {args.outdir}")
    print()
    print("   Variable parameter ranges:")
    for k, (lo, hi) in tpm_ranges.items():
        print(f"      {k:16s} in [{lo:+.6f}, {hi:+.6f}]")
    print("   Fixed parameters:")
    for k, v in tpm_fixed.items():
        print(f"      {k:16s} = {v}")
    print()

    # ---- (1) Base global sample ----
    print("[1/3] Generating base Sobol sample ...")
    t0 = time.time()
    pts = sobol_points(args.base, tpm_ranges, seed=args.seed)
    print(f"      {pts.shape[0]} points; "
          f"Sobol size rounded to 2**{int(np.ceil(np.log2(max(args.base,2))))}.")

    print("[2/3] Evaluating base sample (parallel) ...")
    labels = evaluate_points(pts, cobaya_yaml, tpm_ranges, tpm_fixed, n_workers)
    dt = time.time() - t0
    print(f"      done in {dt:.1f} s "
          f"({dt / len(pts) * 1e3:.1f} ms/pt amortised, "
          f"stable fraction = {labels.mean():.3f})")

    all_pts = [pts]
    all_lab = [labels]
    iter_info = [{"phase": "base", "n": len(pts), "stable_frac": float(labels.mean())}]

    # ---- (2) Adaptive refinement ----
    for it in range(args.refine_iters):
        print(f"[3/3] Refinement pass {it + 1}/{args.refine_iters} ...")
        t0 = time.time()
        new = refine_box(
            np.vstack(all_pts),
            np.concatenate(all_lab),
            args.refine_batch,
            tpm_ranges,
            seed=args.seed + 100 + it,
        )
        new_lab = evaluate_points(new, cobaya_yaml, tpm_ranges, tpm_fixed, n_workers)
        dt = time.time() - t0
        all_pts.append(new)
        all_lab.append(new_lab)
        iter_info.append({"phase": f"refine_{it+1}",
                          "n": len(new),
                          "stable_frac": float(new_lab.mean())})
        print(f"      +{len(new)} points in {dt:.1f} s, "
              f"batch stable fraction = {new_lab.mean():.3f}")

    all_pts_arr = np.vstack(all_pts)
    all_lab_arr = np.concatenate(all_lab)

    # ---- (3) Save ----
    print(f"[done] Saving {len(all_pts_arr)} labelled points to {output_file}")
    with open(output_file, "wb") as f:
        pickle.dump({
            "param_names":   list(tpm_ranges.keys()),
            "param_ranges":  tpm_ranges,
            "fixed_params":  tpm_fixed,
            "points":        all_pts_arr,        # shape (N, d)
            "stable":        all_lab_arr,        # shape (N,)
            "iter_info":     iter_info,
            "cobaya_yaml":   os.path.abspath(cobaya_yaml),
        }, f)

    print()
    print(f"   Total evaluations : {len(all_pts_arr)}")
    print(f"   Global stable frac: {all_lab_arr.mean():.4f}")
    print()
    print(f"Next step:  python tpm_stability_viz.py {output_file}")


if __name__ == "__main__":
    # 'spawn' is safer than 'fork' when CAMB / EFTCAMB hold C/Fortran state.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    sys.exit(main())
