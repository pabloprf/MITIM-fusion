"""
LocalOptimaTools.py
-------------------
Utilities for post-batch local optima mining in MITIM's Bayesian optimization loop.

The core idea (Strategy 3):
  After every n_acq_batches_per_cycle acquisition iterations, scan the current GP
  surrogate for diverse local maxima of the scalarized posterior mean, evaluate those
  points with the physics model, and inject the results back into the training set.

Two-stage algorithm
~~~~~~~~~~~~~~~~~~~
Stage 1 — Gradient ascent:
  Run n_restarts independent L-BFGS-B optimizations of
      f(x) = scalarized_objective(E[GP(x)])
  from diverse starting points (Sobol samples + best observed training points).
  Each restart converges independently to its own local maximum of the surrogate.

Stage 2 — Diversity selection:
  From the n_restarts converged local optima, choose n_optima diverse candidates
  using the algorithm specified by ``diversity_algorithm``.

Supported diversity algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"greedy_max_min_distance" (default):
  Sort all converged solutions by objective value (descending).  Iteratively select
  the next-best candidate whose minimum distance (in normalized DV space) to all
  already-selected points is >= min_distance.  If fewer than n_optima diverse
  candidates are found, fill the remaining slots with the next-best solutions
  regardless of distance (so the returned tensor always has exactly n_optima rows).

Future algorithms (not yet implemented, add dispatch case in _select_diverse):
  "k_medoids"                  — K-medoids clustering on converged solutions
  "determinantal_point_process" — DPP sampling for maximum-volume subset
"""

import torch
import numpy as np
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from mitim_tools.opt_tools import BOTORCHtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
import botorch.acquisition.objective


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def mine_local_optima(
    step,
    bounds,
    scalarized_objective,
    train_X,
    train_Y,
    dfT,
    n_optima=10,
    n_restarts=256,
    n_from_training=32,
    min_distance=0.1,
    diversity_algorithm="greedy_max_min_distance",
    seed=None,
):
    """Find n_optima diverse local maxima of the GP surrogate posterior mean.

    Parameters
    ----------
    step : OPTstep
        A fitted MITIM BO step containing ``step.GP["combined_model"]``.
    bounds : OrderedDict
        Current DV bounds as {name: np.array([lo, hi])}.
    scalarized_objective : callable
        The user-defined scalarized objective.  Signature:
            (Y: Tensor[..., n_ofs]) -> (of, cal, res: Tensor[...])
    train_X : np.ndarray, shape (N, n_dvs)
        Current training inputs (unnormalized).
    train_Y : np.ndarray, shape (N, n_ofs)
        Current training outputs (unnormalized).
    dfT : torch.Tensor
        A dummy tensor whose dtype/device defines the computation context
        (obtained from MITIM_BO.dfT).
    n_optima : int
        Number of diverse local optima to return.
    n_restarts : int
        Number of independent gradient-ascent restarts.
    n_from_training : int
        How many of the best training points to include as warm starts.
    min_distance : float
        Minimum pairwise distance (normalized [0,1] DV space) for diversity.
    diversity_algorithm : str
        Name of the diversity-selection algorithm.  Currently only
        "greedy_max_min_distance" is supported.
    seed : int or None
        Seed used for Sobol restart generation. When provided, the local-optima
        restart pool is deterministic across runs.

    Returns
    -------
    torch.Tensor, shape (n_optima, n_dvs)
        Diverse local optima in unnormalized DV space, on dfT's device/dtype.
    """

    n_dvs = train_X.shape[1]
    dtype = dfT.dtype
    device = dfT.device

    # ------------------------------------------------------------------
    # 1. Build the scalarized posterior-mean acquisition function
    # ------------------------------------------------------------------
    gp_model = step.GP["combined_model"].gpmodel

    def _residual(Y, X=None):
        return scalarized_objective(Y)[2]

    objective = botorch.acquisition.objective.GenericMCObjective(_residual)

    acq = BOTORCHtools.PosteriorMean(gp_model, objective=objective)

    # ------------------------------------------------------------------
    # 2. Generate starting conditions
    # ------------------------------------------------------------------
    bounds_np = np.array([list(v) for v in bounds.values()]).T   # (2, n_dvs)
    bounds_tensor = torch.tensor(bounds_np, dtype=dtype, device=device)

    # Sobol samples
    n_sobol = max(0, n_restarts - n_from_training)
    sobol_ic = draw_sobol_samples(bounds_tensor, n=n_sobol, q=1, seed=seed).to(dtype=dtype, device=device)
    # shape: (n_sobol, 1, n_dvs)

    # Best training points as warm starts
    n_warm = min(n_from_training, len(train_X))
    if n_warm > 0:
        # Scalarize train_Y to find the best observed points
        with torch.no_grad():
            tY = torch.tensor(train_Y, dtype=dtype, device=device)
            _, _, scalar_obj = scalarized_objective(tY)
        best_idx = scalar_obj.argsort(descending=True)[:n_warm]
        warm_X = torch.tensor(train_X[best_idx.cpu().numpy()], dtype=dtype, device=device)
        warm_ic = warm_X.unsqueeze(1)   # (n_warm, 1, n_dvs)
        ic = torch.cat([warm_ic, sobol_ic], dim=0)  # (n_restarts, 1, n_dvs)
    else:
        ic = sobol_ic

    # Clamp starting points to bounds (Sobol already respects bounds;
    # training points may lie outside if bounds were expanded).
    ic = ic.clamp(
        bounds_tensor[0].unsqueeze(0).unsqueeze(0),
        bounds_tensor[1].unsqueeze(0).unsqueeze(0),
    )

    actual_restarts = ic.shape[0]
    print(
        f"\t[LocalOptima] Mining surrogate with {actual_restarts} restarts "
        f"({n_warm} warm-start + {actual_restarts - n_warm} Sobol)",
        typeMsg="i",
    )

    # ------------------------------------------------------------------
    # 3. Optimize acquisition from all starting conditions independently
    # ------------------------------------------------------------------
    try:
        with torch.no_grad():
            acq.eval()

        candidates, acq_values = optimize_acqf(
            acq_function=acq,
            bounds=bounds_tensor,
            q=1,
            num_restarts=actual_restarts,
            raw_samples=None,
            batch_initial_conditions=ic,
            return_best_only=False,
            options={
                "maxiter": 200,
                "batch_limit": min(actual_restarts, 32),
            },
        )
        # candidates : (actual_restarts, 1, n_dvs)
        # acq_values : (actual_restarts,)
    except Exception as e:
        print(
            f"\t[LocalOptima] optimize_acqf failed ({e}); "
            "returning best training points as fallback.",
            typeMsg="w",
        )
        # Fallback: return top-n_optima training points
        with torch.no_grad():
            tY = torch.tensor(train_Y, dtype=dtype, device=device)
            _, _, scalar_obj = scalarized_objective(tY)
        best_idx = scalar_obj.argsort(descending=True)[:n_optima]
        return torch.tensor(train_X[best_idx.cpu().numpy()], dtype=dtype, device=device)

    candidates = candidates.squeeze(1)  # (actual_restarts, n_dvs)

    # ------------------------------------------------------------------
    # 4. Diversity selection
    # ------------------------------------------------------------------
    selected = _select_diverse(
        candidates=candidates,
        acq_values=acq_values,
        bounds_tensor=bounds_tensor,
        n_optima=n_optima,
        min_distance=min_distance,
        algorithm=diversity_algorithm,
    )

    print(
        f"\t[LocalOptima] Selected {len(selected)} diverse local optima "
        f"(algorithm: {diversity_algorithm}, min_distance={min_distance})",
        typeMsg="i",
    )

    return torch.stack(selected).to(dtype=dtype, device=device)  # (n_optima, n_dvs)


# ---------------------------------------------------------------------------
# Diversity selection dispatcher
# ---------------------------------------------------------------------------

def _select_diverse(candidates, acq_values, bounds_tensor, n_optima, min_distance, algorithm):
    """Dispatch to the requested diversity-selection algorithm.

    Parameters
    ----------
    candidates : Tensor (n_restarts, n_dvs)    unnormalized
    acq_values : Tensor (n_restarts,)
    bounds_tensor : Tensor (2, n_dvs)
    n_optima : int
    min_distance : float                       in normalized [0,1] space
    algorithm : str

    Returns
    -------
    list of Tensor, length n_optima (each shape (n_dvs,), unnormalized)
    """
    if algorithm == "greedy_max_min_distance":
        return _greedy_max_min_distance(
            candidates, acq_values, bounds_tensor, n_optima, min_distance
        )
    else:
        raise ValueError(
            f"[LocalOptima] Unknown diversity_algorithm='{algorithm}'. "
            "Supported: 'greedy_max_min_distance'"
        )


def _greedy_max_min_distance(candidates, acq_values, bounds_tensor, n_optima, min_distance):
    """Greedy max-min distance diversity selection.

    Algorithm
    ---------
    1. Normalize all candidates to [0,1] using bounds.
    2. Sort by acquisition value (descending).
    3. Greedy pass: start with the best candidate; for each subsequent
       candidate, add it only if its minimum distance to all already-selected
       points is >= min_distance.
    4. If fewer than n_optima diverse candidates are found, fill the remaining
       slots with the next-best solutions regardless of distance.

    Returns list of n_optima unnormalized Tensors.
    """
    lo = bounds_tensor[0]   # (n_dvs,)
    hi = bounds_tensor[1]   # (n_dvs,)
    denom = (hi - lo).clamp(min=1e-12)

    # Normalize
    cands_norm = (candidates - lo) / denom  # (n_restarts, n_dvs)

    # Sort by acquisition value descending
    order = acq_values.argsort(descending=True)
    cands_sorted = cands_norm[order]           # normalized, sorted
    cands_orig_sorted = candidates[order]      # unnormalized, sorted

    selected_norm = []
    selected_orig = []

    for i in range(len(cands_sorted)):
        c_norm = cands_sorted[i]
        if len(selected_norm) == 0:
            selected_norm.append(c_norm)
            selected_orig.append(cands_orig_sorted[i])
        else:
            stacked = torch.stack(selected_norm)  # (k, n_dvs)
            dists = torch.norm(c_norm.unsqueeze(0) - stacked, dim=-1)  # (k,)
            if dists.min().item() >= min_distance:
                selected_norm.append(c_norm)
                selected_orig.append(cands_orig_sorted[i])

        if len(selected_orig) == n_optima:
            break

    # If not enough diverse points were found, fill with next-best regardless of distance
    if len(selected_orig) < n_optima:
        print(
            f"\t[LocalOptima] Only {len(selected_orig)} diverse candidates found "
            f"(min_distance={min_distance}); filling remaining {n_optima - len(selected_orig)} "
            "slots with next-best solutions.",
            typeMsg="w",
        )
        fill_count = 0
        for i in range(len(cands_orig_sorted)):
            if len(selected_orig) >= n_optima:
                break
            # Only fill with points not already selected
            c = cands_orig_sorted[i]
            already_in = any(torch.allclose(c, s) for s in selected_orig)
            if not already_in:
                selected_orig.append(c)
                fill_count += 1

        # Last resort: if still not enough (e.g. n_optima > n_restarts), duplicate the best
        while len(selected_orig) < n_optima:
            selected_orig.append(cands_orig_sorted[0])

    return selected_orig[:n_optima]
