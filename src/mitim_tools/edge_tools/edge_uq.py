"""
edge_uq.py
----------
Offline Monte Carlo calibration of edge + transport uncertainties for PORTALS-edge.

Two-component decomposition
---------------------------
sigma_target  : Uncertainty in the POWER BALANCE TARGETS (QeMWm2, QiMWm2, etc.)
                caused by uncertain edge-model assumptions:
                G0, ne_target, Te_target, neutral_source_rate,
                impurity_source_rate, D0_imp, V0_imp.
                Measured by running powerstate_edge.calculate() N times with
                sampled edge params and recording how targets vary.

sigma_transport : Uncertainty in the TRANSPORT MODEL PREDICTIONS (QeMWm2_tr, etc.)
                  Two sub-components:
                  (a) sigma_transport_edge: transport sensitivity to the same 7 edge
                      params — measured from the same N calculate() runs.
                  (b) sigma_transport_mult: analytic uncertainty from multiplicative
                      model errors f_qe, f_qi, f_mom (each ~10% rel std):
                      sigma_mult[ch, cp] = f_std_ch * mean(Q_tr[ch, cp])

sigma_transport_total² = sigma_transport_edge² + sigma_transport_mult²
sigma_residual²        = sigma_target²          + sigma_transport_total²

Workflow
--------
Phase 1 (offline, run once, ~100s of TGLF calls):
  1. build_phase1_uq_spec(rel_std)       – define 7 edge-param distributions
  2. run_phase1_mc(powerstate, N, ...)   – MC loop: sample → calculate() → record
  3. calibration saved to JSON

Phase 2 (per-run, in PORTALS BO loop):
  inflate_powerstate_stds_with_edge_uq() – inject sigma_target + sigma_transport
"""

import numpy as np
import torch
import copy
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import json

from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print


# ============================================================================
# Input uncertainty specification
# ============================================================================

class EdgeInputUncertaintySpec:
    """
    Specification of the 7 edge-parameter input distributions for Phase 1 MC.

    All parameters are sampled as RELATIVE MULTIPLIERS around their nominal
    values (nominal = 1.0 by convention), so the same spec works regardless
    of machine-specific absolute values.  The offline workflow multiplies
    the base powerstate's stored option values by the drawn factors.

    Transport multipliers (f_qe, f_qi, f_mom) are NOT included here because
    they are applied analytically in EdgeUQCalibration.sigma_transport_mult()
    without extra TGLF calls.

    Parameter groups
    ----------------
    "bc"      : G0, ne_target_rel, Te_target_rel
    "neutral" : neutral_source_rate_rel
    "cs"      : impurity_source_rate_rel, D0_imp_rel, V0_imp_rel
    """

    def __init__(self):
        self.spec: Dict[str, Dict] = {}

    def add_param(
        self,
        name: str,
        nominal: float,
        rel_std: float,
        dist: str = "lognormal",
        group: str = "edge_param",
    ):
        """
        Add one uncertain parameter using a relative standard deviation.

        Parameters
        ----------
        name : str
            Identifier (e.g. "G0", "ne_target_rel").
        nominal : float
            Baseline / centre value. For relative-multiplier params use 1.0.
        rel_std : float
            Fractional std, e.g. 0.10 for ±10 %.
            For lognormal  : log-std  ≈ rel_std  (exact for small rel_std).
            For normal     : std = abs(nominal) * rel_std.
        dist : str
            "lognormal" (positive quantities) or "normal".
        group : str
            Bookkeeping tag: "bc", "neutral", "cs", or "edge_param".
        """
        nominal = float(nominal)
        rel_std = float(rel_std)

        if dist == "lognormal":
            log_mean = np.log(abs(nominal)) if nominal != 0.0 else 0.0
            self.spec[name] = {
                "nominal": nominal,
                "dist_type": "lognormal",
                "param1": log_mean,
                "param2": rel_std,       # log-std ≈ rel_std for small rel_std
                "relative_std": rel_std,
                "group": group,
            }
        elif dist == "normal":
            std = abs(nominal) * rel_std if nominal != 0.0 else rel_std
            self.spec[name] = {
                "nominal": nominal,
                "dist_type": "normal",
                "param1": nominal,
                "param2": std,
                "relative_std": rel_std,
                "group": group,
            }
        else:
            raise ValueError(f"Unknown dist type '{dist}'. Use 'lognormal' or 'normal'.")

    # ------------------------------------------------------------------
    # Legacy helpers kept for backward compatibility
    # ------------------------------------------------------------------

    def add_edge_param(self, name, nominal, dist_type, param_mean_or_a,
                       param_std_or_b, category="edge_assumption"):
        self.spec[name] = {
            "nominal": float(nominal),
            "dist_type": dist_type,
            "param1": float(param_mean_or_a),
            "param2": float(param_std_or_b),
            "group": category,
        }

    def add_plasma_param(self, name, nominal, relative_std):
        self.add_param(name, nominal, relative_std, dist="lognormal", group="plasma_input")

    # ------------------------------------------------------------------

    def sample(self, n_samples: int, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Draw n_samples from the joint (independent) input distribution."""
        rng = np.random.default_rng(seed)
        samples = {}
        for name, sp in self.spec.items():
            if sp["dist_type"] == "lognormal":
                samples[name] = np.exp(rng.normal(sp["param1"], sp["param2"], n_samples))
            elif sp["dist_type"] == "normal":
                samples[name] = rng.normal(sp["param1"], sp["param2"], n_samples)
            elif sp["dist_type"] == "uniform":
                samples[name] = rng.uniform(sp["param1"], sp["param2"], n_samples)
            else:
                raise ValueError(f"Unknown dist_type '{sp['dist_type']}'")
        return samples

    def to_dict(self) -> Dict:
        return copy.deepcopy(self.spec)

    @classmethod
    def from_dict(cls, spec_dict: Dict) -> "EdgeInputUncertaintySpec":
        obj = cls()
        obj.spec = copy.deepcopy(spec_dict)
        return obj


# ============================================================================
# Full-stack output extractor
# ============================================================================

class FullStackOutputExtractor:
    """
    Record per-channel, per-rhoCP transport and target quantities after
    each powerstate_edge.calculate() call.

    Output naming convention
    ------------------------
    "target_{ch}_cp{i}"    : power-balance target flux for channel ``ch``
                             at rhoCP index ``i``.
                             Keys: QeMWm2  (te), QiMWm2  (ti), Ce  (ne),
                                   MtJm2   (w0), CZ       (nZ).
    "transport_{ch}_cp{i}" : transport model prediction for the same.
                             Keys: QeMWm2_tr, QiMWm2_tr, Ce_tr, MtJm2_tr, CZ_tr.

    sigma_target[ch, cp]       = std of target outputs over N MC samples
    sigma_transport_edge[ch,cp]= std of transport outputs over N MC samples
    sigma_transport_mult[ch,cp]= analytic: f_std * mean(transport) [from calib]
    """

    def __init__(self):
        self.samples: List[Dict[str, float]] = []

    def append_sample(self, powerstate):
        """
        Extract and record fluxes at rhoCP from a *completed* powerstate.

        Parameters
        ----------
        powerstate : powerstate_edge
            Must have been through calculate() so that QeMWm2, QeMWm2_tr, …
            are populated in powerstate.plasma.
        """
        out = {}
        profile_map = powerstate.profile_map   # e.g. {"te": ("QeMWm2", "QeMWm2_tr"), …}

        for ch, (target_key, transport_key) in profile_map.items():
            if ch not in powerstate.predicted_channels:
                continue

            for group, key in [("target", target_key), ("transport", transport_key)]:
                if key not in powerstate.plasma:
                    continue
                # Interpolate fine-grid tensor down to rhoCP
                val_cp = powerstate._interp_tensor_from_rho_to_rhoCP(
                    powerstate.plasma[key]
                )  # (batch, n_cp)
                vals = val_cp[0].detach().cpu().numpy()
                for i, v in enumerate(vals):
                    out[f"{group}_{ch}_cp{i}"] = float(v)

        self.samples.append(out)

    def to_array(self) -> Tuple[np.ndarray, List[str]]:
        if not self.samples:
            raise ValueError("No samples recorded")
        output_names = sorted(self.samples[0].keys())
        Y = np.array([[s.get(n, np.nan) for n in output_names] for s in self.samples])
        return Y, output_names


# Legacy extractor kept for backward compatibility
class EdgeModuleOutputExtractor:
    """
    (Legacy) Collect scalar outputs from edge-module-only evaluations.
    Replaced by FullStackOutputExtractor for Phase 1 full-stack MC.
    """

    def __init__(self):
        self.samples: List[Dict[str, float]] = []

    def append_sample(self, bc_dict: Dict, plasma_dict: Dict, n_sample: int):
        out = {}
        for key in ["ne", "te", "ti", "aLne", "aLte", "aLti"]:
            if key in bc_dict:
                val = bc_dict[key]
                if isinstance(val, torch.Tensor):
                    val = float(val.detach().cpu().numpy())
                out[f"bc_{key}"] = float(val)
        for key in ["n0", "S_ion_main", "nu_ioniz_main", "tau_n0"]:
            if key in plasma_dict:
                val = plasma_dict[key]
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().numpy()
                val = np.atleast_1d(val)
                out[f"neu_{key}_mean"] = float(np.mean(val))
                out[f"neu_{key}_max"]  = float(np.max(val))
        for key in ["nz_all", "qrad_aurora", "Zeff"]:
            if key in plasma_dict:
                val = plasma_dict[key]
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().numpy()
                val = np.atleast_1d(val)
                out[f"imp_{key}_mean"] = float(np.mean(val))
                out[f"imp_{key}_max"]  = float(np.max(val))
        self.samples.append(out)

    def to_array(self) -> Tuple[np.ndarray, List[str]]:
        if not self.samples:
            raise ValueError("No samples recorded")
        output_names = sorted(self.samples[0].keys())
        Y = np.array([[s.get(n, np.nan) for n in output_names] for s in self.samples])
        return Y, output_names


# ============================================================================
# Calibration engine
# ============================================================================

class EdgeUQCalibration:
    """
    Offline MC calibration: edge-param samples → full-stack evaluate → sigma.

    Stores three sigma dictionaries (all keyed by "{ch}_cp{i}"):
      sigma_target          – std of power-balance targets over MC samples
      sigma_transport_edge  – std of transport predictions over MC samples
                              (sensitivity to edge params via changed BC/neutrals)
      sigma_transport_mult  – analytic: f_std * mean(Q_tr)
                              (set by set_transport_multiplier_sigma())

    Combined:
      sigma_transport_total²  = sigma_transport_edge²  + sigma_transport_mult²
      sigma_residual²         = sigma_target²           + sigma_transport_total²
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.input_spec: Optional[EdgeInputUncertaintySpec] = None
        self.samples:    Optional[Dict[str, np.ndarray]]    = None
        self.Y:          Optional[np.ndarray]               = None  # (N, n_out)
        self.output_names: Optional[List[str]]              = None

        # Sigma results
        self.sigma_target:          Dict[str, float] = {}
        self.sigma_transport_edge:  Dict[str, float] = {}
        self.sigma_transport_mult:  Dict[str, float] = {}
        self.sigma_transport_total: Dict[str, float] = {}
        self.sigma_residual:        Dict[str, float] = {}

        # PCA (optional, for compact representation)
        self.mu:        Optional[np.ndarray] = None
        self.sigma_arr: Optional[np.ndarray] = None
        self.var_ratio: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    def generate_samples(
        self,
        input_spec: EdgeInputUncertaintySpec,
        n_samples: int,
        seed: int = 42,
    ):
        """Draw MC samples from the edge-parameter input specification."""
        self.input_spec = input_spec
        self.samples = input_spec.sample(n_samples, seed=seed)
        print(
            f"[EdgeUQ] Generated {n_samples} MC samples from "
            f"{len(input_spec.spec)} parameters",
            typeMsg="i",
        )

    # ------------------------------------------------------------------
    # Register extractor outputs
    # ------------------------------------------------------------------

    def register_outputs(self, extractor: FullStackOutputExtractor):
        """Convert extractor samples to the internal Y array."""
        self.Y, self.output_names = extractor.to_array()
        print(
            f"[EdgeUQ] Registered {self.Y.shape[0]} samples × "
            f"{self.Y.shape[1]} outputs",
            typeMsg="i",
        )

    # ------------------------------------------------------------------
    # Sigma computation
    # ------------------------------------------------------------------

    def compute_sigma(self):
        """
        Compute empirical sigma_target and sigma_transport_edge from Y.

        Splits output_names by prefix ("target_" vs "transport_") and
        computes std over the MC ensemble for each column.
        """
        if self.Y is None or self.output_names is None:
            raise ValueError("Must call register_outputs() first.")

        for j, name in enumerate(self.output_names):
            col_std = float(np.nanstd(self.Y[:, j]))
            if name.startswith("target_"):
                key = name[len("target_"):]
                self.sigma_target[key] = col_std
            elif name.startswith("transport_"):
                key = name[len("transport_"):]
                self.sigma_transport_edge[key] = col_std

        # Combine (transport_mult added separately via set_transport_multiplier_sigma)
        self._recompute_totals()

        n_t  = len(self.sigma_target)
        n_tr = len(self.sigma_transport_edge)
        print(
            f"[EdgeUQ] sigma_target: {n_t} values, "
            f"sigma_transport_edge: {n_tr} values",
            typeMsg="i",
        )

    def set_transport_multiplier_sigma(
        self,
        f_qe_rel_std:  float = 0.10,
        f_qi_rel_std:  float = 0.10,
        f_mom_rel_std: float = 0.10,
    ):
        """
        Compute analytic sigma_transport_mult from transport multiplier uncertainty.

        For each channel/rhoCP, sigma_mult = f_std * |mean(Q_transport)|.

        Parameters
        ----------
        f_qe_rel_std, f_qi_rel_std, f_mom_rel_std : float
            Relative std on f_qe, f_qi, f_mom (e.g. 0.10 for 10 %).
        """
        if self.Y is None or self.output_names is None:
            raise ValueError("Must call register_outputs() before this.")

        channel_fstd = {"te": f_qe_rel_std, "ti": f_qi_rel_std, "w0": f_mom_rel_std}

        for j, name in enumerate(self.output_names):
            if not name.startswith("transport_"):
                continue
            key = name[len("transport_"):]   # e.g. "te_cp0"
            ch  = key.split("_cp")[0]        # e.g. "te"
            f_std = channel_fstd.get(ch, 0.0)
            mean_q = abs(float(np.nanmean(self.Y[:, j])))
            self.sigma_transport_mult[key] = f_std * mean_q

        self._recompute_totals()

    def _recompute_totals(self):
        """Recompute combined sigma values after any sigma update."""
        all_keys = set(self.sigma_target) | set(self.sigma_transport_edge)
        for key in all_keys:
            s_tgt  = self.sigma_target.get(key, 0.0)
            s_tedge = self.sigma_transport_edge.get(key, 0.0)
            s_tmult = self.sigma_transport_mult.get(key, 0.0)
            s_ttotal = np.sqrt(s_tedge**2 + s_tmult**2)
            self.sigma_transport_total[key] = float(s_ttotal)
            self.sigma_residual[key] = float(np.sqrt(s_tgt**2 + s_ttotal**2))

    # ------------------------------------------------------------------
    # PCA (optional, for variance decomposition diagnostics)
    # ------------------------------------------------------------------

    def fit_pca(
        self,
        n_components: Optional[int] = None,
        explained_var_threshold: float = 0.95,
    ):
        """Optional PCA on the full output ensemble for diagnostics."""
        if self.Y is None:
            raise ValueError("Must call register_outputs() first.")
        self.mu = np.nanmean(self.Y, axis=0)
        self.sigma_arr = np.nanstd(self.Y, axis=0)
        Y_std = (self.Y - self.mu) / (self.sigma_arr + 1e-10)
        C = np.cov(Y_std.T)
        lam, U = np.linalg.eigh(C)
        lam, U = lam[::-1], U[:, ::-1]
        explained = np.cumsum(lam) / np.sum(lam)
        if n_components is None:
            n_components = int(np.searchsorted(explained, explained_var_threshold)) + 1
        self.var_ratio = explained[:n_components]
        print(
            f"[EdgeUQ] PCA: {n_components} components explain "
            f"{self.var_ratio[-1]:.1%} variance",
            typeMsg="i",
        )

    # ------------------------------------------------------------------
    # Legacy helper
    # ------------------------------------------------------------------

    def empirical_sigma_model(self) -> Dict[str, float]:
        """Legacy: return sigma_residual dict (combined target + transport)."""
        if not self.sigma_residual:
            if self.Y is not None:
                self.compute_sigma()
            else:
                raise ValueError("No outputs registered.")
        return dict(self.sigma_residual)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_calibration(self, label: str = ""):
        """Save all sigma dicts and metadata to JSON."""
        save_dict = {
            "label": str(label),
            "n_samples":  int(self.Y.shape[0]) if self.Y is not None else 0,
            "n_outputs":  int(self.Y.shape[1]) if self.Y is not None else 0,
            "output_names": self.output_names or [],
            "sigma_target":           self.sigma_target,
            "sigma_transport_edge":   self.sigma_transport_edge,
            "sigma_transport_mult":   self.sigma_transport_mult,
            "sigma_transport_total":  self.sigma_transport_total,
            "sigma_residual":         self.sigma_residual,
            # Optional PCA fields
            "mu":        (self.mu.tolist()        if self.mu        is not None else None),
            "sigma_arr": (self.sigma_arr.tolist() if self.sigma_arr is not None else None),
            "var_ratio": (self.var_ratio.tolist() if self.var_ratio is not None else None),
        }
        filepath = self.output_dir / f"edge_uq_calib_{label}.json"
        with open(filepath, "w") as f:
            json.dump(save_dict, f, indent=2)
        print(f"[EdgeUQ] Saved calibration to {IOtools.clipstr(filepath)}", typeMsg="i")

    def load_calibration(self, label: str = ""):
        """Load calibration from JSON."""
        filepath = self.output_dir / f"edge_uq_calib_{label}.json"
        with open(filepath, "r") as f:
            data = json.load(f)
        self.output_names           = data.get("output_names", [])
        self.sigma_target           = data.get("sigma_target", {})
        self.sigma_transport_edge   = data.get("sigma_transport_edge", {})
        self.sigma_transport_mult   = data.get("sigma_transport_mult", {})
        self.sigma_transport_total  = data.get("sigma_transport_total", {})
        self.sigma_residual         = data.get("sigma_residual", {})
        mu = data.get("mu")
        sa = data.get("sigma_arr")
        vr = data.get("var_ratio")
        self.mu        = np.array(mu) if mu is not None else None
        self.sigma_arr = np.array(sa) if sa is not None else None
        self.var_ratio = np.array(vr) if vr is not None else None
        print(f"[EdgeUQ] Loaded calibration from {IOtools.clipstr(filepath)}", typeMsg="i")


# ============================================================================
# Per-run sigma injection
# ============================================================================

def inflate_powerstate_stds_with_edge_uq(
    powerstate,
    edge_uq_calib: EdgeUQCalibration,
    scale_factor: float = 1.0,
):
    """
    Inflate powerstate target and transport stds using the Phase 1 calibration.

    For each predicted channel ``ch`` and rhoCP index ``i``:
      - Inflates plasma[target_key + "_stds"]    by sigma_target["{ch}_cp{i}"]
      - Inflates plasma[transport_key + "_stds"] by sigma_transport_total["{ch}_cp{i}"]

    Both inflations are added in quadrature to any existing std values.

    Parameters
    ----------
    powerstate : powerstate_edge
        Modified in-place.  Must have been through calculateTargets() so that
        target _stds already exist.
    edge_uq_calib : EdgeUQCalibration
        Loaded calibration with sigma_target and sigma_transport_total populated.
    scale_factor : float
        Global multiplier on all sigma values (default 1.0).
    """
    plasma      = powerstate.plasma
    profile_map = powerstate.profile_map

    n_inflated = 0
    for ch, (target_key, transport_key) in profile_map.items():
        if ch not in powerstate.predicted_channels:
            continue

        n_cp = len(powerstate.rhoCP)
        for i in range(n_cp):
            cp_key = f"{ch}_cp{i}"

            # --- Target inflation ---
            s_tgt = edge_uq_calib.sigma_target.get(cp_key, 0.0) * scale_factor
            if s_tgt > 0.0 and target_key + "_stds" in plasma:
                stds = plasma[target_key + "_stds"]
                if stds.dim() == 2 and stds.shape[1] > i:
                    # Find rhoCP index on fine grid
                    rho_cp_val = float(powerstate.rhoCP[i])
                    rho_fine   = plasma["rho"][0].detach().cpu().numpy()
                    fine_idx   = int(np.argmin(np.abs(rho_fine - rho_cp_val)))
                    stds[:, fine_idx] = torch.sqrt(
                        stds[:, fine_idx] ** 2 + s_tgt ** 2
                    )
                    n_inflated += 1

            # --- Transport inflation ---
            s_tr = edge_uq_calib.sigma_transport_total.get(cp_key, 0.0) * scale_factor
            if s_tr > 0.0:
                tr_std_key = transport_key + "_stds"
                if tr_std_key in plasma:
                    stds = plasma[tr_std_key]
                    if stds.dim() == 2 and stds.shape[1] > i:
                        rho_cp_val = float(powerstate.rhoCP[i])
                        rho_fine   = plasma["rho"][0].detach().cpu().numpy()
                        fine_idx   = int(np.argmin(np.abs(rho_fine - rho_cp_val)))
                        stds[:, fine_idx] = torch.sqrt(
                            stds[:, fine_idx] ** 2 + s_tr ** 2
                        )
                        n_inflated += 1

    print(
        f"[EdgeUQ] Inflated {n_inflated} (channel, rhoCP) stds with edge-UQ calibration",
        typeMsg="i",
    )
