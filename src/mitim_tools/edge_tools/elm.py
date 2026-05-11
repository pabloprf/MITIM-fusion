"""
elm.py  (edge ELM stability models)
------------------------------------
Peeling-ballooning (ELM) stability models for PORTALS-Edge.

All models share the same public interface::

    model = <Model>(options)
    model.solve(powerstate, batch_idx=0)
    f_elm   = model.elm_factor        # (rho,) penalty multiplier on turbulent fluxes
    in_elm  = model.in_elm_region     # (rho,) bool — True where alpha_MHD > alpha_crit

The returned ``elm_factor`` is a per-radial-surface multiplicative factor that
should be applied to turbulent (not neoclassical) transport fluxes whenever the
local pressure gradient exceeds the peeling-ballooning stability boundary.

ELM physics
-----------
Type-I ELMs are driven by the combined peeling-ballooning instability in the
pedestal.  The stability boundary lies in the (s_hat, alpha_MHD) plane and is
bounded from above (ballooning limit) and from the right (peeling limit due to
bootstrap current).

Here we implement a simplified analytic boundary based on the s-alpha stability
diagram (Connor-Hastie ideal MHD ballooning) with empirical shaping corrections,
following the standard ASTRA/JINTRAC parameterization.

Normalised MHD pressure gradient
---------------------------------
Starting from the GACODE/powerstate definition of ``p_prime``::

    p_prime = (mu_0/4pi) * q * a^2 / r / B_unit^2 * dp/dr       [dimensionless]

the standard MHD alpha parameter is::

    alpha_MHD = -2 * mu_0 * R * q^2 / B^2 * dp/dr
              = -8*pi * Rmajoa * roa * q * p_prime              [dimensionless]

Because ``p_prime < 0`` for inward-decreasing pressure, ``alpha_MHD > 0``.

Magnetic shear
--------------
The powerstate ``s_q`` is related to the standard magnetic shear ``s_hat`` by::

    s_q   = (q / roa)^2 * s_hat     (GACODE convention)
    s_hat = s_q * (roa / q)^2

Stability boundary
------------------
Three components contribute:

1.  **Ideal ballooning** (pressure driven, high-n) — Connor-Hastie first
    stability boundary::

        alpha_bal = |s_hat| * geo_factor
        geo_factor = (1 + 1.5 * sqrt(eps)) * (1 + 2 * |delta|)

    where ``eps = roa / Rmajoa`` is the local inverse aspect ratio and
    ``delta`` is the triangularity.  The geometry factor accounts for the
    stabilising effects of toroidicity and triangularity.

2.  **Peeling** (current driven, low-n) — the parallel edge current exceeds
    the kink-mode threshold when::

        s_hat > s_peel = s_peel_frac * alpha_MHD / max(alpha_bal, 1e-3)

    This approximates the upper-right peeling boundary in the s-alpha plane.

3.  **Combined** — the region is ELM-unstable if *either* criterion fires::

        in_elm = (alpha_MHD > alpha_bal) | (s_hat > s_peel)

Penalty factor
--------------
The stiff-transport penalty escalates smoothly above the boundary::

    elm_overshoot = max(0, alpha_MHD / alpha_crit - 1)
    elm_factor    = 1 + stiffness * elm_overshoot ^ power

where ``alpha_crit = min(alpha_bal, alpha_bal_from_peeling)``.  The penalty
factor is broadcast to the same shape as the turbulent flux tensors
``(batch, rho)``.  It is only applied in the pedestal region (roa >= roa_min).

Available models
----------------
"Null"        — no-op; all penalty factors = 1.0 (default)
"AnalyticPB"  — inline s-alpha peeling-ballooning stability criterion above

Model options (AnalyticPB)
--------------------------
stiffness           : float, default 10.0
    Strength of the transport penalty above the stability boundary.
stiffness_power     : float, default 1.0
    Exponent for how quickly the penalty rises with overshoot.
s_hat_min           : float, default 0.1
    Minimum |s_hat| used in computing alpha_bal (avoids division issues at
    low-shear surfaces near the axis).
s_peel_frac         : float, default 1.5
    Multiplier in the peeling shear threshold (larger → harder peeling
    trigger, i.e.  only extremely stiff current gradients cause peeling).
roa_min             : float, default 0.8
    Innermost normalized radius (r/a) at which the ELM penalty is applied.
    Surfaces inside this radius are left untouched.
geometry_correction : bool, default True
    Whether to apply the triangularity / toroidal-geometry correction factor.
verbose             : bool, default False
"""

import math
import hashlib
import json
import os
import shutil
import torch
import numpy as np
from pathlib import Path
from mitim_tools.misc_tools.LOGtools import printMsg as print

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ElmStability:
    """Abstract base for ELM stability models."""

    def __init__(self, options: dict):
        self.verbose = options.get("verbose", False)
        self.elm_factor: torch.Tensor | None = None   # (rho,) – set by solve()
        self.in_elm_region: torch.Tensor | None = None
        self.alpha_MHD: torch.Tensor | None = None
        self.alpha_crit: torch.Tensor | None = None

    def solve(self, powerstate, batch_idx: int = 0) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Null model
# ---------------------------------------------------------------------------

class NullElm(ElmStability):
    """No-op stability model.  All fluxes remain unchanged."""

    def __init__(self, options: dict):
        super().__init__(options)

    def solve(self, powerstate, batch_idx: int = 0) -> None:
        n_rho = powerstate.plasma["roa"].shape[-1]
        device = powerstate.plasma["roa"].device
        dtype  = powerstate.plasma["roa"].dtype

        self.elm_factor    = torch.ones(n_rho, device=device, dtype=dtype)
        self.in_elm_region = torch.zeros(n_rho, device=device, dtype=torch.bool)
        self.alpha_MHD     = torch.zeros(n_rho, device=device, dtype=dtype)
        self.alpha_crit    = torch.ones(n_rho, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Analytic peeling-ballooning model
# ---------------------------------------------------------------------------

class AnalyticPBElm(ElmStability):
    """
    Inline s-alpha peeling-ballooning stability criterion.

    See module docstring for the physics.
    """

    def __init__(self, options: dict):
        super().__init__(options)
        self.stiffness          = float(options.get("stiffness",          10.0))
        self.stiffness_power    = float(options.get("stiffness_power",    1.0))
        self.s_hat_min          = float(options.get("s_hat_min",          0.1))
        self.s_peel_frac        = float(options.get("s_peel_frac",        1.5))
        self.roa_min            = float(options.get("roa_min",            0.8))
        self.geometry_correction = bool(options.get("geometry_correction", True))

    def solve(self, powerstate, batch_idx: int = 0) -> None:
        """
        Evaluate the peeling-ballooning criterion for batch element *batch_idx*.

        Reads from ``powerstate.plasma`` (all tensors have shape (batch, rho)):
          p_prime, q, roa, Rmajoa, s_q, kappa, delta

        Sets (all shape (rho,)):
          self.elm_factor
          self.in_elm_region
          self.alpha_MHD
          self.alpha_crit
        """
        p = powerstate.plasma
        b = batch_idx

        # ── 1. Extract per-surface profiles (shape: rho) ──────────────────────
        p_prime = p["p_prime"][b, :]      # normalised pressure gradient (< 0)
        q       = p["q"][b, :]            # safety factor
        roa     = p["roa"][b, :]          # r/a
        Rmajoa  = p["Rmajoa"][b, :]       # R_maj / a

        # s_q = (q/roa)^2 * s_hat  →  s_hat = s_q * (roa/q)^2
        if "s_q" in p:
            s_q   = p["s_q"][b, :]
            s_hat = s_q * (roa / q.clamp(min=0.01)).pow(2)
        else:
            # Fallback: compute s_hat = rmin/q * dq/drmin from profiles
            rmin  = p["rmin"][b, :].detach().cpu().numpy()
            q_np  = q.detach().cpu().numpy()
            dqdrmin = np.gradient(q_np, rmin)
            s_hat = torch.from_numpy(rmin / np.clip(q_np, 1e-3, None) * dqdrmin).to(p_prime)

        # ── 2. MHD alpha (normalised pressure drive) ──────────────────────────
        # alpha_MHD = -8π * Rmajoa * roa * q * p_prime  (always ≥ 0)
        _8pi    = 8.0 * math.pi
        alpha_MHD = -_8pi * Rmajoa * roa * q * p_prime
        alpha_MHD = alpha_MHD.clamp(min=0.0)   # physically non-negative

        # ── 3. Geometry correction factor ─────────────────────────────────────
        if self.geometry_correction and "kappa" in p and "delta" in p:
            kappa = p["kappa"][b, :]
            delta = p["delta"][b, :]
            eps_local = (roa / Rmajoa.clamp(min=0.1)).clamp(min=0.0, max=1.0)
            # Toroidal correction: (1 + 1.5√ε)
            toroid = 1.0 + 1.5 * eps_local.sqrt()
            # Triangularity stabilisation: (1 + 2|δ|)
            triang = 1.0 + 2.0 * delta.abs()
            geo_factor = toroid * triang
        else:
            geo_factor = torch.ones_like(alpha_MHD)

        # ── 4. Ideal ballooning stability boundary ────────────────────────────
        # alpha_bal = |s_hat| * geo_factor  (Connor-Hastie first stability)
        alpha_bal = s_hat.abs().clamp(min=self.s_hat_min) * geo_factor

        # ── 5. Peeling stability boundary ─────────────────────────────────────
        # Peeling triggers when local shear exceeds:
        #   s_peel = s_peel_frac * alpha_MHD / alpha_bal
        # i.e. very high shear at the edge combined with steep pressure → kink.
        s_peel = self.s_peel_frac * alpha_MHD / alpha_bal.clamp(min=1e-6)

        # Convert the peeling shear limit into an equivalent alpha limit for
        # a unified overshoot calculation.  The effective alpha_crit is the
        # *lower* of the ballooning limit and the peeling-implied alpha limit.
        # When s_hat > s_peel, the peeling-implied alpha ~ alpha_bal * s_hat / s_peel.
        alpha_peel = alpha_bal * s_hat.abs() / s_peel.clamp(min=1e-6)
        alpha_crit = torch.minimum(alpha_bal, alpha_peel)

        # ── 6. ELM-unstable region ────────────────────────────────────────────
        # Only check in the pedestal region (roa ≥ roa_min)
        in_pedestal   = roa >= self.roa_min
        in_elm_bal    = in_pedestal & (alpha_MHD > alpha_bal)
        in_elm_peel   = in_pedestal & (s_hat > s_peel)
        in_elm_region = in_elm_bal | in_elm_peel

        # ── 7. Penalty factor ─────────────────────────────────────────────────
        # Smooth stiff-transport inflation above the stability boundary:
        #   overshoot  = max(0, alpha_MHD / alpha_crit - 1)
        #   elm_factor = 1 + stiffness * overshoot^power
        overshoot  = ((alpha_MHD / alpha_crit.clamp(min=1e-6)) - 1.0).clamp(min=0.0)
        # Apply only in pedestal region
        pedestal_mask = in_pedestal.to(alpha_MHD.dtype)
        overshoot     = overshoot * pedestal_mask
        elm_factor    = 1.0 + self.stiffness * overshoot.pow(self.stiffness_power)

        # ── 8. Store diagnostics ──────────────────────────────────────────────
        self.elm_factor    = elm_factor
        self.in_elm_region = in_elm_region
        self.alpha_MHD     = alpha_MHD
        self.alpha_crit    = alpha_crit

        if self.verbose:
            n_elm = in_elm_region.sum().item()
            roa_np = roa.detach().cpu().numpy()
            alpha_np = alpha_MHD.detach().cpu().numpy()
            crit_np  = alpha_crit.detach().cpu().numpy()
            print(
                f"\t[AnalyticPBElm] batch={b}: "
                f"{n_elm} surfaces in ELM region — "
                f"max alpha/alpha_crit = {(alpha_np / np.clip(crit_np, 1e-6, None)).max():.2f} "
                f"@ roa = {roa_np[(alpha_np / np.clip(crit_np, 1e-6, None)).argmax()]:.3f}",
                typeMsg="i",
            )


# ---------------------------------------------------------------------------
# EPED-based model
# ---------------------------------------------------------------------------

def _pressure_kPa(plasma: dict, b: int) -> torch.Tensor:
    """
    Total thermal pressure at each radial surface for batch element *b*.

    Returns a 1-D tensor of shape (rho,) in **kPa**.

    Units: ne × Te [1e19 m⁻³ × keV] → kPa via factor 1.602e-3.
    """
    _c = 1.602e-3   # (e_J × 1e19 × 1e3 × 1e-3) kPa per (1e19/m³ × keV)
    ne = plasma["ne"][b, :]       # (rho,)       1e19 m⁻³
    te = plasma["te"][b, :]       # (rho,)       keV
    ni = plasma["ni"][b, :, :]    # (rho, ions)  1e19 m⁻³
    ti = plasma["ti"][b, :]       # (rho,)       keV — same for all species
    return _c * (ne * te + (ni * ti.unsqueeze(-1)).sum(-1))


class EpedElm(ElmStability):
    """
    ELM stability model that calls EPED.run() / EPED.read() to obtain a
    physics-based peeling-ballooning stability limit, then translates the
    degree of pedestal overshoot into a stiff-transport penalty on the
    turbulent fluxes.

    Stability criterion
    -------------------
    EPED predicts the maximum stable pedestal-top pressure ``p_top_crit`` [kPa].
    The current pedestal-top pressure ``p_top_current`` is sampled from the
    plasma profiles at the pedestal-top reference surface.  The global overshoot

        overshoot = max(0, p_top_current / p_top_crit - 1)

    drives a stiff-transport penalty applied over the pedestal region with a
    linear spatial ramp (strongest at LCFS, zero inside roa_min)::

        elm_factor(roa) = 1 + stiffness
                            * ramp(roa)
                            * overshoot ^ stiffness_power

        ramp(roa) = clamp((roa - roa_min) / (1 - roa_min),  0, 1)

    EPED run folder management
    --------------------------
    Each call to ``solve()`` for batch element *b* stores its EPED run in::

        eped_folder / f"b{b}_{hash8}"

    where ``hash8`` is an 8-character MD5 hash of the EPED input parameters
    rounded to 3 decimal places.  Identical inputs therefore reuse the cached
    EPED output without re-running, saving cluster time during iterative
    transport solves.  Pass ``cold_start=True`` to force fresh runs.

    Auto-extracted global parameters
    ---------------------------------
    The following EPED inputs are extracted automatically from the powerstate
    object; all can be overridden via ``elm_model_options``:

    ============  ================  ======================================
    Key           Units             Source
    ============  ================  ======================================
    ip            MA                profiles["current(MA)"][0]
    bt            T                 profiles["bcentr(T)"][0]
    r             m                 profiles["rcentr(m)"][0]  (R₀)
    a             m                 plasma["a"]
    kappa         –                 plasma["kappa"][:, -1]  (LCFS)
    delta         –                 plasma["delta"][:, -1]  (LCFS)
    neped         1e19 m⁻³          plasma["ne"][:, ix_pt]  (roa ≈ 0.95)
    nesep         1e19 m⁻³          lcfs_bc["ne"]  or  0.25 × neped
    tesep         eV                lcfs_bc["te"]×1e3  or  plasma["te"][:,-1]×1e3
    zeffped       –                 plasma["Zeff"][:, -1]   (fallback 1.5)
    betan         –                 computed from volume-average pressure + ip
    zeta          –                 0.0  (squareness, rarely available)
    ============  ================  ======================================

    Required options
    ----------------
    eped_folder : str or Path
        Root directory under which EPED run sub-folders are created.

    Optional options
    ----------------
    ip, bt, r, a, kappa, delta, neped, betan, zeffped, nesep, tesep, zeta
        Override any auto-extracted EPED input parameter.
    stiffness         : float, default 10.0
    stiffness_power   : float, default 1.0
    roa_min           : float, default 0.8
    pedestal_top_roa  : float or None, default None
        roa at which ``p_top_current`` is sampled.  If None, estimated as
        ``max(roa_min, 1 - wrped)`` using EPED's ``wrped`` output.
    cold_start        : bool, default False
    nproc_per_run     : int, default 64
    minutes_slurm     : int, default 30
    verbose           : bool, default False
    """

    _EPED_PARAM_KEYS = (
        "ip", "bt", "r", "a", "kappa", "delta",
        "neped", "betan", "zeffped", "nesep", "tesep", "zeta",
    )

    def __init__(self, options: dict):
        super().__init__(options)
        self.stiffness        = float(options.get("stiffness",       10.0))
        self.stiffness_power  = float(options.get("stiffness_power",  1.0))
        self.roa_min          = float(options.get("roa_min",          0.8))
        self.pedestal_top_roa = options.get("pedestal_top_roa",       None)
        self.cold_start       = bool(options.get("cold_start",        False))
        self.nproc_per_run    = int(options.get("nproc_per_run",      64))
        self.minutes_slurm    = int(options.get("minutes_slurm",      30))

        eped_folder = options.get("eped_folder", None)
        if eped_folder is None:
            raise ValueError(
                "[EpedElm] 'eped_folder' is required in elm_model_options."
            )
        self.eped_folder = Path(eped_folder)
        self.eped_folder.mkdir(parents=True, exist_ok=True)
        self._warned_missing_eped_runtime = False
        # If EPED runtime is missing, fallback is always safe; logging is optional.
        # Default to verbose-only to avoid noisy warnings in production runs.
        self.warn_runtime_unavailable = bool(
            options.get("warn_runtime_unavailable", self.verbose)
        )

        # User-supplied overrides for EPED inputs (None = auto-extract)
        self._overrides = {k: options.get(k, None) for k in self._EPED_PARAM_KEYS}

    def _eped_runtime_available(self) -> tuple[bool, str]:
        """
        Check whether EPED external runtime dependencies are available.

        Required:
          - EPED_SOURCE_PATH environment variable pointing to a valid EPED tree
          - EPED template directory under EPED_SOURCE_PATH
          - ips.py available on PATH
        """
        eped_src = os.environ.get("EPED_SOURCE_PATH", "").strip()
        if not eped_src:
            return False, "EPED_SOURCE_PATH is not set"

        template_dir = Path(eped_src) / "template" / "engaging" / "eped_run_template"
        if not template_dir.exists():
            return False, f"EPED template folder not found: {template_dir}"

        if shutil.which("ips.py") is None:
            return False, "ips.py is not available on PATH"

        return True, ""

    # ------------------------------------------------------------------
    # EPED input extraction
    # ------------------------------------------------------------------

    def _extract_eped_inputs(self, powerstate, b: int) -> dict:
        """
        Build the EPED input parameter dict for batch element *b*.
        Auto-extracted values are used for any key absent from ``_overrides``.
        """
        p   = powerstate.plasma
        prf = powerstate.profiles   # gacode_state

        def _ov(key, fallback):
            v = self._overrides.get(key)
            return v if v is not None else fallback()

        # ── Machine / geometry ────────────────────────────────────────
        ip = _ov("ip",  lambda: float(prf.profiles["current(MA)"][0]))
        bt = _ov("bt",  lambda: float(prf.profiles["bcentr(T)"][0]))
        r  = _ov("r",   lambda: float(prf.profiles["rcentr(m)"][0]))
        a  = _ov("a",   lambda: float(p["a"].item()))

        # ── Pedestal-top surface index (roa ≈ 0.95) ──────────────────
        roa_np = p["roa"][b, :].detach().cpu().numpy()
        ix_pt  = int(np.searchsorted(roa_np, 0.95))
        ix_pt  = min(ix_pt, len(roa_np) - 1)

        # ── Shape at LCFS ─────────────────────────────────────────────
        kappa = _ov("kappa", lambda: float(p["kappa"][b, -1].item()))
        delta = _ov("delta", lambda: float(p["delta"][b, -1].item()))
        zeta  = _ov("zeta",  lambda: 0.0)

        # ── Density / temperatures ────────────────────────────────────
        neped = _ov("neped", lambda: float(p["ne"][b, ix_pt].item()))

        def _nesep_default():
            lcfs_bc = getattr(powerstate, "_lcfs_bc", {})
            return float(lcfs_bc["ne"]) if "ne" in lcfs_bc else 0.25 * neped

        def _tesep_default():
            lcfs_bc = getattr(powerstate, "_lcfs_bc", {})
            if "te" in lcfs_bc:
                return float(lcfs_bc["te"]) * 1e3   # keV → eV
            return float(p["te"][b, -1].item()) * 1e3

        nesep   = _ov("nesep",   _nesep_default)
        tesep   = _ov("tesep",   _tesep_default)
        zeffped = _ov("zeffped", lambda: float(
            p["Zeff"][b, ix_pt].item() if "Zeff" in p else 1.5
        ))

        # ── Normalised beta ───────────────────────────────────────────
        betan = _ov("betan", lambda: self._compute_betan(powerstate, b, bt, ip))

        params = dict(
            ip=ip, bt=bt, r=r, a=a,
            kappa=kappa, delta=delta, zeta=zeta,
            neped=neped, betan=betan,
            zeffped=zeffped, nesep=nesep, tesep=tesep,
        )

        if self.verbose:
            print(
                f"\t[EpedElm] batch={b} EPED inputs: "
                + ", ".join(f"{k}={v:.3g}" for k, v in params.items()),
                typeMsg="i",
            )
        return params

    @staticmethod
    def _compute_betan(powerstate, b: int, bt_T: float, ip_MA: float) -> float:
        """
        Estimate the volume-averaged normalised beta::

            β_N = 100 × β × a [m] × B₀ [T] / I_p [MA]
            β   = 2 μ₀ <p> / B₀²

        <p> [Pa] is computed by integrating ``p(r) × volp × drmin``.
        Falls back to the arithmetic mean over the outer half if integration
        geometry is unavailable.
        """
        from scipy.constants import mu_0 as mu0

        p_obj = powerstate.plasma
        a_m   = float(p_obj["a"].item())
        p_kPa = _pressure_kPa(p_obj, b)   # (rho,) in kPa

        try:
            rmin  = p_obj["rmin"][b, :].detach().cpu().double()
            volp  = p_obj["volp"][b, :].detach().cpu().double()
            drmin = torch.diff(rmin, prepend=rmin[:1])
            dV    = (volp * drmin).clamp(min=0.0)
            p_avg_Pa = float(
                (p_kPa.cpu().double() * dV).sum() / dV.sum().clamp(min=1e-30)
            ) * 1e3
        except Exception:
            n = p_kPa.shape[0]
            p_avg_Pa = float(p_kPa[n // 2:].mean()) * 1e3

        beta  = 2.0 * mu0 * p_avg_Pa / max(bt_T ** 2, 1e-6)
        return round(100.0 * beta * a_m * bt_T / max(ip_MA, 1e-6), 4)

    # ------------------------------------------------------------------
    # Pedestal-top pressure
    # ------------------------------------------------------------------

    def _pedestal_top_pressure_kPa(
        self, powerstate, b: int, roa_pt: float
    ) -> float:
        """Interpolate total thermal pressure [kPa] at roa = roa_pt."""
        p      = powerstate.plasma
        roa_np = p["roa"][b, :].detach().cpu().numpy()
        p_np   = _pressure_kPa(p, b).cpu().numpy()
        return float(np.interp(roa_pt, roa_np, p_np))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(self, powerstate, batch_idx: int = 0) -> None:
        """
        Run (or reuse) EPED for batch element *batch_idx*, compare the current
        pedestal pressure to the EPED stability limit, and compute the spatial
        ELM penalty factor.
        """
        from mitim_tools.eped_tools.EPEDtools import EPED

        b      = batch_idx
        p      = powerstate.plasma
        n_rho  = p["roa"].shape[-1]
        device = p["roa"].device
        dtype  = p["roa"].dtype

        # If EPED runtime is unavailable, gracefully fall back to stable.
        runtime_ok, runtime_msg = self._eped_runtime_available()
        if not runtime_ok:
            if self.warn_runtime_unavailable and (not self._warned_missing_eped_runtime):
                print(
                    f"\t[EpedElm] EPED runtime unavailable ({runtime_msg}). "
                    f"Skipping EPED and assuming stable (elm_factor = 1).",
                    typeMsg="i",
                )
            self._warned_missing_eped_runtime = True

            self.elm_factor    = torch.ones(n_rho, device=device, dtype=dtype)
            self.in_elm_region = torch.zeros(n_rho, device=device, dtype=torch.bool)
            self.alpha_MHD     = torch.zeros(n_rho, device=device, dtype=dtype)
            self.alpha_crit    = torch.ones(n_rho, device=device, dtype=dtype)
            return

        # ── 1. Build EPED inputs ──────────────────────────────────────
        params = self._extract_eped_inputs(powerstate, b)

        # ── 2. Hash-based subfolder for caching ───────────────────────
        rounded   = {k: round(float(v), 3) for k, v in params.items()}
        h8        = hashlib.md5(
            json.dumps(rounded, sort_keys=True).encode()
        ).hexdigest()[:8]
        subfolder = f"b{b}_{h8}"

        # ── 3. Run EPED (or reuse cached result) ──────────────────────
        ptop_crit_kPa = np.inf   # assume stable unless EPED says otherwise
        wrped         = 0.05     # fallback pedestal width in rho

        try:
            eped = EPED(self.eped_folder)
            eped.run(
                subfolder=subfolder,
                input_params=params,
                cold_start=self.cold_start,
                nproc_per_run=self.nproc_per_run,
                minutes_slurm=self.minutes_slurm,
            )
            eped.read(subfolder=subfolder, print_results=self.verbose, label="elm")

            data = eped.results.get("elm", {}).get("1", None)
            if data is not None and "ptop" in data.data_vars:
                ptop_crit_kPa = float(data["ptop"].values[0])
                if "wrped" in data.data_vars:
                    wrped = float(data["wrped"].values[0])
                if self.verbose:
                    print(
                        f"\t[EpedElm] batch={b}: EPED ptop_crit = "
                        f"{ptop_crit_kPa:.2f} kPa  (wrped = {wrped:.3f} ρ)",
                        typeMsg="i",
                    )
            else:
                if self.verbose:
                    print(
                        f"\t[EpedElm] batch={b}: EPED found no stability "
                        f"crossing — assuming stable.",
                        typeMsg="i",
                    )

        except Exception as exc:
            print(
                f"\t[EpedElm] batch={b}: EPED call failed ({exc}). "
                f"Assuming stable (elm_factor = 1).",
                typeMsg="w",
            )

        # ── 4. Pedestal-top reference location ────────────────────────
        roa_pt = (
            self.pedestal_top_roa
            if self.pedestal_top_roa is not None
            else max(self.roa_min, 1.0 - wrped)
        )

        # ── 5. Current pedestal-top pressure ─────────────────────────
        ptop_current_kPa = self._pedestal_top_pressure_kPa(powerstate, b, roa_pt)

        if self.verbose:
            ratio_str = f"{ptop_current_kPa / max(ptop_crit_kPa, 1e-3):.3f}"
            print(
                f"\t[EpedElm] batch={b}: p_top_current = {ptop_current_kPa:.2f} kPa  "
                f"p_top_crit = {ptop_crit_kPa:.2f} kPa  (ratio = {ratio_str})",
                typeMsg="i",
            )

        # ── 6. Spatial penalty factor ─────────────────────────────────
        roa_1d    = p["roa"][b, :].detach().cpu()
        overshoot = max(0.0, ptop_current_kPa / max(ptop_crit_kPa, 1e-3) - 1.0)

        # Linear ramp in roa: 0 at roa_min, 1 at LCFS
        ramp = (
            (roa_1d - self.roa_min) / max(1.0 - self.roa_min, 1e-6)
        ).clamp(0.0, 1.0)

        elm_factor    = (
            1.0 + self.stiffness * ramp * (overshoot ** self.stiffness_power)
        ).to(dtype).to(device)
        in_elm_region = (
            (roa_1d >= self.roa_min) & (overshoot > 0.0)
        ).to(device)

        # Use pressure ratio as the "alpha" diagnostic (alpha_crit = 1.0)
        ratio = ptop_current_kPa / max(ptop_crit_kPa, 1e-3)
        self.elm_factor    = elm_factor
        self.in_elm_region = in_elm_region
        self.alpha_MHD     = torch.full((n_rho,), ratio,  dtype=dtype, device=device)
        self.alpha_crit    = torch.ones(  n_rho,           dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ELM_REGISTRY: dict[str, type] = {
    "Null":       NullElm,
    "AnalyticPB": AnalyticPBElm,
    "EPED":       EpedElm,
}


def build_elm_model(name: str, options: dict) -> ElmStability:
    """
    Instantiate an ELM stability model by name.

    Parameters
    ----------
    name    : one of "Null", "AnalyticPB", "EPED"
    options : model-specific keyword options (see module docstring)

    Returns
    -------
    ElmStability instance
    """
    if name not in _ELM_REGISTRY:
        raise ValueError(
            f"[elm] Unknown elm_model '{name}'.  "
            f"Available: {sorted(_ELM_REGISTRY.keys())}"
        )
    return _ELM_REGISTRY[name](options)
