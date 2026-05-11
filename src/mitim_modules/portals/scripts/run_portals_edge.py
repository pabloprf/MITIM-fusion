"""
run_portals_edge.py
-------------------
Launch a PORTALS-Edge run from the command line.

Usage
-----
    run_portals_edge <folder> [options]

The script mirrors ``run_portals.py`` but uses ``portals_edge`` instead of
``portals`` and accepts additional ``--edge-*`` options for the physics models.
All ``--edge-*`` flags are merged into the ``edge_options`` block of the
namelist *in memory*, so the YAML file on disk is never modified.

Required positional argument
----------------------------
folder          : str
    Working directory.  Created if it does not exist.

Optional arguments
------------------
--namelist PATH
    Path to a PORTALS YAML namelist (default: ``./namelist.portals.yaml``).
--input PATH
    Path to ``input.gacode`` (default: ``./input.gacode``).
--cold
    Start fresh, ignoring any previous BO state in *folder*.

Edge-physics options  (all optional; defaults match STATEedge defaults)
----------------------------------------------------------------------
--domain-roa  FLOAT FLOAT
    Restrict the plasma radial grid to roa ∈ [roa_min, roa_max].
--domain-rho  FLOAT FLOAT
    Same restriction in sqrt-toroidal-flux ρ (ignored if --domain-roa given).
--lcfs-te  FLOAT        LCFS electron temperature [keV].
--lcfs-ti  FLOAT        LCFS ion temperature [keV].
--lcfs-ne  FLOAT        LCFS electron density [1e19 m-3].

--bc-model  {FixedInitial,TwoFluid_PeretSSF,TwoFluid_EichManz}
    Boundary-condition model (default: FixedInitial).
--bc-ne-target  FLOAT   Divertor target density for two-point models [1e19 m-3].
--bc-Te-target  FLOAT   Divertor target Te for two-point models [keV].
--bc-Lpar  FLOAT        Override parallel connection length [m].

--neutral-model  {Null,Analytic}
    Main-ion neutral solver (default: Null).
--neutral-source-rate  FLOAT
    D0 source rate crossing the LCFS inward [s-1] (default: 1e21).
--neutral-include-cx    Enable charge-exchange contribution to diffusivity.

--charge-state-model  {Null,Aurora}
    Impurity charge-state solver (default: Null).
--cs-imp  STR           Impurity element symbol, e.g. 'C' or 'W'.
--cs-D  FLOAT           Impurity diffusion coefficient [cm2/s] (default: 1000).
--cs-source-rate  FLOAT  Impurity injection rate [s-1] (default: 1e21).
--cs-cxr            Enable CXR in the Aurora run.

--elm-model  {Null,AnalyticPB,EPED}
    ELM stability model (default: Null).
--elm-stiffness  FLOAT  Transport penalty stiffness above the stability boundary.
--elm-roa-min  FLOAT    Innermost roa where the ELM penalty is applied (default 0.8).
--eped-folder  PATH     Root folder for EPED run sub-directories (required for EPED).

--no-edge-targets
    Use the standard ``analytical_model`` instead of ``analytical_model_edge``
    for target evaluation.

--parameterizer {spline,mtanh,mtanh_spline}
    Override the profile parameterizer for the edge run.
--mtanh-defined-on {y,aLy}
    Control-point type for the MtanhSpline parameterizer (default: y).

--target-multiplier TARGET FACTOR
    Scale a selected fixed target source at x0 and shift the full target profile.
    TARGET can be: qe, qi, ge, gz, mt. Repeat the flag to set multiple channels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mitim_modules.portals.PORTALSedge import portals_edge
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.misc_tools import IOtools


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_portals_edge",
        description="Launch a PORTALS-Edge Bayesian optimisation run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Positional ────────────────────────────────────────────────────────────
    p.add_argument("folder", type=str, help="Working directory for the run.")

    # ── Standard PORTALS args ─────────────────────────────────────────────────
    p.add_argument("--namelist", type=str, default=None,
                   help="PORTALS YAML namelist (default: ./namelist.portals.yaml).")
    p.add_argument("--input", type=str, default=None,
                   help="input.gacode path (default: ./input.gacode).")
    p.add_argument("--cold", action="store_true", default=False,
                   help="Ignore previous BO state and start fresh.")

    # ── Domain trimming ───────────────────────────────────────────────────────
    p.add_argument("--domain-roa", nargs=2, type=float, metavar=("ROA_MIN", "ROA_MAX"),
                   default=None, help="Restrict plasma grid to roa ∈ [ROA_MIN, ROA_MAX].")
    p.add_argument("--domain-rho", nargs=2, type=float, metavar=("RHO_MIN", "RHO_MAX"),
                   default=None, help="Restrict plasma grid to rho ∈ [RHO_MIN, RHO_MAX].")

    # ── LCFS initial BC overrides ──────────────────────────────────────────────
    lcfs = p.add_argument_group("LCFS boundary conditions (initial overrides)")
    lcfs.add_argument("--lcfs-te", type=float, default=None,
                      help="LCFS electron temperature [keV].")
    lcfs.add_argument("--lcfs-ti", type=float, default=None,
                      help="LCFS ion temperature [keV].")
    lcfs.add_argument("--lcfs-ne", type=float, default=None,
                      help="LCFS electron density [1e19 m-3].")

    # ── Boundary-condition model ───────────────────────────────────────────────
    bc = p.add_argument_group("Boundary-condition model")
    bc.add_argument("--bc-model",
                    choices=["FixedInitial", "TwoFluid_PeretSSF", "TwoFluid_EichManz"],
                    default="FixedInitial",
                    help="SOL boundary-condition model (default: FixedInitial).")
    bc.add_argument("--bc-ne-target", type=float, default=None,
                    help="Divertor target density [1e19 m-3] for two-point models.")
    bc.add_argument("--bc-Te-target", type=float, default=None,
                    help="Divertor target Te [keV] for two-point models.")
    bc.add_argument("--bc-Lpar", type=float, default=None,
                    help="Override parallel connection length [m].")

    # ── Neutral model ─────────────────────────────────────────────────────────
    neu = p.add_argument_group("Main-ion neutral model")
    neu.add_argument("--neutral-model", choices=["Null", "Analytic"],
                     default="Null", help="Neutral solver (default: Null).")
    neu.add_argument("--neutral-source-rate", type=float, default=1e21,
                     help="D0 LCFS source rate [s-1] (default: 1e21).")
    neu.add_argument("--neutral-include-cx", action="store_true", default=False,
                     help="Include CX contribution to neutral diffusivity.")

    # ── Charge-state model ────────────────────────────────────────────────────
    cs = p.add_argument_group("Impurity charge-state model")
    cs.add_argument("--charge-state-model", choices=["Null", "Aurora"],
                    default="Null", help="Charge-state solver (default: Null).")
    cs.add_argument("--cs-imp", type=str, default="C",
                    help="Impurity element symbol (default: C).")
    cs.add_argument("--cs-D", type=float, default=1e3,
                    help="Impurity diffusion coefficient [cm2/s] (default: 1000).")
    cs.add_argument("--cs-source-rate", type=float, default=1e21,
                    help="Impurity injection rate [s-1] (default: 1e21).")
    cs.add_argument("--cs-cxr", action="store_true", default=False,
                    help="Enable CXR in Aurora.")

    # ── ELM model ─────────────────────────────────────────────────────────────
    elm = p.add_argument_group("ELM stability model")
    elm.add_argument("--elm-model", choices=["Null", "AnalyticPB", "EPED"],
                     default="Null", help="ELM model (default: Null).")
    elm.add_argument("--elm-stiffness", type=float, default=10.0,
                     help="Transport penalty stiffness above the stability boundary.")
    elm.add_argument("--elm-roa-min", type=float, default=0.8,
                     help="Innermost roa for ELM penalty application (default: 0.8).")
    elm.add_argument("--eped-folder", type=str, default=None,
                     help="Root folder for EPED run sub-directories (required with --elm-model EPED).")

    # ── Target evaluator override ─────────────────────────────────────────────
    p.add_argument("--no-edge-targets", action="store_true", default=False,
                   help="Use standard analytical_model instead of analytical_model_edge.")

    # ── Parameterizer ─────────────────────────────────────────────────────────
    p.add_argument("--parameterizer",
                   choices=["spline", "akima", "mtanh", "MtanhSpline", "mtanh_spline"],
                   default=None,
                   help=(
                       "Override the profile parameterizer for the edge run. "
                       "Use 'mtanh' for true global Mtanh params [log_A, log_Delta_0, delta, m], "
                       "or 'MtanhSpline' for the legacy spline-fitted Mtanh hybrid."
                   ))
    p.add_argument("--mtanh-defined-on", choices=["y", "aLy"], default="y",
                   help="Control-point type for MtanhSpline (ignored by true mtanh).")

    # ── Target profile multipliers at x0 ─────────────────────────────────────
    p.add_argument(
        "--target-multiplier",
        nargs=2,
        action="append",
        metavar=("TARGET", "FACTOR"),
        default=None,
        help=(
            "Scale TARGET at x0 and shift the full fixed target profile. "
            "TARGET in {qe, qi, ge, gz, mt}. Repeat for multiple channels."
        ),
    )

    return p


# ---------------------------------------------------------------------------
# Helper: build edge_options dict from parsed args
# ---------------------------------------------------------------------------

def _build_edge_options(args) -> dict:
    opts: dict = {}

    _target_alias_to_key = {
        "qe": "QeMWm2_fixedtargets",
        "qi": "QiMWm2_fixedtargets",
        "ge": "Ge_fixedtargets",
        "gz": "GZ_fixedtargets",
        "mt": "MtJm2_fixedtargets",
    }

    # Domain trimming
    if args.domain_roa is not None:
        opts["domain_roa"] = args.domain_roa
    if args.domain_rho is not None:
        opts["domain_rho"] = args.domain_rho

    # LCFS overrides
    lcfs_bc: dict = {}
    if args.lcfs_te is not None:
        lcfs_bc["te"] = args.lcfs_te
    if args.lcfs_ti is not None:
        lcfs_bc["ti"] = args.lcfs_ti
    if args.lcfs_ne is not None:
        lcfs_bc["ne"] = args.lcfs_ne
    if lcfs_bc:
        opts["lcfs_bc"] = lcfs_bc

    # BC model
    opts["bc_model"] = args.bc_model
    bc_opts: dict = {}
    if args.bc_ne_target is not None:
        bc_opts["ne_target"] = args.bc_ne_target
    if args.bc_Te_target is not None:
        bc_opts["Te_target"] = args.bc_Te_target
    if args.bc_Lpar is not None:
        bc_opts["Lpar"] = args.bc_Lpar
    if bc_opts:
        opts["bc_model_options"] = bc_opts

    # Neutral model
    opts["neutral_model"] = args.neutral_model
    if args.neutral_model != "Null":
        opts["neutral_model_options"] = {
            "source_rate": args.neutral_source_rate,
            "include_cx":  args.neutral_include_cx,
        }

    # Charge-state model
    opts["charge_state_model"] = args.charge_state_model
    if args.charge_state_model != "Null":
        opts["charge_state_model_options"] = {
            "imp":         args.cs_imp,
            "D_z_cm2_s":  args.cs_D,
            "source_rate": args.cs_source_rate,
            "cxr_flag":    args.cs_cxr,
        }

    # ELM model
    opts["elm_model"] = args.elm_model
    elm_opts: dict = {
        "stiffness": args.elm_stiffness,
        "roa_min":   args.elm_roa_min,
    }
    if args.elm_model == "EPED":
        if args.eped_folder is None:
            raise ValueError("--eped-folder is required when --elm-model EPED is set.")
        elm_opts["eped_folder"] = args.eped_folder
    opts["elm_model_options"] = elm_opts

    # Target evaluator
    opts["use_edge_targets"] = not args.no_edge_targets

    # mtanh defined_on
    opts["defined_on"] = args.mtanh_defined_on

    # Fixed-target multipliers (at x0)
    if args.target_multiplier:
        target_multipliers: dict = {}
        for target_name_raw, factor_raw in args.target_multiplier:
            target_name = target_name_raw.strip().lower()
            if target_name not in _target_alias_to_key:
                raise ValueError(
                    f"Unknown --target-multiplier TARGET '{target_name_raw}'. "
                    "Valid values are: qe, qi, ge, gz, mt."
                )

            try:
                factor = float(factor_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid multiplier FACTOR '{factor_raw}' for target '{target_name_raw}'."
                ) from exc

            target_multipliers[_target_alias_to_key[target_name]] = factor

        if target_multipliers:
            opts["target_multipliers"] = target_multipliers

    return opts


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = _build_parser()
    args = parser.parse_args()

    folder_work  = Path(args.folder)
    namelist     = (Path(args.namelist)
                    if args.namelist is not None
                    else IOtools.expandPath(".") / "namelist.portals.yaml")
    inputgacode  = (Path(args.input)
                    if args.input is not None
                    else IOtools.expandPath(".") / "input.gacode")

    # Instantiate portals_edge
    portals_fun = portals_edge(folder_work, portals_namelist=namelist)

    # Merge edge_options into the loaded namelist *in memory* before prep()
    edge_options = _build_edge_options(args)
    portals_fun.portals_parameters["edge_options"] = edge_options

    # Override parameterizer in the solution block if requested
    if args.parameterizer is not None:
        portals_fun.portals_parameters["solution"]["parameterizer"] = args.parameterizer

    # Also propagate defined_on into the solution block so STATEtools forwards
    # it via evolution_options to powerstate_edge.__init__
    portals_fun.portals_parameters["solution"]["defined_on"] = args.mtanh_defined_on

    portals_fun.prep(inputgacode)

    mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=args.cold)
    mitim_bo.run()


if __name__ == "__main__":
    main()
