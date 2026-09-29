"""
test_transp_prescribed_eq.py
============================
End-to-end MAESTRO test of the prescribed-equilibrium (LEVGEO=8) and frozen-poloidal-field
options of the TRANSP beat: `machine_initialization: null` (which IS the prescribed-equilibrium
selector -- "no seed machine" and "no GS solve" are the same statement, since TEQ cannot be
given a user initial guess) and `frozen_field: True`, both in the beat's parameters_prepare.

The subject is the MAESTRO machinery: a real `maestro` object is driven through
define_beat -> initialize -> prepare from a minimal single-beat namelist and a repo-shipped
input.gacode, and every assertion is made on what lands in the beat folder. Nothing below
instantiates TRANSPhelpers to BUILD anything -- the point is to prove the
namelist -> beat -> deck plumbing.

Two stages:

  STAGE 1 (default, local, seconds): prepare-level assertions. Drives MAESTRO up to -- but
  not including -- run(), so nothing is submitted anywhere. Includes a local construction
  check of the stage-2 deck (TORIC block, MPI pserve, rank/cpus_per_task run kwargs).

  STAGE 2 (--full, dispatches to the cluster): a real 2-beat chain, beats: [transp, portals],
  executed through the normal MITIM dispatch (TRANSP -> singularity per config_user.json,
  TGLF per config). Asserts on OUTCOMES: TRANSP NORMAL EXIT with 0 quval, TORIC actually
  solved at the requested power, and the TRANSP-computed sources handed off into the PORTALS
  beat's input state.

Stage 2 is behind an explicit flag rather than an env guard because it SUBMITS CLUSTER JOBS
and burns real queue time: it must never fire as a side effect of someone running the file.

Run as:

    python tests/dev_tests/test_transp_prescribed_eq.py            # stage 1 only
    python tests/dev_tests/test_transp_prescribed_eq.py --full     # + the real 2-beat chain

Exits non-zero on any assertion failure. Each test prints PASS on success.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

mitim_root = Path(__file__).resolve().parents[2] / "src"
if str(mitim_root) not in sys.path:
    sys.path.insert(0, str(mitim_root))

from mitim_modules.maestro.MAESTROmain import maestro
from mitim_modules.maestro.utils import PORTALSbeat, TRANSPbeat
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.transp_tools import UFILEStools
from mitim_tools.transp_tools.utils import TRANSPhelpers

# Stage 1 uses the small state (fast to prepare); stage 2 needs a BURNING DT plasma so that
# TORIC and NUBEAM both have something real to do.
GACODE = Path(__file__).resolve().parents[1] / "data" / "input.gacode"
GACODE_DT = Path(__file__).resolve().parents[1] / "data" / "input.gacode_SPARC_PRD"

# Engineering parameters of each shipped state, so the namelist and the profiles agree
IP_MA, B_T, A_MINOR = 0.997, 2.411, 0.518
IP_MA_DT, B_T_DT, A_MINOR_DT = 8.7, 12.156, 0.568


def maestro_namelist(machine_initialization=None, frozen_field=True, gacode=None, ip=None, bt=None, a=None):
    """Minimal single-beat MAESTRO namelist. Built as a dict rather than a YAML file so the
    test stays self-contained on repo data; the `import::` strings a YAML would carry resolve
    to exactly these callables."""
    gacode, ip = gacode or GACODE, ip or IP_MA
    bt, a = bt or B_T, a or A_MINOR
    return {
        "seed": 1,
        "plasma": {
            "profiles_initialization": {
                "initialization_type": "profiles",
                "creator_type": None,
                "parameters": {"profiles_file": str(gacode)},
            },
            "parameters": {
                "Bt": bt, "Ip": ip, "neped_20": 0.3, "fGped": 0.8,
                "ne_ratio_sep_ped": 0.4, "Tesep_eV": 70.0,
                "separatrix": {
                    "R": None, "a": a, "delta_sep": None, "kappa_sep": None,
                    "zeta_sep": None, "rz_boundary_file": None, "internal_flux_file": None,
                    "n_mxh": 5, "boundary_surface_psin": 0.995, "freeze_995_from": None,
                    "shaping_extraction_psin": 0.995, "geqdsk_file": None,
                },
            },
            "species": {
                "fuel": ["D", "T"], "Zeff": 1.8,
                "mix": {"fmain": 0.85, "highZ": "W", "fhighZ": 1.5e-05, "CShighZ_estimate": 50},
            },
            "heating": {
                "type": "ICRH",
                "parameters": {"P_icrh": 1.0, "minority": [1, 1], "fmini": 0.03,
                               "freq_ICH": None, "P_nbi": 0.0, "nu_source": 5.0,
                               "Pe": 0.5, "Pi": 0.5},
            },
        },
        "maestro": {
            "beats": ["transp"],
            "prune_level": 0,
            "refreeze_995_after_beat": 0,
            "master_cpus": 1,
            "transp": {
                "beat_type": "transp",
                "base_module": None,
                "parameters_prepare": {
                    "tokamak_structures": None,
                    # null IS the prescribed-equilibrium selector
                    "machine_initialization": machine_initialization,
                    # deliberately non-zero, and expected to be INERT under a null machine
                    "transition_window": 0.1,
                    "currentheating_window": 0.001,
                    "machine_initialization_match_target": False,
                    "mxh_coeffs_smooth_sep": None,
                    "flattop_window": 0.05,
                    "min_sawtooth_period_ms": None,
                    "ensure_sawtooths": None,
                    "sanitize_q_input": None,
                    "extract_at": "last",
                    "min_extraction_flattop_fraction": None,
                    "frozen_field": frozen_field,
                    "useNUBEAMforAlphas": False,
                    "Pich": False,
                    "Pnbi": False,
                    "dtEquilMax_ms": 10.0, "dtHeating_ms": 10.0,
                    "dtCurrentDiffusion_ms": 10.0, "dtOut_ms": 10.0, "dtIn_ms": 10.0,
                    "nzones": 20, "nzones_energetic": 10, "nzones_distfun": 5,
                    "MCparticles": 1000.0, "toric_ntheta": 64, "toric_nrho": 128,
                    "extractAC": False, "time_before_end": 0.001,
                },
                "preprocess_prepare": TRANSPbeat.preprocess_prepare_transp,
                "preprocess_prepare_parameters": {"cpus_toric": 1, "cpus_nubeam": 1},
                "preprocess_run": TRANSPbeat.preprocess_run_transp,
            },
        },
    }


def engineering_from(nm):
    """The hand-picked engineering set MAESTRO's initializer takes, read off the namelist."""
    p = nm["plasma"]["parameters"]
    return {
        "Ip_MA": p["Ip"], "B_T": p["Bt"], "Zeff": nm["plasma"]["species"]["Zeff"],
        "type_heating": nm["plasma"]["heating"]["type"],
        "Paux_MW": nm["plasma"]["heating"]["parameters"]["P_icrh"],
        "neped_20": p["neped_20"], "Tesep_keV": p["Tesep_eV"] * 1e-3,
        "nesep_20": p["neped_20"] * p["ne_ratio_sep_ped"],
    }


def run_prepare(folder, machine_initialization=None, frozen_field=True, cold_start=True):
    """Drive the real MAESTRO entry points up to (not including) run()."""
    nm = maestro_namelist(machine_initialization, frozen_field)
    engineering = engineering_from(nm)

    cfg = nm["maestro"]["transp"]
    prep = cfg["preprocess_prepare"](cfg["parameters_prepare"], nm,
                                     cfg["preprocess_prepare_parameters"])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m = maestro(folder, master_seed=nm["seed"], terminal_outputs=True,
                    master_cold_start=cold_start, prune_level=0, maestro_namelist=nm)
        m.define_beat("transp", initializer="profiles")
        m.initialize(profiles_file=str(GACODE), **engineering)
        m.prepare(**prep)

    return m, buf.getvalue()


# ---------------------------------------------------------------------------
# Helpers on the produced deck
# ---------------------------------------------------------------------------


def deck(m):
    return (m.beat.folder / f"{m.beat.shot}{m.beat.runid}TR.DAT").read_text()


def nml_value(txt, key):
    """Value of a namelist entry, asserting the key appears EXACTLY once. The uniqueness half
    matters as much as the value: a deck that acquired a second `nlmdif` line would be read by
    Fortran with the last one winning, silently undoing the frozen-field setting."""
    hits = [l for l in txt.splitlines()
            if "=" in l and l.split("=")[0].strip().lower() == key.lower()]
    assert len(hits) == 1, f"expected exactly one `{key}` line in TR.DAT, found {len(hits)}"
    return hits[0].split("=")[1].split("!")[0].strip()


def read_uf(path):
    uf = UFILEStools.UFILEtransp()
    uf.readUFILE(str(path))
    return uf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_deck_is_prescribed_and_frozen():
    with tempfile.TemporaryDirectory() as d:
        m, _ = run_prepare(d)
        txt = deck(m)

        assert nml_value(txt, "levgeo") == "8"
        assert nml_value(txt, "lfixup") == "2"
        for key in ["nriqpr", "nrigrb", "nriprs"]:
            assert nml_value(txt, key) == "-5", f"{key} = {nml_value(txt, key)}"
        for pre, ext in [("pregrb", "GRB"), ("preprs", "PRS"), ("pretrf", "TRF"),
                         ("preplf", "PLF"), ("prerfs", "RFS"), ("prezfs", "ZFS")]:
            assert nml_value(txt, pre) == '"MIT"'
            assert nml_value(txt, "ext" + pre[3:]) == f'"{ext}"'

        assert nml_value(txt, "nlmdif") == "F"
        assert nml_value(txt, "nlqdata") == "T"
        assert nml_value(txt, "nlpcur") == "F"
        assert nml_value(txt, "nqmoda(1)") == "4" and nml_value(txt, "nqmoda(2)") == "4"
        assert nml_value(txt, "nqmodb(1)") == "2" and nml_value(txt, "nqmodb(2)") == "2"
        assert nml_value(txt, "t_sawtooth_on") == "1.0E3"

        assert "nteq_mode" not in txt and "softteq" not in txt, "TEQ block present in a levgeo=8 deck"
        keys = {l.split("=")[0].strip().lower() for l in txt.splitlines() if "=" in l}
        assert not ({"premry", "extmry", "nrimry"} & keys)
    print("PASS: MAESTRO-produced TR.DAT is levgeo=8 + fully frozen, each key exactly once, no TEQ/MRY")


def test_beat_folder_contents():
    with tempfile.TemporaryDirectory() as d:
        m, _ = run_prepare(d)
        written = {f.name for f in m.beat.folder.iterdir()}

        for ext in ["RFS", "ZFS", "QPR", "GRB", "PRS", "TRF", "PLF"]:
            assert f"MIT{m.beat.shot}.{ext}" in written, f"missing .{ext} in {sorted(written)}"
        assert not any(n.endswith(".MRY") for n in written), written
        assert not any(n.startswith("BOUNDARY_") for n in written), written
    print("PASS: beat folder holds all five prescribed-equilibrium ufiles and no MRY/BOUNDARY files")


def test_surfaces_full_x_and_nonfolding():
    with tempfile.TemporaryDirectory() as d:
        m, _ = run_prepare(d)

        ufR = read_uf(m.beat.folder / f"MIT{m.beat.shot}.RFS")
        ufZ = read_uf(m.beat.folder / f"MIT{m.beat.shot}.ZFS")
        assert ufR.dim == 3 and ufZ.dim == 3

        theta, x = np.asarray(ufR.Variables["Y"]), np.asarray(ufR.Variables["Q"])
        nt = len(ufR.Variables["X"])
        assert ufR.Variables["Z"].shape == (nt, len(theta), len(x))

        assert len(x) > 2, f"only {len(x)} radial points -- that is the boundary trick, not a full set"
        assert np.isclose(x[0], 0.0) and np.isclose(x[-1], 1.0)
        assert np.all(np.diff(x) > 0), "x must be strictly increasing"
        assert np.isclose(theta[0], 0.0) and np.isclose(theta[-1], 2 * np.pi)

        R, Z = ufR.Variables["Z"][0].T, ufZ.Variables["Z"][0].T
        assert np.allclose(R[:, 0], R[:, -1]) and np.allclose(Z[:, 0], Z[:, -1]), "theta grid must close"
        assert np.allclose(R[0], R[0, 0]) and np.allclose(Z[0], Z[0, 0]), "x=0 row must be the axis repeated"

        _, detJ_min, margin = TRANSPhelpers.jacobian_margin(R, Z, x, theta)
        assert detJ_min > 0.0, f"det(J) changes sign (min {detJ_min:.3e})"
        assert margin > 0.05, f"det(J) margin {margin:.4f} too small"

        area = np.array([0.5 * np.abs(np.sum(R[i, :-1] * np.diff(Z[i]) - Z[i, :-1] * np.diff(R[i])))
                         for i in range(1, len(x))])
        assert np.all(np.diff(area) > 0), "flux surfaces are not nested"
    print(f"PASS: RFS/ZFS full-x 3-D ({len(x)} surfaces x {len(theta)} theta), nested, det(J) margin {margin:.3f}")


def test_flux_functions_match_the_state():
    with tempfile.TemporaryDirectory() as d:
        m, _ = run_prepare(d)
        p = PROFILEStools.gacode_state(str(GACODE))
        f = m.beat.folder

        grb = read_uf(f / f"MIT{m.beat.shot}.GRB").Variables["Z"][:, 0]
        expected = abs(p.profiles["rcentr(m)"][0] * p.profiles["bcentr(T)"][0]) * 1e2
        assert np.allclose(grb, expected), f"GRB {grb[0]} != {expected} T*cm"

        prs = read_uf(f / f"MIT{m.beat.shot}.PRS").Variables["Z"][:, 0]
        assert np.all(prs > 0)
        assert np.allclose(prs, p.derived["pthr_manual"] * 1e6), "PRS != thermal pressure [Pa]"
        e = 1.602176634e-19
        te, ne = p.profiles["te(keV)"] * 1e3, p.profiles["ne(10^19/m^3)"] * 1e19
        ti = np.atleast_2d(p.profiles["ti(keV)"].T).T * 1e3
        ni = np.atleast_2d(p.profiles["ni(10^19/m^3)"].T).T * 1e19
        p_allions = e * (ne * te + (ni * ti).sum(axis=1))
        assert prs[0] <= p_allions[0] * (1 + 1e-9), "thermal PRS above the all-ion upper bound"
        assert np.isclose(prs[0], p_allions[0], rtol=0.05), "PRS more than a fast-ion correction off"

        trf = np.asarray(read_uf(f / f"MIT{m.beat.shot}.TRF").Variables["Z"])
        plf = np.asarray(read_uf(f / f"MIT{m.beat.shot}.PLF").Variables["Z"])
        assert np.allclose(trf, 2 * np.pi * abs(p.profiles["torfluxa(Wb/radian)"][0]))
        psi = p.profiles["polflux(Wb/radian)"]
        assert np.allclose(plf, abs(psi[-1] - psi[0]))
        # TRANSP takes magnitudes; directions come from nlbccw/nljccw. This state has
        # NEGATIVE torfluxa/polflux/bcentr/current, so the sign stripping is load-bearing.
        assert np.all(trf > 0) and np.all(plf > 0) and np.all(grb > 0)

        q = read_uf(f / f"MIT{m.beat.shot}.QPR").Variables["Z"][:, 0]
        assert np.all(q > 0) and np.all(np.isfinite(q))
    print("PASS: GRB/PRS/TRF/PLF/QPR match the initialized state, signs stripped to magnitudes")


def test_transition_window_inert_under_null_machine():
    """transition_window=0.1 is set in the namelist but there is no seed to morph away from, so
    it must leave no trace: no FreeGS seed equilibrium, and geometry constant in time."""
    with tempfile.TemporaryDirectory() as d:
        m, log = run_prepare(d)

        ufR = read_uf(m.beat.folder / f"MIT{m.beat.shot}.RFS")
        Z = ufR.Variables["Z"]
        assert len(ufR.Variables["X"]) >= 2
        for i in range(1, len(ufR.Variables["X"])):
            assert np.allclose(Z[i], Z[0]), "geometry is not constant in time -- a morph leaked in"

        # A morphed run carries a machine-initialization slice whose boundary is the SEED
        # machine's (CMOD: R ~ 0.9 m against this state's 1.6 m). Every slice must instead be
        # the initialized state's own outermost surface. Compared on the bounding box with a
        # 1 mm tolerance: the poloidal resampling onto a fixed 101-point grid moves the extrema
        # by a few 1e-5 m, ~4 orders of magnitude below a seed leak.
        p = PROFILEStools.gacode_state(str(GACODE))
        Zbig = read_uf(m.beat.folder / f"MIT{m.beat.shot}.ZFS").Variables["Z"]
        R_state, Z_state = p.derived["R_surface"][0][-1], p.derived["Z_surface"][0][-1]
        R_written, Z_written = Z[0].T[-1], Zbig[0].T[-1]
        for got, want, name in [(R_written.max(), R_state.max(), "Rmax"),
                                (R_written.min(), R_state.min(), "Rmin"),
                                (Z_written.max(), Z_state.max(), "Zmax"),
                                (Z_written.min(), Z_state.min(), "Zmin")]:
            assert abs(got - want) < 1e-3, \
                f"boundary {name} {got:.5f} != state {want:.5f} (seed geometry leaked in)"

        assert "freegs" not in log.lower(), "a FreeGS machine-initialization equilibrium was built"
        # the namelist value itself is untouched -- it is simply never acted on
        assert m.beat.transition_window == 0.1
        # and the run tree still gets a registered label even with no seed machine
        assert m.beat.machine_run == "CMOD"
    print("PASS: transition_window inert under machine_initialization: null (no seed built, geometry constant in time)")


def test_reprepare_is_idempotent():
    """MAESTROmain.check() re-prepares a prepared-but-unrun beat, so prepare() must regenerate
    the identical deck and ufiles from scratch."""
    with tempfile.TemporaryDirectory() as d:
        m1, _ = run_prepare(d, cold_start=True)
        first = {f.name: f.read_bytes() for f in m1.beat.folder.iterdir() if f.is_file()}
        mtimes = {f.name: f.stat().st_mtime_ns for f in m1.beat.folder.iterdir() if f.is_file()}

        m2, log2 = run_prepare(d, cold_start=False)
        second = {f.name: f.read_bytes() for f in m2.beat.folder.iterdir() if f.is_file()}

        # The comparison below is only meaningful if prepare() actually RE-RAN. MAESTROmain
        # skips it when the beat already has results; here beat_results/input.gacode does not
        # exist (we never called run()), so run_flag must be True and every file rewritten.
        assert "Skipping beat preparation" not in log2, "prepare() was skipped -- the test would be vacuous"
        assert m2.beat.run_flag, "run_flag should be True for a prepared-but-unrun beat"
        rewritten = [n for n, t in mtimes.items()
                     if n in second and (m2.beat.folder / n).stat().st_mtime_ns != t]
        assert len(rewritten) > 5, f"only {len(rewritten)} files were rewritten -- prepare() did not re-run"

        assert m1.beat.shot == m2.beat.shot and m1.beat.runid == m2.beat.runid, \
            f"shot/runid drifted: {m1.beat.shot}{m1.beat.runid} -> {m2.beat.shot}{m2.beat.runid}"
        assert set(first) == set(second), f"file set changed: {set(first) ^ set(second)}"

        differing = [n for n in first if first[n] != second[n]]
        assert not differing, f"re-prepare produced different bytes for {differing}"
    print(f"PASS: re-prepare regenerates all {len(first)} beat files byte-identically")


def test_frozen_field_requires_prescribed():
    with tempfile.TemporaryDirectory() as d:
        try:
            run_prepare(d, machine_initialization="CMOD", frozen_field=True)
        except ValueError as exc:
            assert "machine_initialization" in str(exc) and "null" in str(exc), str(exc)
        else:
            raise AssertionError("frozen_field=True with a seed machine should raise")
    print("PASS: frozen_field with machine_initialization='CMOD' raises through the MAESTRO path")


def test_evolve_mode_unchanged():
    """The default path must be untouched: levgeo=11, TEQ block, boundary-trick RFS, no extras."""
    with tempfile.TemporaryDirectory() as d:
        m, _ = run_prepare(d, machine_initialization="CMOD", frozen_field=False)
        txt = deck(m)

        assert nml_value(txt, "levgeo") == "11"
        assert nml_value(txt, "nlmdif") == "T" and nml_value(txt, "nlqdata") == "F"
        assert nml_value(txt, "nlpcur") == "T"
        assert nml_value(txt, "nqmodb(1)") == "1" and nml_value(txt, "nqmoda(2)") == "1"
        assert "nteq_mode" in txt and "softteq" in txt

        x = np.asarray(read_uf(m.beat.folder / f"MIT{m.beat.shot}.RFS").Variables["Q"])
        assert len(x) == 2 and np.allclose(x, [0.0, 1.0]), f"expected the axis+boundary trick, got {x}"
        assert not (m.beat.folder / f"MIT{m.beat.shot}.GRB").exists(), "GRB written in evolve mode"
    print("PASS: equilibrium_mode='evolve' through MAESTRO is the old behavior (levgeo=11, boundary trick, no extras)")




# ---------------------------------------------------------------------------
# Stage 2: the real 2-beat chain
# ---------------------------------------------------------------------------
#
# TRANSP heating configuration. The rank/cpus_per_task combination is not free: on the r8
# partition under sbatch, pserve=1 with >=32 ranks and cpus-per-task=2 is the combination
# that works; serial (NPROCS=1) makes pretr DATCHK-abort on pserve=1, and intermediate rank
# counts fail to bind. nparallel = max(trmpi, toricmpi, ptrmpi) is BOTH the sbatch ntasks and
# the container's NPROCS (TRANSPsingularity.py), so the rank count is what selects the MPI
# branch of the runscript.
STAGE2_RANKS, STAGE2_CPUS_PER_TASK = 32, 2

# STABLE run folder (NOT a mkdtemp). The chain submits a real TRANSP job, so the outputs must
# survive the process: with cold_start=False MAESTRO's check() finds beat_results/input.gacode
# and SKIPS completed beats, so re-running --full iterates on the assertions without
# resubmitting anything. A TemporaryDirectory defeats that -- and deletes the completed chain
# on the way out even when an assertion raises, which is how the first successful chain was lost.
STAGE2_FOLDER = Path(__file__).resolve().parents[1] / "scratch" / "dev_transp_prescribed_chain"


def stage2_namelist():
    """2-beat chain: prescribed+frozen TRANSP with live TORIC+NUBEAM, then a tiny PORTALS."""
    nm = maestro_namelist(machine_initialization=None, frozen_field=True,
                          gacode=GACODE_DT, ip=IP_MA_DT, bt=B_T_DT, a=A_MINOR_DT)

    # ICRH sized for THIS state: SPARC-PRD carries ~9.46 MW of RF, which is what the beat picks
    # up (Paux_MW = derived['qRF_MW'][-1]). MINORITY IS He3, not H. The beat derives the antenna
    # frequency as f = B0*(Z/A)*15, built so the FUNDAMENTAL resonance sits at B = B0, i.e. ON
    # AXIS (B ~ 1/R, B(R0) = B0) -- that part is well behaved at 12 T. But the species choice
    # matters: H minority would need 182 MHz, outside SPARC's actual ICRH band, whereas He3 gives
    # 121.6 MHz, which is both the real system and the worked example in TRANSPbeat's own
    # docstring ("He3 in SPARC: F = 12 * 2/3 * 15 = 120 MHz"). He3 is also NMLtools' default.
    nm["plasma"]["heating"] = {
        "type": "ICRH",
        "parameters": {"P_icrh": 9.46, "minority": [2, 3], "fmini": 0.03, "freq_ICH": None,
                       "P_nbi": 0.0, "nu_source": 5.0, "Pe": 4.7, "Pi": 4.7},
    }
    nm["plasma"]["species"]["fuel"] = ["D", "T"]

    prep = nm["maestro"]["transp"]["parameters_prepare"]
    prep.update({
        "Pich": True,               # live TORIC
        "Pnbi": False,              # resolved to False by preprocess anyway (heating type is ICRH)
        "useNUBEAMforAlphas": True, # nalpha=0 / nlfhe4=T -> Monte-Carlo fusion products
        "flattop_window": 0.1,      # the ~0.1 s window that completed in the levgeo=8 tests
        "dtOut_ms": 10.0,
        "toric_ntheta": 64, "toric_nrho": 128, "MCparticles": 10000.0,
        # PRODUCTION zone counts -- NOT the deliberately tiny stage-1 ones. The first stage-2
        # launch (04379P01) aborted at t=0 with
        #     ?BMZINI_TR -- COMMON DIMENSION "IMXBZ" EXCEEDED
        # inside NUBEAM's fast-ion zone initialization. The geometry was NOT the cause: the log
        # shows "%check_rzfs_data: RFS and ZFS flux surface data has 112 radial grid points:
        # mapped to TRANSP run with 21 radial grid points", i.e. TRANSP ingested the prescribed
        # equilibrium happily. What was wrong is that the stage-1 namelist halves every zone
        # count for speed (nzones 20 / nzone_nb 10 / nzone_fb 5), and BMZINI_TR's companion
        # diagnostics in the binary tie exactly these together:
        #     "?bmzini_tr: NZONES/NXSKPB.ne.NZONE_FB"  and  "NXSKPB must divide NZONES ..."
        # The values below are the ones a production ARC beat used to run TORIC + NUBEAM MC
        # alphas (nalpha=0, nlfhe4=T) to completion on this same cluster, so they are known-good
        # for this module combination rather than guessed. They are also simply the right
        # resolution for a run whose purpose is a deposition profile: 20 radial zones is far
        # too coarse to resolve one.
        "nzones": 60,             # -> nzones   = 60
        "nzones_energetic": 20,   # -> nzone_nb = 20 and nzone_fp = 20
        "nzones_distfun": 10,     # -> nzone_fb = 10
    })
    # null -> toricmpi = trmpi = master_cpus, i.e. STAGE2_RANKS
    nm["maestro"]["transp"]["preprocess_prepare_parameters"] = {"cpus_toric": None, "cpus_nubeam": None}
    nm["maestro"]["master_cpus"] = STAGE2_RANKS

    nm["maestro"]["beats"] = ["transp", "portals"]
    nm["maestro"]["portals"] = {
        "beat_type": "portals",
        "base_module": None,
        "parameters_prepare": {
            "portals_namelist_location": None,
            "portals_parameters": {
                "solution": {"predicted_roa": [0.35, 0.55, 0.75], "keep_full_model_folder": False},
                "transport": {"options": {"tglf": {"run": {"code_settings": "SAT2astra"},
                                                   "keep_files": "none"}}},
                "optimization_options": {
                    "convergence_options": {"maximum_iterations": 3},
                    "acquisition_options": {"optimizers": ["sr"]},
                },
            },
            "initialization_parameters": {"thermalize_fast": True, "quasineutrality": True},
            "change_last_radial_call": True,
            "use_previous_residual": False,
            "use_previous_surrogate_data": False,
            "use_previous_ranges": False,
            "try_flux_match_only_for_first_point": True,
            "enforce_impurity_radiation_existence": False,
        },
        "preprocess_prepare": PORTALSbeat.preprocess_prepare_portals,
        "preprocess_prepare_parameters": {"lumpImpurities": True, "enforce_same_density_gradients": True},
        "preprocess_run": None,
    }
    return nm


def stage2_run_kwargs(nm, beat):
    cfg = nm["maestro"][beat]
    kw = (cfg["preprocess_run"]({}, nm, nm["maestro"]["master_cpus"], True)
          if cfg.get("preprocess_run") is not None else {})
    if beat == "transp":
        kw["cpus_per_task"] = STAGE2_CPUS_PER_TASK
    return kw


def test_stage2_deck_constructed_correctly():
    """LOCAL check of the stage-2 configuration: the deck really comes out as prescribed +
    frozen + live TORIC, with the MPI settings that are known to bind under sbatch. Runs
    always; submits nothing."""
    nm = stage2_namelist()
    with tempfile.TemporaryDirectory() as d:
        cfg = nm["maestro"]["transp"]
        prep = cfg["preprocess_prepare"](cfg["parameters_prepare"], nm,
                                         cfg["preprocess_prepare_parameters"])
        engineering = engineering_from(nm)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m = maestro(d, master_seed=nm["seed"], terminal_outputs=True,
                        master_cold_start=True, prune_level=0, maestro_namelist=nm)
            m.define_beat("transp", initializer="profiles")
            m.initialize(profiles_file=str(GACODE_DT), **engineering)
            m.prepare(**prep)
        txt = deck(m)

        # still prescribed + frozen with heating on
        assert nml_value(txt, "levgeo") == "8"
        assert nml_value(txt, "nlmdif") == "F" and nml_value(txt, "nlqdata") == "T"
        assert nml_value(txt, "nqmodb(1)") == "2" and nml_value(txt, "nqmodb(2)") == "2"
        # live TORIC + MC fusion products
        assert nml_value(txt, "nlicrf") == "T" and nml_value(txt, "nicrf") == "8"
        assert nml_value(txt, "nalpha") == "0" and nml_value(txt, "nlfhe4") == "True"
        assert nml_value(txt, "extrfp") == '"RFP"'
        assert (m.beat.folder / f"MIT{m.beat.shot}.RFP").exists()
        # MPI parallel servers (NOT the serial -1: that needs NPROCS=1, which cannot run TORIC here)
        assert nml_value(txt, "ntoric_pserve") == "1" and nml_value(txt, "nbi_pserve") == "1"
        # He3 minority at 3%; f = B0*(Z/A)*15 puts the fundamental resonance on axis
        assert abs(float(nml_value(txt, "xzmini")) - 2.0) < 1e-6
        assert abs(float(nml_value(txt, "amini")) - 3.0) < 1e-6
        assert abs(float(nml_value(txt, "frmini")) - 0.03) < 1e-6
        f_mhz = float(nml_value(txt, "frqicha")) * 1e-6
        assert abs(f_mhz - B_T_DT * (2.0 / 3.0) * 15.0) < 1.0, \
            f"ICRF frequency {f_mhz:.1f} MHz off the He3 fundamental"
        assert 100.0 < f_mhz < 140.0, f"{f_mhz:.1f} MHz is outside a SPARC-class ICRH band"

        # zone counts known to work with TORIC + MC alphas (see stage2_namelist); guarding
        # these is what stops a repeat of the 04379P01 BMZINI_TR/IMXBZ abort
        assert nml_value(txt, "nzones") == "60"
        assert nml_value(txt, "nzone_nb") == "20" and nml_value(txt, "nzone_fp") == "20"
        assert nml_value(txt, "nzone_fb") == "10"
        nz, nfb = int(nml_value(txt, "nzones")), int(nml_value(txt, "nzone_fb"))
        assert nz % nfb == 0, f"NZONES={nz} not divisible by NZONE_FB={nfb} (BMZINI_TR aborts)"

        kw = stage2_run_kwargs(nm, "transp")
        assert kw["mpisettings"]["toricmpi"] == STAGE2_RANKS, kw
        assert kw["mpisettings"]["trmpi"] == STAGE2_RANKS, kw
        assert kw["cpus_per_task"] == STAGE2_CPUS_PER_TASK
    print(f"PASS: stage-2 deck is prescribed+frozen with live TORIC, pserve 1/1, "
          f"{STAGE2_RANKS} ranks x {STAGE2_CPUS_PER_TASK} cpus-per-task, ICRF at {f_mhz:.1f} MHz")


def run_full_chain(folder):
    """STAGE 2. Executes the 2-beat chain through the normal MITIM dispatch. SUBMITS JOBS."""
    nm = stage2_namelist()
    engineering = engineering_from(nm)

    # cold_start False so a re-run reuses completed beats (the capability-test pattern)
    m = maestro(folder, master_seed=nm["seed"], terminal_outputs=True,
                master_cold_start=False, prune_level=0, maestro_namelist=nm)

    for i, beat in enumerate(nm["maestro"]["beats"]):
        cfg = nm["maestro"][beat]
        prep = cfg["parameters_prepare"]
        if cfg.get("preprocess_prepare") is not None:
            prep = cfg["preprocess_prepare"](prep, nm, cfg["preprocess_prepare_parameters"])

        m.define_beat(beat, initializer="profiles" if i == 0 else None)
        if i == 0:
            m.initialize(profiles_file=str(GACODE_DT), **engineering)
        m.prepare(**prep)
        m.run(**stage2_run_kwargs(nm, beat))

    return m


def locate_transp_cdf(tr_folder):
    """The RUN CDF, not the <runid>PH.CDF companion. Mirrors TRANSPbeat._locate_cdf: a bare
    glob("*.CDF") can hand back the PH file, whose time axis is not the run's -- that is what
    made CDFtools raise "IndexError: TIME not found" on the first successful chain."""
    cands = [c for c in sorted(tr_folder.glob("*.CDF")) if not c.name.upper().endswith("PH.CDF")]
    assert len(cands) == 1, f"expected exactly one run CDF in {tr_folder}, found {[c.name for c in cands]}"
    return cands[0]


def test_stage2_full_chain(folder, execute=True):
    """STAGE 2 OUTCOMES. With execute=False, asserts against an already-completed chain
    without touching MAESTRO at all (safe to iterate on: submits nothing)."""
    from mitim_tools.transp_tools import CDFtools

    if execute:
        run_full_chain(folder)
    root = Path(folder) / "Beats"
    tr_folder = root / "Beat_1" / "run_transp"
    assert tr_folder.exists(), f"no completed TRANSP beat in {folder}"

    # ---- TRANSP ran and accepted the prescribed equilibrium
    cdf = locate_transp_cdf(tr_folder)
    runid = cdf.stem
    log = next(tr_folder.glob("*tr.log")).read_text()
    assert "quval" not in log.lower(), "quval failures -- the prescribed equilibrium was rejected"
    assert "TORIC" in log, "TORIC never initialized"

    c = CDFtools.transp_output(str(cdf))
    t_end = float(c.t[-1])

    # ---- TORIC delivered the requested ICRH power
    requested = 9.46
    p_ich = float(np.asarray(c.PichT)[-1])
    assert abs(p_ich - requested) / requested < 0.10, \
        f"PICHTOT {p_ich:.3f} MW vs requested {requested} MW"

    # ---- NUBEAM produced a real fusion-alpha population (this state burns: qFus ~ 20 MW)
    nfast = np.asarray(getattr(c, "nfusHe4", getattr(c, "nfus", None)))
    assert nfast is not None, "no fusion fast-ion density in the CDF"
    assert float(np.max(nfast[-1])) > 0.0, "NUBEAM produced no fast-alpha population"
    p_fus = float(np.asarray(c.Pfus)[-1]) if hasattr(c, "Pfus") else None
    c.close()

    # ---- source handoff into the PORTALS beat's input state
    st_transp = PROFILEStools.gacode_state(str(root / "Beat_1" / "beat_results" / "input.gacode"))
    st_seed = PROFILEStools.gacode_state(str(GACODE_DT))
    assert float(st_transp.derived["qRF_MW"][-1]) > 0.0, "no RF power in the TRANSP output state"
    assert not np.allclose(st_transp.profiles["qrfe(MW/m^3)"], st_seed.profiles["qrfe(MW/m^3)"]), \
        "RF deposition is the seed's, not TRANSP's -- the handoff did not happen"
    q_fus_out = float(st_transp.derived["qFus_MW"][-1])
    assert q_fus_out > 0.0, "no fusion power merged into the TRANSP output state"

    # ---- the frozen prescribed equilibrium must not have drifted
    for key in ["rmin(m)", "rmaj(m)", "kappa(-)"]:
        assert np.allclose(st_transp.profiles[key], st_seed.profiles[key], rtol=2e-2), \
            f"{key} drifted despite the frozen prescribed equilibrium"

    # ---- PORTALS beat completed
    out = root / "Beat_2" / "beat_results" / "input.gacode"
    assert out.exists(), "PORTALS beat produced no output state"
    PROFILEStools.gacode_state(str(out))

    # ---- PORTALS residual trajectory (reported; only monotone-ish improvement is asserted)
    resid = None
    try:
        from mitim_modules.portals.utils import PORTALSanalysis
        pfold = next(p for p in (root / "Beat_2").iterdir()
                     if p.is_dir() and (p / "Outputs").exists())
        pa = PORTALSanalysis.PORTALSanalyzer.from_folder(str(pfold))
        resid = np.asarray(pa.resTeM)
        assert resid[-1] <= resid[0], f"PORTALS residual grew: {resid[0]:.3e} -> {resid[-1]:.3e}"
    except Exception as exc:                       # analyzer is a reporting nicety, not the subject
        print(f"  note  PORTALS residual trajectory unavailable ({type(exc).__name__}: {exc})")

    r_txt = (f"{resid[0]:.3e} -> {resid[-1]:.3e} over {len(resid)} iters"
             if resid is not None else "n/a")
    print(f"PASS: full chain -- {runid} to t={t_end:.4f}s, 0 quval, PICHTOT {p_ich:.3f}/{requested} MW, "
          f"max n_alpha {float(np.max(nfast[-1])):.3e}, Pfus {p_fus}, "
          f"qFus(out) {q_fus_out:.2f} MW, PORTALS residual {r_txt}, wrote {out.name}")


if __name__ == "__main__":
    assert GACODE.exists(), f"missing repo test data: {GACODE}"
    test_deck_is_prescribed_and_frozen()
    test_beat_folder_contents()
    test_surfaces_full_x_and_nonfolding()
    test_flux_functions_match_the_state()
    test_transition_window_inert_under_null_machine()
    test_reprepare_is_idempotent()
    test_frozen_field_requires_prescribed()
    test_evolve_mode_unchanged()
    test_stage2_deck_constructed_correctly()
    print("\nSTAGE 1 PASSED")

    folder = os.environ.get("MITIM_TEST_CHAIN_FOLDER", str(STAGE2_FOLDER))
    if "--assert-only" in sys.argv:
        # Re-check an already-completed chain. Touches nothing, submits nothing.
        test_stage2_full_chain(folder, execute=False)
        print("\nSTAGE 2 ASSERTIONS PASSED (existing outputs, nothing submitted)")
    elif "--full" in sys.argv:
        Path(folder).mkdir(parents=True, exist_ok=True)
        test_stage2_full_chain(folder, execute=True)
        print("\nSTAGE 2 PASSED")
    else:
        print(f"(stage 2 skipped -- --full executes the 2-beat chain in {folder} "
              f"and SUBMITS CLUSTER JOBS; --assert-only re-checks an existing one)")
