"""
QuaLiKiz interface for MITIM.

This is a QuaLiKiz counterpart to ``mitim_tools.gacode_tools.TGLFtools``, scoped
to the run/read/save *interface* only (no plotting, no in-process/ctypes
engine -- see CLAUDE.md decision: TGLFtools.py's in-process mixin and ~5000
lines of TGLF-only plotting (SAT comparisons, wavefunctions, ky-spectral
field/fluctuation plots, per-mode QL-flux contributors) have no QuaLiKiz
equivalent and are intentionally not ported).

Input/output handling is NOT hand-rolled here. It delegates to QuaLiKiz's own
Python API, shipped as the ``qualikiz_tools`` package in the
``QuaLiKiz-pythontools`` repo (a submodule of the QuaLiKiz repo) -- install
it (e.g.
``pip install -e <path>/QuaLiKiz/QuaLiKiz-pythontools``) so ``import
qualikiz_tools`` resolves:

    * Inputs are built as a ``qualikiz_tools.qualikiz_io.inputfiles.QuaLiKizPlan``
      (a ``QuaLiKizXpoint`` base + a "parallel" scan across ``self.rhos``), and
      materialized to the real binary ``input/*.bin`` files QuaLiKiz's Fortran
      executable reads via ``qualikiz_tools.qualikiz_io.qualikizrun.QuaLiKizRun``
      (``.prepare()`` + ``.generate_input()``). This also gets quasineutrality
      enforcement and the pure-toroidal-rotation decomposition (Machtor/gammaE
      -> Autor/Machpar/Aupar) "for free" from the library instead of
      hand-rolled approximations.
    * Outputs are read back via
      ``qualikiz_tools.qualikiz_io.outputfiles.qualikiz_folder_to_xarray``,
      which returns a fully-dimensioned ``xarray.Dataset`` (dimx/dimn/nions/
      numsols handled by the library's own datamap, not guessed here).

Because all ``self.rhos`` are mapped onto QuaLiKiz's own scan mechanism
(``dimx``), one MITIM call produces a single QuaLiKiz run directory (one
binary execution, internally MPI-parallel over dimx*dimn) rather than one
folder per rho as in TGLF/NEO.

Caveats (please verify against your setup):
    * ``run_specifications['code_call']`` assumes a ``QuaLiKiz`` executable is
      reachable on PATH on the execution machine (sourced via the `modules`
      string of that machine's entry in ``config_user.json``, exactly like
      `tglf`/`cgyro`/etc.) -- adjust if your build is invoked differently.
    * The mapping from a MITIM ``gacode_state`` to QuaLiKiz's circular/
      s-alpha-like geometry is split across two layers: ``MITIMstate.gacode_state.to_qualikiz``
      (a best-effort analogue of ``to_tglf`` / ``to_neo`` that has no dependency
      on ``qualikiz_tools`` and returns a plain dict) and this module's
      ``_build_plan_from_gacode`` (which turns that dict into the actual
      ``QuaLiKizXpoint`` / ``QuaLiKizPlan`` objects). QuaLiKiz does not support
      Miller/MXH shaping, so shaped-equilibrium information is dropped.
"""

import copy
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np

from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gacode_tools.utils import NORMtools
from mitim_tools.misc_tools import IOtools, FARMINGtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools.PLASMAtools import md_u
from mitim_tools.qualikiz_tools.utils import QLKdefaults

from qualikiz_tools.qualikiz_io.inputfiles import (
    Electron,
    Ion,
    IonList,
    QuaLiKizXpoint,
    QuaLiKizPlan,
)
from qualikiz_tools.qualikiz_io.qualikizrun import QuaLiKizRun, find_qualikiz_binary
from qualikiz_tools.qualikiz_io.outputfiles import qualikiz_folder_to_xarray

MAX_QLK_IONS = 9  # Soft cap on the number of ion species sent to QuaLiKiz


# ======================================================================================
# Main class
# ======================================================================================

class QuaLiKiz:
    """
    QuaLiKiz wrapper. File-based (subprocess / SLURM) only -- no in-process engine.
    """

    def __init__(
        self,
        rhos=[0.4, 0.6],   # rho locations of interest (mapped onto QuaLiKiz's own dimx scan)
        alreadyRun=None,   # Option: reuse an already-run class instance (to inspect without re-running)
    ):

        self.rhos = np.array(rhos) if rhos is not None else None

        self.results, self.scans = {}, {}

        def code_call(folder, n=1, additional_command=""):
            # Assumes a QuaLiKiz executable is on PATH on the execution
            # machine (sourced via that machine's `modules` config), exactly
            # like TGLF's `tglf` / CGYRO's `cgyro` commands. Adjust if your
            # build needs a different invocation.
            return f"cd {folder} && mpirun -np {n} QuaLiKiz {additional_command}"

        self.run_specifications = {
            "code": "qualikiz",
            "code_call": code_call,
        }

        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t QuaLiKiz class module")
        print("-----------------------------------------------------------------------------------------\n")

        if alreadyRun is not None:
            self.__class__ = alreadyRun.__class__
            self.__dict__ = alreadyRun.__dict__
            print("* Readying previously-run QuaLiKiz class", typeMsg="i")
        else:
            # Whole-folder retrieval -- which exact *.dat files exist depends on
            # meta flags (separateflux, phys_meth, write_primi, ...), so the run
            # folder's output/, output/primitive/ and debug/ subfolders are
            # retrieved wholesale rather than enumerated by filename.
            self.output_folders_simulation = ["output", "output/primitive", "debug"]

            self.NormalizationSets = {
                "TRANSP": None,
                "PROFILES": None,
                "TGYRO": None,
                "EXP": None,
                "input_gacode": None,
                "SELECTED": None,
            }

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prep(self, mitim_state, FolderGACODE, code_settings=None, cold_start=False, forceIfcold_start=False):
        """
        Prepare a QuaLiKiz plan from a MITIM state class (or a path to input.gacode).
        """

        print("> Preparation run from MITIM state class")

        self.FolderGACODE = IOtools.expandPath(FolderGACODE)

        if cold_start or not self.FolderGACODE.exists():
            IOtools.askNewFolder(self.FolderGACODE, force=forceIfcold_start)

        if isinstance(mitim_state, (str, Path)):
            self.profiles = PROFILEStools.gacode_state(mitim_state)
        else:
            self.profiles = mitim_state

        self.profiles.write_state(file=self.FolderGACODE / "input.gacode_torun")
        self.profiles.derive_quantities(mi_ref=md_u)

        self.plan = _build_plan_from_gacode(self.profiles, self.rhos, code_settings=code_settings)

        print("> Setting up normalizations")
        self.NormalizationSets, cdf = NORMtools.normalizations(self.profiles)

        return cdf

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def run(
        self,
        subfolder,                 # 'run1/'
        code_settings=None,        # None or a QLKdefaults preset (e.g. "FAST")
        extraOptions={},           # dict of {key: value, or list of length len(rhos) for scanned keys}
        multipliers={},            # dict of {key: multiplicative factor, or list of length len(rhos)}
        cold_start=False,
        forceIfcold_start=False,
        launchSlurm=True,
        allocation=None,           # {'resources_per_call': int, 'minutes': int}
        binaryrelpath=None,        # Path to the QuaLiKiz binary (only used for QuaLiKizRun's local bookkeeping symlink)
        attempts_execution=1,
        extra_name="exe",
    ):
        """
        Synchronous run only (send + execute + wait + retrieve). Async
        submit/check/fetch and ``QuaLiKizBatch``-style multi-run management are
        not ported here -- this module covers the single-shot run/read/save path.
        """

        plan = _modify_plan(copy.deepcopy(self.plan), code_settings=code_settings, extraOptions=extraOptions, multipliers=multipliers)

        return self._submit_plan(
            subfolder,
            plan,
            cold_start=cold_start,
            launchSlurm=launchSlurm,
            allocation=allocation,
            binaryrelpath=binaryrelpath,
            attempts_execution=attempts_execution,
            extra_name=extra_name,
        )

    def run_cases(
        self,
        subfolder,                 # 'turb_drives/'
        cases,                      # list of dicts {scan_dict_key: multiplier}, one per perturbation case
        code_settings=None,        # None or a QLKdefaults preset (e.g. "FAST")
        cold_start=False,
        forceIfcold_start=False,
        launchSlurm=True,
        allocation=None,           # {'resources_per_call': int, 'minutes': int}
        binaryrelpath=None,
        attempts_execution=1,
        extra_name="exe",
    ):
        """
        Run many perturbation cases (e.g. the gradient-scan points used for
        scan-trick uncertainty estimation) in a SINGLE QuaLiKiz execution, by
        stacking them onto QuaLiKiz's own "parallel" dimx scan alongside
        self.rhos, instead of issuing one execution per case.

        Each entry of ``cases`` is a dict of {scan_dict_key: multiplier} (same
        convention as ``run()``'s ``multipliers`` arg). The resulting plan has
        dimx = len(cases) * len(self.rhos), laid out case-major / rho-minor:
        block ``i`` (dimx[i*nrho:(i+1)*nrho]) reuses the base per-rho scan_dict
        values, with every key in cases[i] multiplied by its given factor
        across the whole block (i.e. uniformly over all rho points, same as a
        single-case ``run(multipliers=cases[i])`` would have done).

        Pair with ``read_cases(label, n_cases=len(cases))`` to unstack the
        resulting dataset back into per-case, per-rho slices.

        Note: since one execution now covers len(cases) times as many dimx
        points, `allocation['resources_per_call']` may need to be increased
        relative to a plain ``run()`` call to keep wallclock reasonable.
        """

        nrho = len(self.rhos)

        base_scan_dict = self.plan["scan_dict"]
        stacked_scan_dict = OrderedDict()
        for key, base_vals in base_scan_dict.items():
            stacked_vals = []
            for case in cases:
                mult = case.get(key, 1.0)
                stacked_vals.extend([v * mult for v in base_vals])
            stacked_scan_dict[key] = stacked_vals

        plan = copy.deepcopy(self.plan)
        plan["scan_dict"] = stacked_scan_dict
        if code_settings is not None:
            print(f"\t- Using presets code_settings = {code_settings}", typeMsg="i")
            for ikey, val in QLKdefaults.addQLKcontrol(code_settings).items():
                plan["xpoint_base"][ikey] = val

        return self._submit_plan(
            subfolder,
            plan,
            cold_start=cold_start,
            launchSlurm=launchSlurm,
            allocation=allocation,
            binaryrelpath=binaryrelpath,
            attempts_execution=attempts_execution,
            extra_name=extra_name,
        )

    def _submit_plan(
        self,
        subfolder,
        plan,
        cold_start=False,
        launchSlurm=True,
        allocation=None,
        binaryrelpath=None,
        attempts_execution=1,
        extra_name="exe",
    ):
        """ Shared prepare/generate/run bookkeeping for run() and run_cases(). """

        Folder_sim = self.FolderGACODE / subfolder
        binaryrelpath = binaryrelpath or _resolve_binary(Folder_sim)

        qlk_run = QuaLiKizRun(self.FolderGACODE, subfolder, binaryrelpath, qualikiz_plan=plan)

        needs_run = cold_start or not qlk_run.is_done()

        if needs_run:
            qlk_run.prepare(overwrite=True, overwrite_meta=True, overwrite_imp=True)
            qlk_run.generate_input()

        self.qlk_run = qlk_run
        self.FolderSimLast = qlk_run.rundir

        if needs_run:
            self._run(
                qlk_run,
                launchSlurm=launchSlurm,
                allocation=allocation,
                attempts_execution=attempts_execution,
                extra_name=extra_name,
            )
        else:
            print("\t- QUALIKIZ not run because output already found (please ensure consistency!)", typeMsg="i")
            self.simulation_job = None

        return qlk_run

    def _run(
        self,
        qlk_run,
        launchSlurm=True,
        allocation=None,
        attempts_execution=1,
        extra_name="exe",
    ):

        print("\t- QUALIKIZ needs to run because output was not found", typeMsg="i")

        if allocation is None:
            allocation = {"resources_per_call": 1, "minutes": 10}
        resources_per_call = allocation.get("resources_per_call", 1)

        code = self.run_specifications["code"]
        code_call = self.run_specifications["code_call"]

        rundir = qlk_run.rundir

        # FARMINGtools requires input_folders to be relative to folder_local.
        # rundir (e.g. .../base_qlk) is a direct child of its parent, so using
        # rundir.parent as folder_local satisfies that constraint without an
        # extra copy step.  After retrieval, outputs land at
        # folder_local/base_qlk/... which is already rundir/..., so no
        # post-processing move is needed.
        folder_local = rundir.parent

        self.simulation_job = FARMINGtools.mitim_job(folder_local)
        self.simulation_job.define_machine(
            code,
            f"mitim_{code}_{extra_name}",
            launchSlurm=launchSlurm,
        )

        rundir_red = rundir.name

        command = code_call(folder=rundir_red, n=resources_per_call)

        self.simulation_job.prep(
            command,
            input_folders=[rundir],
            output_folders=[f"{rundir_red}/{f}" for f in self.output_folders_simulation],
        )

        self.simulation_job.run(removeScratchFolders=True, attempts_execution=attempts_execution)

    def _organize_results(self, qlk_run, tmpFolder):

        print("\t- Retrieving output/debug folders")
        rundir = qlk_run.rundir
        rundir_red = rundir.relative_to(rundir.parent).as_posix()

        fineall = True
        for relfolder in self.output_folders_simulation:
            src = tmpFolder / rundir_red / relfolder
            dst = rundir / relfolder
            if not src.exists():
                print(f"\t!! folder {relfolder} could not be retrieved", typeMsg="w")
                fineall = False
                continue
            IOtools.shutil_rmtree(dst)
            src.replace(dst)

        if fineall:
            print("\t\t- All output folders were successfully retrieved")
            IOtools.shutil_rmtree(tmpFolder)
        else:
            print("\t\t- Some output folders were not retrieved", typeMsg="w")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(
        self,
        label="run1",
        folder=None,           # If None, use the last folder run() produced
        input_gacode=None,     # If provided and normalizations aren't set yet, derive them from it
        **kwargs_to_xarray,    # Forwarded to qualikiz_folder_to_xarray (e.g. genfromtxt=True)
    ):
        print("> Reading simulation results")

        if folder is None:
            folder = self.FolderSimLast
        folder = IOtools.expandPath(folder)

        if self.NormalizationSets.get("SELECTED") is None and input_gacode is not None:
            print("\t- Getting normalizations from input.gacode provided in the read function")
            self.NormalizationSets, _ = NORMtools.normalizations(PROFILEStools.gacode_state(input_gacode))

        ds = qualikiz_folder_to_xarray(folder, **kwargs_to_xarray)
        ds = ds.assign_coords(rho=("dimx", np.array(self.rhos)))

        self.results[label] = {
            "dataset": ds,
            "x": np.array(self.rhos),
            # Per-rho slices, mirroring TGLFtools' self.results[label]['output'] list-by-rho shape
            "output": [ds.isel(dimx=i) for i in range(len(self.rhos))],
        }

    def read_cases(
        self,
        label,
        n_cases,                # must match len(cases) passed to the run_cases() call being read
        folder=None,            # If None, use the last folder run_cases() produced
        input_gacode=None,
        **kwargs_to_xarray,
    ):
        """
        Counterpart to read(), for a plan produced by run_cases(): unstacks
        the combined dimx = n_cases * len(self.rhos) dataset (case-major /
        rho-minor, matching run_cases()'s layout) back into per-case,
        per-rho slices.
        """
        print("> Reading stacked-case simulation results")

        if folder is None:
            folder = self.FolderSimLast
        folder = IOtools.expandPath(folder)

        if self.NormalizationSets.get("SELECTED") is None and input_gacode is not None:
            print("\t- Getting normalizations from input.gacode provided in the read function")
            self.NormalizationSets, _ = NORMtools.normalizations(PROFILEStools.gacode_state(input_gacode))

        ds = qualikiz_folder_to_xarray(folder, **kwargs_to_xarray)

        nrho = len(self.rhos)
        if ds.sizes["dimx"] != n_cases * nrho:
            raise ValueError(
                f"read_cases: dimx={ds.sizes['dimx']} does not match "
                f"n_cases({n_cases}) * len(self.rhos)({nrho}) = {n_cases * nrho}"
            )

        ds = ds.assign_coords(
            rho=("dimx", np.tile(np.array(self.rhos), n_cases)),
            case=("dimx", np.repeat(np.arange(n_cases), nrho)),
        )

        self.results[label] = {
            "dataset": ds,
            "x": np.array(self.rhos),
            # output[case][rho], mirroring read()'s per-rho 'output' list with an extra outer per-case axis
            "output": [
                [ds.isel(dimx=icase * nrho + irho) for irho in range(nrho)]
                for icase in range(n_cases)
            ],
        }

    # ------------------------------------------------------------------
    # Saving / restoring
    # ------------------------------------------------------------------

    def prepare_for_save(self):
        if "fn" in self.__dict__:
            del self.fn
        return copy.deepcopy(self)

    def to_netcdf(self, file, label="run1"):
        """ Persist a read-in result as a netCDF file (native xarray format). """
        self.results[label]["dataset"].to_netcdf(file)

    @classmethod
    def from_netcdf(cls, file, rhos):
        import xarray as xr
        instance = cls(rhos=rhos)
        ds = xr.open_dataset(file)
        instance.results["run1"] = {
            "dataset": ds,
            "x": np.array(rhos),
            "output": [ds.isel(dimx=i) for i in range(len(rhos))],
        }
        return instance


# ======================================================================================
# Input construction (QuaLiKizPlan from a MITIM gacode_state)
# ======================================================================================

def _build_plan_from_gacode(profiles, rhos, code_settings=None):
    """
    Turns the physics/geometry mapping computed by
    ``MITIMstate.gacode_state.to_qualikiz`` (a plain dict of arrays/scalars,
    with no dependency on the external ``qualikiz_tools`` package) into an
    actual ``QuaLiKizPlan``, scanned in parallel across ``rhos``.

    The ``qualikiz_tools`` (QuaLiKiz-pythontools) object construction is kept
    here rather than in ``to_qualikiz`` so that optional external dependency
    stays confined to this QuaLiKiz-specific interface -- MITIMstate.py itself
    never needs to import it.
    """

    m = profiles.to_qualikiz(rhos, max_ions=MAX_QLK_IONS)

    n_ions = m["n_ions"]
    geometry_scan = m["geometry_scan"]
    electron_scan = m["electron_scan"]
    ion_scan = m["ion_scan"]
    ion_species = m["ion_species"]

    # Representative (rho[0]) placeholder values for the xpoint_base -- every
    # scanned key below is overwritten per-point for ALL dimx points by
    # QuaLiKizPlan's "parallel" scan, so these never reach the binaries.
    elec = Electron(
        T=float(electron_scan["Te"][0]),
        n=float(electron_scan["ne"][0]),
        At=float(electron_scan["Ate"][0]),
        An=float(electron_scan["Ane"][0]),
        type=1,
        anis=1.0,
        danisdr=0.0,
    )

    ions = IonList(*[
        Ion(
            A=float(ion_species[i]["A"]),
            Z=float(ion_species[i]["Z"]),
            T=float(ion_scan[f"Ti{i}"][0]),
            n=float(ion_scan[f"ni{i}"][0]),
            At=float(ion_scan[f"Ati{i}"][0]),
            An=float(ion_scan[f"Ani{i}"][0]),
            type=ion_species[i]["type"],
            anis=1.0,
            danisdr=0.0,
        )
        for i in range(n_ions)
    ])

    meta_kwargs = QLKdefaults.addQLKcontrol(code_settings)
    kthetarhos = QLKdefaults.default_kthetarhos()

    base = QuaLiKizXpoint(
        kthetarhos=kthetarhos,
        electrons=elec,
        ions=ions,
        # Geometry (placeholders for scanned keys; Rmin/Bo are the only ones held fixed)
        x=float(geometry_scan["x"][0]),
        rho=float(geometry_scan["rho"][0]),
        Ro=float(geometry_scan["Ro"][0]),
        Rmin=m["a"],
        Bo=m["Bo"],
        q=float(geometry_scan["q"][0]),
        smag=float(geometry_scan["smag"][0]),
        alpha=float(geometry_scan["alpha"][0]),
        Machtor=float(geometry_scan["Machtor"][0]),
        Autor=0.0,
        Machpar=0.0,
        Aupar=0.0,
        gammaE=float(geometry_scan["gammaE"][0]),
        # Meta (constant numerical/physics controls)
        **meta_kwargs,
        # Options: quasineutrality + rotation handled by the library, not here.
        set_qn_normni=True,
        set_qn_normni_ion=m["qn_ion_index"],
        set_qn_An=True,
        set_qn_An_ion=m["qn_ion_index"],
        check_qn=True,
        x_eq_rho=False,
        recalc_Nustar=False,       # We supply Te (and hence Nustar) per radius ourselves
        recalc_Ti_Te_rel=False,
        assume_tor_rot=True,
        puretor_abs_var="Machtor",
        puretor_grad_var="gammaE",
        recalc_rot_var=False,
        recalc_Ati=False,
    )

    # geometry_scan / electron_scan dict order already matches the desired
    # scan-key order (see MITIMstate.to_qualikiz).
    scan_dict = OrderedDict()
    for key, vals in geometry_scan.items():
        scan_dict[key] = list(vals)
    for key, vals in electron_scan.items():
        scan_dict[key] = list(vals)
    for i in range(n_ions):
        for prefix in ["Ti", "ni", "Ati", "Ani"]:
            scan_dict[f"{prefix}{i}"] = list(ion_scan[f"{prefix}{i}"])

    return QuaLiKizPlan(scan_dict=scan_dict, scan_type="parallel", mixed_dict=None, xpoint_base=base)


def _modify_plan(plan, code_settings=None, extraOptions={}, multipliers={}):

    if code_settings is not None:
        print(f"\t- Using presets code_settings = {code_settings}", typeMsg="i")
        for ikey, val in QLKdefaults.addQLKcontrol(code_settings).items():
            plan["xpoint_base"][ikey] = val

    if extraOptions:
        print("\t- External options:")
    for ikey, val in extraOptions.items():
        if ikey in plan["scan_dict"]:
            print(f"\t\t- Setting scanned {ikey} = {val}", typeMsg="i")
            plan["scan_dict"][ikey] = list(val)
        else:
            print(f"\t\t- Setting {ikey} = {val}", typeMsg="i")
            plan["xpoint_base"][ikey] = val

    if multipliers:
        print("\t- Variables change (multipliers):")
    for ikey, mult in multipliers.items():
        if ikey in plan["scan_dict"]:
            var_orig = plan["scan_dict"][ikey]
            var_new = [v * m for v, m in zip(var_orig, mult)] if isinstance(mult, (list, np.ndarray)) else [v * mult for v in var_orig]
            print(f"\t\t- Changing scanned {ikey} (x{mult})")
            plan["scan_dict"][ikey] = var_new
        else:
            var_orig = plan["xpoint_base"][ikey]
            var_new = var_orig * mult
            print(f"\t\t- Changing {ikey} from {var_orig} to {var_new} (x{mult})")
            plan["xpoint_base"][ikey] = var_new

    return plan


def _resolve_binary(start_folder):
    import os
    # 1. Explicit override via environment variable
    env_bin = os.environ.get("QUALIKIZ_BIN")
    if env_bin:
        return Path(env_bin)

    # 2. Library search near the run folder
    try:
        return find_qualikiz_binary(startdir=Path(start_folder))
    except FileNotFoundError:
        pass

    # 3. Fall back to PATH lookup.
    # Do NOT call .resolve() — that follows symlinks and may change the
    # filename (e.g. QuaLiKiz -> QuaLiKiz-3.0.0), which would break the
    # symlink name that QuaLiKizRun.prepare() creates from binaryrelpath.name.
    # shutil.which already returns an absolute path with the correct name.
    which = shutil.which("QuaLiKiz")
    if which:
        return Path(which)

    print("\t- Could not locate a QuaLiKiz binary via QUALIKIZ_BIN, filesystem search, or PATH; symlink will be broken", typeMsg="w")
    return "QuaLiKiz"
