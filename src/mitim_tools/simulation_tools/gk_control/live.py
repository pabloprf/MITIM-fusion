"""
Live state of the radii of a running CGYRO evaluation, read from its scratch without touching the job
(the "CGYRO live" tab, mitim_kill_cgyro), and the stop request that mitim_kill_cgyro drops.
"""
import json
import os
import shlex
import shutil
import time
from pathlib import Path
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.simulation_tools.gk_control import markers
from mitim_tools.misc_tools import FARMINGtools
from mitim_tools.misc_tools.LOGtools import printMsg as print


# The minimal read set, plus input.cgyro (PRINT_STEP, MAX_TIME) and .mitim_t0 (simulated time at
# launch). out.cgyro.time goes LAST: copied after bin.cgyro.ky_flux it can only be longer, which
# CGYROoutput._reconcile_time_vector trims.
_LIVE_FILES = [
    "input.cgyro", "input.cgyro.gen", markers.T0,
    "bin.cgyro.geo", "out.cgyro.egrid", "out.cgyro.equilibrium", "out.cgyro.grids", "out.cgyro.hosts",
    "out.cgyro.memory", "out.cgyro.mpi", "out.cgyro.prec", "out.cgyro.rotation", "out.cgyro.startups",
    "out.cgyro.version", "out.cgyro.info", "out.cgyro.timing", markers.BUDGET_TAG, markers.DISCARD_TAG,
    "bin.cgyro.freq", "bin.cgyro.ky_cflux", "bin.cgyro.ky_flux", "out.cgyro.time",
]


def live_source_from_submission(submission_json):
    '''Where a submitted (run_type submit) CGYRO job runs, from its cgyro_submission.json:
    (machineSettings, scratch folder, [(subfolder, rho), ...]).'''
    meta = json.loads(Path(submission_json).read_text())
    pairs = [(sub, float(rho)) for sub, rhos in meta["kwargs_organize"]["code_executor"].items() for rho in rhos]
    return meta["job"]["machineSettings"], meta["job"]["folderExecution"], pairs


def live_source_from_bash(tmp_folder):
    '''
    Where a run_type normal (bash) CGYRO job runs, when no submission JSON exists: the scratch folder
    is the first `cd` of the execution script staged in tmp_<code>, and the radii are the staged
    <subfolder>/rho_* folders. Only LOCAL scratch is supported this way (the machine is not recorded),
    so this returns None when the folder is not on this filesystem.
    '''
    tmp_folder = Path(tmp_folder)
    for script in sorted(tmp_folder.glob("mitim_bash*.src")) + sorted(tmp_folder.glob("mitim_shell_executor*.sh")):
        for line in script.read_text().splitlines():
            if line.strip().startswith("cd "):
                folder_execution = shlex.split(line.strip())[1]
                if not Path(folder_execution).is_dir():
                    return None
                pairs = sorted((d.parent.name, float(d.name.split("rho_")[-1])) for d in tmp_folder.glob("*/rho_*") if d.is_dir())
                return {"machine": "local"}, folder_execution, pairs
    return None


def fetch_live_outputs(machine_settings, folder_execution, pairs, local_folder, files=None):
    '''
    Copy the in-progress outputs of every radius from the scratch folder (local copy, or SFTP get from a
    remote machine) into local_folder as <file>_<rho:.4f>, the layout of retrieved results, so the
    standard reader applies. Read-only on the scratch side: no tarball, no renames, the job is untouched.
    Returns {rho: {'mtime': last write of out.cgyro.time (epoch s) or None, 'machine': name}}.
    '''
    local_folder = Path(local_folder)
    local_folder.mkdir(parents=True, exist_ok=True)
    job = FARMINGtools.mitim_job(local_folder)
    job.machineSettings = machine_settings
    job.connect()
    info = {}
    try:
        for sub, rho in pairs:
            info[rho] = {"machine": machine_settings["machine"], "mtime": None}
            remote = f"{folder_execution}/{sub}/{SIMtools.rho_folder(rho)}"
            for name in (files or _LIVE_FILES):
                src, dst = f"{remote}/{name}", local_folder / f"{name}_{rho:.4f}"
                try:
                    if job.sftp is None:
                        shutil.copyfile(src, dst)
                        mtime = os.stat(src).st_mtime
                    else:
                        job.sftp.get(src, str(dst))
                        mtime = job.sftp.stat(src).st_mtime
                except OSError:
                    continue
                if name == "out.cgyro.time":
                    info[rho]["mtime"] = mtime
    finally:
        job.close()
    return info


def link_finished_radii(run_folder, local_folder, live_rhos, files=None):
    '''
    Radii of the running evaluation that already ended and were retrieved into the run folder (a job
    relaunched for the radii left only lists those): symlink their outputs into local_folder next to
    the live ones, so the live tab shows the whole evaluation. Read-only on the run folder.
    Returns {rho: {'mtime': last write of out.cgyro.time (epoch s), 'machine': 'run folder'}}.
    '''
    run_folder, local_folder = Path(run_folder), Path(local_folder)
    live = {f"{rho:.4f}" for rho in live_rhos}
    info = {}
    for time_file in sorted(run_folder.glob("out.cgyro.time_*")):
        tag = time_file.name.split("_")[-1]
        if tag in live:
            continue
        for name in (files or _LIVE_FILES):
            src = run_folder / f"{name}_{tag}"
            if src.is_file():
                (local_folder / src.name).symlink_to(src)
        info[float(tag)] = {"machine": "run folder", "mtime": time_file.stat().st_mtime}
    return info


# Enough to tell, per radius, how far it got and whether it already ended (EXIT line, watchdog tags)
_STOP_STATUS_FILES = ["input.cgyro", markers.T0, "out.cgyro.info", markers.BUDGET_TAG, markers.DISCARD_TAG, markers.STOP_REQUEST, "out.cgyro.time"]


def live_radii_state(machine_settings, folder_execution, pairs, local_folder):
    '''
    Per-radius state of a running CGYRO job, read from its scratch without touching it:
    {(subfolder, rho): {'t', 't0', 'MAX_TIME', 'exit', 'budget', 'discard', 'stop_requested', 'mtime'}}.
    't' is the last simulated time in out.cgyro.time; 'mtime' its last write (epoch s).
    '''
    local_folder = Path(local_folder)
    state = {}
    for sub, rho in pairs:
        dst = local_folder / sub
        info = fetch_live_outputs(machine_settings, folder_execution, [(sub, rho)], dst, files=_STOP_STATUS_FILES)[rho]
        scal = _read_live_scalars(dst, rho)
        try:
            t = float((dst / f"out.cgyro.time_{rho:.4f}").read_text().split("\n")[-2].split()[0])
        except (OSError, IndexError, ValueError):
            t = None
        state[(sub, rho)] = {
            "t": t, "t0": scal.get("t0"), "MAX_TIME": scal.get("MAX_TIME"), "exit": scal.get("exit"),
            "budget": (dst / f"{markers.BUDGET_TAG}_{rho:.4f}").exists(),
            "discard": (dst / f"{markers.DISCARD_TAG}_{rho:.4f}").exists(),
            "stop_requested": (dst / f"{markers.STOP_REQUEST}_{rho:.4f}").exists(),
            "mtime": info["mtime"],
        }
    return state


def request_stop(machine_settings, folder_execution, pairs, note="mitim_kill_cgyro"):
    '''
    Drop a mitim_stop file in the scratch folder of each (subfolder, rho) in pairs. The watchdog
    around the running launch (CGYRO._wall_budget_wrap) picks it up within ~20 s, waits for the next
    restart write, leaves mitim_budget.tag and stops CGYRO; the radius is then read as finished,
    with its fluxes averaged over what it simulated. Launches started before the watchdog wrapped
    main radii (MITIM older than this function) ignore the file.
    '''
    job = FARMINGtools.mitim_job(Path.cwd())
    job.machineSettings = machine_settings
    job.connect()
    try:
        for sub, rho in pairs:
            path = f"{folder_execution}/{sub}/{SIMtools.rho_folder(rho)}/{markers.STOP_REQUEST}"
            text = f"{note} {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            if job.sftp is None:
                Path(path).write_text(text)
            else:
                with job.sftp.open(path, "w") as f:
                    f.write(text)
            print(f"\t- Stop requested for {sub}/rho_{rho:.4f} ({machine_settings['machine']}:{path})")
    finally:
        job.close()


def _read_live_scalars(folder, rho):
    '''PRINT_STEP, DELTA_T and MAX_TIME from input.cgyro, simulated time at launch (.mitim_t0) and the EXIT line of out.cgyro.info.'''
    out = {}
    try:
        for line in (folder / f"input.cgyro_{rho:.4f}").read_text().splitlines():
            key, _, val = line.partition("=")
            if key.strip() in ("PRINT_STEP", "DELTA_T", "MAX_TIME") and val.split():
                out[key.strip()] = float(val.split()[0])
    except (OSError, ValueError):
        pass
    try:
        out["t0"] = float((folder / f"{markers.T0}_{rho:.4f}").read_text().strip() or 0.0)
    except (OSError, ValueError):
        pass
    try:
        out["exit"] = next((l.strip() for l in (folder / f"out.cgyro.info_{rho:.4f}").read_text().splitlines() if "EXIT" in l), None)
    except OSError:
        pass
    try:
        # a watchdog / cost-guard / manual stop leaves no EXIT line but a tag, and the radius counts as finished
        out["tag"] = (folder / f"{markers.BUDGET_TAG}_{rho:.4f}").read_text().split()[0]
    except (OSError, IndexError):
        pass
    return out


def _find_running_cgyro_evaluation(iter_folders, base_subfolder):
    '''
    Where the evaluation still in flight is running: (iteration, scratch source) with the source as
    live_source_* returns it, or None. A submitted job is the latest evaluation whose
    cgyro_submission.json exists (written at submit, deleted once its results are read); without one,
    a bash-mode job of the LAST evaluation is found from its staged execution script (local scratch only).
    '''
    from mitim_tools.gacode_tools import CGYROtools

    for it, folder in reversed(iter_folders):
        path = folder / base_subfolder / CGYROtools.CGYRO._submission_metadata_filename
        if path.is_file():
            return it, live_source_from_submission(path)
    if iter_folders:
        it, folder = iter_folders[-1]
        tmp = folder / "tmp_cgyro"
        if tmp.is_dir():
            source = live_source_from_bash(tmp)
            if source is not None:
                return it, source
    return None


def locate_running_cgyro(root_folder):
    '''
    The CGYRO evaluation of a PORTALS run still in flight, or None: a dict with the iteration 'it',
    'machine_settings', 'folder_execution' (scratch), 'pairs' [(subfolder, rho), ...], the run's
    'read_kwargs' and 'base_subfolder', plus 'params' (namelist) and 'iter_folders'.
    Settings come from the run's namelist.portals.yaml (the merged parameters prep() writes), not from
    a powerstate, so this works before any evaluation has finished. Plain YAML load on purpose:
    read_mitim_yaml would execute the run's import:: sidecar functions.
    '''
    import yaml
    from mitim_modules.portals.utils.PORTALSanalysis import _model_highest_fidelity, _resolve_code_from_options
    from mitim_modules.portals.utils.PORTALSplot import _iterate_portals_evaluation_folders

    root_folder = Path(root_folder)
    try:
        params = yaml.safe_load((root_folder / "namelist.portals.yaml").read_text())
        transport = params["transport"]
        cgyro_key = _model_highest_fidelity(transport["evaluator_instance_attributes"]["turbulence_model"])
    except (OSError, KeyError, TypeError, yaml.YAMLError) as e:
        print(f"\t- Could not read the run namelist ({type(e).__name__}: {e}) to locate the running CGYRO evaluation", typeMsg='w')
        return None
    if _resolve_code_from_options(transport, cgyro_key) != "cgyro":
        return None
    read_cfg = ((transport.get("options") or {}).get(cgyro_key) or {}).get("read") or {}
    base_subfolder = f"base_{cgyro_key}"
    iter_folders = list(_iterate_portals_evaluation_folders(root_folder))

    found = _find_running_cgyro_evaluation(iter_folders, base_subfolder)
    if found is None:
        return None
    it, (machine_settings, folder_execution, pairs) = found
    return {
        "it": it, "machine_settings": machine_settings, "folder_execution": folder_execution, "pairs": pairs,
        "read_kwargs": {k: v for k, v in read_cfg.items() if k in ("tmin", "tmin_is_rel", "last_tmin_for_linear", "averaging")},
        "base_subfolder": base_subfolder, "params": params, "iter_folders": iter_folders,
    }
