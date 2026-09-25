import shutil
import datetime
import json
import time
import os
import copy
import numpy as np
import dill as pickle_dill
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from mitim_tools import __version__ as mitim_version
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gacode_tools.utils import GACODEdefaults, NORMtools
from mitim_tools.misc_tools import FARMINGtools, IOtools, LOGtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

from mitim_tools.misc_tools.PLASMAtools import md_u

_RUN_TYPE_ALIASES = {'run': 'normal'}


class RunType(Enum):
    '''What `_run` does with the work it has staged.'''
    NORMAL = 'normal'   # send, submit and wait for the results
    SUBMIT = 'submit'   # send and submit, return without waiting
    SEND   = 'send'     # send the inputs, submit nothing
    PREP   = 'prep'     # build the job, send nothing

    @classmethod
    def parse(cls, run_type):
        '''From a RunType or from any of the strings callers pass; 'run' is an alias of 'normal'.'''
        if isinstance(run_type, cls):
            return run_type
        try:
            return cls(_RUN_TYPE_ALIASES.get(run_type, run_type))
        except ValueError:
            raise ValueError(f"[MITIM] run_type {run_type!r} is not one of {[m.value for m in cls]} (or the alias 'run')")

    @property
    def blocking(self):
        '''True when `_run` waits for the machine before returning.'''
        return self in (RunType.NORMAL, RunType.SEND)


def _normalize_run_type(run_type):
    '''Canonical string form, for the callers outside this module that compare against strings.'''
    return _RUN_TYPE_ALIASES.get(run_type, run_type)


class SubmissionType(Enum):
    '''How a work plan reaches the machine; chosen by SLURMtools.resolve, which returns the string.'''
    BASH = 'bash'
    SLURM_STANDARD = 'slurm_standard'
    SLURM_ARRAY = 'slurm_array'

    @classmethod
    def parse(cls, submission_type):
        if isinstance(submission_type, cls):
            return submission_type
        try:
            return cls(submission_type)
        except ValueError:
            raise ValueError(f"[MITIM] submission_type {submission_type!r} is not one of {[m.value for m in cls]}")


class JobStatus(IntEnum):
    '''`mitim_job.status`. An IntEnum, so the plain integer form other modules use still compares equal.'''
    PENDING = 0
    RUNNING = 1
    GONE = 2

# ----------------------------------------------------------------------------------------------------
# Per-radius naming convention (the ONE place it lives; writers and readers must not drift)
# ----------------------------------------------------------------------------------------------------

def rho_suffix(rho):
    '''Suffix of a stored per-radius file: `<file>_<rho>`. Casts, so a rho read back from JSON works.'''
    return f"_{float(rho):.4f}"

def rho_folder(rho):
    '''Name of a per-radius execution folder inside its subfolder.'''
    return f"rho{rho_suffix(rho)}"


_CODE_EXECUTOR_FIELDS = ("folder", "dictionary", "inputs", "extraOptions", "multipliers", "additional_files_to_send")


@dataclass
class RadialCall:
    '''
    One pending (subfolder, rho) execution: what to stage, where its results land,
    and the naming convention that links the two.
    '''
    subfolder: str
    rho: float
    folder: Path = None                  # final destination of the retrieved results
    dictionary: object = None            # per-rho input class
    inputs: str = None                   # text of the input file to stage
    extraOptions: dict = None
    multipliers: dict = None
    additional_files_to_send: list = None
    entry: dict = field(default=None, repr=False)   # the code_executor entry this was built from

    @property
    def rel(self):
        '''Execution folder of this call, relative to the scratch root.'''
        return f"{self.subfolder}/{rho_folder(self.rho)}"

    def result_name(self, file):
        '''Name this call's `file` takes once stored next to the other radii.'''
        return f"{file}{rho_suffix(self.rho)}"

    @classmethod
    def from_entry(cls, subfolder, rho, entry):
        return cls(subfolder, rho, entry=entry,
                   **{k: v for k, v in entry.items() if k in _CODE_EXECUTOR_FIELDS})

    def to_entry(self):
        return self.entry if self.entry is not None else {k: getattr(self, k) for k in _CODE_EXECUTOR_FIELDS}


class WorkPlan:
    '''
    The pending radial calls of one submission, in staging order. Built from (and
    convertible back to) `code_executor`, which stays the external contract.
    '''

    def __init__(self, calls=(), subfolders=None):
        self.calls = list(calls)
        # Kept explicitly: a subfolder whose radii are all cached contributes no call
        self.subfolders = list(subfolders) if subfolders is not None else list(dict.fromkeys(c.subfolder for c in self.calls))

    def __len__(self):
        return len(self.calls)

    def __iter__(self):
        return iter(self.calls)

    @property
    def rel_paths(self):
        return [call.rel for call in self.calls]

    @classmethod
    def from_code_executor(cls, code_executor):
        return cls(
            [RadialCall.from_entry(sub, rho, entry) for sub, rhos in code_executor.items() for rho, entry in rhos.items()],
            subfolders=list(code_executor.keys()),
        )

    def to_code_executor(self):
        code_executor = {sub: {} for sub in self.subfolders}
        for call in self.calls:
            code_executor.setdefault(call.subfolder, {})[call.rho] = call.to_entry()
        return code_executor


@dataclass
class CompletionSpec:
    '''
    How a code says "this radius ran to completion": a substring that must appear in
    `marker_file`, or the presence of `alt_file` (e.g. the tag left when a watchdog
    stops a run on purpose). Names carry the per-radius suffix when a rho is given,
    and are plain when it is not (scratch-folder layout).
    '''
    marker_file: str = None
    marker_text: str = None
    alt_file: str = None

    @classmethod
    def from_run_specifications(cls, run_specifications):
        marker = (run_specifications or {}).get("completion_marker")
        return None if marker is None else cls(marker[0], marker[1], (run_specifications or {}).get("completion_alt_file"))

    @classmethod
    def coerce(cls, completion_marker, alt_file=None):
        '''Accept a CompletionSpec, a (file, substring) tuple or None.'''
        if isinstance(completion_marker, cls):
            return cls(completion_marker.marker_file, completion_marker.marker_text,
                       completion_marker.alt_file if alt_file is None else alt_file)
        if completion_marker is None:
            return cls(None, None, alt_file)
        return cls(completion_marker[0], completion_marker[1], alt_file)

    def _name(self, file, rho):
        return file if rho is None else f"{file}{rho_suffix(rho)}"

    def finished(self, folder, rho=None):
        '''Returns (finished, marker_path).'''
        folder = Path(folder)
        mfile = folder / self._name(self.marker_file, rho) if self.marker_file is not None else folder
        finished = False
        if self.marker_file is not None:
            try:
                finished = self.marker_text in mfile.read_text(errors="ignore")
            except OSError:
                finished = False
        if not finished and self.alt_file is not None:
            finished = (folder / self._name(self.alt_file, rho)).exists()
        return finished, mfile

    def unfinished(self, plan):
        '''(call, marker_path) for every call of the plan that did not run to completion.'''
        if not isinstance(plan, WorkPlan):
            plan = WorkPlan.from_code_executor(plan)
        return [(call, mfile) for call, (ok, mfile) in ((c, self.finished(c.folder, c.rho)) for c in plan) if not ok]


def _submitted_state(sim, attribute, what):
    '''
    State that only the detached path produces. `check()` and `fetch()` read it, so a call
    made before a submit (or before a re-attach) gets one clear message instead of an
    AttributeError from somewhere deeper.
    '''
    value = getattr(sim, attribute, None)
    if value is None:
        raise RuntimeError(
            f"[MITIM] {what} is missing (self.{attribute}). It is produced by _run(run_type='submit') "
            f"and by load_submission_state(); call one of them before check()/fetch()."
        )
    return value


def _background_job_block(command, indent="    "):
    '''
    Wrap a per-code `code_call` command in a brace group launched in the
    background ('{ ...; } &') for the bash and slurm_standard builders.

    This is the single place the trailing-newline contract lives, so the
    per-code `code_call` functions don't have to agree on a convention and the
    builders don't each rstrip defensively. `code_call` shapes differ: TGLF/NEO/GX
    return a single line with no trailing newline; CGYRO returns a multi-line
    block (export prefix + cgyro launch + post-run restart-cleanup if-block)
    ending in a newline. Two failure modes this avoids:
      - a bare '<cmd> &' only backgrounds the last line of a multi-line command
        and leaves a dangling '&' after a trailing 'fi' -> bash syntax error;
      - gluing '} &' onto a command with no trailing newline keeps '}' on the
        command's line, so the group never closes -> 'syntax error near done'.
    Normalizing to exactly one trailing newline, with '} &' on its own line,
    closes the group correctly for both shapes.
    '''
    return f"{indent}{{\n{command.rstrip(chr(10))}\n{indent}}} &\n"

def slurm_allocation_hostnames():
    '''Nodes of the SLURM allocation this process runs in ([] outside SLURM).'''
    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not nodelist:
        return []
    try:
        import subprocess
        out = subprocess.run(["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True, timeout=30)
        return [h for h in out.stdout.split() if h] if out.returncode == 0 else []
    except Exception:
        return []


class JobScript:
    '''
    The shell text that launches every call of one submission, plus the per-folder pieces
    the stall-rescue path needs. Every builder defines `command`, `per_folder_commands` and
    `array_index_by_folder`; the modes that cannot re-issue a single call leave the last two
    empty. `folders` are the execution folders relative to the scratch root, in staging order.
    '''

    def __init__(self, folders, code_call, resources_per_call, exec_folder):
        self.folders = list(folders)
        self.code_call = code_call
        self.resources_per_call = resources_per_call
        self.exec_folder = exec_folder
        self.per_folder_commands = {}
        self.array_index_by_folder = {}
        self.command = self._build()

    def _build(self):
        raise NotImplementedError

    def _call(self, folder, **kwargs):
        return self.code_call(folder=folder, n=self.resources_per_call, p=self.exec_folder, **kwargs)


class BashScript(JobScript):
    '''
    Bash loop over the folders, `max_parallel` calls in flight at a time. Inside a SLURM
    allocation it also exports the node list and a 1-based call counter, so a code_call can
    pin call k to its own node(s) (CGYRO does).
    '''

    def __init__(self, folders, code_call, resources_per_call, exec_folder, max_parallel=1, hosts=()):
        self.max_parallel = max_parallel
        self.hosts = list(hosts)
        super().__init__(folders, code_call, resources_per_call, exec_folder)

    def _build(self):
        command = "#!/usr/bin/env bash\n"
        command += "set -m\n"  # job control, which the `jobs -rp` throttle below needs
        command += f"max_parallel_execution={self.max_parallel}\n\n"

        command += "folders=(\n"
        for folder in self.folders:
            command += f'    "{folder}"\n'
        command += ")\n\n"

        if self.hosts:
            command += "MITIM_HOSTS=( " + " ".join(self.hosts) + " )\nMITIM_CALL=0\n\n"
        command += "for folder in \"${folders[@]}\"; do\n"
        if self.hosts:
            command += "    MITIM_CALL=$((MITIM_CALL+1))\n"
        command += _background_job_block(self._call('"$folder"'))
        # `jobs -rp` prints one PID per line; plain `jobs -r` echoes the job's command text,
        # which for a multi-line brace group spans several lines and inflates the count
        command += "    while (( $(jobs -rp | wc -l) >= max_parallel_execution )); do sleep 1; done\n"
        command += "done\n\n"
        command += "wait\n"
        return command

    def folder_bodies(self):
        '''Per-call bodies with the literal folder, for the in-allocation scheduler.'''
        return {folder: self._call(folder) for folder in self.folders}


class StandardSlurmScript(JobScript):
    '''One allocation for the whole plan, every call backgrounded inside it.'''

    def _build(self):
        command = ""
        for folder in self.folders:
            command += _background_job_block(self._call(folder))
        command += "\nwait"  # so the script does not end before the calls do
        return command


class ArraySlurmScript(JobScript):
    '''One array element per call, indexed into a FOLDERS bash array.'''

    _INDEXED_FOLDER = "${FOLDERS[$SLURM_ARRAY_TASK_ID]}"

    def _redirect(self, folder):
        return (f'1> {self.exec_folder}/{folder}/slurm_output.dat '
                f'2> {self.exec_folder}/{folder}/slurm_error.dat\n')

    def _build(self):
        folders_list = "FOLDERS=( "
        for folder in self.folders:
            folders_list += f"{folder} "
        folders_list += ")"

        command = folders_list + "\n\n"
        command += self._call(self._INDEXED_FOLDER, additional_command=self._redirect(self._INDEXED_FOLDER))

        # Literal-folder bodies and the folder -> element map let the stall-rescue path
        # scancel one array index and resubmit that call alone as a standalone job
        # (mitim_job.resubmit_single_task), keeping that primitive code-agnostic.
        self.array_index_by_folder = {folder: i for i, folder in enumerate(self.folders)}
        self.per_folder_commands = {folder: self._call(folder, additional_command=self._redirect(folder))
                                    for folder in self.folders}
        return command

    @property
    def array_list(self):
        return [str(i) for i in range(len(self.folders))]


@dataclass
class SubmissionRecord:
    '''
    On-disk record of a detached (`run_type='submit'`) job: everything a later process needs
    to re-attach to it instead of resubmitting. `write` and `read` are the only two places
    that know the JSON layout, so the writer and the reader cannot drift apart.
    '''
    SCHEMA = 1

    code: str = None
    mode: str = 'single'
    created_utc: str = None
    base_subfolder: str = None
    slurm_output: str = None
    kwargs_organize: dict = field(default_factory=dict)
    resubmit_ledger: dict = field(default_factory=dict)
    restart_sources: dict = None
    results_per_plasma: dict = None
    job: dict = field(default_factory=dict)
    path: Path = None
    raw: dict = field(default=None, repr=False)

    @classmethod
    def from_simulation(cls, sim, base_subfolder):
        job = sim.simulation_job

        # Only the field _organize_results reads back per (subfolder, rho). Keys at full
        # precision (repr): a rho rounded to 6 decimals can land on a rho_{:.4f} tie that
        # names a different folder after reload (0.29434978 -> "0.294350" -> rho_0.2944)
        code_executor_serial = {
            sub: {repr(float(rho)): {"folder": str(v["folder"])} for rho, v in rhos.items()}
            for sub, rhos in sim.kwargs_organize["code_executor"].items()
        }

        results_per_plasma_serial = None
        if getattr(sim, "results_per_plasma", None):
            results_per_plasma_serial = {
                str(int(p)): {"subfolder": info["subfolder"], "folder": str(info["folder"])}
                for p, info in sim.results_per_plasma.items()
            }

        return cls(
            code=sim.run_specifications.get("code"),
            mode="batched" if results_per_plasma_serial is not None else "single",
            created_utc=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
            base_subfolder=base_subfolder,
            slurm_output=sim.slurm_output,
            kwargs_organize={
                "tmpFolder": str(sim.kwargs_organize["tmpFolder"]),
                "filesToRetrieve": list(sim.kwargs_organize["filesToRetrieve"]),
                "optional_files_to_retrieve": list(sim.kwargs_organize.get("optional_files_to_retrieve", [])),
                "code_executor": code_executor_serial,
                # Populated only for slurm_array submissions; persisted so the rescue path
                # survives a PORTALS restart between submit and the first stall decision
                "array_index_by_folder": dict(sim.kwargs_organize.get("array_index_by_folder", {})),
                "per_folder_commands": dict(sim.kwargs_organize.get("per_folder_commands", {})),
            },
            # Per-folder stall-rescue ledger (empty until the first resubmit); it spans poll
            # cycles within one PORTALS iteration, so a re-attach must pick up child jobids
            # spawned before the prior process was killed
            resubmit_ledger=dict(getattr(sim, "_resubmit_ledger", {})),
            # CGYRO warm-start parents, embedded so the plotter's per-(rho,iter) map survives
            # a kill+reattach even if the local restart_sources.json is wiped. None elsewhere
            restart_sources=getattr(sim, "_restart_sources_payload", None),
            results_per_plasma=results_per_plasma_serial,
            job={
                "folder_local": str(job.folder_local),
                "folderExecution": str(job.folderExecution),
                "jobid": job.jobid,
                "launchSlurm": bool(job.launchSlurm),
                "slurm_settings": job.slurm_settings,
                "machineSettings": job.machineSettings,
                "output_files": [str(f) for f in getattr(job, "output_files", [])],
                "output_folders": [str(f) for f in getattr(job, "output_folders", [])],
                "check_files_in_folder": getattr(job, "check_files_in_folder", {}),
                "output_folders_selective": getattr(job, "output_folders_selective", {}),
                "output_file_fallbacks": getattr(job, "output_file_fallbacks", {}),
                "log_simulation_file": str(job.log_simulation_file) if job.log_simulation_file else None,
                "run_in_place": bool(getattr(job, "run_in_place", False)),
            },
        )

    def _payload(self):
        return {
            "schema_version": self.SCHEMA,
            "mode": self.mode,
            "code": self.code,
            "created_utc": self.created_utc,
            "base_subfolder": self.base_subfolder,
            "slurm_output": self.slurm_output,
            "kwargs_organize": self.kwargs_organize,
            "resubmit_ledger": self.resubmit_ledger,
            "restart_sources": self.restart_sources,
            "results_per_plasma": self.results_per_plasma,
            "job": self.job,
        }

    def write(self, path):
        # TODO(multi-process-safety): concurrent PORTALS drivers writing to the same folder
        # could race here — add fcntl.flock if that becomes real.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._payload(), f, indent=2, default=str)
        self.path = path
        return path

    @classmethod
    def read(cls, path):
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        schema = data.get("schema_version")
        if schema != cls.SCHEMA:
            raise RuntimeError(
                f"[MITIM] {path} carries submission-record schema {schema!r}, but this MITIM reads schema "
                f"{cls.SCHEMA}. Re-run the evaluation instead of re-attaching to that job."
            )

        return cls(
            code=data.get("code"),
            mode=data.get("mode", "single"),
            created_utc=data.get("created_utc"),
            base_subfolder=data.get("base_subfolder"),
            slurm_output=data["slurm_output"],
            kwargs_organize=data["kwargs_organize"],
            resubmit_ledger=dict(data.get("resubmit_ledger", {})),
            restart_sources=data.get("restart_sources"),
            results_per_plasma=data.get("results_per_plasma"),
            job=data["job"],
            path=path,
            raw=data,
        )

    def apply_to(self, sim):
        '''
        Put this record back on a simulation object, so `check()` / `fetch()` talk to the
        already-running job without resubmitting.
        '''
        job = FARMINGtools.mitim_job(self.job["folder_local"], log_simulation_file=self.job["log_simulation_file"])
        job.folderExecution = self.job["folderExecution"]
        job.jobid = self.job["jobid"]
        job.launchSlurm = self.job["launchSlurm"]
        job.slurm_settings = self.job["slurm_settings"]
        job.machineSettings = self.job["machineSettings"]
        job.output_files = list(self.job["output_files"])
        job.output_folders = list(self.job["output_folders"])
        job.check_files_in_folder = self.job["check_files_in_folder"]
        job.output_folders_selective = self.job["output_folders_selective"]
        job.output_file_fallbacks = self.job.get("output_file_fallbacks", {})
        job.run_in_place = self.job.get("run_in_place", False)

        # On the original submit path `_run` creates this folder; on re-attach the process is
        # fresh and it may not exist, while `retrieve()` writes its tarball here
        job.folder_local.mkdir(parents=True, exist_ok=True)

        sim.simulation_job = job
        sim.slurm_output = self.slurm_output

        sim.kwargs_organize = {
            "code_executor": {
                sub: {sim._exact_rho(float(rho)): {"folder": Path(v["folder"])} for rho, v in rhos.items()}
                for sub, rhos in self.kwargs_organize["code_executor"].items()
            },
            "tmpFolder": Path(self.kwargs_organize["tmpFolder"]),
            "filesToRetrieve": list(self.kwargs_organize["filesToRetrieve"]),
            "optional_files_to_retrieve": list(self.kwargs_organize.get("optional_files_to_retrieve", [])),
            "array_index_by_folder": dict(self.kwargs_organize.get("array_index_by_folder", {})),
            "per_folder_commands": dict(self.kwargs_organize.get("per_folder_commands", {})),
        }

        # The base_subfolder is needed to rewrite this record after a future resubmit
        sim._resubmit_ledger = dict(self.resubmit_ledger)
        sim._base_subfolder = self.base_subfolder

        # restart_sources.json is re-derived from the embedded copy, which reflects what was
        # actually staged at submit time; the local file may have been wiped since
        sim._restart_sources_payload = self.restart_sources
        if self.restart_sources and isinstance(self.restart_sources, dict) and self.path is not None:
            local_json = self.path.parent / "restart_sources.json"
            try:
                local_json.parent.mkdir(parents=True, exist_ok=True)
                with open(local_json, "w") as f:
                    json.dump(self.restart_sources, f, indent=2)
                print(f"\t- Restored restart_sources.json from submission metadata at {local_json}", typeMsg='i')
            except OSError as e:
                print(f"\t- Could not restore restart_sources.json at {local_json}: {e}", typeMsg='w')

        # results_per_plasma is rebuilt in memory by the caller via `_prepare_plasmas_state`,
        # which also restores the profiles / inputs_files / NormalizationSets that read_plasma
        # needs, so it is deliberately not restored here.


@dataclass
class _RunSettings:
    '''Everything the steps of `_run` read out of the caller's kwargs, resolved once.'''
    run_type: RunType
    code: str
    input_file: str
    code_call: object
    name: str
    job_name_suffix: str
    launch_slurm: bool
    allocation: dict
    resources_per_call: int
    minutes: int
    submission_type_override: str
    exclusive: bool
    attempts_execution: int
    cold_start: bool
    helper_lostconnection: bool
    base_subfolder: str
    tmpFolder: Path
    files_to_retrieve: list
    optional_files_to_retrieve: list


class mitim_simulation:
    '''
    Main class for running GACODE simulations.
    '''

    # Subclasses that want `_run(run_type='submit')` to persist slurm-job metadata
    # (so a later process can re-attach instead of resubmitting) set this to a
    # filename (e.g. "cgyro_submission.json"). Leave as None to opt out.
    _submission_metadata_filename = None

    def __init__(
        self,
        rhos=None,  # rho locations of interest, e.g. [0.4,0.6,0.8]
    ):
        # Float-dtype even when empty, so `np.asarray(self.rhos, dtype=float)` always works
        self.rhos = np.array(rhos) if rhos is not None else np.array([])

        # A simulation may have multiple ways to run (e.g. linear, nonlinear, etc) with different outputs, or not desirable to bring everything locally
        self.output_files_simulation = {
            'complete': [],
            'minimal': [],
        }

        # Optional primary/fallback output pairs: if the primary is present
        # on the remote it is tarred; if only the fallback is present, the
        # fallback is renamed to the primary on the remote just before the
        # tar (see FARMINGtools.mitim_job.retrieve). Subclasses populate as
        # e.g. {"bin.cgyro.restart": "bin.cgyro.restart.old"}.
        self.output_file_fallbacks = {}
        
        self.nameRunid = "0"
        
        self.results, self.scans = {}, {}
        
        self.run_specifications = None

        self.NormalizationSets = {'SELECTED': None}

        # harvest_recorder (mitim_tools.harvest_tools.HARVESTtools) attached by PORTALS when the
        # user opted into harvesting; None -> read() records nothing
        self.harvest = None

    def _harvest(self, label, folder=None):
        if self.harvest is not None:
            self.harvest.record(self, label, folder=folder)

    def harvest_records(self, label, folder=None):
        '''
        One harvest record per radius of results[label]: the full input file (parsed dict, or the
        output object's harvest_inputs() when the output class owns the inputs, e.g. CGYRO/GX) and
        the scalar fluxes from the output object's harvest_outputs(). Subclasses that store their
        results differently (QuaLiKiz) override this whole method.
        '''
        res = self.results[label]
        code = self.run_specifications['code']
        sim_folder = Path(folder).name if folder is not None else Path(getattr(self, 'FolderSimLast', '') or '').name
        machine = _harvest_machine_info(getattr(self, 'simulation_job', None))
        records = []
        for irho, rho in enumerate(self.rhos):
            out = res['output'][irho]
            inputs = out.harvest_inputs()
            if inputs is None:
                inputs = res['parsed'][irho] if res.get('parsed') and res['parsed'][irho] is not None else None
            if inputs is None and rho in getattr(self, 'inputs_files', {}):
                inputs = {**self.inputs_files[rho].controls, **self.inputs_files[rho].plasma}
            records.append({
                'code': code,
                'inputs': inputs or {},
                'outputs': out.harvest_outputs(),
                'hash_extra': out.harvest_hash_extra(),
                'meta': {'label': label, 'sim_folder': sim_folder, 'rho': float(rho), 'roa': float(getattr(out, 'roa', np.nan)),
                         'code_version': out.harvest_version(), 'in_process': bool(getattr(self, 'in_process', False)), **machine,
                         **out.harvest_provenance()},
            })
        return records

    def prep(
        self,
        mitim_state,                # A MITIM state class
        FolderGACODE,               # Main folder where all caculations happen (runs will be in subfolders)
        cold_start=False,           # If True, do not use what it potentially inside the folder, run again
        forceIfcold_start=False,    # Extra flag
        ):
        '''
        This method prepares the GACODE run from a MITIM state class by setting up the necessary input files and directories.
        '''

        print("> Preparation run from MITIM state class (direct conversion)")

        if self.run_specifications is None:
            raise Exception("[MITIM] Simulation child class did not define run specifications")

        state_converter = self.run_specifications['state_converter']    # e.g. to_tglf
        input_class     = self.run_specifications['input_class']        # e.g. TGLFinput
        input_file      = self.run_specifications['input_file']         # e.g. input.tglf

        self.FolderGACODE = IOtools.expandPath(FolderGACODE)
        
        if cold_start or not self.FolderGACODE.exists():
            IOtools.askNewFolder(self.FolderGACODE, force=forceIfcold_start)
            
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        if isinstance(mitim_state, str) or isinstance(mitim_state, Path):
            # If a string, assume it's a path to input.gacode
            self.profiles = PROFILEStools.gacode_state(mitim_state)
        else:
            self.profiles = mitim_state
            
        # Keep a copy of the file
        self.profiles.write_state(file=self.FolderGACODE / "input.gacode_torun")

        self.profiles.derive_quantities(mi_ref=md_u)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize from state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # Call the method dynamically based on state_converter
        conversion_method = getattr(self.profiles, state_converter)
        self.inputs_files = conversion_method(r=self.rhos, r_is_rho=True)

        for rho in self.inputs_files:
            
            # Initialize class
            self.inputs_files[rho] = input_class.initialize_in_memory(self.inputs_files[rho])
                
            # Write input.tglf file
            self.inputs_files[rho].file = self.FolderGACODE / f'{input_file}{rho_suffix(rho)}'
            self.inputs_files[rho].write_state()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Definining normalizations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print("> Setting up normalizations")
        self.NormalizationSets, cdf = NORMtools.normalizations(self.profiles)

        return cdf
    
    def run(
        self,
        subfolder,  # 'neo1/',
        code_settings=None,
        extraOptions={},
        multipliers={},
        minimum_delta_abs={},
        ApplyCorrections=True,  # Removing ions with too low density and that are fast species
        Quasineutral=False,  # Ensures quasineutrality. By default is False because I may want to run the file directly
        launchSlurm=True,
        cold_start=False,
        forceIfcold_start=False,
        extra_name="exe",
        allocation=None,   # {'resources_per_call': int, 'minutes': int, 'mem': str|None}
        attempts_execution=1,
        only_minimal_files=False,
        run_type = 'normal', # 'normal': send, submit and wait; 'submit': send and submit and do not wait; 'send': send and do not submit; 'prep': do not submit
        additional_files_to_send = None, # Dict (rho keys) of files to send along with the run (e.g. for restart). Each list entry is either a path or a (src_path, dst_basename) tuple — tuples let the file be renamed on stage-in (e.g. CGYRO restart per-rho blobs -> out.cgyro.restart).
        helper_lostconnection=False, # If True, it means that the connection to the remote machine was lost, but the files are there, so I just want to retrieve them not execute the commands
        job_name_suffix='_sim', # Suffix appended to the code name for the slurm --job-name (e.g. "cgyro" + "_sim" -> "cgyro_sim"). PORTALS overrides with "_ev{evaluation_number}" to tag submissions by iteration.
        rescue_interrupted=False, # If True, radii whose previous execution was interrupted in the scratch folder (restart + tag files present, identical input) are continued in place instead of wiped and re-run (codes that declare `rescue_spec`, e.g. CGYRO)
    ):

        run_type = _normalize_run_type(run_type)

        if allocation is None:
            allocation = self._default_allocation(self.run_specifications.get('code', ''), minutes=10)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare inputs
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        code_executor, code_executor_full = self._run_prepare(
            #
            subfolder,
            code_executor={},
            code_executor_full={},
            #
            code_settings=code_settings,
            extraOptions=extraOptions,
            multipliers=multipliers,
            #
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            only_minimal_files=only_minimal_files,
            #
            launchSlurm=launchSlurm,
            allocation=allocation,
            #
            additional_files_to_send=additional_files_to_send,
            #
            ApplyCorrections=ApplyCorrections,
            minimum_delta_abs=minimum_delta_abs,
            Quasineutral=Quasineutral,
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Run NEO
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self._run(
            code_executor,
            code_executor_full=code_executor_full,
            code_settings=code_settings,
            ApplyCorrections=ApplyCorrections,
            Quasineutral=Quasineutral,
            launchSlurm=launchSlurm,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            extra_name=extra_name,
            allocation=allocation,
            only_minimal_files=only_minimal_files,
            attempts_execution=attempts_execution,
            run_type=run_type,
            helper_lostconnection=helper_lostconnection,
            base_subfolder=subfolder,
            job_name_suffix=job_name_suffix,
            rescue_interrupted=rescue_interrupted,
        )

        return code_executor_full

    def _run_prepare(
        self,
        # ********************************
        # Required options
        # ********************************
        subfolder_simulation,
        code_executor=None,
        code_executor_full=None,
        # ********************************
        # Run settings
        # ********************************
        code_settings=None,
        extraOptions={},
        multipliers={},
        # ********************************
        # IO settings
        # ********************************
        cold_start=False,
        forceIfcold_start=False,
        only_minimal_files=False,
        # ********************************
        # Slurm settings (for warnings)
        # ********************************
        launchSlurm=True,
        allocation=None,
        # ********************************
        # Additional files to send (e.g. restarts). Must be a dictionary with rho keys;
        # each list entry is either a path or a (src_path, dst_basename) tuple — tuples
        # let the file be renamed on stage-in (e.g. CGYRO per-rho restart blobs -> out.cgyro.restart).
        # ********************************
        additional_files_to_send = None,
        # ********************************
        # Additional settings to correct/modify inputs
        # ********************************
        **kwargs_control
        ):

        if allocation is None:
            allocation = self._default_allocation(self.run_specifications.get('code', ''), minutes=5)

        if self.run_specifications is None:
            raise Exception("[MITIM] Simulation child class did not define run specifications")

        # Because of historical relevance, I allow both TGLFsettings and code_settings #TODO #TOREMOVE
        if "TGLFsettings" in kwargs_control:
            if code_settings is not None:
                raise Exception('[MITIM] Cannot use both TGLFsettings and code_settings')
            else:
                code_settings = kwargs_control["TGLFsettings"]
                del kwargs_control["TGLFsettings"]
        # ------------------------------------------------------------------------------------

        if code_executor is None:
            code_executor = {}
        if code_executor_full is None:
            code_executor_full = {}

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare for run
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        rhos = self.rhos

        inputs = copy.deepcopy(self.inputs_files)
        Folder_sim = self.FolderGACODE / subfolder_simulation

        # ------------------------------------------------
        # Selection of files to retrieve
        # ------------------------------------------------
        
        if only_minimal_files:
            filesToRetrieve = self.output_files_simulation["minimal"]
        else:
            filesToRetrieve = self.output_files_simulation["complete"]

        # Do I need to run all radii?
        rhosEvaluate = cold_start_checker(
            rhos,
            filesToRetrieve,
            Folder_sim,
            cold_start=cold_start,
            completion_marker=self.run_specifications.get("completion_marker"),
            completion_alt_file=self.run_specifications.get("completion_alt_file"),
        )

        if len(rhosEvaluate) == len(rhos):
            # All radii need to be evaluated
            IOtools.askNewFolder(Folder_sim, force=forceIfcold_start)
            
        # Once created, expand here
        Folder_sim = IOtools.expandPath(Folder_sim)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Change this specific run
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        latest_inputsFile, latest_inputsFileDict = change_and_write_code(
            rhos,
            inputs,
            Folder_sim,
            code_settings=code_settings,
            extraOptions=extraOptions,
            multipliers=multipliers,
            addControlFunction=self.run_specifications['control_function'],
            controls_file=self.run_specifications['controls_file'],
            allocation=allocation,
            **kwargs_control
        )
        
        code_executor_full[subfolder_simulation] = {}
        code_executor[subfolder_simulation] = {}
        for irho in self.rhos:
            code_executor_full[subfolder_simulation][irho] = {
                "folder": Folder_sim,
                "dictionary": latest_inputsFileDict[irho],
                "inputs": latest_inputsFile[irho],
                "extraOptions": extraOptions,
                "multipliers": multipliers,
                # .get: a radius may have nothing to send (restart_from_cases "best" cold-starts the radii with no parent)
                "additional_files_to_send": (additional_files_to_send or {}).get(irho)
            }
            if irho in rhosEvaluate:
                code_executor[subfolder_simulation][irho] = code_executor_full[subfolder_simulation][irho]

        # Check input file problems
        for irho in latest_inputsFileDict:
            latest_inputsFileDict[irho].anticipate_problems()

        self.FolderSimLast = Folder_sim

        return code_executor, code_executor_full

    def _default_allocation(self, code, minutes=10):
        '''Allocation used when the caller passes none: the code's own default resources for one call.'''
        from mitim_tools.misc_tools import SLURMtools
        return {
            "resources_per_call": SLURMtools.CODE_HINTS.get(code, {}).get("default_resources_per_call", 1),
            "minutes": minutes,
        }

    def _extra_point_hooks(self, resources_per_call):
        '''Codes that can use nodes freed by early radial calls (load_balance 'extra_points') return
        the SCHEDULERtools hook dict here; None keeps the plain bash loop.'''
        return None

    def _rescue_interrupted_runs(self, kwargs_run, folders, folders_red, input_file):
        '''
        Continue, in place, radial runs that a previous execution left unfinished in
        the scratch folder (driver killed, allocation expired). Enabled by
        run(rescue_interrupted=True) for codes declaring `run_specifications["rescue_spec"]`:
            {"required": [files that must exist, e.g. restart blob + tag],
             "progress_file": file holding the time the code will resume from,
             "progress_line": its 1-based line number (None = last line); first token is read,
             "time_key": input-file key holding the run length (e.g. "MAX_TIME"),
             "checksum_ignore": input-file key prefixes excluded from the identity md5
                (optional, defaults to [time_key]; list every key `after_trim` rewrites,
                otherwise the next rescue of the same folder fails the identity check),
             "after_trim": optional callable (text, remaining) -> (text, log_note) applied
                to the staged input after the time_key rewrite, for keys whose value must
                follow the trimmed run length (CGYRO: RESTART_STEP),
             "report_files": files whose sizes are logged for forensics (optional)}
        A rho sub-folder is rescued only if the required files are there and its
        input file is byte-identical (md5) to the one just generated, so a changed
        namelist, preset or gradient never continues a stale run. Rescued folders
        are kept across the scratch wipe (mitim_job.preserve_subfolders) and receive
        no staged restart file, so the code's own continuation logic takes over
        (CGYRO: restart + out.cgyro.tag -> restart_flag=1, t continues). Codes take
        `time_key` as a number of steps to ADD from the restart point (CGYRO:
        n_time = MAX_TIME/DELTA_T), so for rescued radii that key is rewritten in the
        staged input to the remaining time (original minus the resume time) and
        excluded from the identity md5. The resume time must be the one the code
        restarts from (CGYRO: t_current in out.cgyro.tag, second line), NOT the last
        printed time: the restart blob predates the last output by up to one restart
        interval, and counting from the wrong point over- or under-shoots the target.
        Everything else is staged and run as usual.
        '''
        spec = self.run_specifications.get("rescue_spec")
        self.simulation_job.preserve_subfolders = []
        if not (kwargs_run.get("rescue_interrupted", False) and spec):
            return
        if getattr(self.simulation_job, "run_in_place", False):
            print("\t- rescue_interrupted requested but the run is in-place (no scratch folder): nothing to rescue", typeMsg="i")
            return

        time_key = spec.get("time_key")
        ignore_prefixes = list(spec.get("checksum_ignore") or ([time_key] if time_key else []))
        found = self.simulation_job.probe_interrupted_runs(
            folders_red, spec.get("required", []), input_file,
            progress_file=spec.get("progress_file"), progress_line=spec.get("progress_line"),
            checksum_ignore_prefix=ignore_prefixes, report_files=spec.get("report_files"),
        )
        if not found:
            return

        import hashlib, re
        rescued = []
        for folder_sim_this, rel in zip(folders, folders_red):
            if rel not in found:
                continue
            md5_remote, progress, report = found[rel]
            text = (folder_sim_this / input_file).read_text()
            kept = "".join(l for l in text.splitlines(keepends=True) if not any(l.startswith(p) for p in ignore_prefixes))
            if hashlib.md5(kept.encode()).hexdigest() != md5_remote:
                print(f"\t- [rescue] {rel}: interrupted run found but its {input_file} differs from the new one; discarding it", typeMsg="w")
                continue
            # Without a resume time the trim below cannot happen and the radius would run
            # the full time_key again on top of the restart point, so it is not rescued
            try:
                done = float(progress)
            except (TypeError, ValueError):
                print(f"\t- [rescue] {rel}: interrupted run found but no resume time could be read ({progress!r}); discarding it", typeMsg="w")
                continue
            # Keep only the input file locally: staged restarts would overwrite the
            # orphan's own (more advanced) restart on extraction
            for f in folder_sim_this.iterdir():
                if f.name != input_file:
                    f.unlink()
            # Trim the run length to what is left (the code counts steps from the restart point)
            remaining_msg = ""
            m = re.search(rf"^({time_key}\s*=\s*)(\S+)", text, flags=re.M) if time_key else None
            if m is not None:
                total = float(m.group(2))
                remaining = max(total - done, 1.0)
                text = text[:m.start(2)] + f"{remaining:.5E}" + text[m.end(2):]
                remaining_msg = f", {time_key} {total:g} -> {remaining:g} remaining"
                after_trim = spec.get("after_trim")
                if after_trim is not None:
                    text, note = after_trim(text, remaining)
                    remaining_msg += note
                (folder_sim_this / input_file).write_text(text)
            rescued.append(rel)
            print(f"\t- [rescue] {rel}: continuing interrupted run in place (resuming from t={progress}{remaining_msg}) [{report}]", typeMsg="i")

        self.simulation_job.preserve_subfolders = rescued

    def _run(
        self,
        code_executor,
        run_type = 'normal', # 'normal': submit and wait; 'submit': submit and do not wait; 'prep': do not submit
        **kwargs_run
    ):
        """
        extraOptions and multipliers are not being grabbed from kwargs_NEOrun, but from code_executor for WF
        """

        settings = self._run_settings(run_type, kwargs_run)

        # Internal view of the pending work; code_executor stays the external contract
        plan = WorkPlan.from_code_executor(code_executor)

        if len(plan) == 0:
            print(f"\t- {settings.code.upper()} not run because all results files found (please ensure consistency!)",typeMsg="i")
            self.simulation_job = None
            return

        print(f"\t- {settings.code.upper()} needs to run because not all results files found",typeMsg="i")

        folders, folders_red = self._stage_inputs(plan, settings)
        self._rescue_interrupted_runs(kwargs_run, folders, folders_red, settings.input_file)
        resolved = self._resolve_allocation(folders_red, settings)
        script = self._build_script(folders_red, resolved, settings)
        self._prepare_job(folders, folders_red, script, resolved, settings)

        if settings.run_type.blocking:
            self._dispatch_blocking(code_executor, folders, settings)
        elif settings.run_type is RunType.SUBMIT:
            self._dispatch_detached(code_executor, script, settings)

    def _run_settings(self, run_type, kwargs_run):
        '''
        The caller's kwargs, resolved once into the values every step of `_run` shares.
        launchSlurm=True asks for a batch job on the selected machine (if a partition is
        configured); launchSlurm=False runs it there as a bash script.
        '''
        code = self.run_specifications.get('code', 'tglf')
        allocation = kwargs_run.get("allocation") or {}

        if kwargs_run.get("only_minimal_files", False):
            files_to_retrieve = self.output_files_simulation["minimal"]
        else:
            files_to_retrieve = self.output_files_simulation["complete"]

        run_type = RunType.parse(run_type)
        if run_type is RunType.SUBMIT and "minutes" not in allocation:
            # A detached submission is a long job by definition; a silent default wall clock
            # would just time every element out
            raise ValueError(f"[MITIM] run_type 'submit' for {code} needs allocation['minutes'] (the wall clock of each submitted element)")
        return _RunSettings(
            run_type=run_type,
            code=code,
            input_file=self.run_specifications.get('input_file', 'input.tglf'),
            code_call=self.run_specifications.get('code_call', None),
            name=f"{code}_{self.nameRunid}{kwargs_run.get('extra_name', '')}",
            # Slurm --job-name suffix ('_sim' by default, '_ev{N}' for PORTALS)
            job_name_suffix=kwargs_run.get('job_name_suffix', '_sim'),
            launch_slurm=kwargs_run.get("launchSlurm", True),
            allocation=allocation,
            resources_per_call=allocation.get("resources_per_call", self._default_allocation(code)["resources_per_call"]),
            minutes=allocation.get("minutes", 10),
            # allocation.submission_type ('slurm_array' | 'slurm_standard' | 'bash') overrides
            # the resolver heuristic, and itself wins over the per-code default
            submission_type_override=allocation.get("submission_type") or self.run_specifications.get('force_submission_type'),
            # allocation.exclusive forces/disables --exclusive, e.g. to guarantee whole-node
            # array elements on clusters without strict per-job GPU isolation
            exclusive=allocation.get("exclusive"),
            attempts_execution=kwargs_run.get("attempts_execution", 1),
            cold_start=kwargs_run.get("cold_start", False),
            helper_lostconnection=kwargs_run.get("helper_lostconnection", False),
            base_subfolder=kwargs_run.get("base_subfolder"),
            tmpFolder=self.FolderGACODE / f"tmp_{code}",
            files_to_retrieve=files_to_retrieve,
            # Best-effort retrievals: tarred if present on the remote, but their absence only
            # emits a warning (no 60s retry, no cold-start trigger)
            optional_files_to_retrieve=list(self.output_files_simulation.get("optional", [])),
        )

    def _stage_inputs(self, plan, settings):
        '''
        Write every pending call's input file (and the extra files it was given) into a fresh
        scratch tree, and build the mitim_job that will ship it. Returns the absolute and the
        scratch-relative execution folders, in staging order.
        '''
        IOtools.askNewFolder(settings.tmpFolder, force=True)

        kkeys = [str(sub).replace('/', '') for sub in plan.subfolders]
        self.simulation_job = FARMINGtools.mitim_job(
            settings.tmpFolder,
            log_simulation_file=self.FolderGACODE / f"mitim_simulation_{kkeys[0]}.log",   # refer with the first folder
        )
        # connect_ssh() retry config (set from the PORTALS namelist for PORTALS-CGYRO); stays
        # None for every other caller, which preserves the historical retry behavior
        self.simulation_job.connection_retry_settings = getattr(self, "connection_retry_settings", None)
        self.simulation_job.define_machine_quick(settings.code, f"mitim_{settings.name}")

        folders, folders_red = [], []
        for call in plan:
            print(f"\t- Preparing {settings.code.upper()} execution ({call.subfolder}) at rho={call.rho:.4f}")

            folder_sim_this = settings.tmpFolder / call.rel
            folders.append(folder_sim_this)

            folder_sim_this_rel = folder_sim_this.relative_to(settings.tmpFolder)
            folders_red.append(folder_sim_this_rel.as_posix() if self.simulation_job.machineSettings['machine'] != 'local' else str(folder_sim_this_rel))

            folder_sim_this.mkdir(parents=True, exist_ok=True)

            with open(folder_sim_this / settings.input_file, "w") as f:
                f.write(call.inputs)

            # Entries are a bare path (staged with its own basename) or a (src, dst_basename)
            # tuple to rename on stage-in (CGYRO per-rho restart blobs named
            # "out.cgyro.restart_<rho>" -> "out.cgyro.restart")
            for entry in (call.additional_files_to_send or []):
                src, dst_name = entry if isinstance(entry, tuple) else (entry, Path(entry).name)
                shutil.copy(src, folder_sim_this / dst_name)

        return folders, folders_red

    def _machine_limits(self, settings):
        '''
        Machine settings handed to the resolver. A local bash run is additionally capped by
        what this process really has: the limits of the SLURM allocation it sits in when
        nested, otherwise the machine's own cpu_count.
        '''
        machineSettings = FARMINGtools.mitim_job.grab_machine_settings(settings.code)

        if (machineSettings["machine"] != "local") or \
            (settings.launch_slurm and ("partition" in self.simulation_job.machineSettings["slurm"])):
            return machineSettings

        ntasks = os.environ.get('SLURM_NTASKS')
        cpus_per_task = os.environ.get('SLURM_CPUS_PER_TASK')
        if (cpus_per_task is not None) and (ntasks is not None):
            env_cores = int(cpus_per_task) * int(ntasks)
        elif cpus_per_task is not None:
            env_cores = int(cpus_per_task)
        elif ntasks is not None:
            env_cores = int(ntasks)
        else:
            env_cores = int(os.cpu_count() or 16)

        cores_per_node = machineSettings.get("cores_per_node")
        if cores_per_node is None or env_cores < cores_per_node:
            print(f"\t- Local execution capped at {env_cores} cores (machine config says {cores_per_node})", typeMsg="i")
            machineSettings = dict(machineSettings)
            machineSettings["cores_per_node"] = env_cores

        return machineSettings

    def _resolve_allocation(self, folders_red, settings):
        '''
        Submission type, sbatch dict, mpi layout and concurrency for the whole plan, in one
        resolve call. The array indices are the positions of the staged folders, so they are
        known here whether or not the submission turns out to be an array.
        '''
        from mitim_tools.misc_tools import SLURMtools

        if settings.code not in SLURMtools.CODE_HINTS:
            raise Exception(
                f"[MITIM] Code '{settings.code}' is not registered in SLURMtools.CODE_HINTS. "
                f"Add a hints entry ({{'default_resources_per_call', 'uses_gpu', ...}})."
            )

        n_calls = len(folders_red)   # every (subfolder, rho) work unit, i.e. one staged folder each
        return SLURMtools.resolve(
            code=settings.code,
            allocation={"resources_per_call": settings.resources_per_call, "minutes": settings.minutes,
                        "mem": settings.allocation.get("mem"), "max_concurrent_calls": settings.allocation.get("max_concurrent_calls")},
            n_rhos=n_calls, n_subfolders=1,
            machine_settings=self._machine_limits(settings),
            launch_slurm=settings.launch_slurm,
            force_submission_type=settings.submission_type_override,
            job_name=settings.code + settings.job_name_suffix,
            array_list=[str(i) for i in range(n_calls)],
            exclusive=settings.exclusive,
        )

    def _build_script(self, folders_red, resolved, settings):
        '''The shell text for the resolved submission type, plus its per-folder rescue pieces.'''
        submission_type = SubmissionType.parse(resolved.submission_type)
        code = settings.code.upper()
        total_cores_required = int(settings.resources_per_call) * len(folders_red)
        exec_folder = self.simulation_job.folderExecution

        if submission_type is SubmissionType.BASH:
            max_parallel_execution = max(1, resolved.concurrency)
            n_sequential = -(-len(folders_red) // max_parallel_execution)  # ceil division
            print(f"\t- {code} will be executed as bash script (total cores: {total_cores_required},  cores per simulation: {settings.resources_per_call}). MITIM will launch {n_sequential} sequential execution(s)",typeMsg="i")
            script = BashScript(folders_red, settings.code_call, settings.resources_per_call, exec_folder,
                                max_parallel=max_parallel_execution, hosts=slurm_allocation_hostnames())
            self._attach_scheduler(script, settings)
            return script

        if (getattr(self, "_load_balance", None) or {}).get("strategy") == "extra_points":
            print("\t- load_balance 'extra_points' needs the driver inside the allocation (bash mode); "
                  + ("array elements release their nodes on their own" if submission_type is SubmissionType.SLURM_ARRAY
                     else "in slurm mode radii just wait for the slowest one"), typeMsg="w")

        if submission_type is SubmissionType.SLURM_STANDARD:
            print(f"\t- {code} will be executed in SLURM as standard job (cpus: {total_cores_required})",typeMsg="i")
            return StandardSlurmScript(folders_red, settings.code_call, settings.resources_per_call, exec_folder)

        print(f"\t- {code} will be executed in SLURM as job array due to its size (cpus: {total_cores_required})",typeMsg="i")
        return ArraySlurmScript(folders_red, settings.code_call, settings.resources_per_call, exec_folder)

    def _attach_scheduler(self, script, settings):
        '''
        load_balance 'extra_points': the same per-call bodies run through a Python scheduler
        that gives the nodes freed by early calls to extra cases. The bash script is still
        written for reference but not executed.
        '''
        hooks = self._extra_point_hooks(settings.resources_per_call) if script.hosts else None
        if hooks is None:
            return
        from mitim_tools.simulation_tools.utils import SCHEDULERtools
        self._scheduler_to_attach = SCHEDULERtools.InAllocationScheduler(
            script.folder_bodies(), script.hosts, script.max_parallel,
            completion_marker=self.run_specifications.get("completion_marker"), **hooks)

    def _prepare_job(self, folders, folders_red, script, resolved, settings):
        '''Hand the script, the staged folders and the retrieval lists to the mitim_job.'''
        self.simulation_job.define_machine(
            settings.code,
            f"mitim_{settings.name}",
            launchSlurm=settings.launch_slurm,
            slurm_settings=resolved.sbatch,
        )
        self.simulation_job.scheduler = getattr(self, "_scheduler_to_attach", None)
        self._scheduler_to_attach = None

        # `files_we_must_check` is what check_all_received flags as missing (triggers the
        # retry). The tarball additionally carries the optional files, so they come down if
        # the remote has them.
        files_we_must_check = {folder: list(settings.files_to_retrieve) for folder in folders_red}
        files_we_want_to_tar = {folder: list(settings.files_to_retrieve) + list(settings.optional_files_to_retrieve)
                                for folder in folders_red}

        self.simulation_job.prep(
            script.command,
            input_folders=folders,
            output_folders=folders_red,
            output_folders_selective=files_we_want_to_tar,
            output_file_fallbacks=self.output_file_fallbacks,
            check_files_in_folder=files_we_must_check,
        )

    def _dispatch_blocking(self, code_executor, folders, settings):
        '''
        Run now and wait. 'normal' executes the script and collects the results; 'send' only
        stages the inputs, so it keeps the scratch folder and organizes nothing.
        '''
        executes = settings.run_type is RunType.NORMAL

        attempts = 0
        while True:
            try:
                self.simulation_job.run(
                    waitYN=executes,
                    removeScratchFolders=True,
                    attempts_execution=settings.attempts_execution,
                    helper_lostconnection=settings.helper_lostconnection,
                    execute_case_flag=executes,
                    )
                break
            except LOGtools.InteractiveTerminalError:
                # A failed retrieval already removed the local rho folders (they are
                # both the staged inputs and the retrieval targets), so a repeat would
                # die in the tarball step with a confusing FileNotFoundError.
                if any(not Path(f).exists() for f in folders):
                    raise RuntimeError(
                        f"[MITIM] {settings.code.upper()} run did not return its expected outputs and the staged "
                        f"inputs under {settings.tmpFolder} are gone; not retrying. Check {settings.tmpFolder}/mitim_farming.err "
                        f"and the code's own logs in the scratch folder."
                    )
                if attempts >= 1:
                    # The retry was already spent; a second failure is not random,
                    # and falling through would organize results of a run that never produced them
                    raise
                print('\n\t Run wanted to crash because interactive terminal is not allowed in this bash job, but repeating once to see if error was random')
                attempts += 1

        if not executes:
            return

        self._organize_results(code_executor, settings.tmpFolder, settings.files_to_retrieve,
                               optional_files_to_retrieve=settings.optional_files_to_retrieve)
        self._verify_completion(code_executor, settings.code)

    def _dispatch_detached(self, code_executor, script, settings):
        '''Submit and return; the caller polls with check() and collects with fetch().'''
        if not settings.launch_slurm:
            raise RuntimeError(
                "[MITIM] run_type='submit' needs launchSlurm=True: with launchSlurm=False the work runs as a "
                "plain bash script, so there is no queued job for check()/fetch() to re-attach to. Use "
                "run_type='normal' to run it as a bash script and wait."
            )

        self.simulation_job.run(
            waitYN=False,
            check_if_files_received=False,
            removeScratchFolders=False,
            removeScratchFolders_goingIn=settings.cold_start,
        )

        self.kwargs_organize = {
            "code_executor": code_executor,
            "tmpFolder": settings.tmpFolder,
            "filesToRetrieve": settings.files_to_retrieve,
            "optional_files_to_retrieve": settings.optional_files_to_retrieve,
            # Empty for non-array submissions; the array builder fills them so the per-rho
            # stall-rescue path can map a stalled folder back to its element and its body.
            "array_index_by_folder": script.array_index_by_folder,
            "per_folder_commands": script.per_folder_commands,
        }

        self.slurm_output = "slurm_output.dat"

        # Prepare how to search for the job without waiting for it
        self.simulation_job.launchSlurm = True
        self.simulation_job.slurm_settings['name'] = Path(self.simulation_job.folderExecution).name

        # Persist submission metadata so a future process can re-attach to this in-flight job
        # instead of resubmitting. Opt-in via subclass `_submission_metadata_filename` (e.g.
        # CGYRO). The base_subfolder is also stashed on `self` so the stall-rescue path
        # (CGYROtools._cgyro_handle_stalled_tasks) can re-write the metadata after every
        # successful resubmit without threading the arg through the polling-loop callback.
        if self._submission_metadata_filename is not None:
            self._base_subfolder = settings.base_subfolder
            self._write_submission_metadata(self._base_subfolder)

    def _child_jobids(self):
        '''
        Deduplicated rescue jobids recorded in the auto-resubmit ledger (insertion
        order). Empty until a stalled radius has been resubmitted.
        '''
        seen = []
        for entry in (getattr(self, "_resubmit_ledger", None) or {}).values():
            for jid in entry.get("child_jobids", []):
                if jid not in seen:
                    seen.append(jid)
        return seen

    def _live_child_jobids(self, alive_if_unreachable=False):
        '''
        Rescue jobids that squeue still reports as queued/running. The parent array can
        drain while a rescue child is still integrating, so the run is over only when
        both are gone.

        alive_if_unreachable: answer when the probe cannot reach the remote. The poll
        loop passes True (keep waiting rather than fetch a half-done radius); the
        re-attach decision tree passes False ("no signal", fall through to its own checks).
        '''
        child_ids = self._child_jobids()
        job = getattr(self, "simulation_job", None)
        if not child_ids or job is None:
            return []

        cmd = f'squeue -h -j {",".join(child_ids)} -o "%.15i %.10T"'
        try:
            job.connect()
            out, _err = job.execute(cmd, printYN=False)
            job.close()
        except Exception as e:
            print(f"\t- [child-jobid liveness] squeue failed ({type(e).__name__}: {e}); treating rescue children as "
                  f"{'alive' if alive_if_unreachable else 'not alive'}", typeMsg='w')
            return list(child_ids) if alive_if_unreachable else []

        if isinstance(out, bytes):
            out = out.decode(errors='replace')

        alive = []
        for line in (out or "").strip().splitlines():
            toks = line.split()
            if toks and toks[0] in child_ids and toks[0] not in alive:
                print(f"\t- [child-jobid liveness] rescue child jobid {toks[0]} is still in the queue (state={toks[1] if len(toks) > 1 else '?'})", typeMsg='i')
                alive.append(toks[0])
        return alive

    def _any_child_job_alive(self, alive_if_unreachable=False):
        '''Boolean form of `_live_child_jobids`.'''
        return len(self._live_child_jobids(alive_if_unreachable=alive_if_unreachable)) > 0

    def check(self, every_n_minutes=None, skip_first_iteration_squeue=False, max_completing_polls=2, custom_checker=None):
        '''
        Poll slurm until the job leaves the queue (state "NOT FOUND" / status=2).

        skip_first_iteration_squeue: when True, the first loop iteration does NOT
        run a fresh `squeue`; instead it reuses `self.simulation_job.status`
        already populated by an earlier call (e.g. the re-attach liveness probe
        in transport_cgyro.py) so the poller does not immediately redo a squeue
        that we just ran seconds ago.

        max_completing_polls: safety valve for jobs stuck in slurm state
        "COMPLETING" (usually a node/IO/epilog hang after the compute already
        finished). When squeue reports COMPLETING for this many consecutive
        polls, exit the loop with a warning so `fetch()` can pull whatever is
        on disk rather than sleeping forever on a dead node. Set to None to
        disable the heuristic.

        custom_checker: optional callable invoked as `custom_checker(self)`
        after every squeue poll. Intended for subclass-specific inspection
        (e.g. CGYRO's per-(subfolder,rho) out.cgyro.info / out.cgyro.timing
        walk) — the generic check() stays code-agnostic. Exceptions are
        caught and logged so a flaky remote query does not abort the poll.
        '''

        if self.simulation_job is None:
            print("- Not checking status because simulation job is not defined (not run)", typeMsg="i")
            return

        if self.simulation_job.launchSlurm:
            print("- Checker job status")

            slurm_output = _submitted_state(self, "slurm_output", "the name of the job's slurm output file")

            first = True
            completing_streak = 0
            while True:
                if first and skip_first_iteration_squeue:
                    print(f"\t- Reusing status from the earlier liveness probe (skipping redundant squeue)", typeMsg='i')
                else:
                    self.simulation_job.check(file_output = slurm_output)
                first = False
                state = self.simulation_job.infoSLURM.get("STATE")
                print(f'\t- Current status (as of  {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}): {self.simulation_job.status} ({state})')

                if custom_checker is not None:
                    try:
                        custom_checker(self)
                    except Exception as _e:
                        print(f"\t- custom_checker raised: {_e}; continuing poll", typeMsg='w')

                if self.simulation_job.status == JobStatus.GONE:
                    # squeue on the parent jobid only: an auto-resubmit rescue child is an
                    # independent job, so the parent array can drain while a rescued radius
                    # is still integrating. Fetching then pulls that radius half-done.
                    live_children = self._live_child_jobids(alive_if_unreachable=True)
                    if not live_children:
                        print("\n\t* Job considered finished (please do .fetch() to retrieve results)",typeMsg="i")
                        break
                    print(f"\t- Parent job left the queue but rescue child jobid(s) {live_children} are still in it; continuing to poll", typeMsg='i')

                # Track consecutive COMPLETING polls — slurm's "COMPLETING"
                # state normally lasts seconds; when it persists it is almost
                # always an epilog/filesystem hang on the node rather than the
                # user code still running. Give up after max_completing_polls
                # so fetch() can run against whatever has been flushed.
                if state == "COMPLETING":
                    completing_streak += 1
                    if max_completing_polls is not None and completing_streak >= max_completing_polls:
                        print(f"\n\t* Slurm state has been COMPLETING for {completing_streak} consecutive polls — likely node/IO/epilog hang; giving up on the poll loop and letting fetch() pull whatever is on the remote", typeMsg='w')
                        break
                else:
                    completing_streak = 0

                if every_n_minutes is None:
                    print("\n\t* Job not finished yet")
                    break
                else:
                    print(f"\n\t* Waiting {every_n_minutes} minutes")
                    time.sleep(every_n_minutes * 60)
        else:
            print("- Not checking status because this was run command line (not slurm)")

    def fetch(self):
        """
        For a job that has been submitted but not waited for, once it is done, get the results
        """

        if self.simulation_job is None:
            print("- Not fetching because simulation job is not defined (not run)", typeMsg="i")
            return

        print("\n\n\t- Fetching results")

        if self.simulation_job.launchSlurm:
            kwargs_organize = _submitted_state(self, "kwargs_organize", "the retrieval spec (kwargs_organize)")

            self.simulation_job.connect()
            self.simulation_job.retrieve()
            self.simulation_job.close()

            self._organize_results(**kwargs_organize)

            # Same gate the 'normal' path applies after its own _organize_results: the
            # submit/fetch path used to hand back truncated radii as finished results
            self._verify_completion(kwargs_organize["code_executor"], self.run_specifications.get("code", ""))

        else:
            print("- Not retrieving results because this was run command line (not slurm)")

    def delete(self):
        '''
        Cancel the submitted job. The scratch folder is left alone: it holds the only record
        of what the job did, and it is still the running job's working directory until the
        scancel lands.
        '''
        print("\n\n\t- Deleting job")

        job = self.simulation_job
        launch_slurm = job.launchSlurm
        job.launchSlurm = False   # the scancel is a plain shell command, not a job to queue
        try:
            job.prep(
                f"scancel -n {job._squeue_job_name()}",
                label_log_files="_finish",
            )
            job.run(removeScratchFolders=False)
        finally:
            job.launchSlurm = launch_slurm

    def _verify_completion(self, code_executor, code):
        '''
        Refuse to hand back a run that did not finish. The retrieval check only proves the
        output files exist, and codes like CGYRO write them all from the first step: a step
        killed mid-run (preemption, crash, GPU OOM, node failure) leaves a complete-looking
        set, and the launch script's exit status is not a reliable signal. So the code's own
        completion marker is checked on the retrieved files - the same test cold_start_checker
        applies before a launch, which otherwise only catches a truncated run on the NEXT call,
        after this one has already been consumed.
        '''
        spec = CompletionSpec.from_run_specifications(self.run_specifications)
        if spec is None:
            return

        unfinished = []
        for call, mfile in spec.unfinished(code_executor):
            try:
                lines = [l.strip() for l in mfile.read_text(errors="ignore").splitlines() if l.strip()]
                last = lines[-1] if lines else "empty"
            except OSError:
                last = "file missing"
            unfinished.append(f"{call.subfolder} rho={call.rho:.4f} (last line of {mfile.name}: {last!r})")

        if unfinished:
            raise RuntimeError(
                f"[MITIM] {code.upper()} returned without finishing at {len(unfinished)} radius(es) - no "
                f"'{spec.marker_text}' line in {spec.marker_file}" + (f" and no {spec.alt_file}" if spec.alt_file else "") + ":\n\t"
                + "\n\t".join(unfinished)
                + "\nTheir outputs are truncated and are not used. Re-running this evaluation re-runs only these "
                "radii (cold_start_checker applies the same test before launching)."
            )

    def _organize_results(self, code_executor, tmpFolder, filesToRetrieve, optional_files_to_retrieve=None, **_unused_kwargs_organize):
        # **_unused_kwargs_organize tolerates any forward-compatible keys the
        # submit path adds to kwargs_organize for the rescue path (e.g.
        # array_index_by_folder, per_folder_commands). fetch() always splats
        # the full dict; this method only consumes the four fields it needs.

        # ---------------------------------------------
        # Organize
        # ---------------------------------------------

        optional_files_to_retrieve = list(optional_files_to_retrieve) if optional_files_to_retrieve else []

        print("\t- Retrieving files and changing names for storing")
        fineall = True
        missing_optional = []
        for call in WorkPlan.from_code_executor(code_executor):

            for file in filesToRetrieve:
                original_file = call.result_name(file)
                final_destination = call.folder / f"{original_file}"

                temp_file = tmpFolder / call.rel / f"{file}"

                # A file that did not come back leaves the previous result in place
                # (removing it first would destroy a good result on a failed retrieval)
                if not temp_file.exists():
                    print(f"\t!! file {file} ({original_file}) could not be retrieved", typeMsg="w")
                    fineall = False
                    continue

                final_destination.unlink(missing_ok=True)
                temp_file.replace(final_destination)

                fineall = fineall and final_destination.exists()

                if not final_destination.exists():
                    print(f"\t!! file {file} ({original_file}) could not be retrived",typeMsg="w",)

            # Optional retrievals — move if present, silently skip if not;
            # we aggregate the misses and emit one summary warning so the
            # log is not flooded with "restart not found" noise per rho.
            for file in optional_files_to_retrieve:
                final_destination = call.folder / f"{call.result_name(file)}"
                final_destination.unlink(missing_ok=True)
                temp_file = tmpFolder / call.rel / f"{file}"
                if not temp_file.exists():
                    missing_optional.append((call.subfolder, float(call.rho), file))
                    continue
                temp_file.replace(final_destination)

        if missing_optional:
            distinct = sorted({f for _, _, f in missing_optional})
            print(f"\t- Optional file(s) not present on the remote (ok, proceeding): {distinct}   [{len(missing_optional)} (subfolder, rho, file) tuples]", typeMsg="w")

        # Extra cases accepted by the in-allocation scheduler (load_balance 'extra_points')
        # come back under tmpFolder/<rel>; keep them next to the main results before the wipe
        accepted = (getattr(self.simulation_job, "scheduler_result", None) or {}).get("accepted", [])
        for rel in accepted:
            src, dst = tmpFolder / rel, Path(self.FolderGACODE) / rel
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                for f in src.iterdir():
                    f.replace(dst / f.name)
        if accepted:
            print(f"\t- Extra case(s) stored under {Path(self.FolderGACODE) / accepted[0].split('/')[0]}: {accepted}", typeMsg="i")

        if fineall:
            print("\t\t- All files were successfully retrieved")

            # Remove temporary folder
            IOtools.shutil_rmtree(tmpFolder)

        else:
            print("\t\t- Some files were not retrieved", typeMsg="w")

    # ------------------------------------------------------------------
    # Submission-state persistence: lets a later process re-attach to an
    # already-submitted (run_type='submit') slurm job instead of resubmitting.
    # Write is triggered from `_run()`'s submit branch when the subclass opts in
    # via `_submission_metadata_filename`. Load is invoked by the caller of
    # `run()` (e.g. transport_cgyro.py) before `check()` / `fetch()` / `read()`.
    # ------------------------------------------------------------------

    def _submission_metadata_path(self, base_subfolder):
        if self._submission_metadata_filename is None or base_subfolder is None:
            return None
        return self.FolderGACODE / base_subfolder / self._submission_metadata_filename

    def _write_submission_metadata(self, base_subfolder):
        path = self._submission_metadata_path(base_subfolder)
        if path is None:
            return
        SubmissionRecord.from_simulation(self, base_subfolder).write(path)
        print(f"\t- Submission metadata written to {path}", typeMsg="i")

    def _exact_rho(self, rho):
        '''
        Radius as this object knows it (self.rhos, set from the caller's exact rho list), matched
        within 1e-5: submission files written before full-precision keys stored rho with 6
        decimals, which can format to a different rho_{:.4f} folder name than the one the run used.
        '''
        if self.rhos is not None and len(self.rhos) > 0:
            k = int(np.argmin(np.abs(np.asarray(self.rhos, dtype=float) - rho)))
            if abs(float(self.rhos[k]) - rho) < 1e-5:
                return float(self.rhos[k])
        return rho

    def load_submission_state(self, path):
        '''
        Rehydrate `self.simulation_job`, `self.kwargs_organize`, `self.slurm_output`,
        and the stall-rescue ledger from a JSON file written by
        `_write_submission_metadata`, so `check()` / `fetch()` can talk to the
        already-running slurm job without resubmitting. Returns the raw JSON.
        '''
        record = SubmissionRecord.read(path)
        record.apply_to(self)
        return record.raw

    def _local_results_complete(self):
        '''
        True when every expected result file (per rho per subfolder) from
        `kwargs_organize` already exists on disk and, for codes declaring a
        `completion_marker`, every radius also carries it — used by the re-attach path
        to skip check()+fetch() when the prior job finished while no PORTALS
        process was watching. Files alone do not prove completion (CGYRO writes them
        all from its first step). Purely local: no remote access.
        '''
        if not getattr(self, "kwargs_organize", None):
            return False
        files_to_retrieve = self.kwargs_organize["filesToRetrieve"]
        spec = CompletionSpec.from_run_specifications(self.run_specifications)
        for call in WorkPlan.from_code_executor(self.kwargs_organize["code_executor"]):
            folder = Path(call.folder)
            for fname in files_to_retrieve:
                if not (folder / call.result_name(fname)).exists():
                    return False
            if spec is not None and not spec.finished(folder, call.rho)[0]:
                return False
        return True

    def run_over_plasmas(
        self,
        list_of_states,           # List of mitim_state / input.gacode path / gacode_state objects, one per plasma
        base_subfolder,           # 'base' -> produces subfolder labels base_plasma0, base_plasma1, ...
        cold_start=False,
        forceIfcold_start=False,
        code_settings=None,
        extraOptions=None,
        multipliers=None,
        minimum_delta_abs=None,
        ApplyCorrections=True,
        Quasineutral=False,
        launchSlurm=True,
        extra_name="exe",
        allocation=None,
        attempts_execution=1,
        only_minimal_files=False,
        run_type='normal',
        additional_files_to_send=None,
        helper_lostconnection=False,
        job_name_suffix='_sim',
        rescue_interrupted=False,
        load_balance=None,   # see CGYROtools.CGYRO.run; consumed by _run via self._load_balance
    ):
        '''
        Phase-1 multi-plasma runner. Runs the same simulation configuration (same rhos,
        same code_settings, same non-scan multipliers, ...) for N independent plasma
        states in a single parallel submission by reusing the subfolder_simulation
        axis that scans already rely on.

        Each plasma p ends up under subfolder_simulation = f"{base_subfolder}_plasma{p}"
        and its prep-time state (profiles, inputs_files, normalizations, FolderSim) is
        cached on self.results_per_plasma[p] so the caller can read each plasma's
        results afterwards via self.read_plasma(p).

        Notes:
            - Requires self.FolderGACODE to be set. Call prep(...) once before this
              helper, or rely on the first iteration's prep call to create it.
            - The i-th prep(...) call overwrites the single staging input.gacode_torun
              under self.FolderGACODE, then this helper copies it to
              input.gacode_torun_plasma{p} for traceability. The in-memory
              self.inputs_files is snapshotted into the code_executor before the next
              plasma's prep touches it, so per-plasma inputs do not leak across.
        '''

        run_type = _normalize_run_type(run_type)

        if extraOptions is None:
            extraOptions = {}
        if multipliers is None:
            multipliers = {}
        if minimum_delta_abs is None:
            minimum_delta_abs = {}

        if allocation is None:
            allocation = self._default_allocation(self.run_specifications.get('code', ''), minutes=10)

        if not hasattr(self, 'FolderGACODE') or self.FolderGACODE is None:
            raise Exception(
                "[MITIM] run_over_plasmas requires FolderGACODE to be set. Either call "
                "prep(...) once before run_over_plasmas or pass FolderGACODE via a "
                "preceding prep call."
            )

        code_executor, code_executor_full, plasma_labels = self._prepare_plasmas_state(
            list_of_states,
            base_subfolder,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            code_settings=code_settings,
            extraOptions=extraOptions,
            multipliers=multipliers,
            minimum_delta_abs=minimum_delta_abs,
            only_minimal_files=only_minimal_files,
            launchSlurm=launchSlurm,
            allocation=allocation,
            additional_files_to_send=additional_files_to_send,
            ApplyCorrections=ApplyCorrections,
            Quasineutral=Quasineutral,
            announce=True,
        )

        # Parallel dispatch of every (plasma, rho) work unit via the existing _run path.
        # load_balance is carried on the instance (as CGYRO.run does for the single-plasma path)
        self._load_balance = load_balance
        try:
            self._run(
                code_executor,
                code_executor_full=code_executor_full,
                code_settings=code_settings,
                ApplyCorrections=ApplyCorrections,
                Quasineutral=Quasineutral,
                launchSlurm=launchSlurm,
                cold_start=cold_start,
                forceIfcold_start=forceIfcold_start,
                extra_name=extra_name,
                allocation=allocation,
                only_minimal_files=only_minimal_files,
                attempts_execution=attempts_execution,
                run_type=run_type,
                helper_lostconnection=helper_lostconnection,
                base_subfolder=base_subfolder,
                job_name_suffix=job_name_suffix,
                rescue_interrupted=rescue_interrupted,
            )
        finally:
            self._load_balance = None

        return plasma_labels

    def _prepare_plasmas_state(
        self,
        list_of_states,
        base_subfolder,
        cold_start=False,
        forceIfcold_start=False,
        code_settings=None,
        extraOptions=None,
        multipliers=None,
        minimum_delta_abs=None,
        only_minimal_files=False,
        launchSlurm=True,
        allocation=None,
        additional_files_to_send=None,
        ApplyCorrections=True,
        Quasineutral=False,
        announce=False,
    ):
        '''
        Build per-plasma `code_executor` and populate `self.results_per_plasma`
        in memory without submitting anything. Extracted from `run_over_plasmas`
        so the re-attach path (transport_cgyro.py) can rebuild the per-plasma
        state that `read_plasma` needs after a prior process submitted the job.
        '''
        if extraOptions is None:
            extraOptions = {}
        if multipliers is None:
            multipliers = {}
        if minimum_delta_abs is None:
            minimum_delta_abs = {}

        code_executor, code_executor_full = {}, {}
        self.results_per_plasma = {}
        plasma_labels = {}

        for p, state in enumerate(list_of_states):
            subfolder_simulation = f"{base_subfolder}_plasma{p}"
            plasma_labels[p] = subfolder_simulation

            if announce:
                print(f"\n=============================================================")
                print(f"  run_over_plasmas: preparing plasma {p} -> {subfolder_simulation}")
                print(f"=============================================================")

            # Populate self.profiles / self.inputs_files / self.NormalizationSets for plasma p.
            # Only the first prep may need to create FolderGACODE; subsequent plasmas reuse it.
            self.prep(
                state,
                self.FolderGACODE,
                cold_start=cold_start if p == 0 else False,
                forceIfcold_start=forceIfcold_start,
            )

            # Keep a per-plasma copy of the input.gacode alongside the shared staging one
            # (subprocess path writes this file; in-process prep uses synthetic in-memory
            # paths that never touch disk, so the file may not exist — skip gracefully).
            src_gacode = self.FolderGACODE / "input.gacode_torun"
            if src_gacode.exists():
                shutil.copy2(src_gacode, self.FolderGACODE / f"input.gacode_torun_plasma{p}")

            # _run_prepare snapshots self.inputs_files into code_executor[subfolder][rho]['inputs']
            # so plasma p's inputs are frozen here; subsequent per-plasma prep calls do not
            # corrupt already-queued work.
            self._run_prepare(
                subfolder_simulation,
                code_executor=code_executor,
                code_executor_full=code_executor_full,
                code_settings=code_settings,
                extraOptions=extraOptions,
                multipliers=multipliers,
                cold_start=cold_start,
                forceIfcold_start=forceIfcold_start,
                only_minimal_files=only_minimal_files,
                launchSlurm=launchSlurm,
                allocation=allocation,
                additional_files_to_send=additional_files_to_send,
                ApplyCorrections=ApplyCorrections,
                minimum_delta_abs=minimum_delta_abs,
                Quasineutral=Quasineutral,
            )

            self.results_per_plasma[p] = {
                'subfolder': subfolder_simulation,
                'folder': self.FolderSimLast,
                'profiles': self.profiles,
                'inputs_files': copy.deepcopy(self.inputs_files),
                'NormalizationSets': self.NormalizationSets,
            }

        return code_executor, code_executor_full, plasma_labels

    def read_plasma(self, plasma_idx, label=None, **read_kwargs):
        '''
        Read results for a specific plasma produced by run_over_plasmas.

        Temporarily activates that plasma's in-memory prep state (profiles /
        inputs_files / NormalizationSets / FolderSimLast) so that the existing
        read(...) path can reuse per-plasma profiles and normalizations without
        being aware of the plasma axis.
        '''
        if not hasattr(self, 'results_per_plasma') or plasma_idx not in self.results_per_plasma:
            raise KeyError(
                f"No cached plasma run for index {plasma_idx}. Call run_over_plasmas(...) first."
            )

        info = self.results_per_plasma[plasma_idx]
        effective_label = label if label is not None else info['subfolder']

        saved_profiles = self.profiles
        saved_inputs_files = self.inputs_files
        saved_NormalizationSets = self.NormalizationSets
        saved_FolderSimLast = getattr(self, 'FolderSimLast', None)

        self.profiles = info['profiles']
        self.inputs_files = info['inputs_files']
        self.NormalizationSets = info['NormalizationSets']
        self.FolderSimLast = info['folder']

        try:
            self.read(label=effective_label, folder=info['folder'], **read_kwargs)
        finally:
            self.profiles = saved_profiles
            self.inputs_files = saved_inputs_files
            self.NormalizationSets = saved_NormalizationSets
            if saved_FolderSimLast is not None:
                self.FolderSimLast = saved_FolderSimLast

        return effective_label

    def run_scan(
        self,
        subfolder,  # 'scan1',
        multipliers={},
        minimum_delta_abs={},
        variable="RLTS_1",
        varUpDown=[0.5, 1.0, 1.5],
        variables_scanTogether=[],
        relativeChanges=True,
        **kwargs_run,
    ):

        # -------------------------------------
        # Add baseline
        # -------------------------------------
        if (1.0 not in varUpDown) and relativeChanges:
            print("\n* Since variations vector did not include base case, I am adding it",typeMsg="i",)
            varUpDown_new = []
            added = False
            for i in varUpDown:
                if i > 1.0 and not added:
                    varUpDown_new.append(1.0)
                    added = True
                varUpDown_new.append(i)
        else:
            varUpDown_new = varUpDown


        code_executor, code_executor_full, folders, varUpDown_new = self._prepare_scan(
            subfolder,
            multipliers=multipliers,
            minimum_delta_abs=minimum_delta_abs,
            variable=variable,
            varUpDown=varUpDown_new,
            variables_scanTogether=variables_scanTogether,
            relativeChanges=relativeChanges,
            **kwargs_run,
        )

        # Run them all
        self._run(
            code_executor,
            code_executor_full=code_executor_full,
            **kwargs_run,
        )
        
        # Read results
        for cont_mult, mult in enumerate(varUpDown_new):
            name = f"{variable}_{mult}"
            self.read(
                label=f"{self.subfolder_scan}_{name}",
                folder=folders[cont_mult],
                cold_startWF = False,
                require_all_files=not kwargs_run.get("only_minimal_files",False),
            )

        return code_executor_full

    def _prepare_scan(
        self,
        subfolder,  # 'scan1',
        multipliers=None,
        minimum_delta_abs=None,
        variable="RLTS_1",
        varUpDown=[0.5, 1.0, 1.5],
        variables_scanTogether=None,
        relativeChanges=True,
        **kwargs_run,
    ):
        """
        Multipliers will be modified by adding the scaning variables, but I don't want to modify the original
        multipliers, as they may be passed to the next scan

        Set relativeChanges=False if varUpDown contains the exact values to change, not multipleiers
        """
        
        if multipliers is None:
            multipliers = {}
        if minimum_delta_abs is None:
            minimum_delta_abs = {}
        if variables_scanTogether is None:
            variables_scanTogether = []
        
        completeVariation = self.run_specifications['complete_variation']
        
        multipliers_mod = copy.deepcopy(multipliers)

        self.subfolder_scan = subfolder

        if relativeChanges:
            for i in range(len(varUpDown)):
                varUpDown[i] = round(varUpDown[i], 6)

        print(f"\n- Proceeding to scan {variable}{' together with '+', '.join(variables_scanTogether) if len(variables_scanTogether)>0 else ''}:")

        code_executor = {}
        code_executor_full = {}
        folders = []
        for cont_mult, mult in enumerate(varUpDown):
            mult = round(mult, 6)

            if relativeChanges:
                print(f"\n + Multiplier: {mult} -----------------------------------------------------------------------------------------------------------")
            else:
                print(f"\n + Value: {mult} ----------------------------------------------------------------------------------------------------------------")

            # If multipliers already had the variable, make sure I account for that variation as well
            if variable in multipliers:
                base_mult = multipliers[variable]
                mult = round(base_mult * mult, 6)
            
            multipliers_mod[variable] = mult

            for variable_scanTogether in variables_scanTogether:
                multipliers_mod[variable_scanTogether] = mult

            name = f"{variable}_{mult}"

            species = self.inputs_files[self.rhos[0]]  # Any rho will do

            if completeVariation is not None:
                multipliers_mod = completeVariation(multipliers_mod, species)

            if not relativeChanges:
                for ikey in multipliers_mod:
                    kwargs_run["extraOptions"][ikey] = multipliers_mod[ikey]
                multipliers_mod = {}

            # Force ensure quasineutrality if the
            if variable in ["AS_3", "AS_4", "AS_5", "AS_6"]:
                kwargs_run["Quasineutral"] = True

            # Only ask the cold_start in the first round
            kwargs_run["forceIfcold_start"] = cont_mult > 0 or ("forceIfcold_start" in kwargs_run and kwargs_run["forceIfcold_start"])

            code_executor, code_executor_full = self._run_prepare(
                f"{self.subfolder_scan}_{name}",
                code_executor=code_executor,
                code_executor_full=code_executor_full,
                multipliers=multipliers_mod,
                minimum_delta_abs=minimum_delta_abs,
                **kwargs_run,
            )

            folders.append(copy.deepcopy(self.FolderSimLast))

        return code_executor, code_executor_full, folders, varUpDown

    def read(
        self,
        label="run1",
        folder=None,  # If None, search in the previously run folder
        suffix=None,  # If None, search with my standard _0.55 suffixes corresponding to rho of this TGLF class
        input_gacode=None,  # If provided, will try to get normalizations from it if they are not already populated in the class
        **kwargs_to_class_output
    ):
        print("> Reading simulation results")

        class_output = self.run_specifications['output_class']

        # If no specified folder, check the last one
        if folder is None:
            folder = self.FolderSimLast
            
        self.results[label] = {
            'output':[],
            'parsed': [],
            "x": np.array(self.rhos),
            }

        # Try get normalizations if they weren't populated (e.g. this is just a "read" of an already run folder)
        if self.NormalizationSets["SELECTED"] is None and input_gacode is not None:
            print("\t- Getting normalizations from input.gacode provided in the read function")
            from mitim_tools.gacode_tools import PROFILEStools
            self.NormalizationSets, _ = NORMtools.normalizations(PROFILEStools.gacode_state(input_gacode))

        for rho in self.rhos:

            SIMout = class_output(
                folder,
                suffix=(rho_suffix(rho) if rho is not None else "") if suffix is None else suffix,
                **kwargs_to_class_output
            )
            
            # Unnormalize
            if 'NormalizationSets' in self.__dict__:
                SIMout.unnormalize(
                    self.NormalizationSets["SELECTED"],
                    rho=rho,
                )
            else:
                print("No normalization sets found.")

            self.results[label]['output'].append(SIMout)

            self.results[label]['parsed'].append(buildDictFromInput(SIMout.inputFile) if SIMout.inputFile else None)

        self._harvest(label, folder=folder)

    def read_scan(
        self,
        label="scan1",
        subfolder=None,
        variable="RLTS_1",
        ion_OI_position_in_total_padded_list=2,
        variable_mapping=None,
        variable_mapping_unn=None
    ):
        '''
        ion_OI_position_in_total_padded_list is the index in the input.tglf file... so if you want for ion RLNS_5, ion_OI_position_in_total_padded_list=5
        The name comes from the fact that in input.tglf files electrons are 1, first ion is 2, etc.
        '''

        if subfolder is None:
            subfolder = self.subfolder_scan
            
        if variable_mapping is None:
            variable_mapping = {}
        if variable_mapping_unn is None:
            variable_mapping_unn = {}

        self.scans[label] = {}
        self.scans[label]["variable"] = variable
        self.scans[label]["positionBase"] = None
        self.scans[label]["unnormalization_successful"] = True
        self.scans[label]["results_tags"] = []

        self.ion_OI_position_in_total_padded_list_scan = ion_OI_position_in_total_padded_list

        # ----
        
        scan = {}
        for ikey in variable_mapping | variable_mapping_unn:
            scan[ikey] = []

        cont = 0
        for ikey in self.results:
            
            isThisTheRightReadResults = (subfolder in ikey) and (variable== "_".join(ikey.split("_")[:-1]).split(subfolder + "_")[-1])

            if isThisTheRightReadResults:

                self.scans[label]["results_tags"].append(ikey)
                
                # Initialize lists
                scan0 = {}
                for ikey2 in variable_mapping | variable_mapping_unn:
                    scan0[ikey2] = []

                # Loop over radii
                for irho_cont in range(len(self.rhos)):
                    irho = np.where(self.results[ikey]["x"] == self.rhos[irho_cont])[0][0]

                    for ikey2 in variable_mapping:
                        
                        obj = self.results[ikey][variable_mapping[ikey2][0]][irho]
                        if not hasattr(obj, '__dict__'):
                            obj_dict = obj
                        else:
                            obj_dict = obj.__dict__
                        var0 = obj_dict[variable_mapping[ikey2][1]]
                        scan0[ikey2].append(var0 if variable_mapping[ikey2][2] is None else var0[variable_mapping[ikey2][2]])

                    # Unnormalized
                    self.scans[label]["unnormalization_successful"] = True
                    for ikey2 in variable_mapping_unn:
                        obj = self.results[ikey][variable_mapping_unn[ikey2][0]][irho]
                        if not hasattr(obj, '__dict__'):
                            obj_dict = obj
                        else:
                            obj_dict = obj.__dict__
                            
                        if variable_mapping_unn[ikey2][1] not in obj_dict:
                            self.scans[label]["unnormalization_successful"] = False
                            break
                        var0 = obj_dict[variable_mapping_unn[ikey2][1]]
                        scan0[ikey2].append(var0 if variable_mapping_unn[ikey2][2] is None else var0[variable_mapping_unn[ikey2][2]])
                
                for ikey2 in variable_mapping | variable_mapping_unn:
                    scan[ikey2].append(scan0[ikey2])

                if float(ikey.split('_')[-1]) == 1.0:
                    self.scans[label]["positionBase"] = cont
                cont += 1

        self.scans[label]["x"] = np.array(self.rhos)

        for ikey2 in variable_mapping | variable_mapping_unn:
            self.scans[label][ikey2] = np.atleast_2d(np.transpose(scan[ikey2]))

    def prepare_for_save(self, class_to_store = None):
        """
        Remove potential unpickleable objects
        """

        if class_to_store is None:
            class_to_store = self

        if 'fn' in class_to_store.__dict__:
            print('\t- Removing Qt object before pickling')
            del class_to_store.fn

        return class_to_store

    def save_pickle(self, file, class_to_store = None):
        
        print('...Pickling simulation class...')
                
        class_to_store = self.prepare_for_save(class_to_store=class_to_store)

        with open(file, "wb") as handle:
            pickle_dill.dump(class_to_store, handle, protocol=4)
            
def restore_class_pickle(file):
    
    print('...Restoring pickled simulation class...')
    
    return IOtools.unpickle_mitim(file)

def change_and_write_code(
    rhos,
    inputs0,
    Folder_sim,
    code_settings=None,
    extraOptions={},
    multipliers={},
    minimum_delta_abs={},
    ApplyCorrections=True,
    Quasineutral=False,
    addControlFunction=None,
    controls_file='input.tglf.controls',
    **kwargs
):
    """
    Received inputs classes and gives text.
    ApplyCorrections refer to removing ions with too low density and that are fast species
    """

    inputs = copy.deepcopy(inputs0)

    mod_input_file = {}
    ns_max = []
    for i, rho in enumerate(rhos):
        print(f"\t- Changing input file for rho={rho:.4f}")
        input_sim_rho = modifyInputs(
            inputs[rho],
            code_settings=code_settings,
            extraOptions=extraOptions,
            multipliers=multipliers,
            minimum_delta_abs=minimum_delta_abs,
            position_change=i,
            addControlFunction=addControlFunction,
            controls_file=controls_file,
            NS=inputs[rho].num_recorded,
            allocation=kwargs.get("allocation", None),
        )

        input_file = input_sim_rho.file.name.split('_')[0]

        newfile = Folder_sim / f"{input_file}{rho_suffix(rho)}"

        if code_settings is not None:
            # Apply corrections
            if ApplyCorrections:
                print("\t- Applying corrections")
                input_sim_rho.removeLowDensitySpecie()
                input_sim_rho.remove_fast()

            # Ensure that plasma to run is quasineutral
            if Quasineutral:
                input_sim_rho.ensureQuasineutrality()
        else:
            print('\t- Not applying corrections because settings is None')

        input_sim_rho.write_state(file=newfile)

        mod_input_file[rho] = input_sim_rho

        ns_max.append(inputs[rho].num_recorded)
        
    # Convert back to a string because that's how the run operates
    inputFile = inputToVariable(Folder_sim, rhos, file=input_file)

    if (np.diff(ns_max) > 0).any():
        print("> Each radial location has its own number of species... probably because of removal of fast or low density...",typeMsg="w")
        print("\t * Reading of simulation results will fail... consider doing something before launching run",typeMsg="q")

    return inputFile, mod_input_file

def inputToVariable(folder, rhos, file='input.tglf'):
    """
    Entire text file to variable
    """

    inputFilesTGLF = {}
    for rho in rhos:
        fileN = folder / f"{file}{rho_suffix(rho)}"

        with open(fileN, "r") as f:
            lines = f.readlines()
        inputFilesTGLF[rho] = "".join(lines)

    return inputFilesTGLF

def radius_finished(folder, rho, completion_marker, completion_alt_file=None):
    """
    Whether the run stored in `folder` as `<file>_<rho>` ran to completion.

    completion_marker: (file, substring); `<file>_<rho>` must CONTAIN the substring. For
    CGYRO that is ('out.cgyro.info', 'EXIT'), a line written only on an orderly finish
    (cgyro_final_kernel.F90) - a crash writes 'ERROR: (CGYRO)' and a signal writes nothing.
    completion_alt_file: file whose presence (`<file>_<rho>`) also counts as finished,
    e.g. the 'mitim_budget.tag' left when CGYRO's wall-budget watchdog stops a run on purpose.

    Returns (finished, marker_path).
    """
    return CompletionSpec.coerce(completion_marker, alt_file=completion_alt_file).finished(folder, rho)


def cold_start_checker(
    rhos,
    output_files_simulation_select,
    Folder_sim,
    cold_start=False,
    print_each_time=False,
    completion_marker=None,
    completion_alt_file=None,
):
    """
    This function checks if the TGLF inputs are already in the folder. If they are, it returns True

    completion_marker: optional (file, substring). A radius counts as done only if
    `<file>_<rho>` also CONTAINS the substring — e.g. ('out.cgyro.info', 'EXIT') for
    CGYRO, whose output files exist from the first step on: a run killed mid-way
    (job cancelled or timed out) leaves a complete-looking file set with a few a/cs
    of data, which used to be accepted as a finished evaluation.
    completion_alt_file: optional file name whose presence (`<file>_<rho>`) also counts
    as done, e.g. the 'mitim_budget.tag' left by CGYRO's wall-budget watchdog.
    """
    cont_each = 0
    if cold_start:
        rhosEvaluate = rhos
    else:
        rhosEvaluate = []
        for ir in rhos:
            existsRho = True
            for j in output_files_simulation_select:
                ffi = Folder_sim / f"{j}{rho_suffix(ir)}"
                existsThis = ffi.exists()
                existsRho = existsRho and existsThis
                if not existsThis:
                    if print_each_time:
                        print(f"\t* {ffi} does not exist")
                    else:
                        cont_each += 1
            if existsRho and completion_marker is not None:
                finished, mfile = radius_finished(Folder_sim, ir, completion_marker, completion_alt_file)
                if not finished:
                    print(f"\t* {mfile.name} has no '{completion_marker[1]}' marker: run was interrupted, re-running this radius", typeMsg='w')
                    existsRho = False
            if not existsRho:
                rhosEvaluate.append(ir)

    if not print_each_time and cont_each > 0:
        print(f'\t* {cont_each} files from expected set are missing')

    if len(rhosEvaluate) < len(rhos) and len(rhosEvaluate) > 0:
        print("~ Not all radii are found, but not removing folder and running only those that are needed",typeMsg="i",)

    return rhosEvaluate

def modifyInputs(
    input_class,
    code_settings=None,
    extraOptions=None,
    multipliers=None,
    minimum_delta_abs=None,
    position_change=0,
    addControlFunction=None,
    controls_file = 'input.tglf.controls',
    **kwargs_to_function,
):

    if extraOptions is None:
        extraOptions = {}
    if multipliers is None:
        multipliers = {}
    if minimum_delta_abs is None:
        minimum_delta_abs = {}

    # Check that those are valid flags
    GACODEdefaults.review_controls(extraOptions, control = controls_file)
    GACODEdefaults.review_controls(multipliers, control = controls_file)
    # -------------------------------------------

    if code_settings is not None:
        CodeOptions = addControlFunction(code_settings, extraOptions=extraOptions,**kwargs_to_function)

        # ~~~~~~~~~~ Change with presets
        print(f" \t- Using presets code_settings = {code_settings}", typeMsg="i")
        # Show the actual preset overrides (post-inheritance) so the log
        # records what knobs the named preset is flipping on top of the
        # defaults, not just its label.
        preset_resolved = GACODEdefaults.resolve_preset(code_settings, controls_file=controls_file)
        preset_controls = preset_resolved.get("controls", {}) if isinstance(preset_resolved, dict) else {}
        if preset_controls:
            width = max(len(str(k)) for k in preset_controls)
            print("\t     controls applied by preset:")
            for k, v in preset_controls.items():
                print(f"\t        {str(k):<{width}} = {v}")
        input_class.controls = CodeOptions

    else:
        print("\t- Input file was not modified by code_settings, using what was there before",typeMsg="i")

    # Make all upper case
    #extraOptions = {ikey.upper(): value for ikey, value in extraOptions.items()}

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Change with external options -> Input directly, not as multiplier
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if len(extraOptions) > 0:
        print("\t- External options:")
    for ikey in extraOptions:
        if isinstance(extraOptions[ikey], (list, np.ndarray)):
            value_to_change_to = extraOptions[ikey][position_change]
        else:
            value_to_change_to = extraOptions[ikey]
            
        try:
            isspecie = ikey.split("_")[0] in input_class.species[1]
        except:
            isspecie = False

        # is a species parameter?
        if isspecie:
            specie = int(ikey.split("_")[-1])
            varK = "_".join(ikey.split("_")[:-1])
            var_orig = input_class.species[specie][varK]
            var_new = value_to_change_to
            input_class.species[specie][varK] = var_new
        # is a another parameter?
        else:
            if ikey in input_class.controls:
                var_orig = input_class.controls[ikey]
                var_new = value_to_change_to
                input_class.controls[ikey] = var_new
            elif ikey in input_class.plasma:
                var_orig = input_class.plasma[ikey]
                var_new = value_to_change_to
                input_class.plasma[ikey] = var_new
            else:
                # If the variable in extraOptions wasn't in there, consider it a control param
                print(f"\t\t- Variable {ikey} to change did not exist previously, creating now",typeMsg="i")
                var_orig = None
                var_new = value_to_change_to
                input_class.controls[ikey] = var_new

        print(f"\t\t- Changing {ikey} from {var_orig} to {var_new}",typeMsg="i",)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Change with multipliers -> Input directly, not as multiplier
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if len(multipliers) > 0:
        print("\t\t- Variables change:")
    for ikey in multipliers:
    
        if isinstance(multipliers[ikey], (list, np.ndarray)):
            value_to_change_to = multipliers[ikey][position_change]
        else:
            value_to_change_to = multipliers[ikey]
    
        # is a specie one?
        if "species" in input_class.__dict__.keys() and ikey.split("_")[0] in input_class.species[1]:
            specie = int(ikey.split("_")[-1])
            varK = "_".join(ikey.split("_")[:-1])
            var_orig = input_class.species[specie][varK]
            var_new = multiplier_input(var_orig, value_to_change_to, minimum_delta_abs = minimum_delta_abs.get(ikey,None))
            input_class.species[specie][varK] = var_new
        else:
            if ikey in input_class.controls:
                var_orig = input_class.controls[ikey]
                var_new = multiplier_input(var_orig, value_to_change_to, minimum_delta_abs = minimum_delta_abs.get(ikey,None))
                input_class.controls[ikey] = var_new
            
            elif ikey in input_class.plasma:
                var_orig = input_class.plasma[ikey]
                var_new = multiplier_input(var_orig, value_to_change_to, minimum_delta_abs = minimum_delta_abs.get(ikey,None))
                input_class.plasma[ikey] = var_new
            
            else:
                print("\t- Variable to scan did not exist in original file, add it as extraOptions first",typeMsg="w",)

        print(f"\t\t\t- Changing {ikey} from {var_orig} to {var_new} (x{value_to_change_to})")

    return input_class

def multiplier_input(var_orig, multiplier, minimum_delta_abs = None):

    delta = var_orig * (multiplier - 1.0)

    if minimum_delta_abs is not None:
        if (multiplier != 1.0) and abs(delta) < minimum_delta_abs:
            print(f"\t\t\t- delta = {delta} is smaller than minimum_delta_abs = {minimum_delta_abs}, enforcing",typeMsg="i")
            # Direction from the multiplier when the base value is exactly zero:
            # np.sign(0.0) = 0 made this floor a no-op precisely in the zero-value
            # case it exists for (e.g. aLn ~ 0.0, zero-rotation shear)
            direction = np.sign(delta) if delta != 0.0 else np.sign(multiplier - 1.0)
            delta = direction * minimum_delta_abs

    return var_orig + delta

def buildDictFromInput(inputFile):
    parsed = {}

    lines = inputFile.split("\n")
    for line in lines:
        if "=" in line:
            splits = [i.split()[0] for i in line.split("=")]
            if ("." in splits[1]) and (splits[1][0].split()[0] != "."):
                try:
                    parsed[splits[0].split()[0]] = float(splits[1].split()[0])
                    continue
                except:
                    pass
                    
            try:
                parsed[splits[0].split()[0]] = int(splits[1].split()[0])
            except:
                parsed[splits[0].split()[0]] = splits[1].split()[0]

    for i in parsed:
        if isinstance(parsed[i], str):
            if (
                parsed[i].lower() == "t"
                or parsed[i].lower() == "true"
                or parsed[i].lower() == ".true."
            ):
                parsed[i] = True
            elif (
                parsed[i].lower() == "f"
                or parsed[i].lower() == "false"
                or parsed[i].lower() == ".false."
            ):
                parsed[i] = False

    return parsed

def _harvest_machine_info(job):
    ms = getattr(job, 'machineSettings', None) or {}
    return {'machine': str(ms.get('machine', '') or ''), 'modules': str(ms.get('modules', '') or '')}

class GACODEoutput:
    def __init__(self, *args, **kwargs):
        self.inputFile = None

    def unnormalize(self, *args, **kwargs):
        print("No unnormalization implemented.")

    # ---- harvest interface (see mitim_tools.harvest_tools.HARVESTtools); override where the layout differs
    def harvest_outputs(self):
        '''Scalar fluxes of this radius, as the code returned them (GB units): TGLF and NEO layout'''
        out = {}
        for k in ('Qe', 'Qi', 'Ge', 'Mt', 'Se', 'Qifast'):
            if hasattr(self, k):
                out[k] = getattr(self, k)
        GiAll = getattr(self, 'GiAll', None)
        if GiAll is not None:
            for i, g in enumerate(np.atleast_1d(GiAll)):
                out[f'Gi_{i+1}'] = float(g)
        return out

    def harvest_inputs(self):
        '''None -> the simulation object uses its parsed input file'''
        return None

    def harvest_hash_extra(self):
        '''Extra payload that distinguishes two runs with identical inputs (e.g. a longer CGYRO trace)'''
        return None

    def harvest_provenance(self):
        '''Per-(run, code) descriptors stored once with the provenance, e.g. the flux-averaging method'''
        return {}

    def harvest_version(self):
        for attr in ('tglf_version', 'neo_version', 'cgyro_version', 'gx_version'):
            if getattr(self, attr, ''):
                return str(getattr(self, attr))
        return ''

class GACODEinput:
    def __init__(self, file=None, controls_file=None, code='', n_species=None):
        self.file = IOtools.expandPath(file) if isinstance(file, (str, Path)) else None
        
        self.controls_file = controls_file
        self.code = code
        self.n_species = n_species
        
        self.num_recorded = 100

        if self.file is not None and self.file.exists():
            with open(self.file, "r") as f:
                lines = f.readlines()
            file_txt = "".join(lines)
        else:
            file_txt = ""
        input_dict = buildDictFromInput(file_txt)

        self.process(input_dict)

    @classmethod
    def initialize_in_memory(cls, input_dict):
        instance = cls()
        instance.process(input_dict)
        return instance

    def process(self, input_dict):

        if self.controls_file is not None:
            options_check = [key for key in IOtools.generateMITIMNamelist(self.controls_file, caseInsensitive=False).keys()]
        else:
            options_check = []

        self.controls, self.plasma = {}, {}
        for key in input_dict.keys():
            if key in options_check:
                self.controls[key] = input_dict[key]
            else:
                self.plasma[key] = input_dict[key]

        # Get number of recorded species
        if self.n_species is not None and self.n_species in input_dict:
            self.num_recorded = int(input_dict[self.n_species])

    def write_state(self, file=None):
        
        if file is None:
            file = self.file

        # Local formatter: floats -> 6 significant figures in exponential (uppercase),
        # ints stay as ints, bools as 0/1, sequences space-separated with same rule.
        def _fmt_num(x):
            import numpy as _np
            if isinstance(x, (bool, _np.bool_)):
                return "True" if x else "False"
            if isinstance(x, (_np.floating, float)):
                # 6 significant figures in exponential => 5 digits after decimal
                return f"{float(x):.5E}"
            if isinstance(x, (_np.integer, int)):
                return f"{int(x)}"
            return str(x)

        def _fmt_value(val):
            import numpy as _np
            if isinstance(val, (list, tuple, _np.ndarray)):
                # Flatten numpy arrays but keep ordering; join with spaces
                if isinstance(val, _np.ndarray):
                    flat = val.flatten().tolist()
                else:
                    flat = list(val)
                return " ".join(_fmt_num(v) for v in flat)
            return _fmt_num(val)

        with open(file, "w") as f:
            f.write("#-------------------------------------------------------------------------\n")
            f.write(f"# {self.code} input file modified by MITIM {mitim_version}\n")
            f.write("#-------------------------------------------------------------------------\n")

            f.write("\n\n# Control parameters\n")
            f.write("# ------------------\n\n")
            for ikey in self.controls:
                var = self.controls[ikey]
                f.write(f"{ikey.ljust(23)} = {_fmt_value(var)}\n")

            f.write("\n\n# Plasma/Geometry parameters\n")
            f.write("# ------------------\n\n")
            for ikey in self.plasma:
                var = self.plasma[ikey]
                f.write(f"{ikey.ljust(23)} = {_fmt_value(var)}\n")

    def anticipate_problems(self):
        pass

    def remove_fast(self):
        pass

    def removeLowDensitySpecie(self, *args):
        pass
    