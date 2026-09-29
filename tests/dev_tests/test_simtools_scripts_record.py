"""
The refactored pieces of SIMtools must be drop-in replacements for what HEAD did inline.

Checks:
  1. The three JobScript builders produce byte-identical command strings to the inline
     builders of commit 33212239, driven through `_run(run_type='prep')` on both versions
     with the same fake mitim_job and the same fake SLURMtools.resolve.
  2. SubmissionRecord round-trips, and is byte-compatible with the old
     `_write_submission_metadata` / `load_submission_state` pair in both directions.
  3. RunType.parse('run') is NORMAL, and the run-type/submission-type parsers reject junk.
  4. check() / fetch() before a submit raise a RuntimeError that names the missing state.

Run: python tests/dev_tests/test_simtools_scripts_record.py
"""

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

from mitim_tools.simulation_tools import SIMtools
from mitim_tools.misc_tools import FARMINGtools, SLURMtools

HEAD = "33212239"


# ---------------------------------------------------------------------------
# The HEAD version of SIMtools, imported under another name as the reference
# ---------------------------------------------------------------------------

def _load_old_simtools(workdir):
    repo = Path(__file__).resolve().parents[2]
    src = subprocess.run(
        ["git", "show", f"{HEAD}:src/mitim_tools/simulation_tools/SIMtools.py"],
        cwd=repo, capture_output=True, text=True, check=True).stdout
    path = workdir / "old_simtools.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("old_simtools", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["old_simtools"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fakes: enough of mitim_job and of the resolver to reach the prep() call
# ---------------------------------------------------------------------------

EXEC_FOLDER = "/scratch/mitim_run"


class FakeJob:
    """Records the command handed to prep(); everything else is inert."""

    captured = {}

    def __init__(self, folder_local, log_simulation_file=None):
        self.folder_local = Path(folder_local)
        self.log_simulation_file = log_simulation_file
        self.folderExecution = EXEC_FOLDER
        self.machineSettings = {"machine": "cluster", "slurm": {"partition": "gpu"}}
        self.scheduler = None
        self.preserve_subfolders = []
        self.run_in_place = False
        self.launchSlurm = True
        self.slurm_settings = {}
        self.jobid = None
        self.connection_retry_settings = None

    @staticmethod
    def grab_machine_settings(code):
        return {"machine": "cluster", "cores_per_node": 64, "gpus_per_node": 4, "slurm": {"partition": "gpu"}}

    def define_machine_quick(self, code, name):
        pass

    def define_machine(self, code, name, launchSlurm=True, slurm_settings=None):
        self.launchSlurm = launchSlurm
        self.slurm_settings = dict(slurm_settings or {})

    def prep(self, command, **kwargs):
        FakeJob.captured["command"] = command
        FakeJob.captured["kwargs"] = kwargs


def _fake_resolve(submission_type, concurrency=2):
    def resolve(**kwargs):
        return SLURMtools.ResolvedAllocation(
            use_slurm=submission_type != "bash", sbatch={"name": "fake"},
            concurrency=concurrency, submission_type=submission_type, resources_per_call=kwargs.get("n_rhos", 1))
    return resolve


def _code_call(folder, n, p, additional_command="", **kwargs):
    """Multi-line shape, like CGYRO's: exercises _background_job_block's newline contract."""
    return (f"export MITIM_N={n}\n"
            f"cd {p}/{folder} && run_code -n {n} {additional_command}\n"
            f"if [ -f done ]; then rm -f blob; fi\n")


def _build_sim(module, folder, rhos):
    sim = module.mitim_simulation(rhos=rhos)
    sim.FolderGACODE = folder
    sim.nameRunid = "7"
    sim.run_specifications = {
        "code": "cgyro",
        "input_file": "input.cgyro",
        "code_call": _code_call,
    }
    sim.output_files_simulation = {"complete": ["out.cgyro.info"], "minimal": ["out.cgyro.info"], "optional": ["bin.cgyro.restart"]}
    sim.output_file_fallbacks = {}
    return sim


def _code_executor(folder, rhos):
    return {"base_cgyro": {rho: {"folder": folder / "base_cgyro", "dictionary": None,
                                 "inputs": f"MAX_TIME={i}\n", "extraOptions": {}, "multipliers": {},
                                 "additional_files_to_send": None}
                           for i, rho in enumerate(rhos)}}


def _command_for(module, submission_type, hosts, workdir, concurrency=2):
    rhos = [0.3486, 0.6712]
    folder = Path(tempfile.mkdtemp(dir=workdir))
    sim = _build_sim(module, folder, rhos)

    old_job, old_resolve, old_hosts = FARMINGtools.mitim_job, SLURMtools.resolve, module.slurm_allocation_hostnames
    FARMINGtools.mitim_job = FakeJob
    SLURMtools.resolve = _fake_resolve(submission_type, concurrency=concurrency)
    module.slurm_allocation_hostnames = lambda: list(hosts)
    try:
        FakeJob.captured = {}
        with contextlib.redirect_stdout(io.StringIO()):
            sim._run(_code_executor(folder, rhos), run_type="prep",
                     allocation={"resources_per_call": 4, "minutes": 30}, launchSlurm=True)
    finally:
        FARMINGtools.mitim_job, SLURMtools.resolve = old_job, old_resolve
        module.slurm_allocation_hostnames = old_hosts

    return FakeJob.captured["command"], FakeJob.captured["kwargs"]


def test_script_byte_identity(old):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cases = [
            ("bash", ()),
            ("bash", ("nodeA", "nodeB")),
            ("slurm_standard", ()),
            ("slurm_array", ()),
        ]
        for submission_type, hosts in cases:
            new_cmd, new_kwargs = _command_for(SIMtools, submission_type, hosts, tmp)
            old_cmd, old_kwargs = _command_for(old, submission_type, hosts, tmp)
            assert new_cmd == old_cmd, (
                f"{submission_type} (hosts={bool(hosts)}) script differs:\n--- new ---\n{new_cmd}\n--- old ---\n{old_cmd}")
            assert sorted(new_kwargs["output_folders"]) == sorted(old_kwargs["output_folders"])
            assert new_kwargs["check_files_in_folder"].keys() == old_kwargs["check_files_in_folder"].keys()
            print(f"PASS: {submission_type} (hosts={bool(hosts)}) script is byte-identical to {HEAD} ({len(new_cmd)} chars)")


def test_array_rescue_pieces():
    """The array builder is still the only one that fills the per-folder rescue maps."""
    folders = ["base_cgyro/rho_0.3486", "base_cgyro/rho_0.6712"]
    array = SIMtools.ArraySlurmScript(folders, _code_call, 4, EXEC_FOLDER)
    assert array.array_index_by_folder == {folders[0]: 0, folders[1]: 1}, array.array_index_by_folder
    assert set(array.per_folder_commands) == set(folders)
    assert array.array_list == ["0", "1"], array.array_list
    for builder in (SIMtools.BashScript(folders, _code_call, 4, EXEC_FOLDER),
                    SIMtools.StandardSlurmScript(folders, _code_call, 4, EXEC_FOLDER)):
        assert builder.per_folder_commands == {} and builder.array_index_by_folder == {}
    print("PASS: every builder defines the three attributes; only the array one fills the rescue maps")


# ---------------------------------------------------------------------------
# SubmissionRecord
# ---------------------------------------------------------------------------

RHOS = [0.3486, 0.6712]


def _record_sim(module, folder):
    sim = _build_sim(module, folder, RHOS)
    sim._submission_metadata_filename = "cgyro_submission.json"
    sim.slurm_output = "slurm_output.dat"
    sim.simulation_job = FakeJob(folder / "tmp_cgyro")
    sim.simulation_job.jobid = "918273"
    sim.simulation_job.slurm_settings = {"name": "cgyro_sim", "minutes": 30}
    sim.simulation_job.output_files = ["mitim.out"]
    sim.simulation_job.output_folders = ["base_cgyro/rho_0.3486"]
    sim.simulation_job.check_files_in_folder = {"base_cgyro/rho_0.3486": ["out.cgyro.info"]}
    sim.simulation_job.output_folders_selective = {"base_cgyro/rho_0.3486": ["out.cgyro.info"]}
    sim.simulation_job.output_file_fallbacks = {"bin.cgyro.restart": "bin.cgyro.restart.old"}
    sim.kwargs_organize = {
        "code_executor": {"base_cgyro": {rho: {"folder": folder / "base_cgyro"} for rho in RHOS}},
        "tmpFolder": folder / "tmp_cgyro",
        "filesToRetrieve": ["out.cgyro.info"],
        "optional_files_to_retrieve": ["bin.cgyro.restart"],
        "array_index_by_folder": {"base_cgyro/rho_0.3486": 0, "base_cgyro/rho_0.6712": 1},
        "per_folder_commands": {"base_cgyro/rho_0.3486": "cd . && cgyro"},
    }
    sim._resubmit_ledger = {"base_cgyro/rho_0.6712": {"status": "RESUBMITTED", "child_jobids": ["918300"]}}
    sim._restart_sources_payload = {"0.6712": {"iter": 3}}
    return sim


def _strip_volatile(payload):
    payload = dict(payload)
    payload.pop("created_utc")
    return payload


def test_submission_record_compat(old):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        old_folder, new_folder = tmp / "old", tmp / "new"
        for folder in (old_folder, new_folder):
            (folder / "base_cgyro").mkdir(parents=True)

        with contextlib.redirect_stdout(io.StringIO()):
            old.mitim_simulation._write_submission_metadata(_record_sim(old, old_folder), "base_cgyro")
            SIMtools.mitim_simulation._write_submission_metadata(_record_sim(SIMtools, new_folder), "base_cgyro")

        old_json = json.loads((old_folder / "base_cgyro" / "cgyro_submission.json").read_text())
        new_json = json.loads((new_folder / "base_cgyro" / "cgyro_submission.json").read_text())
        # Folder paths differ by design (two temp trees); compare everything else
        old_norm = json.dumps(_strip_volatile(old_json)).replace(str(old_folder), "ROOT")
        new_norm = json.dumps(_strip_volatile(new_json)).replace(str(new_folder), "ROOT")
        assert old_norm == new_norm, f"payloads differ:\n--- old ---\n{old_norm}\n--- new ---\n{new_norm}"
        assert list(old_json) == list(new_json), (list(old_json), list(new_json))
        assert new_json["created_utc"].endswith("Z") and "+00:00" not in new_json["created_utc"], new_json["created_utc"]
        print(f"PASS: the new writer's JSON matches {HEAD}'s key-for-key ({len(new_json)} top-level keys)")

        # Old writer -> new reader
        target = _build_sim(SIMtools, new_folder, RHOS)
        with contextlib.redirect_stdout(io.StringIO()):
            data = target.load_submission_state(old_folder / "base_cgyro" / "cgyro_submission.json")
        assert data["schema_version"] == SIMtools.SubmissionRecord.SCHEMA
        assert target.slurm_output == "slurm_output.dat"
        assert target.simulation_job.jobid == "918273"
        assert sorted(target.kwargs_organize["code_executor"]["base_cgyro"]) == sorted(RHOS)
        assert target.kwargs_organize["array_index_by_folder"]["base_cgyro/rho_0.6712"] == 1
        assert target._resubmit_ledger["base_cgyro/rho_0.6712"]["child_jobids"] == ["918300"]
        assert target._base_subfolder == "base_cgyro"
        assert (old_folder / "base_cgyro" / "restart_sources.json").exists()
        print("PASS: a record written by the old function loads through SubmissionRecord.read/apply_to")

        # New writer -> old reader
        target_old = _build_sim(old, old_folder, RHOS)
        with contextlib.redirect_stdout(io.StringIO()):
            data = old.mitim_simulation.load_submission_state(target_old, new_folder / "base_cgyro" / "cgyro_submission.json")
        assert target_old.slurm_output == "slurm_output.dat"
        assert sorted(target_old.kwargs_organize["code_executor"]["base_cgyro"]) == sorted(RHOS)
        assert target_old._resubmit_ledger["base_cgyro/rho_0.6712"]["status"] == "RESUBMITTED"
        print(f"PASS: a record written by SubmissionRecord loads through {HEAD}'s load_submission_state")


def test_submission_record_schema_guard():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cgyro_submission.json"
        path.write_text(json.dumps({"schema_version": 99, "slurm_output": "x", "kwargs_organize": {}, "job": {}}))
        try:
            SIMtools.SubmissionRecord.read(path)
        except RuntimeError as e:
            assert "schema 99" in str(e) and f"schema {SIMtools.SubmissionRecord.SCHEMA}" in str(e), str(e)
            print("PASS: a record from another schema is refused by name")
        else:
            raise AssertionError("a mismatched schema_version must be refused")


# ---------------------------------------------------------------------------
# Enums and the missing-state guards
# ---------------------------------------------------------------------------

def test_enums():
    assert SIMtools.RunType.parse("run") is SIMtools.RunType.NORMAL
    assert SIMtools.RunType.parse("normal") is SIMtools.RunType.NORMAL
    assert SIMtools.RunType.parse(SIMtools.RunType.SEND) is SIMtools.RunType.SEND
    assert SIMtools._normalize_run_type("run") == "normal"
    assert [r.blocking for r in SIMtools.RunType] == [True, False, True, False]
    assert SIMtools.JobStatus.GONE == 2 and SIMtools.JobStatus.PENDING == 0
    assert SIMtools.SubmissionType.parse("bash") is SIMtools.SubmissionType.BASH
    for bad, enum in (("nope", SIMtools.RunType), ("slurm", SIMtools.SubmissionType)):
        try:
            enum.parse(bad)
        except ValueError as e:
            assert bad in str(e), str(e)
        else:
            raise AssertionError(f"{enum.__name__}.parse({bad!r}) must raise")
    print("PASS: RunType.parse('run') == NORMAL, .blocking covers normal+send, JobStatus compares as int")


def test_state_guards():
    sim = SIMtools.mitim_simulation()
    sim.run_specifications = {"code": "cgyro"}
    sim.simulation_job = types.SimpleNamespace(launchSlurm=True)

    for call, attribute in ((sim.check, "slurm_output"), (sim.fetch, "kwargs_organize")):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                call()
        except RuntimeError as e:
            assert attribute in str(e) and "run_type='submit'" in str(e), str(e)
        else:
            raise AssertionError(f"{call.__name__}() before a submit must raise")
    print("PASS: check()/fetch() before a submit name the missing state instead of AttributeError")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as workdir:
        old_module = _load_old_simtools(Path(workdir))
        test_script_byte_identity(old_module)
        test_array_rescue_pieces()
        test_submission_record_compat(old_module)
        test_submission_record_schema_guard()
        test_enums()
        test_state_guards()
    print("\nALL PASS")
