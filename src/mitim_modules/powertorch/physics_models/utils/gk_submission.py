'''
The detached-submission lifecycle of a gyrokinetic run: re-attach to a job a previous process
submitted, poll it, fetch it, and drop its metadata once the results are in.

Backend-agnostic: the metadata file name comes from the simulation object
(`_submission_metadata_filename`: cgyro_submission.json, gx_submission.json), so a backend that
writes none simply never re-attaches.
'''

from dataclasses import dataclass
from pathlib import Path

from mitim_tools.misc_tools.LOGtools import printMsg as print


@dataclass
class ReattachOutcome:
    '''
    reattached:      the job is already on the machine, so do not submit again.
    skip_check_fetch: its results are already on local disk, so go straight to read().
    '''
    reattached: bool = False
    skip_check_fetch: bool = False


class GKSubmission:
    '''
    Owns <folder>/<base_subfolder>/<metadata file> and the decisions it drives.

    try_reattach:

      | metadata file        | slurm job        | local results | outcome        | side effects              |
      |----------------------|------------------|---------------|----------------|---------------------------|
      | backend writes none  | -                | -             | (False, False) | one warning per process   |
      | disabled by namelist | -                | -             | (False, False) | nothing                   |
      | absent               | -                | -             | (False, False) | "no prior submission"     |
      | present              | parent or rescue |               |                |                           |
      |                      | child alive      | -             | (True,  False) | poll it                   |
      | present              | gone             | complete      | (True,  True)  | no fetch, read directly   |
      | present              | gone             | incomplete,   |                |                           |
      |                      |                  | fetch fills   | (True,  True)  | one fetch()               |
      | present              | gone             | incomplete,   |                | metadata unlinked,        |
      |                      |                  | fetch fails   | (False, False) | on_fresh_fallback() runs  |

    The last row is the only one that submits again, and the only one that re-resolves the
    restart chain: every other row must keep the parent pick of the original submission.
    '''

    _warned_backends = set()

    def __init__(self, sim, folder, base_subfolder, every_n_minutes=10, enabled=True,
                 connection_retry_settings=None, label="CGYRO",
                 submit_name="run()", reader_name="read()", organize_label="per-rho folders"):
        self.sim = sim
        self.folder = Path(folder)
        self.base_subfolder = base_subfolder
        self.every_n_minutes = every_n_minutes
        self.enabled = enabled
        self.connection_retry_settings = connection_retry_settings
        self.label = label
        self.submit_name = submit_name
        self.reader_name = reader_name
        self.organize_label = organize_label
        self._skip_check_fetch = False

        filename = getattr(sim, "_submission_metadata_filename", None)
        self.path = None if filename is None else self.folder / base_subfolder / filename

    @property
    def name(self):
        return self.path.name if self.path is not None else "<no metadata file>"

    # ------------------------------------------------------------------

    def exists(self):
        return self.path is not None and self.path.exists()

    def try_reattach(self, on_fresh_fallback=None, before_load=None):
        '''
        `before_load` runs after the banner and before load_submission_state, for the state a
        reader needs but a re-attach never staged (FolderSimLast, the per-plasma folders).
        `on_fresh_fallback` runs only when the re-attach gives up and the caller will submit.
        '''
        self._skip_check_fetch = False
        if not self.enabled:
            return ReattachOutcome()

        if self.path is None:
            backend = type(self.sim).__name__
            if backend not in self._warned_backends:
                self._warned_backends.add(backend)
                print(
                    f"\t- check_existing_runs=True but {backend} writes no submission metadata; "
                    "re-attach is unavailable for this backend and every evaluation submits fresh",
                    typeMsg='w',
                )
            return ReattachOutcome()

        if not self.path.exists():
            self._announce_no_metadata()
            return ReattachOutcome()

        self._announce_metadata()
        if before_load is not None:
            before_load()
        data = self.sim.load_submission_state(self.path)
        # load_submission_state builds a fresh mitim_job from the JSON, so the namelist-tunable
        # retry config has to be put back on it (construction-site propagation only fires for run())
        if self.connection_retry_settings is not None and getattr(self.sim, "simulation_job", None) is not None:
            self.sim.simulation_job.connection_retry_settings = self.connection_retry_settings
        self._announce_prior_job(data)

        if self._job_alive():
            return ReattachOutcome(reattached=True, skip_check_fetch=False)

        print(f"\t- Slurm reports job is NOT in the queue (state={self.sim.simulation_job.infoSLURM.get('STATE')})", typeMsg='i')
        if self.sim._local_results_complete():
            print(f"\t- All expected {self.label} output files are already on local disk — skipping check()/fetch() and jumping to {self.reader_name}", typeMsg='i')
            self._skip_check_fetch = True
            return ReattachOutcome(reattached=True, skip_check_fetch=True)

        print("\t- Local results incomplete; attempting fetch() from remote scratch folder in case the job finished while we were offline...", typeMsg='i')
        try:
            self.sim.fetch()
        except Exception as e:
            print(f"\t- fetch() raised ({e})", typeMsg='w')
        if self.sim._local_results_complete():
            print(f"\t- Remote scratch had the results — fetch complete, skipping check()/fetch() in the main loop and jumping to {self.reader_name}", typeMsg='i')
            self._skip_check_fetch = True
            return ReattachOutcome(reattached=True, skip_check_fetch=True)

        print(f"\t- Even after fetch() the expected {self.label} output files are incomplete — the prior submission apparently failed.", typeMsg='w')
        print(f"\t  Removing {self.name} and falling back to a fresh submission", typeMsg='w')
        self.path.unlink(missing_ok=True)
        if on_fresh_fallback is not None:
            on_fresh_fallback()
        return ReattachOutcome()

    def poll_and_fetch(self, reattached, every_n_minutes=None):
        '''
        Wait for the detached job to leave the queue and pull its results. `reattached` only
        decides whether the first poll reuses the squeue the liveness probe just ran.
        '''
        if self._skip_check_fetch:
            print("")
            print("\t- [submit] Results were already local — reading them directly without polling or fetching.", typeMsg='i')
            print("")
            return

        every_n_minutes = self.every_n_minutes if every_n_minutes is None else every_n_minutes
        print("")
        print(f"\t- [submit] Polling slurm every {every_n_minutes} min until the job leaves the queue (state NOT FOUND / squeue returns nothing).", typeMsg='i')
        print(f"\t  You can ^C at any time; {self.name} is on disk so re-attach will resume from where we left off.", typeMsg='i')
        print("")
        self.sim.check(
            every_n_minutes=every_n_minutes,
            skip_first_iteration_squeue=reattached,
            custom_checker=getattr(self.sim, "_custom_check_callback", None),
        )

        print("")
        print(f"\t- [submit] Job finished on the cluster — pulling the result tarball and organizing files into {self.organize_label}.", typeMsg='i')
        print("")
        self.sim.fetch()

    def cleanup(self, remove_scratch=False):
        '''
        End of the successful path: optionally drop the remote scratch folder, then drop the
        metadata. The invariant is "metadata present => job in flight (or retrieval not yet
        complete)", so the file only goes once the results have been read and used — a raise
        anywhere earlier must leave it behind for the next process to re-attach to.
        '''
        if remove_scratch:
            # A flaky connection must not abort the PORTALS iteration just because rm -rf failed
            try:
                self.sim.simulation_job.connect()
                self.sim.simulation_job.remove_scratch_folder()
                self.sim.simulation_job.close()
                print(f"\t- [submit] remove_scratch_after_fetch=true — removed remote scratch {self.sim.simulation_job.folderExecution}", typeMsg='i')
            except Exception as e:
                print(f"\t- remote scratch removal raised ({e}); leaving the folder in place", typeMsg='w')

        if self.path is None:
            return
        if self.path.exists():
            print(f"\t- [check_existing_runs] Read finished — removing stale submission metadata at {self.name} so the next PORTALS iteration submits fresh", typeMsg='i')
        self.path.unlink(missing_ok=True)

    # ------------------------------------------------------------------

    def _job_alive(self):
        '''
        The parent array, or any rescue child the auto-resubmit path spawned before the prior
        driver died. An unreachable probe counts as "not alive" here (no signal: fall through
        to the local-results checks), unlike in the poll loop.
        '''
        print("\t- Liveness probe via squeue...", typeMsg='i')
        self.sim.simulation_job.check(file_output=self.sim.slurm_output)
        parent_alive = (self.sim.simulation_job.status != 2)
        any_child_alive = self.sim._any_child_job_alive()
        print("")
        if not (parent_alive or any_child_alive):
            return False

        live_summary = f"jobid={self.sim.simulation_job.jobid}, state={self.sim.simulation_job.infoSLURM.get('STATE')}"
        if any_child_alive:
            live_summary += f"; rescue child jobid(s) still alive: {self.sim._child_jobids()}"
        print(f"\t- Slurm reports job is still live ({live_summary}); proceeding with check()/fetch()", typeMsg='i')
        print("")
        return True

    def _announce_metadata(self):
        print("")
        print(f"\t==================== [check_existing_runs] Re-attach to existing {self.label} submission ====================", typeMsg='i')
        print("")
        print("\t- Submission metadata found at:", typeMsg='i')
        print(f"\t     {self.path}", typeMsg='i')
        print("")

    def _announce_no_metadata(self):
        print("")
        print(f"\t==================== [check_existing_runs] No prior {self.label} submission to re-attach ====================", typeMsg='i')
        print("")
        print("\t- Looked for metadata at:", typeMsg='i')
        print(f"\t     {self.path}", typeMsg='i')
        print("\t- File does not exist; this is a fresh PORTALS evaluation, submitting CGYRO normally", typeMsg='i')
        print("")

    def _announce_prior_job(self, data):
        job = data.get("job", {})
        print(f"\t- Prior submission: jobid={job.get('jobid')} on {job.get('machineSettings', {}).get('machine')}", typeMsg='i')
        print(f"\t     remote folder: {job.get('folderExecution')}", typeMsg='i')
        print(f"\t     submitted at:  {data.get('created_utc')} (schema v{data.get('schema_version')})", typeMsg='i')
        print("")
        print(f"\t- Skipping {self.submit_name}/sbatch; polling this job with every_n_minutes={self.every_n_minutes}", typeMsg='i')
        print("")
