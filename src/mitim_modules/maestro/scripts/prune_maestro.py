"""
Post-hoc replication of MAESTRO's `keep_all_files: false` space-saving on a run
that was already FINISHED with `keep_all_files: true`.

A live MAESTRO run with `maestro.keep_all_files: false` shrinks its on-disk
footprint in two places (the source of truth):

  1. MAESTROmain._run_beat (the `if not self.keep_all_files:` block): after every
     beat, the contents of that beat's `run_<name>/` folder are wiped -- everything
     a downstream beat or a replot needs has already been persisted into
     `beat_results/`. The multi-GB TRANSP CDF lives in `run_transp/` and is
     deliberately NOT copied to `beat_results/` (only the small `transp_results.npy`
     subset is), so under `keep_all_files: false` the CDF is discarded here.

  2. PORTALSbeat.optional_postprocessing (run once at the end of the whole run):
       - the LAST PORTALS beat is KEPT but its `optimization_object.pkl` is re-saved
         lean (the fitted GP `steps` dropped -- the bulk of the pickle) so the final
         core solution still replots;
       - every INTERMEDIATE PORTALS beat drops the heavy items nothing downstream
         reads: `optimization_object.pkl`, `optimization_extra.pkl`,
         `optimization_log.txt`, the per-iteration `portals_profiles/` snapshots, and
         that beat's `Outputs/Logs/beat_<n>_*.log`. Chaining keeps `surrogate_data.csv`
         and `beat_results/input.gacode`.

This tool reproduces exactly that end-state on an already-finished
`keep_all_files: true` folder. It never touches `beat_results/` except for the
PORTALS pruning in (2), and never touches the `initializer_*` folders (neither does
the live cleanup). Dry-run by default; pass --apply to actually delete.

    mitim_prune_maestro FOLDER1 FOLDER2 ...          # dry-run: report what would be freed
    mitim_prune_maestro FOLDER1 --apply              # actually prune

Keep the pruned/slimmed set in sync with PORTALSbeat.optional_postprocessing and the
run-folder wipe in MAESTROmain._run_beat if those ever change.
"""

import argparse
from pathlib import Path
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print

# Heavy items an INTERMEDIATE PORTALS beat drops (relative to beat_results/Outputs/).
# Mirrors PORTALSbeat.optional_postprocessing.
_PORTALS_INTERMEDIATE_DROP = [
    'optimization_object.pkl',
    'optimization_extra.pkl',
    'optimization_log.txt',
    'portals_profiles',   # directory
]


def _bytes_of(path):
    '''Total bytes of a file or (recursively) a directory. 0 if missing.'''
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


def _human(nbytes):
    size = float(nbytes)
    for unit in ('B', 'K', 'M', 'G', 'T'):
        if size < 1024 or unit == 'T':
            return f'{size:.0f}{unit}' if unit == 'B' else f'{size:.1f}{unit}'
        size /= 1024
    return f'{size:.1f}T'


class MaestroPruner:
    '''Replicate `keep_all_files: false` on a finished MAESTRO root folder.'''

    def __init__(self, root, apply=False):
        self.root = Path(root)
        self.apply = apply
        self.beats_dir = self.root / 'Beats'
        self.logs_dir = self.root / 'Outputs' / 'Logs'
        self.freed = 0          # bytes deleted outright (run wipe + intermediate prune)
        self.slim_before = 0    # current size of last-beat pickles that get slimmed (not fully freed)

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------
    def discover_beats(self):
        '''
        Return a list of {counter, folder, name, run_dir} for each Beats/Beat_<n>,
        sorted by the integer counter (NOT lexically -- Beat_10 must follow Beat_9).
        `name` is taken from the run_<name>/ subfolder; when that folder was already
        removed, it is inferred from beat_results (portals if it carries PORTALS Outputs).
        `run_dir` is None when no run_<name>/ folder is present.
        '''
        beats = []
        for beat_folder in self.beats_dir.iterdir():
            if not (beat_folder.is_dir() and beat_folder.name.startswith('Beat_')):
                continue
            try:
                counter = int(beat_folder.name.split('_')[1])
            except (IndexError, ValueError):
                print(f'\t- Skipping unrecognized beat folder {beat_folder.name}', typeMsg='w')
                continue
            run_dirs = [d for d in beat_folder.iterdir() if d.is_dir() and d.name.startswith('run_')]
            run_dir = run_dirs[0] if run_dirs else None
            if run_dir is not None:
                name = run_dir.name[len('run_'):]
            elif (beat_folder / 'beat_results' / 'Outputs' / 'surrogate_data.csv').exists():
                name = 'portals'   # run_ already wiped-and-removed; classify from outputs
            else:
                name = 'unknown'
            beats.append({'counter': counter, 'folder': beat_folder, 'name': name, 'run_dir': run_dir})
        return sorted(beats, key=lambda b: b['counter'])

    # -------------------------------------------------------------------------
    # (1) Per-beat run_<name>/ wipe -- mirrors MAESTROmain._run_beat
    # -------------------------------------------------------------------------
    def wipe_run_folders(self, beats):
        for b in beats:
            run_dir = b['run_dir']
            if run_dir is None:
                continue
            items = list(run_dir.iterdir())
            if not items:
                continue
            size = _bytes_of(run_dir)
            self.freed += size
            print(f"\t- Beat {b['counter']} ({b['name']}): wipe run_{b['name']}/ "
                  f"[{_human(size)}, {len(items)} item(s)]{'' if self.apply else '  (dry-run)'}")
            if self.apply:
                for item in items:
                    IOtools.shutil_rmtree(item) if item.is_dir() else item.unlink()

    # -------------------------------------------------------------------------
    # (2) PORTALS slim/prune -- mirrors PORTALSbeat.optional_postprocessing
    # -------------------------------------------------------------------------
    def prune_portals(self, beats):
        portals = [b for b in beats if b['name'] == 'portals']
        if not portals:
            return
        last_counter = max(b['counter'] for b in portals)
        for b in portals:
            outputs = b['folder'] / 'beat_results' / 'Outputs'
            if not outputs.is_dir():
                print(f"\t- Beat {b['counter']} (portals): no beat_results/Outputs -- skipping", typeMsg='w')
                continue
            if b['counter'] == last_counter:
                self._slim_last_portals(b, outputs)
            else:
                self._prune_intermediate_portals(b, outputs)

    def _slim_last_portals(self, b, outputs):
        '''
        Last PORTALS beat: re-save optimization_object.pkl lean (drop GP steps).

        This is the only step that must UNPICKLE the PORTALS object. On old runs whose
        pickled classes have since drifted from the current code (e.g. a removed
        PORTALStools attribute) the load can fail; we then leave the full pickle in place
        and warn rather than abort -- the run-wipe and intermediate prune (which never
        unpickle) have already reclaimed the bulk of the space.
        '''
        pkl = outputs / 'optimization_object.pkl'
        if not pkl.exists():
            return
        size = _bytes_of(pkl)
        print(f"\t- Beat {b['counter']} (portals, LAST): slim optimization_object.pkl "
              f"[currently {_human(size)}, keeps GP-less pickle]{'' if self.apply else '  (dry-run)'}")
        if self.apply:
            from mitim_tools.opt_tools import STRATEGYtools
            try:
                m = STRATEGYtools.read_from_scratch(pkl)
                m.folderOutputs = outputs
                m.save(lean=True)
            except Exception as e:
                print(f"\t\t- Could not slim last PORTALS pickle (kept full): {type(e).__name__}: {e}", typeMsg='w')
                return
        self.slim_before += size

    def _prune_intermediate_portals(self, b, outputs):
        '''Intermediate PORTALS beat: drop the heavy items nothing downstream reads.'''
        targets = [outputs / n for n in _PORTALS_INTERMEDIATE_DROP]
        targets += list(self.logs_dir.glob(f"beat_{b['counter']}_*.log")) if self.logs_dir.is_dir() else []
        present = [t for t in targets if t.exists()]
        if not present:
            return
        size = sum(_bytes_of(t) for t in present)
        self.freed += size
        names = ', '.join(t.name for t in present)
        print(f"\t- Beat {b['counter']} (portals, intermediate): drop [{_human(size)}] {names}"
              f"{'' if self.apply else '  (dry-run)'}")
        if self.apply:
            for t in present:
                IOtools.shutil_rmtree(t) if t.is_dir() else t.unlink()

    # -------------------------------------------------------------------------
    def run(self):
        if not self.beats_dir.is_dir():
            print(f'- {IOtools.clipstr(self.root)}: no Beats/ folder -- not a MAESTRO run, skipping', typeMsg='w')
            return
        print(f"\n- {'Pruning' if self.apply else 'Dry-run for'} {IOtools.clipstr(self.root)}")
        beats = self.discover_beats()
        self.wipe_run_folders(beats)
        self.prune_portals(beats)
        verb = 'Freed' if self.apply else 'Would free'
        msg = f"\t=> {verb} {_human(self.freed)}"
        if self.slim_before:
            msg += f" (+ slim ~{_human(self.slim_before)} of last-beat pickle)"
        print(msg, typeMsg='i')


def main():
    parser = argparse.ArgumentParser(
        description='Replicate MAESTRO keep_all_files:false space-saving on a finished keep_all_files:true run.')
    parser.add_argument('folders', type=str, nargs='+', help='MAESTRO run folder(s) to prune.')
    parser.add_argument('--apply', action='store_true',
                        help='Actually delete/slim. Without it, only report what would be freed (dry-run).')
    args = parser.parse_args()

    if not args.apply:
        print('\n[DRY-RUN] Nothing will be deleted. Re-run with --apply to prune.', typeMsg='w')

    for folder in args.folders:
        MaestroPruner(folder, apply=args.apply).run()


if __name__ == '__main__':
    main()
