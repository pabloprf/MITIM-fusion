"""
Post-hoc application of a MAESTRO `prune_level` to a run that already FINISHED at a
lower level (typically level 0, i.e. everything kept).

The levels are exactly those a live run applies via `maestro.prune_level` -- see
templates/namelist.maestro.yaml for the authoritative description, and MAESTRObeat for
the code (the per-beat `scratch_patterns` / `_scratch_to_drop`, `prune_run_folder`,
`prune_initializer`, and `portals_beat.optional_postprocessing`). This script imports
those definitions rather than restating them, so the two cannot drift:

  level 1 : per-beat execution scratch inside `run_<name>/` (the transp `results/`
            duplicate of the multi-GB CDF, the eped per-height TOQ/ELITE work dirs, the
            portals `Execution/` trees). Every plot tab still works afterwards.
  level 2 : 1 + wipe each `run_<name>/` entirely. The TRANSP CDF goes here -- it is
            deliberately never copied into `beat_results/` (only the small
            `transp_results.npy` subset travels forward).
  level 3 : 2 + the PORTALS end-of-run pass (LAST beat's `optimization_object.pkl`
            re-saved lean, INTERMEDIATE beats' pickles / `portals_profiles/` / logs
            dropped; `surrogate_data.csv` and `beat_results/input.gacode` kept) and the
            initializer prune (throwaway geqdsk intermediates + any nested run folder
            such as `initializer_eped/run_eped/`).

`beat_results/` is never touched except by the PORTALS pass at level 3, and
`initializer_*/input.gacode`, `input.geqdsk` and `beat_results/` are never touched at
all. Dry-run by default; pass --apply to actually delete.

    mitim_prune_maestro FOLDER1 FOLDER2 ...              # dry-run at the default level (3)
    mitim_prune_maestro FOLDER1 --level 1                # dry-run, scratch only
    mitim_prune_maestro FOLDER1 --level 2 --apply        # actually prune
    mitim_prune_maestro scan_dir/case_* --apply          # every matching run (shell-expanded glob)

Multiple folders are summed into a grand total at the end. A shell glob (scan_dir/case_*)
is expanded by the shell into one argument per match before this script sees it.
"""

import argparse
from pathlib import Path
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import (
    PRUNE_SCRATCH, PRUNE_RUN, PRUNE_OUTPUTS, PRUNE_LEVELS, _INITIALIZER_SCRATCH)
from mitim_modules.maestro.utils.TRANSPbeat import transp_beat
from mitim_modules.maestro.utils.PORTALSbeat import portals_beat

# Heavy items an INTERMEDIATE PORTALS beat drops (relative to beat_results/Outputs/).
# Mirrors PORTALSbeat.optional_postprocessing.
_PORTALS_INTERMEDIATE_DROP = [
    'optimization_object.pkl',
    'optimization_extra.pkl',
    'optimization_log.txt',
    'portals_profiles',   # directory
]

# Level-1 scratch per beat name, taken from the beat classes so this script cannot drift from
# the live run. eped is special-cased: its selection is dirs-only under case1/run1 (see
# eped_beat._scratch_to_drop), which a plain glob list cannot express.
_SCRATCH_PATTERNS = {
    'transp': transp_beat.scratch_patterns,
    'portals': portals_beat.scratch_patterns,
}


def _scratch_targets(run_dir, name):
    '''Level-1 targets inside a run_<name>/ folder, mirroring the beat classes.'''
    if name == 'eped':
        return [p for p in sorted(run_dir.glob('case1/run1/*')) if p.is_dir()]
    targets = []
    for pattern in _SCRATCH_PATTERNS.get(name, []):
        targets += sorted(run_dir.glob(pattern))
    return targets


def _bytes_of(path):
    return IOtools.path_size_bytes(path)


def _human(nbytes):
    return IOtools.human_readable_size(nbytes)


class MaestroPruner:
    '''Apply a MAESTRO prune level to a finished run folder.'''

    def __init__(self, root, apply=False, level=PRUNE_OUTPUTS):
        self.root = Path(root)
        self.apply = apply
        self.level = level
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

            if self.level >= PRUNE_RUN:
                items = sorted(run_dir.iterdir())
                what = f"wipe run_{b['name']}/"
            else:
                items = _scratch_targets(run_dir, b['name'])
                what = f"drop run_{b['name']}/ scratch"

            if not items:
                continue
            size = sum(_bytes_of(item) for item in items)
            self.freed += size
            print(f"\t- Beat {b['counter']} ({b['name']}): {what} "
                  f"[{_human(size)}, {len(items)} item(s)]{'' if self.apply else '  (dry-run)'}")
            if self.apply:
                for item in items:
                    IOtools.shutil_rmtree(item) if item.is_dir() else item.unlink()

    # -------------------------------------------------------------------------
    # (3) Initializer prune -- mirrors MAESTRObeat.beat.prune_initializer
    # -------------------------------------------------------------------------
    def prune_initializers(self, beats):
        for b in beats:
            targets = []
            for initializer_folder in sorted(b['folder'].glob('initializer_*')):
                targets += [initializer_folder / name for name in _INITIALIZER_SCRATCH]
                for nested_run in sorted(initializer_folder.glob('run_*')):
                    targets += sorted(nested_run.iterdir()) if nested_run.is_dir() else []
            targets = [t for t in targets if t.exists()]
            if not targets:
                continue
            size = sum(_bytes_of(t) for t in targets)
            self.freed += size
            print(f"\t- Beat {b['counter']} ({b['name']}): prune initializer scratch "
                  f"[{_human(size)}, {len(targets)} item(s)]{'' if self.apply else '  (dry-run)'}")
            if self.apply:
                for t in targets:
                    IOtools.shutil_rmtree(t) if t.is_dir() else t.unlink()

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
        '''Prune (or dry-run) this folder. Returns True if it was a MAESTRO run, False if skipped.'''
        if not self.beats_dir.is_dir():
            print(f'- {IOtools.clipstr(self.root)}: no Beats/ folder -- not a MAESTRO run, skipping', typeMsg='w')
            return False
        print(f"\n- {'Pruning' if self.apply else 'Dry-run for'} {IOtools.clipstr(self.root)} at level {self.level}")
        beats = self.discover_beats()
        self.wipe_run_folders(beats)
        if self.level >= PRUNE_OUTPUTS:
            self.prune_portals(beats)
            self.prune_initializers(beats)
        verb = 'Freed' if self.apply else 'Would free'
        msg = f"\t=> {verb} {_human(self.freed)}"
        if self.slim_before:
            msg += f" (+ slim ~{_human(self.slim_before)} of last-beat pickle)"
        print(msg, typeMsg='i')
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Apply a MAESTRO prune_level to an already-finished run (see maestro.prune_level).')
    parser.add_argument('folders', type=str, nargs='+', help='MAESTRO run folder(s) to prune.')
    parser.add_argument('--level', type=int, default=PRUNE_OUTPUTS, choices=list(PRUNE_LEVELS[1:]),
                        help='Prune level to apply: 1 execution scratch only (every plot tab still works), '
                             '2 also wipe run_<name>/, 3 also prune outputs+initializers (default).')
    parser.add_argument('--apply', action='store_true',
                        help='Actually delete/slim. Without it, only report what would be freed (dry-run).')
    args = parser.parse_args()

    if not args.apply:
        print('\n[DRY-RUN] Nothing will be deleted. Re-run with --apply to prune.', typeMsg='w')

    total_freed, total_slim, n = 0, 0, 0
    for folder in args.folders:
        pruner = MaestroPruner(folder, apply=args.apply, level=args.level)
        if pruner.run():
            n += 1
            total_freed += pruner.freed
            total_slim += pruner.slim_before

    # Grand total across every folder specified on the command line.
    if n > 1:
        verb = 'Freed' if args.apply else 'Would free'
        msg = f"\n=== {verb} {_human(total_freed)} across {n} run(s)"
        if total_slim:
            msg += f" (+ slim ~{_human(total_slim)} of last-beat pickles)"
        print(msg, typeMsg='i')


if __name__ == '__main__':
    main()
