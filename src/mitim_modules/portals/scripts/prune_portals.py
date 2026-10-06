"""
Wipe the CGYRO restart blobs (bin.cgyro.restart*, including .old) of every PORTALS evaluation
except the result one. The same function runs at the end of a converged PORTALS run when
transport.options.cgyro.prune_restarts_at_convergence is true (see templates/namelist.portals.yaml
for what is deleted and kept). Everything else (outputs, traces, inputs, fluxes_*.json,
restart_sources.json) stays. Irreversible: the pruned evaluations can no longer seed a warm start.

Dry-run by default; pass --apply to actually delete.

    mitim_prune_portals FOLDER1 FOLDER2 ...          # dry-run, keep the result (best) evaluation
    mitim_prune_portals FOLDER1 --apply              # actually delete
    mitim_prune_portals FOLDER1 --keep last --apply  # keep the last evaluation instead
    mitim_prune_portals FOLDER1 --force --apply      # also on a run that did not converge
"""

import argparse
from pathlib import Path
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print

_RESTART_GLOB = "bin.cgyro.restart*"


def _evaluation_restarts(folder):
    '''{evaluation folder: [restart files]} for every BO evaluation and simple-relax step.'''
    roots = sorted(folder.glob("Execution/Evaluation.*")) + sorted(folder.glob("Initialization/initialization_simple_relax/portals_sr_ev_*"))
    return {root: sorted((root / "transport_simulation_folder").rglob(_RESTART_GLOB)) for root in roots}


def prune_cgyro_restarts(folder, keep_index, apply=False):
    '''
    Delete the CGYRO restart files of every evaluation except Execution/Evaluation.<keep_index>.
    Returns the bytes freed (or that would be freed in dry-run).

    The parallel simple-relax initializer leaves Evaluation.<i> restarts as symlinks into
    portals_sr_ev_<s>/; the targets of the kept evaluation's links are kept too. Sizes use
    lstat so a link counts ~0 and its target is counted once.
    '''
    folder = Path(folder)
    per_evaluation = _evaluation_restarts(folder)

    kept = per_evaluation.get(folder / "Execution" / f"Evaluation.{keep_index}", [])
    keep = set(kept) | {p.resolve() for p in kept}

    targets = [p for files in per_evaluation.values() for p in files if p not in keep and p.resolve() not in keep]
    freed = sum(p.lstat().st_size for p in targets)

    print(f"\t- CGYRO restarts: {'deleting' if apply else 'would delete'} {len(targets)} file(s) [{IOtools.human_readable_size(freed)}] "
          f"across {len(per_evaluation)} evaluation folder(s); keeping {len(kept)} of Evaluation.{keep_index}"
          f"{'' if apply else '  (dry-run)'}")
    if apply:
        for p in targets:
            p.unlink()

    return freed


def resolve_result(folder, keep="best"):
    '''(evaluation index to keep, converged?) of a finished PORTALS run, as PORTALS itself reports them.'''
    from mitim_modules.portals.utils import PORTALSanalysis

    portals = PORTALSanalysis.PORTALSanalyzer.from_folder(folder)
    if not hasattr(portals, "ibest"):
        return None, False

    mitim_bo = portals.opt_fun.mitim_model
    converged = getattr(mitim_bo, "converged", None)
    if converged is None:
        # Pickles predating MITIM_BO.converged: re-run the stopping criteria on the loaded object
        convergence_options = mitim_bo.optimization_options["convergence_options"]
        converged, _ = convergence_options["stopping_criteria"](mitim_bo, parameters=convergence_options["stopping_criteria_parameters"])

    return (portals.ibest if keep == "best" else portals.ilast), bool(converged)


def main():
    parser = argparse.ArgumentParser(
        description='Wipe the CGYRO restart files of every evaluation of a finished PORTALS run except the result one '
                    '(see transport.options.cgyro.prune_restarts_at_convergence).')
    parser.add_argument('folders', type=str, nargs='+', help='PORTALS run folder(s).')
    parser.add_argument('--keep', choices=['best', 'last'], default='best',
                        help='Evaluation whose restarts survive: best = the result PORTALS reports (default), last = the last one run.')
    parser.add_argument('--apply', action='store_true', help='Actually delete. Without it, only report what would be freed (dry-run).')
    parser.add_argument('--force', action='store_true', help='Prune also a run that did not converge.')
    args = parser.parse_args()

    if not args.apply:
        print('\n[DRY-RUN] Nothing will be deleted. Re-run with --apply to prune.', typeMsg='w')

    total, n = 0, 0
    for folder in args.folders:
        folder = IOtools.expandPath(folder)
        index, converged = resolve_result(folder, keep=args.keep)
        if index is None:
            print(f'- {IOtools.clipstr(folder)}: PORTALS results not readable, skipping', typeMsg='w')
            continue
        if not converged and not args.force:
            print(f'- {IOtools.clipstr(folder)}: run did not converge (it may still be extended with warm starts), skipping; use --force to prune anyway', typeMsg='w')
            continue
        print(f"\n- {'Pruning' if args.apply else 'Dry-run for'} {IOtools.clipstr(folder)} (keeping Evaluation.{index}, {args.keep})")
        total += prune_cgyro_restarts(folder, index, apply=args.apply)
        n += 1

    if n > 1:
        print(f"\n=== {'Freed' if args.apply else 'Would free'} {IOtools.human_readable_size(total)} across {n} run(s)", typeMsg='i')


if __name__ == '__main__':
    main()
