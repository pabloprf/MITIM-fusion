import argparse
import tempfile
import time
from pathlib import Path
from mitim_tools.gacode_tools.utils import CGYROplot
from mitim_modules.portals.utils.PORTALSplot import locate_running_cgyro
from mitim_tools.misc_tools.LOGtools import printMsg as print

"""
Stop radii of the CGYRO evaluation a PORTALS run is waiting on, keeping what they simulated.

e.g.	mitim_kill_cgyro <portals_folder>                 # status only
	mitim_kill_cgyro <portals_folder> --rho 0.8334    # stop that radius
	mitim_kill_cgyro <portals_folder> --all           # stop every radius still running

Each stopped radius ends at its next restart write (so later iterations can warm-start from it),
is accepted as finished (mitim_budget.tag), and its fluxes are averaged over the simulated trace.
Caveat: if the trace never saturated, howard_gkav falls back to its second half (with a warning),
and the std it reports does not account for a mean that is still drifting.
"""

def main():

    parser = argparse.ArgumentParser(description="Stop running radii of a PORTALS-CGYRO evaluation, using what they simulated so far.")
    parser.add_argument("folder", type=str, help="PORTALS run folder")
    parser.add_argument("--rho", type=float, nargs="*", default=None, help="Radii to stop (as printed in the status table)")
    parser.add_argument("--all", action="store_true", help="Stop every radius still running")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    args = parser.parse_args()

    run = locate_running_cgyro(Path(args.folder).expanduser().resolve())
    if run is None:
        print("No CGYRO evaluation in flight in this PORTALS run (a bash-mode run is only found on the filesystem where its scratch lives)", typeMsg="w")
        return
    base = run["base_subfolder"]
    pairs = [(sub, rho) for sub, rho in run["pairs"] if sub == base]
    print(f"CGYRO evaluation {run['it']} running on {run['machine_settings']['machine']}:{run['folder_execution']}")

    with tempfile.TemporaryDirectory(prefix="mitim_cgyro_kill_") as tmp:
        state = CGYROplot.live_radii_state(run["machine_settings"], run["folder_execution"], pairs, tmp)
    running = [key for key, st in state.items() if not (st["exit"] or st["budget"] or st["discard"])]
    _print_status(state)

    if args.all:
        targets = running
    elif args.rho:
        targets = [_match(rho, pairs) for rho in args.rho]
        if any(t is None for t in targets):
            return
        for t in [t for t in targets if t not in running]:
            print(f"\t- rho={t[1]:.4f} already ended; nothing to stop", typeMsg="w")
        targets = [t for t in targets if t in running]
    else:
        return
    if not targets:
        print("Nothing to stop")
        return

    for key in targets:
        st = state[key]
        if st["mtime"] is not None and time.time() - st["mtime"] > 1800:
            print(f"\t- rho={key[1]:.4f}: out.cgyro.time not written for {(time.time() - st['mtime']) / 60:.0f} min; if its launch is dead the stop does nothing (a relaunch clears it)", typeMsg="w")
    if not args.yes and not print(f"Stop {', '.join(f'rho={rho:.4f}' for _, rho in targets)} at the next restart write and use the fluxes simulated so far?", typeMsg="q"):
        return
    CGYROplot.request_stop(run["machine_settings"], run["folder_execution"], targets)
    print("The watchdog picks the request up within ~20 s and stops each radius right after its next restart write (RESTART_STEP)")


def _match(rho, pairs, tol=5e-3):
    best = min(pairs, key=lambda p: abs(p[1] - rho))
    if abs(best[1] - rho) > tol:
        print(f"rho={rho} matches no running radius ({', '.join(f'{r:.4f}' for _, r in pairs)})", typeMsg="w")
        return None
    return best


def _print_status(state):
    now = time.time()
    print(f"\n\t{'rho':>8} {'t':>8} {'ends at':>9} {'last write':>11}  status")
    for (sub, rho), st in sorted(state.items(), key=lambda kv: kv[0][1]):
        t = st["t"]
        # CGYRO runs MAX_TIME a/cs past the launch start t0 (warm starts restart at 0, in-place rescues at the tag time)
        end = st["MAX_TIME"] + (st["t0"] or 0.0) if st["MAX_TIME"] is not None else None
        age = f"{(now - st['mtime']) / 60:.0f} min ago" if st["mtime"] is not None else "-"
        if st["exit"]:
            status = "finished (EXIT)"
        elif st["budget"]:
            status = "stopped (mitim_budget.tag)"
        elif st["discard"]:
            status = "discarded"
        else:
            status = "running" + (", stop requested" if st["stop_requested"] else "")
        print(f"\t{rho:8.4f} {t if t is not None else float('nan'):8.1f} {end if end is not None else float('nan'):9.1f} {age:>11}  {status}")
    print("")


if __name__ == "__main__":
    main()
