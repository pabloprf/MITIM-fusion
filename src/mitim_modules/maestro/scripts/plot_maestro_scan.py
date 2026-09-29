"""
Scan-level interpretation of a folder of MAESTRO case_* runs (see MAESTROscan):
seed-spread violins of performance scalars, per-beat evolution, cumulative beat
timing, and a compiled per-case PDF report, all written to <folder>/interpretation/.

e.g.
    mitim_plot_maestro_scan run_folder/ --x neped --series nsep
    mitim_plot_maestro_scan run_folder/ --x fG --series nsep \
        --benchmark ref_run/ --benchmark_x 0.766 --benchmark_series 0.40 --benchmark_label "V3A"
"""

import argparse

import matplotlib
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.maestro.utils.MAESTROscan import maestro_scan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Scan folder containing the case_* runs")
    parser.add_argument("--x", type=str, required=True, help="Scan parameter for the x-axis (e.g. neped, fG)")
    parser.add_argument("--series", type=str, default=None, help="Scan parameter for the color series (e.g. nsep)")
    parser.add_argument("--tag", type=str, default=None, help="Restrict to one case tag (init method / machine)")
    parser.add_argument("--output", type=str, default=None, help="Output folder (default <folder>/interpretation)")
    parser.add_argument("--benchmark", type=str, default=None, help="Reference MAESTRO run folder to overlay")
    parser.add_argument("--benchmark_x", type=float, default=None, help="x value at which to place the benchmark")
    parser.add_argument("--benchmark_series", type=float, default=None, help="Series value coloring the benchmark")
    parser.add_argument("--benchmark_label", type=str, default="benchmark run")
    parser.add_argument("--no_report", action="store_true", help="Skip the per-case PDF report")
    args = parser.parse_args()

    matplotlib.use("Agg")

    scan = maestro_scan(IOtools.expandPath(args.folder))
    if not scan.cases:
        return
    if args.output is not None:
        scan.set_output_folder(args.output)
    print(f"* {len(scan.cases)} cases, tags {scan.tags}, scan parameters "
          + ", ".join(f"{p}={scan.values[p]}" for p in scan.scan_params))

    overlays, overlays_swapped = [], []
    if args.benchmark is not None:
        overlays.append(scan.benchmark_overlay(
            args.benchmark, x_value=args.benchmark_x, series_value=args.benchmark_series,
            label=args.benchmark_label))
        if args.series is not None:
            overlays_swapped.append(scan.benchmark_overlay(
                args.benchmark, x_value=args.benchmark_series, series_value=args.benchmark_x,
                label=args.benchmark_label))

    scan.plot_performance(x=args.x, series=args.series, tag=args.tag, overlays=overlays)
    if args.series is not None and len(scan.values.get(args.series, [])) > 1:
        scan.plot_performance(x=args.series, series=args.x, tag=args.tag, overlays=overlays_swapped)
    scan.plot_beat_evolution(color_by=args.x, tag=args.tag)
    scan.plot_beat_timing(color_by=args.x, panel_by=[args.series] if args.series else [],
                          tag=args.tag,
                          benchmark_timing=(maestro_scan.cumulative_timing(args.benchmark)
                                            if args.benchmark else None))
    if not args.no_report:
        scan.compile_report()


if __name__ == "__main__":
    main()
