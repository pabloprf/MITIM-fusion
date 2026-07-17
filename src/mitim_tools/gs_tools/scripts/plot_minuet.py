import argparse
from pathlib import Path
from IPython import embed
from mitim_tools.misc_tools import IOtools


def main():
    """Read one (or several) MINUET .minuet save files and open the tabbed
    notebook. Mirrors the other mitim_plot_* scripts.

    Several files are appended to ONE window as differently-colored tab-sets
    (using MINUET's multi-case notebook support), each prefixed by the file
    stem so the tabs are unambiguous:

        mitim_plot_minuet run.minuet
        mitim_plot_minuet a.minuet b.minuet          # two cases, one window
        mitim_plot_minuet run.minuet --save          # headless, save figures
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="*",
                        help="One or more MINUET .minuet save files.")
    parser.add_argument("--title", type=str, required=False,
                        default="minuet discharge",
                        help="Notebook window title.")
    parser.add_argument("--save", type=str, nargs="?", const=IOtools.SAVE_FOLDER_AUTO_SENTINEL, required=False, default=None,
                        help=f"Folder to save the figures. If flag given without a value, defaults to '<dir of first file>/{IOtools.SAVE_FOLDER_DEFAULT_SUBDIR}'. Implies --noshow.")
    parser.add_argument("--dpi", type=int, required=False, default=120,
                        help="DPI to save the figures.")
    parser.add_argument("--noshow", required=False, default=False, action="store_true",
                        help="If set, it will not show the figures on screen.")
    args = parser.parse_args()

    # --save implies --noshow (headless save; no point re-rendering on screen).
    if args.save is not None:
        args.noshow = True

    if not args.files:
        parser.error("need at least one .minuet file")

    # lazy import: MINUET is an optional (git) dependency, so importing it
    # only inside main() keeps the entry point registration robust when it
    # is not installed
    from minuet import minuet

    files = [IOtools.expandPath(file) for file in args.files]
    folder_save = IOtools.resolve_save_folder(args.save, Path(files[0]).parent)
    noshow = args.noshow

    multi = len(files) > 1
    fn = None
    for i, file in enumerate(files):
        m = minuet.load(file)
        last = i == len(files) - 1
        fn = m.notebook(
            title=args.title,
            fn=fn,                                   # append to the same window
            tab_color=i if multi else None,          # one color per case
            label_prefix=f"{Path(file).stem}: " if multi else "",
            show=last and not noshow)                # show() blocks: last only

    if folder_save is not None:
        if not folder_save.exists():
            folder_save.mkdir(parents=True)
        fn.save(folder_save, dpi=args.dpi)

    if not noshow:                    # a headless (--save) run must not block
        embed()


if __name__ == "__main__":
    main()
