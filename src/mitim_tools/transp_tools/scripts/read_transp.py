import argparse
from pathlib import Path
from IPython import embed
from mitim_tools.transp_tools import CDFtools

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="*")
    parser.add_argument(
        "--full", "-f", required=False, default=True, action="store_true"  # Full read
    )
    parser.add_argument(
        "--read", "-r", required=False, default=False, action="store_true"  # Only read
    )
    parser.add_argument("--save", type=str, nargs="?", const="figs", required=False, default=None,
                        help="Folder to save the figures. If flag given without a value, defaults to 'figs'. Implies --noshow.")
    parser.add_argument("--dpi", type=int, required=False, default=120,
                        help="DPI to save the figures.")
    parser.add_argument("--noshow", required=False, default=False, action="store_true",
                        help="If set, it will not show the figures on screen.")
    args = parser.parse_args()

    # --save implies --noshow (headless save; no point re-rendering on screen).
    if args.save is not None:
        args.noshow = True

    expl = args.files
    plotYN = not args.read
    fullYN = args.full
    folder_save = Path(args.save) if args.save is not None else None
    noshow = args.noshow
    dpi_fig = args.dpi

    cdfs = []

    ZerothTime = False
    if fullYN:
        readFBM = True
        readTGLF = True
        readTORIC = True
        readGFILE = True
        readStructures = True
        readGEQDSK = True
    else:
        readFBM = False
        readTGLF = False
        readTORIC = False
        readGFILE = False
        readStructures = False
        readGEQDSK = False


    for i in expl:
        cdfs.append(
            CDFtools.transp_output(
                i,
                readFBM=readFBM,
                readTGLF=readTGLF,
                readTORIC=readTORIC,
                readGFILE=readGFILE,
                readGEQDSK=readGEQDSK,
                readStructures=readStructures,
                ZerothTime=ZerothTime,
            )
        )

    if plotYN:
        from mitim_tools.misc_tools.GUItools import FigureNotebook

        fn = FigureNotebook("TRANSP run", show=not noshow)
        for i in range(len(cdfs)):
            cdfs[i].plot(fn=fn, tab_color=i)

        if not noshow:
            fn.show()

        if folder_save is not None:
            if not folder_save.exists():
                folder_save.mkdir(parents=True)
            fn.save(folder_save, dpi=dpi_fig)

    embed()

if __name__ == "__main__":
    main()
