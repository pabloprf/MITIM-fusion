import argparse
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
    args = parser.parse_args()

    expl = args.files
    plotYN = not args.read
    fullYN = args.full

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
        if ":" in i:
            cdfs.append(
                CDFtools.transp_output(
                    i.split(":")[1],
                    ssh=i.split(":")[0],
                    readFBM=readFBM,
                    readTGLF=readTGLF,
                    readTORIC=readTORIC,
                    readGFILE=readGFILE,
                    readStructures=readStructures,
                    readGEQDSK=readGEQDSK,
                    ZerothTime=ZerothTime,
                )
            )
        else:
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

        fn = FigureNotebook("TRANSP run")
        for i in range(len(cdfs)):
            cdfs[i].plot(fn=fn, counter=i)

        fn.show()

    embed()

if __name__ == "__main__":
    main()
