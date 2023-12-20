import argparse, socket
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.transp_tools import CDFtools

from mitim_tools.misc_tools.CONFIGread import read_verbose_level

verbose_level = read_verbose_level()

parser = argparse.ArgumentParser()
parser.add_argument("--files", required=True, type=str, nargs="*")
args = parser.parse_args()

expl = args.files

cdfs = []
readFBM = True
readTGLF = True
readTORIC = True
readGFILE = True
readStructures = True
ZerothTime = False
printCheckPoints = False
readGEQDSK = True

plt.ioff()

for i in expl:
    if ":" in i:
        cdfs.append(
            CDFtools.CDFreactor(
                i.split(":")[1],
                ssh=i.split(":")[0],
                printCheckPoints=printCheckPoints,
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
            CDFtools.CDFreactor(
                i,
                printCheckPoints=printCheckPoints,
                readFBM=readFBM,
                readTGLF=readTGLF,
                readTORIC=readTORIC,
                readGFILE=readGFILE,
                readGEQDSK=readGEQDSK,
                readStructures=readStructures,
                ZerothTime=ZerothTime,
            )
        )
