from IPython import embed
from mitim_tools.misc_tools import CONFIGread

"""
Overall routine to handle runs depending on being globus or singularity
"""


def TRANSP(FolderTRANSP, tokamak):
    s = CONFIGread.load_settings()

    if s["preferences"]["transp"] == "globus":
        from mitim_tools.transp_tools.src.TRANSPglobus import (
            TRANSPglobus as TRANSPclass,
        )
    else:
        from mitim_tools.transp_tools.src.TRANSPsingularity import (
            TRANSPsingularity as TRANSPclass,
        )

    return TRANSPclass(FolderTRANSP, tokamak)
