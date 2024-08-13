import datetime, time, os
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.transp_tools.utils import NMLtools


def populateFromMDS(self, runidMDS):
    """
    This routine grabs NML and UFILES from MDS+ and puts them in the right folder
    """

    if self.tok == "CMOD":
        from mitim_tools.experiment_tools.CMODtools import getTRANSP_MDS
    else:
        raise Exception("Tokamak MDS+ not implemented")

    # shotNumber = self.runid[-3:]
    getTRANSP_MDS(
        runidMDS,
        self.runid,
        folderWork=self.FolderTRANSP,
        toric_mpi=self.mpisettings["toricmpi"],
        shotnumber=self.shotnumberReal,
    )


def defaultbasedMDS(self, outtims=[], PRFmodified=False):
    """
    This routine creates a default nml for the given tokamak, and modifies it according to an
    existing nml that, e.g. has come from MDS+

    PRFmodified = True doesn't care about original model settings, I use mine
    """

    if self.tok == "CMOD":
        from mitim_tools.experiment_tools.CMODtools import updateTRANSPfromNML
    else:
        raise Exception("Tokamak MDS+ not implemented")

    os.system("cp {0} {0}_old".format(self.nml_file))
    TRANSPnamelist_dict = updateTRANSPfromNML(
        self.nml_file + "_old",
        self.nml_file,
        self.FolderTRANSP,
        PRFmodified=PRFmodified,
    )

    # Write NML

    ntoric = int(self.mpisettings["toricmpi"] > 1)
    nbi = int(self.mpisettings["trmpi"] > 1)
    nptr = int(self.mpisettings["ptrmpi"] > 1)

    # Write generic namelist for this tokamak
    nml = NMLtools.default_nml(self.shotnumber, self.tok, pservers=[nbi, ntoric, nptr])
    nml.write(BaseFile=self.nml_file)
    nml.appendNMLs()

    # To allow the option of giving negative outtimes to reflect from the end
    outtims_new = []
    ftime = IOtools.findValue(self.nml_file, "ftime", "=")
    for i in outtims:
        if i < 0.0:
            outtims_new.append(ftime - i)
        else:
            outtims_new.append(i)

    # Modify according to TRANSPnamelist_dict
    NMLtools.changeNamelist(
        self.nml_file,
        self.shotnumber,
        TRANSPnamelist_dict,
        self.FolderTRANSP,
        outtims=outtims_new,
    )
