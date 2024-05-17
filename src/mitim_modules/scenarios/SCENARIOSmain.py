import torch
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.im_tools import IMtools


class opt_class(STRATEGYtools.opt_evaluator):
    def __init__(self, folder, IMnamelist):
        # Store folder, namelist. Read namelist
        super().__init__(
            folder, namelist=IOtools.expandPath("$MITIM_PATH/templates/main.namelist")
        )
        # ----------------------------------------

        # Define dimension
        self.name_objectives = ["Q"]

        # Define setup
        self.IMnamelist = IMnamelist  # f'{GeneralFolder}/im.namelist'

    def run(self, paramsfile, resultsfile):
        # Read stuff
        FolderEvaluation, numEval, dictDVs, dictOFs = self.read(paramsfile, resultsfile)

        # Operations

        # MISSING CHANGE INPUTS

        im = IMtools.runIMworkflow(self.IMnamelist, FolderEvaluation, numEval)

        # ------------------------------------------------------------------------------------
        # Evaluate metrics
        # ------------------------------------------------------------------------------------

        dictOFs = evaluateMetrics(cdf, dictOFs)

        # Write stuff
        self.write(dictOFs, resultsfile)

    def scalarized_objective(self, Y):
        """
        TO DO
        """

        # ofs_ordered_names = np.array(self.Optim['ofs'])

        # of 	= Y[:,ofs_ordered_names == 'z'].unsqueeze(1)
        # cal = Y[:,ofs_ordered_names == 'zval'].unsqueeze(1)
        # res = -(of-cal).abs().mean(axis=1)

        return of, cal, res


def evaluateMetrics(Reactor, dictOFsCVs):
    for iOF in dictOFsCVs:
        print(f'>> "{iOF}" has been requested as OF')

        # Adding _pen to the name of the OFs adds a multiplication by the penalties
        if "_pen" in iOF:
            print("\t- Penalization requested")
            iOFn, multipl = iOF.replace("_pen", ""), Reactor.reactor["penalties"]
        elif "_hmode" in iOF:
            print("\t- LH-Penalization requested")
            iOFn, multipl = iOF.replace("_hmode", ""), Reactor.reactor["LHdiff"]
        else:
            iOFn, multipl = iOF, 1.0

        if "_neg" in iOFn:
            print("\t- Negative value requested")
            iOFn, multipl = iOFn.replace("_neg", ""), -multipl

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluate value from mitim_toolsNML.Reactor.reactor
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        dictOFsCVs[iOF]["value"] = Reactor.reactor[iOFn] * multipl

        print(f"\t>> Value passed to optimizer:\t{dictOFsCVs[iOF]['value']}")

    return dictOFsCVs
