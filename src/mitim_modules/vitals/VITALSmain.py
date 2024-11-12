import copy
import os
import pickle
import torch
import numpy as np
from mitim_tools.gacode_tools import TGLFtools
from mitim_tools.opt_tools import STRATEGYtools
from IPython import embed

def default_namelist(optimization_options):
    """
    This is to be used after reading the namelist, so self.optimization_options should be completed with main defaults.
    """

    optimization_options["initial_training"] = 8
    optimization_options["BO_iterations"] = 20
    optimization_options["newPoints"] = 4
    optimization_options["parallel_evaluations"] = (
        1  # each TGLF is run with 4 cores, so 16 total cores consumed with this default
    )
    optimization_options["surrogateOptions"]["TypeMean"] = 2
    optimization_options["StrategyOptions"]["AllowedExcursions"] = [0.1, 0.1]
    optimization_options["StrategyOptions"]["HitBoundsIncrease"] = [1.1, 1.1]
    optimization_options["StrategyOptions"]["TURBO"] = True
    optimization_options["StrategyOptions"]["TURBO_addPoints"] = 16

    # Acquisition
    optimization_options["optimizers"] = "root_5-botorch-ga"
    optimization_options["acquisition_type"] = "posterior_mean"

    return optimization_options


class vitals(STRATEGYtools.opt_evaluator):
    def __init__(self, folder, namelist=None):
        print(
            "\n-----------------------------------------------------------------------------------------"
        )
        print("\t\t\t VITALS class module")
        print(
            "-----------------------------------------------------------------------------------------\n"
        )

        # Store folder, namelist. Read namelist
        super().__init__(
            folder,
            namelist=namelist,
            default_namelist_function=default_namelist if (namelist is None) else None,
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Default (please change to your desire after instancing the object)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.TGLFparameters = {"TGLFsettings": 2, "extraOptions": {}}

        self.VITALSparameters = {
            "rel_error": 0.02,  # Standard deviation (relative to value)
            "UsingMultipliers": True,
            "launchSlurm": True,
        }

    def prep(
        self,
        tglf_class_file,
        rho,
        ofs,
        dvs,
        dvs_min,
        dvs_max,
        dvs_base=None,
        classLoaded=False,
        grabErrors=True,
    ):
        if not classLoaded:
            with open(tglf_class_file, "rb") as f:
                tglf_read = pickle.load(f)
            self.tglf = TGLFtools.TGLF(alreadyRun=tglf_read)
            self.tglf.FolderGACODE, self.tglf.rhos = self.folder, [rho]

        else:
            self.tglf = tglf_class_file

        norm = self.tglf.NormalizationSets["EXP"]

        # ----------------------------------------------------------------------------
        # Grab fluxes
        # ----------------------------------------------------------------------------

        x, Qe, Qi = np.array(norm["rho"]), norm["exp_Qe"], norm["exp_Qi"]
        Qe_exp, Qi_exp = Qe[np.argmin(np.abs(x - rho))], Qi[np.argmin(np.abs(x - rho))]

        if not grabErrors:
            Qe_std = Qi_std = None
        else:
            Qe_rho = np.array(norm["exp_Qe_rho"])
            Qe_std = norm["exp_Qe_error"][np.argmin(np.abs(Qe_rho - rho))]
            Qi_rho = np.array(norm["exp_Qi_rho"])
            Qi_std = norm["exp_Qi_error"][np.argmin(np.abs(Qi_rho - rho))]

        # ----------------------------------------------------------------------------
        # Grab fluctuations
        # ----------------------------------------------------------------------------

        if not "TeFluct" in ofs:
            fluct_exp = fluct_std = None
        else:
            fluct_rho = np.array(norm["exp_TeFluct_rho"])
            fluct = norm["exp_TeFluct"]
            fluct_exp = fluct[np.argmin(np.abs(fluct_rho - rho))]

            if not grabErrors:
                fluct_std = None
            else:
                fluct_std = norm["exp_TeFluct_error"][
                    np.argmin(np.abs(fluct_rho - rho))
                ]

        # ----------------------------------------------------------------------------
        # Grab phase
        # ----------------------------------------------------------------------------

        if not "neTe" in ofs:
            neTe_exp = neTe_std = None
        else:
            fluct_rho = np.array(norm["exp_neTe_rho"])
            fluct = norm["exp_neTe"]
            neTe_exp = fluct[np.argmin(np.abs(fluct_rho - rho))]

            if not grabErrors:
                neTe_std = None
            else:
                neTe_std = norm["exp_neTe_error"][np.argmin(np.abs(fluct_rho - rho))]

        # ----------------------------------------------------------------------------
        # Calofs depend on the definition of the metric (top of the file)
        # ----------------------------------------------------------------------------

        self.optimization_options["ofs"] = ofs
        o1 = copy.deepcopy(self.optimization_options["ofs"])

        self.name_objectives = []
        for iof in o1:
            self.name_objectives.append(iof + "_devstd")
            self.optimization_options["ofs"].append(iof + "_exp")

        self.optimization_options["dvs"] = dvs
        self.optimization_options["dvs_min"] = dvs_min
        self.optimization_options["dvs_max"] = dvs_max

        if dvs_base is None:
            self.optimization_options["dvs_base"] = [1.0 for i in dvs]
        else:
            self.optimization_options["dvs_base"] = dvs_base

        # ----------------------------------------------------------------------------
        #
        # ----------------------------------------------------------------------------

        self.VITALSparameters["experimentalVals"] = {
            "Qe": Qe_exp,
            "Qi": Qi_exp,
            "TeFluct": fluct_exp,
            "neTe": neTe_exp,
        }
        self.VITALSparameters["std_deviation"] = {
            "Qe": Qe_std,
            "Qi": Qi_std,
            "TeFluct": fluct_std,
            "neTe": neTe_std,
        }

    def run(self, paramsfile, resultsfile):
        # Read stuff
        FolderEvaluation, _, dictDVs, dictOFs = self.read(paramsfile, resultsfile)

        # Modify param
        variables = []
        variation = {}
        for variable in dictDVs:
            vart = dictDVs[variable]["value"]
            variation[variable] = vart
            variables.append(variable)

        # RUN
        tglf = copy.deepcopy(self.tglf)
        launchSlurm = self.VITALSparameters["launchSlurm"]
        runTGLF(
            self,
            tglf,
            FolderEvaluation,
            variation,
            label="tglf1",
            launchSlurm=launchSlurm,
            evNum=f'{paramsfile}'.split(".")[-1],
        )

        # Evaluate

        for iquant in dictOFs:
            if "_exp" not in iquant:
                if iquant == "Qe":
                    value = tglf.results["tglf1"]["TGLFout"][0].Qe_unn
                elif iquant == "Qi":
                    value = tglf.results["tglf1"]["TGLFout"][0].Qi_unn
                elif iquant == "TeFluct":
                    value = tglf.results["tglf1"]["TGLFout"][
                        0
                    ].AmplitudeSpectrum_Te_level
                elif iquant == "neFluct":
                    value = tglf.results["tglf1"]["TGLFout"][
                        0
                    ].AmplitudeSpectrum_ne_level
                elif iquant == "neTe":
                    value = tglf.results["tglf1"]["TGLFout"][0].neTeSpectrum_level

                dictOFs[iquant]["value"] = value
                dictOFs[iquant]["error"] = np.abs(
                    dictOFs[iquant]["value"] * self.VITALSparameters["rel_error"]
                )
            else:
                dictOFs[iquant]["value"] = self.VITALSparameters["experimentalVals"][
                    iquant[:-4]
                ]
                dictOFs[iquant]["error"] = self.VITALSparameters["std_deviation"][
                    iquant[:-4]
                ]

        # Write stuff
        self.write(dictOFs, resultsfile)

    def analyze_results(
        self, plotYN=True, fn=None, cold_start=False, storeResults=True, analysis_level=2
    ):
        analyze_results(
            self,
            plotYN=plotYN,
            fn=fn,
            cold_start=cold_start,
            storeResults=storeResults,
            analysis_level=analysis_level,
        )

    def scalarized_objective(self, Y):
        """
        Notes
        -----
                - Y is the multi-output evaluation of the model in the shape of (dim1...N, num_ofs), i.e. this function should not care
                  about number of dimensions
        """

        ofs_ordered_names = np.array(self.optimization_options["ofs"])

        of, cal, res = torch.Tensor().to(Y), torch.Tensor().to(Y), torch.Tensor().to(Y)
        for iquant in ofs_ordered_names:
            if "_exp" not in iquant:
                of0 = Y[..., ofs_ordered_names == iquant]
                cal0 = Y[..., ofs_ordered_names == iquant + "_exp"]

                # In VITALS, calibration is zero and objective is the absolute difference divided by stdev
                of_pass = (of0 - cal0).abs() / self.VITALSparameters["std_deviation"][
                    iquant
                ]
                of = torch.cat((of, of_pass), dim=-1)
                cal = torch.cat((cal, cal0 * 0.0), dim=-1)

        # Residual is defined as the negative (bc it's maximization) normalized (1/N) norm of radial & channel residuals -> L1 or L2
        res = -1 / of.shape[-1] * torch.norm(of, p=2, dim=-1)

        # res must have shape (dim1...N)
        return of, cal, res


def runTGLF(
    self,
    tglf,
    FolderEvaluation,
    variation,
    label="tglf1",
    folder_label=None,
    launchSlurm=True,
    evNum=1,
    cold_start=True,
):
    # Change folder
    initializationFolder = copy.deepcopy(tglf.FolderGACODE)
    tglf.FolderGACODE = FolderEvaluation

    numSim = self.folder.name

    variation = TGLFtools.completeVariation(variation, tglf.inputsTGLF[tglf.rhos[0]])

    extraOptions = self.TGLFparameters["extraOptions"]
    multipliers = {}

    # Run
    if self.VITALSparameters["UsingMultipliers"]:
        multipliers = variation
    else:
        extraOptions.update(variation)

    if folder_label is None:
        folder_label = label

    tglf.run(
        subFolderTGLF=f"{folder_label}",
        cold_start=cold_start,
        TGLFsettings=self.TGLFparameters["TGLFsettings"],
        forceIfcold_start=True,
        extraOptions=extraOptions,
        multipliers=multipliers,
        extra_name=f"{numSim}_{evNum}",
        launchSlurm=launchSlurm,
    )
    tglf.read(label=label)


def analyze_results(
    self, plotYN=True, fn=None, cold_start=False, storeResults=True, analysis_level=2, **kwargs
):
    # ----------------------------------------------------------------------------------------------------------------
    # Interpret stuff
    # ----------------------------------------------------------------------------------------------------------------

    (
        multipliers_original,
        multipliers_best,
        self_complete,
    ) = self.analyze_optimization_results()

    if analysis_level > 1:
        # ----------------------------------------------------------------------------------------------------------------
        # Running cases: Original and Best
        # ----------------------------------------------------------------------------------------------------------------

        self.tglf_final = self_complete.tglf
        FolderEvaluation = self.folder / "Outputs" / "final_analysis"
        FolderEvaluation.mkdir(parents=True, exist_ok=True)

        launchSlurm = True
        print("\t- Running original case")
        runTGLF(
            self_complete,
            self.tglf_final,
            FolderEvaluation,
            multipliers_original,
            label="Original",
            launchSlurm=launchSlurm,
            evNum=0,
            cold_start=cold_start,
        )
        print(f"\t- Running best case #{self.res.best_absolute_index}")
        runTGLF(
            self_complete,
            self.tglf_final,
            FolderEvaluation,
            multipliers_best,
            label="VITALS",
            folder_label=f"VITALS_{self.res.best_absolute_index}",
            launchSlurm=launchSlurm,
            evNum=self.res.best_absolute_index,
            cold_start=cold_start,
        )

        # ----------------------------------------------------------------------------------------------------------------
        # Storing and Plotting
        # ----------------------------------------------------------------------------------------------------------------

        # Plot
        if plotYN:
            self.tglf_final.plot(
                labels=["Original", "VITALS"], fn=fn, extratitle="TGLF- "
            )

        if storeResults:
            # Save tglf file
            file = file = self.folder / "Outputs" / "final_analysis" / "tglf.pkl"
            self.tglf_final.save_pkl(file)

            # Store dictionary of results (unfortunately so far the dictionary is created at plotting... I have to improve this)
            # if plotYN:
            # 	file ='{0}/Outputs/final_analysis/analysis_results.pkl'.format(self.folder)
            # 	with open(file,'wb') as handle:	pickle.dump(self.tglf_final.simple_dict,handle)
            # 	print('> Dictionary with main results written...',file)
