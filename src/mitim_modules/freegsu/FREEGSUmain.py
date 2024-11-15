import copy
import pickle
import torch
import numpy as np
from collections import OrderedDict
from mitim_modules.freegsu import FREEGSUtools
from mitim_modules.freegsu.utils import FREEGSUplotting
from mitim_tools.misc_tools import IOtools, GUItools
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.opt_tools.utils import BOgraphics
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed


def default_namelist(optimization_options):
    """
    This is to be used after reading the namelist, so self.optimization_options should be completed with main defaults.
    """

    optimization_options["initial_training"] = 32
    optimization_options["BO_iterations"] = 100
    optimization_options["parallel_evaluations"] = 16
    optimization_options['stopping_criteria_parameters']["maximum_value"] = -1e-2  # This is 0.1mm, enough accuracy
    optimization_options["points_per_step"] = 16  # I found this better
    optimization_options["surrogateOptions"]["FixedNoise"] = False
    optimization_options["StrategyOptions"]["TURBO"] = True

    # Acquisition
    optimization_options["optimizers"] = "root_5-botorch-ga"
    optimization_options["acquisition_type"] = "posterior_mean"

    return optimization_options


class freegsu(STRATEGYtools.opt_evaluator):
    def __init__(self, folder, namelist=None, function_parameters={}):
        print(
            "\n-----------------------------------------------------------------------------------------"
        )
        print("\t\t\t FREEGSU class module")
        print(
            "-----------------------------------------------------------------------------------------\n"
        )

        self.function_parameters = function_parameters

        # Store folder, namelist. Read namelist
        super().__init__(
            folder,
            namelist=namelist,
            default_namelist_function=default_namelist if (namelist is None) else None,
        )

    def prep(
        self,
        ofs_dict,
        setCoils,
        setCoils_lower=None,
        dvs_base=None,
        rangeVar=None,
        is_rangeVar_kA=True,
    ):
        self.ofs_dict = ofs_dict

        # Affect funcitons params
        if "xpdRsep_1" not in self.ofs_dict:
            self.function_parameters["optionsFREEGS"]["symmetricX"] = True
            self.function_parameters["CoilCurrents_lower"] = None
        else:
            self.function_parameters["CoilCurrents"]["vs1"] = None
            self.function_parameters["optionsFREEGS"]["symmetricX"] = False
            self.function_parameters["optionsFREEGS"]["outMidplane_matching"] = [
                0,
                self.ofs_dict["xpdRsep_1"] * 1e-2,
            ]

            if self.function_parameters["CoilCurrents_lower"] is None:
                self.function_parameters["CoilCurrents_lower"] = copy.deepcopy(
                    self.function_parameters["CoilCurrents"]
                )

        # Limitys

        (
            maxVar_I,
            maxVar_V,
            minVar_I,
            minVar_V,
        ) = FREEGSUtools.readCoilCalcsMatrices_Supplies(
            IOtools.expandPath(self.function_parameters["params"]["RequirementsFile"])
        )

        coilLimits_kA = {}
        for i in setCoils:
            coilLimits_kA[i] = [minVar_I[i], maxVar_I[i]]





        # From SPARC_PATH in PYTHONPATH
        try:
            from FREEGS_SPARC import GSsparc_coils
        except ImportError as e:
            raise Exception(
                "[mitim] The FREEGS_SPARC module is not available. Please ensure it is installed and accessible."
            )


        sparc_coils = GSsparc_coils.SPARCcoils(
            None, coilsVersion=self.function_parameters["optionsFREEGS"]["coilsVersion"]
        )

        # Apply rangeVar on kA ---------------------------------------------------------------------------------------------
        if (is_rangeVar_kA) and (rangeVar is not None):
            coilLimits_kA = apply_rangeVar(
                setCoils,
                coilLimits_kA,
                self.function_parameters,
                rangeVar,
                turns_real=sparc_coils.turns_real,
            )
        # ------------------------------------------------------------------------------------------------------------------

        coilLimits = {}
        for i in setCoils:
            coilLimits[i] = [
                coilLimits_kA[i][0] * sparc_coils.turns_real[i + "u"] * 1e-3,
                coilLimits_kA[i][1] * sparc_coils.turns_real[i + "u"] * 1e-3,
            ]

        # Apply rangeVar on MAt --------------------------------------------------------------------------------------------
        if (not is_rangeVar_kA) and (rangeVar is not None):
            coilLimits = apply_rangeVar(
                setCoils, coilLimits, self.function_parameters, rangeVar
            )
        # ------------------------------------------------------------------------------------------------------------------

        (
            self.optimization_options["dvs"],
            self.optimization_options["dvs_base"],
            self.optimization_options["dvs_min"],
            self.optimization_options["dvs_max"],
        ) = ([], [], [], [])
        for i in setCoils:
            self.optimization_options["dvs"].append(i)
            self.optimization_options["dvs_base"].append(self.function_parameters["CoilCurrents"][i])
            self.optimization_options["dvs_min"].append(coilLimits[i][0])
            self.optimization_options["dvs_max"].append(coilLimits[i][1])

        if self.function_parameters["CoilCurrents_lower"] is not None:
            if setCoils_lower is None:
                setCoils_lower = setCoils
            for i in setCoils_lower:
                self.optimization_options["dvs"].append(i + "_l")
                self.optimization_options["dvs_base"].append(
                    self.function_parameters["CoilCurrents_lower"][i]
                )
                self.optimization_options["dvs_min"].append(coilLimits[i][0])
                self.optimization_options["dvs_max"].append(coilLimits[i][1])

        if dvs_base is not None:
            self.optimization_options["dvs_base"] = dvs_base

        self.optimization_options["ofs"] = []
        self.name_objectives = []
        for i in self.ofs_dict:
            self.optimization_options["ofs"].append(i)
            self.optimization_options["ofs"].append(i + "_goal")

            self.name_objectives.append(i + "_dev")

    def run(self, paramsfile, resultsfile):
        # ------------
        # Read stuff
        # ------------

        FolderEvaluation, numEval, dictDVs, dictOFs = self.read(paramsfile, resultsfile)

        metrics_opt = {}
        for i in dictOFs:
            metrics_opt[i] = np.inf

        _, _, metrics_opt = runFreeGS(self, dictDVs)
        # except: pass

        # Write stuff
        for i in dictOFs:
            if "_goal" not in i:
                dictOFs[i]["value"] = metrics_opt[i]
            else:
                dictOFs[i]["value"] = self.ofs_dict[i[:-5]]

        # ------------
        # Write stuff
        # ------------

        self.write(dictOFs, resultsfile)

    def scalarized_objective(self, Y):
        """
        Metric is the max deviation in standard deviations
        """

        ofs_ordered_names = np.array(self.optimization_options["ofs"])

        of, cal, res = torch.Tensor().to(Y), torch.Tensor().to(Y), torch.Tensor().to(Y)
        for iquant in ofs_ordered_names:
            if "_goal" not in iquant:
                of0 = Y[..., ofs_ordered_names == iquant]
                cal0 = Y[..., ofs_ordered_names == iquant + "_goal"]

                of = torch.cat((of, of0), dim=-1)
                cal = torch.cat((cal, cal0), dim=-1)

        source = of - cal

        # Residual is defined as the negative (bc it's maximization) normalized (1/N) norm of radial & channel residuals -> L1
        res = -1 / source.shape[-1] * torch.norm(source, p=1, dim=-1)

        # res must have shape (dim1...N)
        return of, cal, res

    def analyze_results(
        self, plotYN=True, fn=None, storeResults=True, analysis_level=2
    ):
        analyze_results(
            self,
            plotYN=plotYN,
            fn=fn,
            storeResults=storeResults,
            analysis_level=analysis_level,
        )


def runFreeGS(self, dictDVs, plot=False, figs=None, onlyPrepare=False, debug=False):
    out = FREEGSUtools.evaluator(
        dictDVs,
        CoilCurrents=self.function_parameters["CoilCurrents"],
        CoilCurrents_lower=(
            self.function_parameters["CoilCurrents_lower"]
            if "CoilCurrents_lower" in self.function_parameters
            else None
        ),
        Constraints=self.function_parameters["Constraints"],
        optionsFREEGS=self.function_parameters["optionsFREEGS"],
        plot=plot,
        figs=figs,
        debug=debug,
        ProblemExtras=(
            self.function_parameters["params"]
            if "params" in self.function_parameters
            else None
        ),
        onlyPrepare=onlyPrepare,
    )

    return out  # prfs,metrics,metrics_opt


def analyze_results(
    self,
    plotYN=True,
    fn=None,
    storeResults=True,
    onlyBest=False,
    analysis_level=2,
    onlyPrepare=False,
    **kwargs
):
    # ----------------------------------------------------------------------------------------------------------------
    # Interpret stuff
    # ----------------------------------------------------------------------------------------------------------------

    (
        variations_original,
        variations_best,
        self_complete,
    ) = self.analyze_optimization_results()

    self.function_parameters = self_complete.function_parameters

    dictDVs = {}
    for i in variations_best:
        if onlyBest:
            dictDVs[i] = {"value": [variations_best[i]]}
        else:
            dictDVs[i] = {"value": [variations_original[i], variations_best[i]]}

    if onlyBest:
        self.function_parameters["params"]["times"] = [0]
    else:
        print(
            "\n*** PRF WARNING: Remember to check the timing for voltages!!\n",
            typeMsg="w",
        )
        self.function_parameters["params"]["times"] = [0, 0.3]

    # ----------------------------------------------------------------------------------------------------------------
    # Re-run
    # ----------------------------------------------------------------------------------------------------------------

    if plotYN:
        fig1 = fn.add_figure(label="FreeGSU - Eq. & Coils")
        fig2 = fn.add_figure(label="FreeGSU - Metrics")
        fig3 = fn.add_figure(label="FreeGSU - Solution")
        figP = fn.add_figure(label="FreeGSU - Profiles")
        figa = fn.add_figure(label="FreeGSU - Powers")
        figb = fn.add_figure(label="FreeGSU - Maxima")
        figMach = fn.add_figure(label="FreeGSU - Machine")
        figRes = fn.add_figure(label="FreeGSU - Summary")

        figs = [fig1, fig2, fig3, figP, figa, figb, figMach, figRes]
    else:
        figs = None

    # Higher resolution
    self.function_parameters["optionsFREEGS"]["n"] = 129

    if onlyPrepare:
        return dictDVs
    else:
        prfs, metrics, metrics_opt = runFreeGS(self, dictDVs, plot=True, figs=figs)

        FolderEvaluation = self.folder / "Outputs" / "final_analysis/"
        if storeResults:
            gs = FREEGSUplotting.writeResults(
                FolderEvaluation,
                prfs,
                metrics,
                self.function_parameters,
                namePkl="results",
                plotGs=True,
                params=self.function_parameters["params"],
                fn=fn,
            )


def combined_analysis(
    opt_funs,
    n=10,
    fn=None,
    times=None,
    folderToStore=None,
    orderInEquil=None,
    nResol=None,
):
    CoilCurrents_all, CoilCurrents_lower_all = [], []
    for opt_fun in opt_funs:
        p = opt_fun
        dictDVs = analyze_results(
            p, plotYN=False, storeResults=False, onlyBest=True, onlyPrepare=True
        )

        CoilCurrents0 = p.function_parameters["CoilCurrents"]
        CoilCurrents_lower0 = p.function_parameters["CoilCurrents_lower"]
        for ikey in dictDVs:
            if "_l" in ikey:
                CoilCurrents_lower0[ikey] = dictDVs[ikey]["value"][0]
            else:
                CoilCurrents0[ikey] = dictDVs[ikey]["value"][0]

        CoilCurrents_all.append(CoilCurrents0)
        CoilCurrents_lower_all.append(CoilCurrents_lower0)

    Constraints = p.function_parameters["Constraints"]

    # Combine
    CoilCurrents = {}
    CoilCurrents_lower = None  # CHANGE
    for ikey in CoilCurrents_all[0]:
        CoilCurrents[ikey] = []
        for i in range(len(CoilCurrents_all)):
            CoilCurrents[ikey].append(CoilCurrents_all[i][ikey])

    CoilCurrents, Constraints, times_mod = FREEGSUplotting.extendSweep(
        CoilCurrents, Constraints, n=n, orderInEquil=orderInEquil, times=None
    )

    # Made to one
    p.function_parameters["CoilCurrents"] = CoilCurrents
    p.function_parameters["CoilCurrents_lower"] = CoilCurrents_lower
    p.function_parameters["Constraints"] = Constraints
    if times is None:
        times = np.linspace(0, 1, n)
        print(
            "\n*** PRF WARNING: Remember to check the timing for voltages!!\n",
            typeMsg="w",
        )
    p.function_parameters["params"]["times"] = times

    # Higher resolution
    p.function_parameters["optionsFREEGS"]["n"] = nResol if nResol is not None else 129

    # Plot

    if fn is None:
        fn = GUItools.FigureNotebook("Combined analysis")

    fig1 = fn.add_figure(label="FreeGSU - Eq. & Coils")
    fig2 = fn.add_figure(label="FreeGSU - Metrics")
    fig3 = fn.add_figure(label="FreeGSU - Solution")
    figP = fn.add_figure(label="FreeGSU - Profiles")
    figa = fn.add_figure(label="FreeGSU - Powers")
    figb = fn.add_figure(label="FreeGSU - Maxima")
    figMach = fn.add_figure(label="FreeGSU - Machine")
    figRes = fn.add_figure(label="FreeGSU - Summary")

    figs = [fig1, fig2, fig3, figP, figa, figb, figMach, figRes]

    prfs, metrics, metrics_opt = runFreeGS(p, {}, plot=True, figs=figs)

    if folderToStore is not None:
        IOtools.askNewFolder(folderToStore, force=True)
        folderEvaluation_out = folderToStore / "Outputs"
        IOtools.askNewFolder(folderEvaluation_out, force=True)
        folderEvaluation = folderEvaluation_out / "final_analysis"
        IOtools.askNewFolder(folderEvaluation, force=True)
        gs = FREEGSUplotting.writeResults(
            folderEvaluation,
            prfs,
            metrics,
            p.function_parameters,
            namePkl="results",
            plotGs=True,
            params=p.function_parameters["params"],
            fn=fn,
        )

    return fn


def apply_rangeVar(
    setCoils, coilLimits, function_parameters, rangeVar, turns_real=None
):
    """
    Pass turns_real when coilLimits are in kA
    """

    if IOtools.isnum(rangeVar):
        rangeVar_dict = {}
        for i in setCoils:
            rangeVar_dict[i] = rangeVar
    else:
        rangeVar_dict = rangeVar

    if turns_real is not None:
        CoilCurrents = copy.deepcopy(function_parameters["CoilCurrents"])
        CoilCurrents_lower = copy.deepcopy(function_parameters["CoilCurrents_lower"])

        for i in setCoils:
            f = turns_real[i + "u"] * 1e-3
            CoilCurrents[i] = CoilCurrents[i] / f
            if CoilCurrents_lower is not None:
                CoilCurrents_lower[i] = CoilCurrents_lower[i] / f

    # Define limits based around the basseline, but not going outside bounds
    for i in setCoils:
        if CoilCurrents_lower is not None:
            minc = np.max(
                [
                    np.min([CoilCurrents[i], CoilCurrents_lower[i]]) - rangeVar_dict[i],
                    coilLimits[i][0],
                ]
            )
            maxc = np.min(
                [
                    np.max([CoilCurrents[i], CoilCurrents_lower[i]]) + rangeVar_dict[i],
                    coilLimits[i][1],
                ]
            )
        else:
            minc = np.max([CoilCurrents[i] - rangeVar_dict[i], coilLimits[i][0]])
            maxc = np.min([CoilCurrents[i] + rangeVar_dict[i], coilLimits[i][1]])
        coilLimits[i] = [minc, maxc]

    return coilLimits


def initializeGSgrab(subfolders, evaluations, superfolder="./"):
    if type(subfolders) == str:
        subfolders = [subfolders]
    subfolders = [IOtools.expandPath(subfolder) for subfolder in subfolders]
    superfolder = IOtools.expandPath(superfolder)

    # Initialize the currents (in case they have different DVs)
    coils = []
    for ff, ev in zip(subfolders, evaluations):
        pklf = superfolder / f"{ff}" / "Outputs" / "optimization_object.pkl"

        mitim = pickle.load(open(pklf, "rb"))

        coils0 = mitim.optimization_object.function_parameters["CoilCurrents"]

        for ikey in coils0:
            if type(coils0[ikey]) == float or coils0[ikey] is None:
                c = coils0[ikey]
            else:
                c = coils0[ikey][0]
            coils0[ikey] = [c]

        coils.append(coils0)

    CoilCurrents = copy.deepcopy(coils[0])
    for ikey in CoilCurrents:
        for i in range(len(coils) - 1):
            CoilCurrents[ikey].append(coils[i + 1][ikey][0])

    # --
    CoilCurrents, _ = grabBestSpecificParams(
        subfolders,
        evaluations,
        CoilCurrents,
        superfolder=superfolder,
        alreadyAssigned=True,
    )

    return CoilCurrents

def grabBestSpecificParams(
    subfolders,
    evaluations,
    SpecificParams,
    superfolder="./",
    avoidExtraOFs=True,
    alreadyAssigned=False,
    ):
    if type(subfolders) == str:
        subfolders = [subfolders]
    subfolders = [IOtools.expandPath(subfolder) for subfolder in subfolders]
    superfolder = IOtools.expandPath(superfolder)

    # None evaluation to find best

    useDict = "evaluations"  #'optima'

    cont = 0
    for ff, ev in zip(subfolders, evaluations):
        res = BOgraphics.optimization_results(
            superfolder / f"{ff}" / "Outputs" / "optimization_results.out"
        )
        res.read()
        useDict_class = res.__dict__[useDict]

        x, yT, _ = BOgraphics.plotAndGrab(
            None,
            None,
            None,
            None,
            res.evaluations,
            res.calibrations,
            res.summations,
            res.OF_labels,
            None,
        )

        yMax = yT.max(axis=1)
        yMean = yT.mean(axis=1)

        if ev is None:
            if len(yMax) == 0:
                ev, yev, yev2, maxev = 0, 1e6, 1e6, 0
            else:
                ev, yev, yev2, maxev = (
                    x[np.nanargmin(yMean)],
                    yMax[np.nanargmin(yMax)],
                    yMean[np.nanargmin(yMax)],
                    x[-1],
                )
            print(
                '\t- For {0}, read "{5}", and best evaluation found at {1} (/{2}) with a residual of {3:.3f} (max) and {4:.3f} (mean)'.format(
                    ff, ev, maxev, yev, yev2, useDict
                )
            )
        else:
            print(f"\t- For {ff}, using user-specified evaluation {ev}")

        # First equilibrium
        for i in res.DV_labels:
            if "_delta" not in i:
                if i not in SpecificParams or (type(SpecificParams[i]) == float):
                    SpecificParams[i] = []

                if not alreadyAssigned:
                    SpecificParams[i].append(useDict_class[ev]["x"][i])
                else:
                    try:
                        SpecificParams[i][cont] = useDict_class[ev]["x"][i]
                    except:
                        SpecificParams[i].append(useDict_class[ev]["x"][i])

        # Second equilibrium, when double optimization is enabled
        for i in res.DV_labels:
            if "_delta" in i:
                j = i[:-6]
                SpecificParams[j].append(
                    useDict_class[ev]["x"][i] + useDict_class[ev]["x"][j]
                )

        cont += 1

    return SpecificParams, ev


def getBestCoils(folderWork, startFrom):
    try:
        preFolder, _ = IOtools.reducePathLevel(folderWork, level=1, isItFile=False)
    except:
        preFolder = IOtools.expandPath("./")
    CoilCurrents = initializeGSgrab(startFrom, [None], superfolder=preFolder)

    for i in CoilCurrents:
        CoilCurrents[i] = CoilCurrents[i][0]

    CoilCurrents_lower = OrderedDict()
    CoilCurrents_upper = OrderedDict()
    for i in CoilCurrents:
        if "_l" in i:
            CoilCurrents_lower[i.split("_l")[0]] = copy.deepcopy(CoilCurrents[i])
        else:
            CoilCurrents_upper[i] = copy.deepcopy(CoilCurrents[i])

    if len(CoilCurrents_lower) == 0:
        CoilCurrents_lower = None
    else:
        if CoilCurrents_upper["vs1"] is None:
            CoilCurrents_lower["vs1"] = None

    return CoilCurrents_upper, CoilCurrents_lower
