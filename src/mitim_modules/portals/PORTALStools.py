import torch
import gpytorch
import copy
import numpy as np
from collections import OrderedDict
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.misc_tools import PLASMAtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

def surrogate_selection_portals(output, surrogate_options, CGYROrun=False):

    print(f'\t- Selecting surrogate options for "{output}" to be run')

    if output is not None:
        # If it's a target, just linear
        if output[3:6] == "tar":
            surrogate_options["TypeMean"] = 1
            surrogate_options["TypeKernel"] = 2  # Constant kernel
        # If it's not, standard case for fluxes
        else:
            surrogate_options["TypeMean"] = 2  # Linear in gradients, constant in rest
            surrogate_options["TypeKernel"] = 1  # RBF

    surrogate_options["additional_constraints"] = {
        'lenghtscale_constraint': gpytorch.constraints.constraints.GreaterThan(0.05) # inputs normalized to [0,1], this is  5% lengthscale
    }

    return surrogate_options

def default_portals_transformation_variables(additional_params = []):
    """
    Physics-informed parameters to fit surrogates
    ---------------------------------------------
        Note: Dict value indicates what variables need to change at this location to add this one (only one of them is needed)
        Note 2: index key indicates when to transition to next (in terms of number of individuals available for fitting)
        Things to add:
                'aLte': ['aLte'],   'aLti': ['aLti'],      'aLne': ['aLne'],
                'nuei': ['te','ne'],'tite': ['te','ti'],   'c_s': ['te'],      'w0_n': ['w0'],
                'beta_e':  ['te','ne']

        transition_evaluations is the number of points to be fitted that require a parameter transition.
            Note that this ignores ExtraData or ExtraPoints.
                - transition_evaluations[0]: max to only consider gradients
                - transition_evaluations[1]: no beta_e
                - transition_evaluations[2]: full
    """

    transition_evaluations = [10, 30, 10000]
    portals_transformation_variables = {
        transition_evaluations[0]: OrderedDict(
            {
                "aLte": ["aLte"],
                "aLti": ["aLti"],
                "aLne": ["aLne"],
                "aLw0_n": ["aLw0"],
            }
        ),
        transition_evaluations[1]: OrderedDict(
            {
                "aLte": ["aLte"],
                "aLti": ["aLti"],
                "aLne": ["aLne"],
                "aLw0_n": ["aLw0"],
                "nuei": ["te", "ne"],
                "tite": ["te", "ti"],
                "w0_n": ["w0"],
            }
        ),
        transition_evaluations[2]: OrderedDict(
            {
                "aLte": ["aLte"],
                "aLti": ["aLti"],
                "aLne": ["aLne"],
                "aLw0_n": ["aLw0"],
                "nuei": ["te", "ne"],
                "tite": ["te", "ti"],
                "w0_n": ["w0"],
                "beta_e": ["te", "ne"],
            }
        ),
    }

    # Add additional parameters (to be used as fixed parameters but changing in between runs)
    for key in additional_params:
        portals_transformation_variables[transition_evaluations[-1]][key] = [key]

    # If doing trace impurities, alnZ only affects that channel, but the rest of turbulent state depends on the rest of parameters
    portals_transformation_variables_trace = copy.deepcopy(portals_transformation_variables)
    portals_transformation_variables_trace[transition_evaluations[0]]["aLnZ"] = ["aLnZ"]
    portals_transformation_variables_trace[transition_evaluations[1]]["aLnZ"] = ["aLnZ"]
    portals_transformation_variables_trace[transition_evaluations[2]]["aLnZ"] = ["aLnZ"]

    return portals_transformation_variables, portals_transformation_variables_trace

def input_transform_portals(Xorig, output, surrogate_parameters, surrogate_transformation_variables):

    """
    - Xorig will be a tensor (batch1...N,dim) unnormalized (with or without gradients).
    - Provides new Xorig unnormalized
    - Provides new dictionary with intermediate steps, useful to avoid repeated calculations
    """

    """
	1. Make sure all batches are squeezed into a single dimension
	------------------------------------------------------------------
		E.g.: (batch1,batch2,batch3,dim) -> (batch1*batch2*batch3,dim)
	"""

    shape_orig = np.array(Xorig.shape)
    X = Xorig.view(np.prod(shape_orig[:-1]), shape_orig[-1])

    """
	2. Calculate kinetic profiles to use during transformations and update powerstate with them
	-------------------------------------------------------------------------------------------
	"""

    powerstate = constructEvaluationProfiles(X, surrogate_parameters, recalculateTargets = True) # This is the only place where I recalculate targets, so that I have the target transformation

    """
	3. Local parameters to fit surrogate to
	-------------------------------------------------------------------------------------------
		Note: 	Reason why I do ":X" is because the powerstate may have more values in it b/c I
				initialize it with a larger batch
	"""

    num = output.split("_")[-1]
    index = powerstate.indexes_simulation[int(num)]  # num=1 -> pos=1, so that it takes the second value in vectors

    xFit = torch.Tensor().to(X)
    for ikey in surrogate_transformation_variables[output]:
        xx = powerstate.plasma[ikey][: X.shape[0], index]
        xFit = torch.cat((xFit, xx.unsqueeze(-1)), dim=-1).to(X)

    parameters_combined = {"powerstate": powerstate}

    """
	4. Go back to the original batching system
	------------------------------------------------------------------------
		E.g.: (batch1*batch2*batch3,dimNEW) -> (batch1,batch2,batch3,dimNEW) 
	"""

    shape_orig[-1] = xFit.shape[-1]
    xFit = xFit.view(tuple(shape_orig))

    return xFit, parameters_combined

# ----------------------------------------------------------------------
# Transformation of Outputs
# ----------------------------------------------------------------------

def output_transform_portals(X, surrogate_parameters, output):
    """
    1. Make sure all batches are squeezed into a single dimension
    ------------------------------------------------------------------
            E.g.: (batch1,batch2,batch3,dim) -> (batch1*batch2*batch3,dim)
    """
    shape_orig = np.array(X.shape)
    X = X.view(np.prod(shape_orig[:-1]), shape_orig[-1])

    """
	2. Produce Factor
	------------------------------------------------------------------
		Works with (batch,dimX) -> (batch,1)
	"""

    # Produce relevant quantities here (in particular, GB will be used)
    powerstate = constructEvaluationProfiles(X, surrogate_parameters)

    # --- Original model output is in real units, transform to GB here b/c that's how GK codes work
    factorGB = GBfromXnorm(X, output, powerstate)
    # --- Specific to output
    factorImp = ImpurityGammaTrick(X, surrogate_parameters, output, powerstate)

    compounded = factorGB * factorImp

    """
	3. Go back to the original batching system
	------------------------------------------------------------------------
		E.g.: (batch1*batch2*batch3,1) -> (batch1,batch2,batch3,1) 
	"""
    shape_orig[-1] = compounded.shape[-1]
    compounded = compounded.view(tuple(shape_orig))

    return compounded


def computeTurbExchangeIndividual(QieMWm3_tr_turb, powerstate):
    """
    Volume integrate energy exchange from MW/m^3 to a flux MW/m^2 to be added
    """

    """
	1. Make sure all batches are squeezed into a single dimension
	------------------------------------------------------------------
		E.g.: (batch1,batch2,batch3,dimR) -> (batch1*batch2*batch3,dimR)
	"""
    shape_orig = np.array(QieMWm3_tr_turb.shape)
    QieMWm3_tr_turb = QieMWm3_tr_turb.view(np.prod(shape_orig[:-1]), shape_orig[-1])

    """
	2. Integrate
	------------------------------------------------------------------------
		qExch is in MW/m^3
		powerstate.volume_integrate produces in MW/m^2
	"""

    # Add zeros at zero
    qExch = torch.cat((torch.zeros(QieMWm3_tr_turb.shape).to(QieMWm3_tr_turb)[..., :1], QieMWm3_tr_turb), dim=-1)

    QieMWm3_tr_turb_integrated = powerstate.volume_integrate(qExch, force_dim=qExch.shape[0])[..., 1:]

    """
	3. Go back to the original batching system
	------------------------------------------------------------------------
		E.g.: (batch1*batch2*batch3,dimR) -> (batch1,batch2,batch3,dimR) 
	"""
    QieMWm3_tr_turb_integrated = QieMWm3_tr_turb_integrated.view(tuple(shape_orig))

    return QieMWm3_tr_turb_integrated

def GBfromXnorm(x, output, powerstate):
    # Decide, depending on the output here, which to use as normalization and at what location
    varFull = '_'.join(output.split("_")[:-1])
    pos = int(output.split("_")[-1])

    # Select GB unit
    if varFull[:2] == "Qe":
        quantity = "Qgb"
    elif varFull[:2] == "Qi":
        quantity = "Qgb"
    elif varFull[:2] == "Mt":
        quantity = "Pgb"
    elif varFull[:2] == "Ge":
        quantity = "Qgb_convection"
    elif varFull[:2] == "GZ":
        quantity = "Qgb_convection"
    elif varFull[:5] == "Pexch":
        quantity = "Sgb"

    T = powerstate.plasma[quantity][: x.shape[0], powerstate.indexes_simulation[pos]].unsqueeze(-1)

    return T


def ImpurityGammaTrick(x, surrogate_parameters, output, powerstate):
    """
    Trick to make GZ a function of a/Lnz only (flux as GammaZ_hat = GammaZ /nZ )
    """

    pos = int(output.split("_")[-1])

    if ("GZ" in output) and surrogate_parameters["applyImpurityGammaTrick"]:
        factor = powerstate.plasma["ni"][: x.shape[0],powerstate.indexes_simulation[pos],powerstate.impurityPosition].unsqueeze(-1)

    else:
        factor = torch.ones(tuple(x.shape[:-1]) + (1,)).to(x)

    return factor

def constructEvaluationProfiles(X, surrogate_parameters, recalculateTargets=False):
    """
    Prepare powerstate for another evaluation with batches
    ------------------------------------------------------

    Notes:
        - Only calculate it once per ModelList, so make sure it's not in parameters_combined before
          computing it, since it is common for all.

    """

    if ("parameters_combined" in surrogate_parameters) and ("powerstate" in surrogate_parameters["parameters_combined"]):
        powerstate = surrogate_parameters["parameters_combined"]["powerstate"]

    else:
        powerstate = surrogate_parameters["powerstate"]

        if X.shape[0] > 0:

            '''
            ----------------------------------------------------------------------------------------------------------------
            Copying the powerstate object (to proceed separately) would be very expensive (~30% increase in foward pass)...
            So, it is better to detach what pythorch has done in other backward evals and repeat with the X shape.
            '''
            powerstate._detach_tensors()
            if powerstate.batch_size != X.shape[0]:
                powerstate._repeat_tensors(batch_size=X.shape[0])
            # --------------------------------------------------------------------------------------------------------------
            
            num_x = powerstate.plasma["rho"].shape[-1] - 1

            # Obtain modified profiles
            CPs = torch.zeros((X.shape[0], num_x + 1)).to(X)
            for iprof, var in enumerate(powerstate.ProfilesPredicted):
                # Specific part of the input vector that deals with this profile and introduce to CP vector (that starts with 0,0)
                CPs[:, 1:] = X[:, (iprof * num_x) : (iprof * num_x) + num_x]

                # Update profile in powerstate
                _ = powerstate.update_var(var, CPs)

            # Update normalizations and targets
            powerstate.calculateProfileFunctions()

            # Targets only if needed (for speed, GB doesn't need it)
            if recalculateTargets:
                powerstate.target_options["ModelOptions"]["targets_evaluator_method"] = "powerstate"  # For surrogate evaluation, always powerstate, logically.
                powerstate.calculateTargets()

    return powerstate


def stopping_criteria_portals(mitim_bo, parameters = {}):

    # Standard stopping criteria
    converged_by_default, yvals = STRATEGYtools.stopping_criteria_default(mitim_bo, parameters)

    # Ricci metric
    ricci_value = parameters["ricci_value"]
    d0 = parameters.get("ricci_d0", 2.0)
    la = parameters.get("ricci_lambda", 1.0)

    print(f"\t- Checking Ricci metric (d0 = {d0}, lamdba = {la})...")

    Y = torch.from_numpy(mitim_bo.train_Y).to(mitim_bo.dfT)
    of, cal, _ = mitim_bo.scalarized_objective(Y)
    
    Ystd = torch.from_numpy(mitim_bo.train_Ystd).to(mitim_bo.dfT)
    of_u, cal_u, _ = mitim_bo.scalarized_objective(Y+Ystd)
    of_l, cal_l, _ = mitim_bo.scalarized_objective(Y-Ystd)

    # If the transformation is linear, they should be the same
    of_stdu, cal_stdu = (of_u-of), (cal_u-cal)
    of_stdl, cal_stdl = (of-of_l), (cal-cal_l)

    of_std, cal_std = (of_stdu+of_stdl)/2, (cal_stdu+cal_stdl)/2

    _, chiR = PLASMAtools.RicciMetric(of, cal, of_std, cal_std, d0=d0, l=la)

    print(f"\t\t* Best Ricci metric: {chiR.min():.3f} (threshold: {ricci_value:.3f})")

    converged_by_ricci = chiR.min() < ricci_value

    if converged_by_default:
        print("\t- Default stopping criteria converged, providing as iteration values the scalarized objective")
        return True, yvals
    elif converged_by_ricci:
        print("\t- Ricci metric converged, providing as iteration values the Ricci metric")
        return True, chiR
    else:
        print("\t- No convergence yet, providing as iteration values the scalarized objective")
        return False, yvals



def calculate_residuals(powerstate, PORTALSparameters, specific_vars=None):
    """
    Notes
    -----
        - Works with tensors
        - It should be independent on how many dimensions it has, except that the last dimension is the multi-ofs
    """

    # Case where I have already constructed the dictionary (i.e. in scalarized objective)
    if specific_vars is not None:
        var_dict = specific_vars
    # Prepare dictionary from powerstate (for use in Analysis)
    else:
        var_dict = {}

        mapper = {
            "Qe_tr_turb": "QeMWm2_tr_turb",
            "Qi_tr_turb": "QiMWm2_tr_turb",
            "Ge_tr_turb": "Ce_tr_turb",
            "GZ_tr_turb": "CZ_tr_turb",
            "Mt_tr_turb": "MtJm2_tr_turb",
            "Qe_tr_neoc": "QeMWm2_tr_neoc",
            "Qi_tr_neoc": "QiMWm2_tr_neoc",
            "Ge_tr_neoc": "Ce_tr_neoc",
            "GZ_tr_neoc": "CZ_tr_neoc",
            "Mt_tr_neoc": "MtJm2_tr_neoc",
            "Qe_tar": "QeMWm2",
            "Qi_tar": "QiMWm2",
            "Ge_tar": "Ce",
            "GZ_tar": "CZ",
            "Mt_tar": "MtJm2",
            "Qie_tr_turb": "QieMWm3_tr_turb"
        }

        for ikey in mapper:
            var_dict[ikey] = powerstate.plasma[mapper[ikey]][..., 1:]
            if mapper[ikey] + "_stds" in powerstate.plasma:
                var_dict[ikey + "_stds"] = powerstate.plasma[mapper[ikey] + "_stds"][..., 1:]
            else:
                var_dict[ikey + "_stds"] = None

    dfT = list(var_dict.values())[0]  # as a reference for sizes

    # -------------------------------------------------------------------------
    # Volume integrate energy exchange from MW/m^3 to a flux MW/m^2 to be added
    # -------------------------------------------------------------------------

    if PORTALSparameters["surrogateForTurbExch"]:
        QieMWm3_tr_turb_integrated = computeTurbExchangeIndividual(var_dict["Qie_tr_turb"], powerstate)
    else:
        QieMWm3_tr_turb_integrated = torch.zeros(dfT.shape).to(dfT)

    # ------------------------------------------------------------------------
    # Go through each profile that needs to be predicted, calculate components
    # ------------------------------------------------------------------------

    of, cal, res = (
        torch.Tensor().to(dfT),
        torch.Tensor().to(dfT),
        torch.Tensor().to(dfT),
    )
    for prof in powerstate.ProfilesPredicted:
        if prof == "te":
            var = "Qe"
        elif prof == "ti":
            var = "Qi"
        elif prof == "ne":
            var = "Ge"
        elif prof == "nZ":
            var = "GZ"
        elif prof == "w0":
            var = "Mt"

        """
		-----------------------------------------------------------------------------------
		Transport (_tr_turb+_tr_neoc)
		-----------------------------------------------------------------------------------
		"""
        of0 = var_dict[f"{var}_tr_turb"] + var_dict[f"{var}_tr_neoc"]

        """
		-----------------------------------------------------------------------------------
		Target (Sum here the turbulent exchange power)
		-----------------------------------------------------------------------------------
		"""
        if var == "Qe":
            cal0 = var_dict[f"{var}_tar"] + QieMWm3_tr_turb_integrated
        elif var == "Qi":
            cal0 = var_dict[f"{var}_tar"] - QieMWm3_tr_turb_integrated
        else:
            cal0 = var_dict[f"{var}_tar"]

        """
		-----------------------------------------------------------------------------------
		Ad-hoc modifications for different weighting
		-----------------------------------------------------------------------------------
		"""

        if var == "Qe":
            of0, cal0 = (
                of0 * PORTALSparameters["Pseudo_multipliers"][0],
                cal0 * PORTALSparameters["Pseudo_multipliers"][0],
            )
        elif var == "Qi":
            of0, cal0 = (
                of0 * PORTALSparameters["Pseudo_multipliers"][1],
                cal0 * PORTALSparameters["Pseudo_multipliers"][1],
            )
        elif var == "Ge":
            of0, cal0 = (
                of0 * PORTALSparameters["Pseudo_multipliers"][2],
                cal0 * PORTALSparameters["Pseudo_multipliers"][2],
            )
        elif var == "GZ":
            of0, cal0 = (
                of0 * PORTALSparameters["Pseudo_multipliers"][3],
                cal0 * PORTALSparameters["Pseudo_multipliers"][3],
            )
        elif var == "MtJm2":
            of0, cal0 = (
                of0 * PORTALSparameters["Pseudo_multipliers"][4],
                cal0 * PORTALSparameters["Pseudo_multipliers"][4],
            )

        of, cal = torch.cat((of, of0), dim=-1), torch.cat((cal, cal0), dim=-1)

    # -----------
    # Composition
    # -----------

    # Source term is (TARGET - TRANSPORT)
    source = cal - of

    # Residual is defined as the negative (bc it's maximization) normalized (1/N) norm of radial & channel residuals -> L2
    res = -1 / source.shape[-1] * torch.norm(source, p=2, dim=-1)

    return of, cal, source, res


def calculate_residuals_distributions(powerstate, PORTALSparameters):
    """
    - Works with tensors
    - It should be independent on how many dimensions it has, except that the last dimension is the multi-ofs
    """

    # Prepare dictionary from powerstate (for use in Analysis)
    
    mapper = {
        "Qe_tr_turb": "QeMWm2_tr_turb",
        "Qi_tr_turb": "QiMWm2_tr_turb",
        "Ge_tr_turb": "Ce_tr_turb",
        "GZ_tr_turb": "CZ_tr_turb",
        "Mt_tr_turb": "MtJm2_tr_turb",
        "Qe_tr_neoc": "QeMWm2_tr_neoc",
        "Qi_tr_neoc": "QiMWm2_tr_neoc",
        "Ge_tr_neoc": "Ce_tr_neoc",
        "GZ_tr_neoc": "CZ_tr_neoc",
        "Mt_tr_neoc": "MtJm2_tr_neoc",
        "Qe_tar": "QeMWm2",
        "Qi_tar": "QiMWm2",
        "Ge_tar": "Ce",
        "GZ_tar": "CZ",
        "Mt_tar": "MtJm2",
        "Qie_tr_turb": "QieMWm3_tr_turb"
    }

    var_dict = {}
    for ikey in mapper:
        var_dict[ikey] = powerstate.plasma[mapper[ikey]][:, 1:]
        if mapper[ikey] + "_stds" in powerstate.plasma:
            var_dict[ikey + "_stds"] = powerstate.plasma[mapper[ikey] + "_stds"][:, 1:]
        else:
            var_dict[ikey + "_stds"] = None

    dfT = var_dict["Qe_tr_turb"]  # as a reference for sizes

    # -------------------------------------------------------------------------
    # Volume integrate energy exchange from MW/m^3 to a flux MW/m^2 to be added
    # -------------------------------------------------------------------------

    if PORTALSparameters["surrogateForTurbExch"]:
        QieMWm3_tr_turb_integrated = computeTurbExchangeIndividual(var_dict["Qie_tr_turb"], powerstate)
        QieMWm3_tr_turb_integrated_stds = computeTurbExchangeIndividual(var_dict["Qie_tr_turb_stds"], powerstate)
    else:
        QieMWm3_tr_turb_integrated = torch.zeros(dfT.shape).to(dfT)
        QieMWm3_tr_turb_integrated_stds = torch.zeros(dfT.shape).to(dfT)

    # ------------------------------------------------------------------------
    # Go through each profile that needs to be predicted, calculate components
    # ------------------------------------------------------------------------

    of, cal = torch.Tensor().to(dfT), torch.Tensor().to(dfT)
    ofE, calE = torch.Tensor().to(dfT), torch.Tensor().to(dfT)
    for prof in powerstate.ProfilesPredicted:
        if prof == "te":
            var = "Qe"
        elif prof == "ti":
            var = "Qi"
        elif prof == "ne":
            var = "Ge"
        elif prof == "nZ":
            var = "GZ"
        elif prof == "w0":
            var = "Mt"

        """
		-----------------------------------------------------------------------------------
		Transport (_tr_turb+_tr_neoc)
		-----------------------------------------------------------------------------------
		"""
        of0 = var_dict[f"{var}_tr_turb"] + var_dict[f"{var}_tr_neoc"]
        of0E = (var_dict[f"{var}_tr_turb_stds"] ** 2 + var_dict[f"{var}_tr_neoc_stds"] ** 2) ** 0.5

        """
		-----------------------------------------------------------------------------------
		Target (Sum here the turbulent exchange power)
		-----------------------------------------------------------------------------------
		"""
        if var == "Qe":
            cal0 = var_dict[f"{var}_tar"] + QieMWm3_tr_turb_integrated
            cal0E = (var_dict[f"{var}_tar_stds"] ** 2 + QieMWm3_tr_turb_integrated_stds**2) ** 0.5
        elif var == "Qi":
            cal0 = var_dict[f"{var}_tar"] - QieMWm3_tr_turb_integrated
            cal0E = (var_dict[f"{var}_tar_stds"] ** 2 + QieMWm3_tr_turb_integrated_stds**2) ** 0.5
        else:
            cal0 = var_dict[f"{var}_tar"]
            cal0E = var_dict[f"{var}_tar_stds"]

        of, cal = torch.cat((of, of0), dim=-1), torch.cat((cal, cal0), dim=-1)
        ofE, calE = torch.cat((ofE, of0E), dim=-1), torch.cat((calE, cal0E), dim=-1)

    return of, cal, ofE, calE
