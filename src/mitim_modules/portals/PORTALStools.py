import torch
import gpytorch
import copy
import numpy as np
from mitim_tools.opt_tools import STRATEGYtools
from mitim_tools.misc_tools import PLASMAtools
from collections import OrderedDict
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

def surrogate_selection_portals(output, surrogate_options, CGYROrun=False):

    print(f'\t- Selecting surrogate options for "{output}" to be run')

    if output is not None:
        # If it's a target, just linear
        if output[2:5] == "Tar":
            surrogate_options["TypeMean"] = 1
            surrogate_options["TypeKernel"] = 2  # Constant kernel
        # If it's not, stndard
        else:
            surrogate_options["TypeMean"] = 2  # Linear in gradients, constant in rest
            surrogate_options["TypeKernel"] = 1  # RBF
            # surrogate_options['ExtraNoise']  = True

    surrogate_options["additional_constraints"] = {
        'lenghtscale_constraint': gpytorch.constraints.constraints.GreaterThan(0.01) # inputs normalized to [0,1], this is  1% lengthscale
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

    _, num = output.split("_")
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
    # --- Ratio of fluxes (quasilinear)
    factorRat = ratioFactor(X, surrogate_parameters, output, powerstate)
    # --- Specific to output
    factorImp = ImpurityGammaTrick(X, surrogate_parameters, output, powerstate)

    compounded = factorGB * factorRat * factorImp

    """
	3. Go back to the original batching system
	------------------------------------------------------------------------
		E.g.: (batch1*batch2*batch3,1) -> (batch1,batch2,batch3,1) 
	"""
    shape_orig[-1] = compounded.shape[-1]
    compounded = compounded.view(tuple(shape_orig))

    return compounded


def computeTurbExchangeIndividual(PexchTurb, powerstate):
    """
    Volume integrate energy exchange from MW/m^3 to a flux MW/m^2 to be added
    """

    """
	1. Make sure all batches are squeezed into a single dimension
	------------------------------------------------------------------
		E.g.: (batch1,batch2,batch3,dimR) -> (batch1*batch2*batch3,dimR)
	"""
    shape_orig = np.array(PexchTurb.shape)
    PexchTurb = PexchTurb.view(np.prod(shape_orig[:-1]), shape_orig[-1])

    """
	2. Integrate
	------------------------------------------------------------------------
		qExch is in MW/m^3
		powerstate.volume_integrate produces in MW/m^2
	"""

    # Add zeros at zero
    qExch = torch.cat((torch.zeros(PexchTurb.shape).to(PexchTurb)[..., :1], PexchTurb), dim=-1)

    PexchTurb_integrated = powerstate.volume_integrate(qExch, force_dim=qExch.shape[0])[..., 1:]

    """
	3. Go back to the original batching system
	------------------------------------------------------------------------
		E.g.: (batch1*batch2*batch3,dimR) -> (batch1,batch2,batch3,dimR) 
	"""
    PexchTurb_integrated = PexchTurb_integrated.view(tuple(shape_orig))

    return PexchTurb_integrated

def GBfromXnorm(x, output, powerstate):
    # Decide, depending on the output here, which to use as normalization and at what location
    varFull = output.split("_")[0]
    pos = int(output.split("_")[1])

    # Select GB unit
    if varFull[:2] == "Qe":
        quantity = "Qgb"
    elif varFull[:2] == "Qi":
        quantity = "Qgb"
    elif varFull[:2] == "Mt":
        quantity = "Pgb"
    elif varFull[:2] == "Ge":
        quantity = "Ggb" if (not powerstate.useConvectiveFluxes) else "Qgb_convection"
    elif varFull[:2] == "GZ":
        quantity = "Ggb" if (not powerstate.useConvectiveFluxes) else "Qgb_convection"
    elif varFull[:5] == "Pexch":
        quantity = "Sgb"

    T = powerstate.plasma[quantity][: x.shape[0], powerstate.indexes_simulation[pos]].unsqueeze(-1)

    return T


def ImpurityGammaTrick(x, surrogate_parameters, output, powerstate):
    """
    Trick to make GZ a function of a/Lnz only (flux as GammaZ_hat = GammaZ /nZ )
    """

    pos = int(output.split("_")[1])

    if ("GZ" in output) and surrogate_parameters["applyImpurityGammaTrick"]:
        factor = powerstate.plasma["ni"][: x.shape[0],powerstate.indexes_simulation[pos],powerstate.impurityPosition].unsqueeze(-1)

    else:
        factor = torch.ones(tuple(x.shape[:-1]) + (1,)).to(x)

    return factor


def ratioFactor(X, surrogate_parameters, output, powerstate):
    """
    This defines the vector to divide by.

    THIS IS BROKEN RIGHT NOW
    """

    v = torch.ones(tuple(X.shape[:-1]) + (1,)).to(X)

    # """
    # Apply diffusivities (not real value, just capturing dependencies,
    # work on normalization, like e_J). Or maybe calculate gradients within powerstate
    # Remember that for Ti I'm using ne...
    # """
    # if surrogate_parameters["useDiffusivities"]:
    #     pos = int(output.split("_")[-1])
    #     var = output.split("_")[0]

    #     if var == "te":
    #         grad = x[:, i] * (
    #             powerstate.plasma["te"][:, powerstate.indexes_simulation[pos]]
    #             / powerstate.plasma["a"]
    #         )  # keV/m
    #         v[:] = grad * powerstate.plasma["ne"][:, powerstate.indexes_simulation[pos]]

    #     if var == "ti":
    #         grad = x[:, i] * (
    #             powerstate.plasma["ti"][:, powerstate.indexes_simulation[pos]]
    #             / powerstate.plasma["a"]
    #         )  # keV/m
    #         v[:] = grad * powerstate.plasma["ne"][:, powerstate.indexes_simulation[pos]]

    #     # if var == 'ne':
    #     #     grad = x[:,i] * ( powerstate.plasma['ne'][:,pos]/powerstate.plasma['a']) # keV/m
    #     #     v[:] = grad

    # """
    # Apply flux ratios
    # For example [1,Qi,Qi] means I will fit to [Qi, Qe/Qi, Ge/Qi]
    # """

    # if surrogate_parameters["useFluxRatios"]:
    #     """
    #     Not ready yet... since my code is not dealing with other outputs at a time so
    #     I don't know Qi if I'm evaluating other fluxes...
    #     """
    #     pass

    return v


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
                powerstate.TargetOptions["ModelOptions"]["TargetCalc"] = "powerstate"  # For surrogate evaluation, always powerstate, logically.
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
