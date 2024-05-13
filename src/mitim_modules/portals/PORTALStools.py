import torch
import copy
import numpy as np
from collections import OrderedDict
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed


def selectSurrogate(output, surrogateOptions, CGYROrun=False):

    print(
        f'\t- Selecting surrogate options for "{output}" to be run'
    )

    if output is not None:
        # If it's a target, just linear
        if output[2:5] == "Tar":
            surrogateOptions["TypeMean"] = 1
            surrogateOptions["TypeKernel"] = 2  # Constant kernel
        # If it's not, stndard
        else:
            surrogateOptions["TypeMean"] = 2  # Linear in gradients, constant in rest
            surrogateOptions["TypeKernel"] = 1  # RBF
            # surrogateOptions['ExtraNoise']  = True

    return surrogateOptions


def produceNewInputs(Xorig, output, surrogate_parameters, physicsInformedParams):
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

    powerstate = constructEvaluationProfiles(X, surrogate_parameters)

    """
	3. Local parameters to fit surrogate to
	-------------------------------------------------------------------------------------------
		Note: 	Reason why I do ":X" is because the powerstate may have more values in it b/c I
				initialize it with a larger batch
	"""

    _, num = output.split("_")
    index = powerstate.indexes_simulation[
        int(num)
    ]  # num=1 -> pos=1, so that it takes the second value in vectors

    xFit = torch.Tensor().to(X)
    for ikey in physicsInformedParams[output]:
        xx = powerstate.plasma[ikey][: X.shape[0], index]
        xFit = torch.cat((xFit, xx.unsqueeze(1)), dim=1).to(X)

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


def transformmitim(
    X, surrogate_parameters, output
):  # TO REMOVE: call it transformPORTALS
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
    powerstate = constructEvaluationProfiles(
        X, surrogate_parameters, recalculateTargets=False
    )

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
    qExch = torch.cat(
        (torch.zeros(PexchTurb.shape).to(PexchTurb)[..., :1], PexchTurb), dim=-1
    )

    PexchTurb_integrated = powerstate.volume_integrate(qExch, dim=qExch.shape[0])[:, 1:]

    """
	3. Go back to the original batching system
	------------------------------------------------------------------------
		E.g.: (batch1*batch2*batch3,dimR) -> (batch1,batch2,batch3,dimR) 
	"""
    PexchTurb_integrated = PexchTurb_integrated.view(tuple(shape_orig))

    return PexchTurb_integrated


# def transformmitim(X,Y,Yvar,surrogate_parameters,output):
# 	'''
# 	Transform direct evaluation output to something that the model understands better.

# 		- Receives unnormalized X (batch1,...,dim) to construct QGB (batch1,...,1) corresponding to what output I'm looking at
# 		- Transforms and produces Y and Yvar (batch1,...,1)

# 	Output of this function is what the surrogate model will be fitting, so make sure it has a physics-based
# 	meaning behind it (e.g. GB fluxes), that makes sense to fit to variables
# 	'''

# 	factor = factorProducer(X,surrogate_parameters,output)

# 	Ytr     = Y    / factor
# 	Ytr_var = Yvar / factor**2

# 	return Ytr,Ytr_var


# def untransformmitim(X, mean, upper, lower, surrogate_parameters, output):
# 	'''
# 	Transform direct model output to the actual evaluation output (must be the opposite to transformmitim)

# 		- Receives unnormalized X (batch1,...,dim) to construct QGB (batch1,...,1) corresponding to what output I'm looking at
# 		- Transforms and produces Y and confidence bounds (batch1,...,)

# 	This untransforms whatever has happened in the transformmitim function
# 	'''

# 	factor = factorProducer(X,surrogate_parameters,output).squeeze(-1)

# 	mean    = mean  * factor
# 	upper   = upper * factor
# 	lower   = lower * factor

# 	return mean, upper, lower


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

    T = powerstate.plasma[quantity][
        : x.shape[0], powerstate.indexes_simulation[pos]
    ].unsqueeze(-1)

    return T


def ImpurityGammaTrick(x, surrogate_parameters, output, powerstate):
    """
    Trick to make GZ a function of a/Lnz only (flux as GammaZ_hat = GammaZ /nZ )
    """

    pos = int(output.split("_")[1])

    if ("GZ" in output) and surrogate_parameters["applyImpurityGammaTrick"]:
        factor = powerstate.plasma["ni"][
            : x.shape[0],
            powerstate.indexes_simulation[pos],
            powerstate.impurityPosition - 1,
        ].unsqueeze(-1)

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


def constructEvaluationProfiles(X, surrogate_parameters, recalculateTargets=True):
    """
    Prepare powerstate for another evaluation with batches
    ------------------------------------------------------

    Note: Copying this object will be very expensive (~30% increase in foward pass)...
              So, method that is better is to detach what pythorch has done in other backward evals.
              So I'm avoiding doing copy.deepcopy(surrogate_parameters['powerstate'])

    Note: Only calculate it once per ModelList, so make sure it's not in parameters_combined before
              computing it, since it is common for all.
    """

    if ("parameters_combined" in surrogate_parameters) and (
        "powerstate" in surrogate_parameters["parameters_combined"]
    ):
        powerstate = surrogate_parameters["parameters_combined"]["powerstate"]

    else:
        powerstate = surrogate_parameters["powerstate"]

        if X.shape[0] > 0:
            powerstate.repeat(
                batch_size=X.shape[0], pos=0, includeDerived=False
            )  # This is an expensive step (to unrepeat), but can't do anything else...
            powerstate.detach_tensors(includeDerived=False)

            num_x = powerstate.plasma["rho"].shape[-1] - 1

            CPs = torch.zeros((X.shape[0], num_x + 1)).to(X)

            # Obtain modified profiles
            for iprof, var in enumerate(powerstate.ProfilesPredicted):
                # Specific part of the input vector that deals with this profile and introduce to CP vector (that starts with 0,0)
                CPs[:, 1:] = X[:, (iprof * num_x) : (iprof * num_x) + num_x]

                # Update profile in powerstate
                _ = powerstate.update_var(var, CPs)

            # Update normalizations and targets
            powerstate.calculateProfileFunctions()

            # Targets only if needed (for speed, GB doesn't need it)
            if recalculateTargets:
                powerstate.TargetCalc = "powerstate"  # For surrogate evaluation, always powerstate, logically.
                powerstate.calculateTargets()

    return powerstate


def default_physicsBasedParams():
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

    transition_evaluations = [10, 30, 100]
    physicsBasedParams = {
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

    # If doing trace impurities, alnZ only affects that channel, but the rest of turbulent state depends on the rest of parameters
    physicsBasedParams_trace = copy.deepcopy(physicsBasedParams)
    physicsBasedParams_trace[transition_evaluations[0]]["aLnZ"] = ["aLnZ"]
    physicsBasedParams_trace[transition_evaluations[1]]["aLnZ"] = ["aLnZ"]
    physicsBasedParams_trace[transition_evaluations[2]]["aLnZ"] = ["aLnZ"]

    return physicsBasedParams, physicsBasedParams_trace
