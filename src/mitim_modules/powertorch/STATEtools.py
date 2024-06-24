import copy
import torch
import datetime
import os
import matplotlib.pyplot as plt
import dill as pickle
from mitim_tools.misc_tools import PLASMAtools, IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.powertorch.aux import TRANSFORMtools, POWERplot, ITtools
from mitim_modules.powertorch.physics import TARGETStools, CALCtools, TRANSPORTtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

UseCUDAifAvailable = True

# ------------------------------------------------------------------
# POWERSTATE Class
# ------------------------------------------------------------------

class powerstate:
    def __init__(
        self,
        profiles,
        EvolutionOptions={},
        TransportOptions={
            "transport_evaluator": None,
            "ModelOptions": {}
            },
        TargetOptions={
            "targets_evaluator": TARGETStools.analytical_model,
            "ModelOptions": {
                "TypeTarget": 3,
                "TargetCalc": "powerstate"
                },
        },
    ):
        '''
        Inputs:
            - profiles: PROFILES_GACODE object
            - EvolutionOptions:
                - rhoPredicted: radial grid (MUST NOT CONTAIN ZERO, it will be added internally)
                - ProfilesPredicted: list of profiles to predict
                - useConvectiveFluxes: boolean = whether to use convective fluxes instead of particle fluxes for FM
                - impurityPosition: int = position of the impurity in the ions set
                - fineTargetsResolution: int = resolution of the fine targets
            - TransportOptions: dictionary with transport_evaluator and ModelOptions
            - TargetOptions: dictionary with targets_evaluator and ModelOptions
        '''

        self.TransportOptions = TransportOptions
        self.TargetOptions = TargetOptions

        # Default options
        self.ProfilesPredicted = EvolutionOptions.get("ProfilePredicted", ["te", "ti", "ne"])
        self.useConvectiveFluxes = EvolutionOptions.get("useConvectiveFluxes", True)
        self.impurityPosition = EvolutionOptions.get("impurityPosition", 1)
        self.fineTargetsResolution = EvolutionOptions.get("fineTargetsResolution", None)
        rho_vec = EvolutionOptions.get("rhoPredicted", [0.2, 0.4, 0.6, 0.8])

        # Default type and device tensor
        self.dfT = torch.randn(
            (2, 2),
            dtype=torch.double,
            device=torch.device(
                "cpu"
                if ((not UseCUDAifAvailable) or (not torch.cuda.is_available()))
                else "cuda"
            ),
        )

        '''
        Potential profiles to evolve (aLX) and their corresponding flux matching
        ------------------------------------------------------------------------
            The order in the P and P_tr (and therefore the source S)
            tensors will be the same as in self.ProfilesPredicted
        '''
        self.profile_map = {
            "te": ("Pe", "Pe_tr"),
            "ti": ("Pi", "Pi_tr"),
            "ne": ("Ce", "Ce_tr"),
            "nZ": ("CZ", "CZ_tr"),
            "w0": ("Mt", "Mt_tr")
        }

        # -------------------------------------------------------------------------------------
        # Populate plama with radial grid
        # -------------------------------------------------------------------------------------

        # Add a zero on the beginning
        self.plasma = {
            'rho' : torch.cat((torch.zeros(1, device=self.dfT.device, dtype=self.dfT.dtype), (rho_vec if isinstance(rho_vec, torch.Tensor) else torch.from_numpy(rho_vec)).to(self.dfT)))
        }

        # Have the posibility of storing profiles in the class
        self.profiles_stored = []
        self.FluxMatch_Xopt, self.FluxMatch_Yopt = torch.Tensor().to(
            self.dfT
        ), torch.Tensor().to(self.dfT)

        self.labelsFM = []
        for profile in self.ProfilesPredicted:
            self.labelsFM.append([f'aL{profile}', list(self.profile_map[profile])[0], list(self.profile_map[profile])[1]])

        # -------------------------------------------------------------------------------------
        # input.gacode
        # -------------------------------------------------------------------------------------

        # Use a copy because I'm deriving, it may be expensive and I don't want to carry that out outside of this class
        self.profiles = copy.deepcopy(profiles)
        if "derived" not in self.profiles.__dict__:
            self.profiles.deriveQuantities()

        # -------------------------------------------------------------------------------------
        # Fine targets (need to do it here so that it's only once per definition of powerstate)
        # -------------------------------------------------------------------------------------

        if self.fineTargetsResolution is None:
            self.plasma_fine, self.positions_targets = None, None
        else:
            self._fine_grid()

        # -------------------------------------------------------------------------------------
        # Standard creation of plasma dictionary
        # -------------------------------------------------------------------------------------

        TRANSFORMtools.fromGacodeToPower(self, self.profiles, self.plasma["rho"])

        # Convert into a batch so that always the quantities are (batch,dimX)
        self.batch_size = 0
        self._repeat_tensors(batch_size=1)

    def _high_res_rho(self):

        rho_new = torch.linspace(
            self.plasma["rho"][0], self.plasma["rho"][-1], self.fineTargetsResolution
        ).to(self.plasma["rho"])
        for i in self.plasma["rho"]:
            if not torch.isclose(
                rho_new, torch.Tensor([i]).to(self.dfT), atol=1e-10
            ).any():
                rho_new = torch.cat((rho_new, i.unsqueeze(0).to(self.dfT)))
        rho_new = torch.sort(rho_new)[0]

        return rho_new

    def _fine_grid(self):
        """
		-------------------------------------------------------------------------------------------------------------------
		Fine targets procedure:
			Create a plasma_fine dictionary that will contain high resolution profiles, to be used only in calculateTargets.
			For the rest, keep using simply the coarse grid
		-------------------------------------------------------------------------------------------------------------------
		"""

        # Copy coarse plasma to insert back later
        plasma_copy = copy.deepcopy(self.plasma)

        # High resolution rho and interpolation positions
        rho_new = self._high_res_rho()

        self.positions_targets = []
        for i in self.plasma["rho"]:
            self.positions_targets.append(
                (torch.isclose(rho_new, torch.Tensor([i]).to(self.dfT), atol=1e-10))
                .nonzero()[0][0]
                .item()
            )

        # Recalculate with higher resolution
        TRANSFORMtools.fromGacodeToPower(self, self.profiles, rho_new)
        self.plasma_fine = copy.deepcopy(self.plasma)

        # Revert plasma back
        self.plasma = plasma_copy

    def insertProfiles(
        self,
        profiles_base,
        writeFile=None,
        position_in_powerstate_batch=0,
        applyCorrections={},
        insert_highres_powers=False,
        rederive_profiles=True,
        reread_ions=True,
    ):
        '''
        Notes:
            - profiles_base is a PROFILES_GACODE object to use as basecase
            - insert_highres_powers: whether to insert high resolution powers (will calculate them with powerstate, not TGYRO)
        '''
        print(">> Inserting powerstate profiles into input.gacode")

        profiles = TRANSFORMtools.fromPowerToGacode(
            self,
            profiles_base,
            position_in_powerstate_batch=position_in_powerstate_batch,
            options=applyCorrections,
            insert_highres_powers=insert_highres_powers,
            rederive=rederive_profiles,
        )

        if writeFile is not None:
            print(f"\t- Writing input.gacode file: {IOtools.clipstr(writeFile)}")
            if not os.path.exists(os.path.dirname(writeFile)):
                os.makedirs(os.path.dirname(writeFile))
            profiles.writeCurrentStatus(file=writeFile)

        # If corrections modify the ions set... it's better to re-read, otherwise powerstate will be confused
        if reread_ions:
            TRANSFORMtools.defineIons(
                self, profiles, self.plasma["rho"][position_in_powerstate_batch, :], self.dfT
            )

            # Repeat, that's how it's done earlier
            self._repeat_tensors(batch_size=self.plasma["rho"].shape[0], specific_keys=["ni","ions_set_mi","ions_set_Zi","ions_set_Dion","ions_set_Tion","ions_set_c_rad"], positionToUnrepeat=None)

        return profiles

    # ------------------------------------------------------------------
    # Storing and combining
    # ------------------------------------------------------------------

    def save(self, file):
        print(f"\t- Writing power state file: {IOtools.clipstr(file)}")
        with open(file, "wb") as handle:
            pickle.dump(self, handle)

    def combine_states(self, states, includeTransport=True):
        self.TransportOptions_set = [self.TransportOptions]
        self.profiles_stored_set = self.profiles_stored

        for state in states:
            for key in self.plasma:
                self.plasma[key] = torch.cat((self.plasma[key], state[key])).to(
                    self.plasma[key]
                )

            self.TransportOptions_set.append(state.TransportOptions)
            self.profiles_stored_set += state.profiles_stored

            if includeTransport:
                for key in ["chi_e", "chi_i"]:
                    self.TransportOptions["ModelOptions"][key] = torch.cat(
                        (
                            self.TransportOptions["ModelOptions"][key],
                            state.TransportOptions["ModelOptions"][key],
                        )
                    ).to(self.TransportOptions["ModelOptions"][key])

    def copy_state(self):

        # ~~~ Perform copy of the state but without the unpickled fn
        state_shallow_copy = copy.copy(self)
        if hasattr(state_shallow_copy, 'fn'):
            del state_shallow_copy.fn
        if hasattr(state_shallow_copy, 'model_results') and hasattr(state_shallow_copy.model_results, 'fn'):
            del state_shallow_copy.model_results.fn
        state_temp = copy.deepcopy(state_shallow_copy)

        return state_temp

    # ------------------------------------------------------------------
    # Flux-matching and iteration tools
    # ------------------------------------------------------------------

    def calculate(
        self, X, nameRun="test", folder="~/scratch/", evaluation_number=0
    ):
        """
        Inputs:
            - X: torch tensor with gradients
            - nameRun: name of the run (to pass to calculation methods in case black-box simulations are used)
            - folder: folder to save the results, if used by calculation methods (e.g. to write profiles and/or run black-box simulations)
            - evaluation_number
        """

        # 1. Modify gradients (X -> aL.. -> te,ti,ne,nZ,w0)
        self.modify(X)

        # 2. Plasma parameters (te,ti,ne,nZ,w0 -> Qgb,Ggb,Pgb,Sgb,nuei,rho_s,c_s,tite,fZ,beta_e,w0_n,aLw0_n)
        self.calculateProfileFunctions()

        # 3. Sources and sinks (populates components and Pe,Pi,...)
        assumedPercentError = self.TransportOptions["ModelOptions"].get("percentError", [5, 1, 0.5])[-1]
        self.calculateTargets(assumedPercentError=assumedPercentError)  # Calculate targets based on powerstate functions (it may be overwritten in next step, if chosen)

        # 4. Turbulent and neoclassical transport (populates components and Pe_tr,Pi_tr,...)
        self.calculateTransport(
            nameRun=nameRun,
            folder=folder,
            evaluation_number=evaluation_number,
        )

        # 5. Residual powers
        self.metric()

        return (
            self.plasma["P_tr"],
            self.plasma["P"],
            self.plasma["S"],
            self.plasma["residual"],
        )

    def modify(self, X):
        self.Xcurrent = X
        numeach = self.plasma["rho"].shape[1] - 1

        for c, i in enumerate(self.ProfilesPredicted):
            if X is not None:
                self.plasma[f"aL{i}"][:, 1:] = self.Xcurrent[
                    :, numeach * c : numeach * (c + 1)
                ]
            self.update_var(i)

    def findFluxMatchProfiles(
        self, algorithm="root", algorithmOptions={}, bounds=None,
    ):
        self.FluxMatch_plasma_orig = copy.deepcopy(self.plasma)

        print(
            f'\n- Flux matching of powerstate file ({self.plasma["rho"].shape[0]} parallel batches of {self.plasma["rho"].shape[1]-1} radii) has been requested...'
        )
        print(
            "**********************************************************************************************"
        )
        timeBeginning = datetime.datetime.now()

        if algorithm == "root":
            self.FluxMatch_Xopt, self.FluxMatch_Yopt = ITtools.fluxMatchRoot(
                self,
                algorithmOptions=algorithmOptions)
        if algorithm == "simple_relax":
            self.FluxMatch_Xopt, self.FluxMatch_Yopt = ITtools.fluxMatchSimpleRelax(
                self,
                algorithmOptions=algorithmOptions,
                bounds=bounds)
        if algorithm == "picard":
            ITtools.fluxMatchPicard(self)


        print(
            "**********************************************************************************************"
        )
        print(
            f"- Flux matching of powerstate file has been found, and took {IOtools.getTimeDifference(timeBeginning)}\n"
        )

    # ------------------------------------------------------------------
    # Plotting tools
    # ------------------------------------------------------------------

    def plot(self, axs=None, axsRes=None, figs=None, c="r", label="", batch_num=0, compare_to_orig=None, c_orig = 'b'):
        if axs is None:
            axsNotGiven = True
            from mitim_tools.misc_tools.GUItools import FigureNotebook

            fn = FigureNotebook("PowerState", geometry="1800x900")

            # Powerstate
            figMain = fn.add_figure(label="PowerState", tab_color='r')
            # Optimization
            figOpt = fn.add_figure(label="Optimization", tab_color='r')
            grid = plt.GridSpec(3, 1+len(self.ProfilesPredicted), hspace=0.5, wspace=0.5)

            axsRes = [figOpt.add_subplot(grid[:, 0])]
            for i in range(len(self.ProfilesPredicted)):
                for j in range(3):
                    axsRes.append(figOpt.add_subplot(grid[j, i+1]))

            # Profiles
            figs = PROFILEStools.add_figures(fn, tab_color='b')

            axs = add_axes_powerstate_plot(figMain, num_kp = len(self.ProfilesPredicted))
        
        else:
            axsNotGiven = False
            fn = None

        # Make sure tensors are detached
        self._detach_tensors()
        if compare_to_orig is not None:
            compare_to_orig._detach_tensors()

        POWERplot.plot(self, axs, axsRes, figs, c=c, label=label, batch_num=batch_num, compare_to_orig=compare_to_orig, c_orig = c_orig)

        if axsNotGiven:
            fn.show()

        return fn

    # ------------------------------------------------------------------
    # Main tools
    # ------------------------------------------------------------------

    def _detach_tensors(self):
        
        # -------------------------------------------------------------------------------------
        # Detach plasma tensors
        # -------------------------------------------------------------------------------------

        self.plasma = {key: tensor.detach() if tensor.requires_grad else tensor for key, tensor in self.plasma.items()}

        # -------------------------------------------------------------------------------------
        # Detach plasma_fine tensors
        # -------------------------------------------------------------------------------------

        if self.plasma_fine is not None:
            self.plasma_fine = {key: tensor.detach() if tensor.requires_grad else tensor for key, tensor in self.plasma_fine.items()}

        # -------------------------------------------------------------------------------------
        # Detach optimization tensors
        # -------------------------------------------------------------------------------------

        if hasattr(self, 'Xcurrent') and self.Xcurrent is not None and self.Xcurrent.requires_grad:
            self.Xcurrent = self.Xcurrent.detach()

        if hasattr(self, 'FluxMatch_Yopt') and self.FluxMatch_Yopt is not None and self.FluxMatch_Yopt.requires_grad:
            self.FluxMatch_Yopt = self.FluxMatch_Yopt.detach()

    def _repeat_tensors(self, batch_size=1, specific_keys=None, positionToUnrepeat=0):
        """
        Repeat 1D profiles [...] or [positionToUnrepeat,...] (unrepeat first) to [batch_size,...] so that the MITIM calculations are fine
        Notes:
            - The reason for repeat and working in batches is so that calculations can occur in parallel for different plasmas / data points
        """

        def _handle_repeating(tensor, batch_size, positionToUnrepeat):
            if tensor.dim() == 0:
                return tensor.repeat(batch_size)
            elif tensor.dim() == 1:
                return tensor.repeat(batch_size, 1)
            elif tensor.dim() == 2:
                return tensor.repeat(batch_size, 1, 1)
            else:
                return tensor

        tensor_dictionaries = [self.plasma]
        if self.plasma_fine is not None:
            tensor_dictionaries.append(self.plasma_fine)

        for plasma_dict in tensor_dictionaries:
            plasma = {
                key: _handle_repeating( plasma_dict[key][positionToUnrepeat, ...] if (self.batch_size > 0 and positionToUnrepeat is not None) else plasma_dict[key], batch_size, positionToUnrepeat)
                for key in (plasma_dict.keys() if specific_keys is None else specific_keys)
            }

            plasma_dict.update(plasma)

        # New batch size
        self.batch_size = batch_size

    def update_var(self, name, var=None, printMessages=True, specific_deparametrizer=None):
        """
        This inserts gradients and updates coarse profiles

        Notes:
                - If no var is given, assume that I have already assign it and I need to update the profile
                - Gradients must be defined on axis. (0,0) please
                - This keeps the thermal ion concentrations constant to the original case
        """

        # -------------------------------------------------------------------------------------
        # General function to update a variable
        # -------------------------------------------------------------------------------------

        deparametrizers_choice = (
            self.deparametrizers_coarse
            if specific_deparametrizer is None
            else specific_deparametrizer
        )

        def _update_plasma_var(var_key, clamp_min=0, clamp_max=200, factor_mult=1):
            if var is not None:
                self.plasma[f"aL{var_key}"][: var.shape[0], :] = var[:, :]
            aLT_withZero = self.plasma[f"aL{var_key}"]
            _, varN = deparametrizers_choice[var_key](
                self.plasma["roa"], aLT_withZero, printMessages=printMessages
            )
            self.plasma[var_key] = varN.clamp(min=clamp_min, max=clamp_max) * factor_mult
            self.plasma[f"aL{var_key}"] = torch.cat(
                (
                    self.plasma[f"aL{var_key}"][..., : -self.plasma["rho"].shape[1] + 1],
                    self.plasma[f"aL{var_key}"][..., 1:],
                ),
                axis=1,
            )
            return aLT_withZero

        # -------------------------------------------------------------------------------------
        # Update variables
        # -------------------------------------------------------------------------------------

        # Prepare variables (some require special treatment)
        factor_mult = 1
        if name == "w0":
            factor_mult = 1 / TRANSFORMtools.factorMult_w0(self)
        if name == "ne":
            ne_0orig, ni_0orig = self.plasma["ne"].clone(), self.plasma["ni"].clone()

        # UPDATE *******************************************************
        aLT_withZero = _update_plasma_var(name, factor_mult=factor_mult)
        # **************************************************************

        # Postprocessing (some require special treatment)
        if name == "ne":
            self.plasma["ni"] = ni_0orig.clone()
            for i in range(self.plasma["ni"].shape[2]):
                self.plasma["ni"][..., i] = self.plasma["ne"] * (
                    ni_0orig[..., i] / ne_0orig
                )
        elif name == "nZ":
            self.plasma["ni"][..., self.impurityPosition - 1] = self.plasma["nZ"]

        return aLT_withZero

    # ------------------------------------------------------------------
    # Toolset for calculation
    # ------------------------------------------------------------------

    def calculateProfileFunctions(self, calculateRotationStuff=True, mref_u=2.01355):
        """
        Update the normalizations of the current state
        """

        # gyro-Bohm
        (
            self.plasma["Qgb"],
            self.plasma["Ggb"],
            self.plasma["Pgb"],
            self.plasma["Sgb"],
            self.plasma["Qgb_convection"],
        ) = PLASMAtools.gyrobohmUnits(
            self.plasma["te"],
            self.plasma["ne"] * 1e-1,
            mref_u,
            self.plasma["B_unit"],
            self.plasma["a"].unsqueeze(-1),
        )

        # Collisionality
        self.plasma["nuei"] = PLASMAtools.xnue(
            self.plasma["te"], self.plasma["ne"] * 1e-1, self.plasma["a"].unsqueeze(-1), mref_u
        )

        # Gyro-radius
        self.plasma["rho_s"] = PLASMAtools.rho_s(
            self.plasma["te"], mref_u, self.plasma["B_unit"]
        )
        self.plasma["c_s"] = PLASMAtools.c_s(self.plasma["te"], mref_u)

        # Other
        self.plasma["tite"] = self.plasma["ti"] / self.plasma["te"]
        self.plasma["fZ"] = self.plasma["nZ"] / self.plasma["ne"]
        self.plasma["beta_e"] = PLASMAtools.betae(
            self.plasma["te"], self.plasma["ne"] * 1e-1, self.plasma["B_unit"]
        )

        """
		Rotation stuff
		--------------
			Reason why I not always calculate this is because here I'm combining quantities derived (w0,cs)
			which can be accessed in a finer grid, with aLw0 which is a primitive one, that works in the coarse
			grid. Therefore, aLw0_n can only be calculated on the coarse grid.
		"""
        if calculateRotationStuff:
            self.plasma["w0_n"] = self.plasma["w0"] / self.plasma["c_s"]
            self.plasma["aLw0_n"] = (
                self.plasma["aLw0"] * self.plasma["w0"] / self.plasma["c_s"]
            )  # aLw0 * w0 = -a*dw0/dr; then aLw0_n = -dw0/dr * a/c_s

    def calculateTargets(self, assumedPercentError=1.0):
        """
        Update the targets of the current state
        """

        # If no targets evaluator is given or the targets will come from TGYRO, assume them as zero
        if (self.TargetOptions["targets_evaluator"] is None) or (self.TargetOptions["ModelOptions"]["TargetCalc"] == "tgyro"):
            targets = TARGETStools.power_targets(self)
        else:
            targets = self.TargetOptions["targets_evaluator"](self)

        # [Optional] Calculate local targets and integrals on a fine grid
        if self.fineTargetsResolution is not None:
            targets.fine_grid()

        # Evaluate local quantities
        targets.evaluate()

        # Integrate
        targets.flux_integrate()

        # Come back to original grid
        if self.fineTargetsResolution is not None:
            targets.coarse_grid()

        # Merge targets, calculate errors and normalize
        targets.postprocessing(
            assumedPercentError=assumedPercentError,
            useConvectiveFluxes=self.useConvectiveFluxes,
            forceZeroParticleFlux=self.TransportOptions["ModelOptions"].get("forceZeroParticleFlux", False))

    def calculateTransport(
        self, nameRun="test", folder="~/scratch/", evaluation_number=0):
        """
        Update the transport of the current state.
        """

        # Select transport evaluator
        if self.TransportOptions["transport_evaluator"] is None:
            transport = TRANSPORTtools.power_transport( self, name=nameRun, folder=folder, evaluation_number=evaluation_number )
        else:
            transport = self.TransportOptions["transport_evaluator"]( self, name=nameRun, folder=folder, evaluation_number=evaluation_number )
        
        # Produce profile object (for certain transport evaluators, this is necessary)
        transport.produce_profiles()

        # Evaluate transport
        transport.evaluate()

        # Pass the results as part of the powerstate class
        self.model_results = transport.model_results

    def metric(self):
        """
        Calculation of residual transport, dP/dt, ignoring the first (a zero)
        Define convergence metric of current state
        """

        # All fluxes in a single vector
        def _concatenate_flux(plasma, profile_key, flux_key):
            plasma["P"] = torch.cat((plasma["P"], plasma[profile_key][:, 1:]), dim=1).to(plasma["P"].device)
            plasma["P_tr"] = torch.cat((plasma["P_tr"], plasma[flux_key][:, 1:]), dim=1).to(plasma["P"].device)

        self.plasma["P"], self.plasma["P_tr"] = torch.Tensor().to(self.plasma["Pe"]), torch.Tensor().to(self.plasma["Pe"])

        for profile in self.ProfilesPredicted:
            _concatenate_flux(self.plasma, *self.profile_map[profile])
            
        self.plasma["S"] = self.plasma["P"] - self.plasma["P_tr"]
        self.plasma["residual"] = self.plasma["S"].abs().mean(axis=1, keepdim=True)

    def volume_integrate(self, var, force_dim=None):
        """
        If var in MW/m^3, this gives as output the MW/m^2 profile
        """

        if force_dim is None:
            return CALCtools.integrateQuadPoly(
                self.plasma["rmin"], var * self.plasma["volp"]
            ) / self.plasma["volp"]
        else:
            return CALCtools.integrateQuadPoly(
                self.plasma["rmin"][0,:].repeat(force_dim,1), var * self.plasma["volp"][0,:].repeat(force_dim,1),
            ) / self.plasma["volp"][0,:].repeat(force_dim,1)

    def determinePerformance(self, nameRun="test", folder="~/scratch/"):
        '''
        At this moment, this recalculates fusion and radiation, etc
        '''
        folder = IOtools.expandPath(folder)

        # ************************************
        # Calculate state
        # ************************************

        self.calculate(None, nameRun=nameRun, folder=folder)

        # ************************************
        # Postprocessing
        # ************************************

        self.plasma["Pfus"] = (
            self.volume_integrate(
                (self.plasma["qfuse"] + self.plasma["qfusi"]) * 5.0
            )
            * self.plasma["volp"]
        )[..., -1]
        self.plasma["Prad"] = (
            self.volume_integrate(self.plasma["qrad"]) * self.plasma["volp"]
        )[..., -1]

        self.profiles.deriveQuantities()
        
        self.insertProfiles(
            self.profiles,
            writeFile=f"{folder}/input.gacode.new.powerstate",
            position_in_powerstate_batch=0,
            applyCorrections={
                "Tfast_ratio": False,
                "Ti_thermals": False,
                "ni_thermals": False,
                "recompute_ptot": False,
                "ensureMachNumber": None,
            },
            insert_highres_powers=True,
            rederive_profiles=False,
            reread_ions=False,
        )

        self.plasma["Pin"] = (
            (self.plasma["Paux_e"] + self.plasma["Paux_i"]) * self.plasma["volp"]
        )[..., -1]
        self.plasma["Q"] = self.plasma["Pfus"] / self.plasma["Pin"]

        # ************************************
        # Print Info
        # ************************************

        print(
            f"Q = {self.plasma['Q'].item():.2f} (Pfus = {self.plasma['Pfus'].item():.2f}MW, Pin = {self.plasma['Pin'].item():.2f}MW)"
        )

        print(f"Prad = {self.plasma['Prad'].item():.2f}MW")

def add_axes_powerstate_plot(figMain, num_kp=3):

    grid = plt.GridSpec(4, num_kp, hspace=0.5, wspace=0.5)

    axs = []
    for i in range(num_kp):
        for j in range(4):
            axs.append(figMain.add_subplot(grid[j, i]))

    return axs

def read_saved_state(file):
    print(f"\t- Reading state file {IOtools.clipstr(file)}")
    with open(file, "rb") as handle:
        state = pickle.load(handle)
    return state
