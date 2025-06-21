import copy
import torch
import datetime
import shutil
from types import MethodType
import matplotlib.pyplot as plt
import dill as pickle
from mitim_tools.misc_tools import PLASMAtools, IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.powertorch.utils import TRANSFORMtools, POWERplot
from mitim_tools.opt_tools.optimizers import optim
from mitim_modules.powertorch.utils import TARGETStools, CALCtools, TRANSPORTtools
from mitim_modules.powertorch.physics_models import targets_analytic
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

# ------------------------------------------------------------------
# POWERSTATE Class
# ------------------------------------------------------------------

class powerstate:
    def __init__(
        self,
        profiles_object,
        increase_profile_resol=True,
        EvolutionOptions=None,
        TransportOptions=None,
        TargetOptions=None,
        tensor_opts=None,
    ):
        '''
        Inputs:
            - profiles_object: Object for gacode_state or others
            - EvolutionOptions:
                - rhoPredicted: radial grid (MUST NOT CONTAIN ZERO, it will be added internally)
                - ProfilesPredicted: list of profiles to predict
                - useConvectiveFluxes: boolean = whether to use convective fluxes instead of particle fluxes for FM
                - impurityPosition: int = position of the impurity in the ions set
                - fineTargetsResolution: int = resolution of the fine targets
            - TransportOptions: dictionary with transport_evaluator and ModelOptions
            - TargetOptions: dictionary with targets_evaluator and ModelOptions
        '''

        if EvolutionOptions is None:
            EvolutionOptions = {}
        if TransportOptions is None:
            TransportOptions = {
            "transport_evaluator": None,
            "ModelOptions": {}
            }
        if TargetOptions is None:
            TargetOptions = {
            "targets_evaluator": targets_analytic.analytical_model,
            "ModelOptions": {
                "TypeTarget": 3,
                "TargetCalc": "powerstate"
                },
            }
        if tensor_opts is None:
            tensor_opts = {
                "dtype": torch.double,
                "device": torch.device("cpu"),
            }

        # -------------------------------------------------------------------------------------
        # Check inputs
        # -------------------------------------------------------------------------------------

        print('>> Creating powerstate object...')

        self.TransportOptions = TransportOptions
        self.TargetOptions = TargetOptions

        # Default options
        self.ProfilesPredicted = EvolutionOptions.get("ProfilePredicted", ["te", "ti", "ne"])
        self.useConvectiveFluxes = EvolutionOptions.get("useConvectiveFluxes", True)
        self.impurityPosition = EvolutionOptions.get("impurityPosition", 1)
        self.impurityPosition_transport = copy.deepcopy(self.impurityPosition)
        self.fineTargetsResolution = EvolutionOptions.get("fineTargetsResolution", None)
        self.scaleIonDensities = EvolutionOptions.get("scaleIonDensities", True)
        rho_vec = EvolutionOptions.get("rhoPredicted", [0.2, 0.4, 0.6, 0.8])

        if rho_vec[0] == 0:
            raise ValueError("[MITIM] The radial grid must not contain the initial zero")

        # Ensure that nZ is always after ne, because of how the scaling of ni rules are imposed
        def _ensure_ne_before_nz(lst):
            if "ne" in lst and "nZ" in lst:
                ne_index = lst.index("ne")
                nz_index = lst.index("nZ")
                if ne_index > nz_index:
                    # Swap "ne" and "nZ" positions
                    lst[ne_index], lst[nz_index] = lst[nz_index], lst[ne_index]
            return lst
        self.ProfilesPredicted = _ensure_ne_before_nz(self.ProfilesPredicted)

        # Default type and device tensor
        self.dfT = torch.randn((2, 2), **tensor_opts)

        '''
        Potential profiles to evolve (aLX) and their corresponding flux matching
        ------------------------------------------------------------------------
            The order in the P and P_tr (and therefore the source S)
            tensors will be the same as in self.ProfilesPredicted
        '''
        self.profile_map = {
            "te": ("QeMWm2", "QeMWm2_tr"),
            "ti": ("QiMWm2", "QiMWm2_tr"),
            "ne": ("Ce", "Ce_tr"),
            "nZ": ("CZ", "CZ_tr"),
            "w0": ("MtJm2", "MtJm2_tr")
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
        # Object type (e.g. input.gacode)
        # -------------------------------------------------------------------------------------

        if isinstance(profiles_object, PROFILEStools.gacode_state):
            self.to_powerstate = TRANSFORMtools.gacode_to_powerstate
            self.from_powerstate = MethodType(TRANSFORMtools.to_gacode, self)

            # Use a copy because I'm deriving, it may be expensive and I don't want to carry that out outside of this class
            self.profiles = copy.deepcopy(profiles_object)
            if "derived" not in self.profiles.__dict__:
                self.profiles.derive_quantities()

        else:
            raise ValueError("[MITIM] The input profile object is not recognized, please use gacode_state")

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

        # Resolution of input.gacode
        if increase_profile_resol:
            TRANSFORMtools.improve_resolution_profiles(self.profiles, rho_vec)

        # Convert to powerstate
        self.to_powerstate(self)

        # Convert into a batch so that always the quantities are (batch,dimX)
        self.batch_size = 0
        self._repeat_tensors(batch_size=1)

        self.Xcurrent = None

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
        TRANSFORMtools.gacode_to_powerstate(self, rho_vec = rho_new)
        self.plasma_fine = copy.deepcopy(self.plasma)

        # Revert plasma back
        self.plasma = plasma_copy

    # ------------------------------------------------------------------
    # Storing and combining
    # ------------------------------------------------------------------

    def save(self, file):
        print(f"\t- Writing power state file: {IOtools.clipstr(file)}")
        with open(file, "wb") as handle:
            pickle.dump(self, handle, protocol=4)

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
        
        # Plasma dictionary may have gradients that I cannot copy, I need to detach them
        state_shallow_copy.plasma = {}
        for key in self.plasma:
            state_shallow_copy.plasma[key] = self.plasma[key].detach()
        state_shallow_copy.Xcurrent = self.Xcurrent.detach() if self.Xcurrent is not None else None

        state_temp = copy.deepcopy(state_shallow_copy)

        return state_temp

    def to_cpu_tensors(self):
        self._cpu_tensors()
        return self

    # ------------------------------------------------------------------
    # Flux-matching and iteration tools
    # ------------------------------------------------------------------

    def calculate(
        self, X=None, nameRun="test", folder="~/scratch/", evaluation_number=0
    ):
        """
        Inputs:
            - X: torch tensor with gradients
            - nameRun: name of the run (to pass to calculation methods in case black-box simulations are used)
            - folder: folder to save the results, if used by calculation methods (e.g. to write profiles and/or run black-box simulations)
            - evaluation_number
        """
        folder = IOtools.expandPath(folder)

        # 1. Modify gradients (X -> aL.. -> te,ti,ne,nZ,w0)
        self.modify(X)

        # 2. Plasma parameters (te,ti,ne,nZ,w0 -> Qgb,Ggb,Pgb,Sgb,nuei,rho_s,c_s,tite,fZ,beta_e,w0_n,aLw0_n)
        self.calculateProfileFunctions()

        # 3. Sources and sinks (populates components and Pe,Pi,...)
        relative_error_assumed = self.TransportOptions["ModelOptions"].get("percentError", [5, 1, 0.5])[-1]
        self.calculateTargets(relative_error_assumed=relative_error_assumed)  # Calculate targets based on powerstate functions (it may be overwritten in next step, if chosen)

        # 4. Turbulent and neoclassical transport (populates components and Pe_tr,Pi_tr,...)
        self.calculateTransport(
            nameRun=nameRun,
            folder=folder,
            evaluation_number=evaluation_number,
        )

        # 5. Residual powers
        self.calculateMetrics()

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

                aLx_before = self.plasma[f"aL{i}"][:, 1:].clone()

                self.plasma[f"aL{i}"][:, 1:] = self.Xcurrent[:, numeach * c : numeach * (c + 1)]

                # For now, scale also the ion gradients by same ammount (for p_prime) #TODO: improve this treatment
                if i == "ne":
                    for j in range(self.plasma["ni"].shape[-1]):
                        f = aLx_before / self.plasma[f"aLni{j}"][:, 1:].clone()
                        self.plasma[f"aLni{j}"][:, 1:] *= f

            self.update_var(i)

    def flux_match(self, algorithm="root", solver_options=None, bounds=None, debugYN=False):
        self.FluxMatch_plasma_orig = copy.deepcopy(self.plasma)
        self.bounds_current = bounds

        print(f'\t- Flux matching of powerstate file ({self.plasma["rho"].shape[0]} parallel batches of {self.plasma["rho"].shape[1]-1} radii) has been requested...')
        print("**********************************************************************************************")
        timeBeginning = datetime.datetime.now()

        if algorithm == "root":
            solver_fun = optim.scipy_root
        elif algorithm == "simple_relax":
            solver_fun = optim.simple_relaxation
        else:
            raise ValueError(f"[MITIM] Algorithm {algorithm} not recognized")
    
        if solver_options is None:
            solver_options = {}

        folder_main = solver_options.get("folder", None)
        namingConvention = solver_options.get("namingConvention", "powerstate_sr_ev")

        cont = 0
        def evaluator(X, y_history=None, x_history=None, metric_history=None):

            nonlocal cont

            nameRun = f"{namingConvention}_{cont}"

            if folder_main is not None:
                folder = IOtools.expandPath(folder_main) /  f"{namingConvention}_{cont}"
                if issubclass(self.TransportOptions["transport_evaluator"], TRANSPORTtools.power_transport):
                    (folder / "model_complete").mkdir(parents=True, exist_ok=True)

            # ***************************************************************************************************************
            # Calculate
            # ***************************************************************************************************************

            folder_run = folder / "model_complete" if folder_main is not None else IOtools.expandPath('~/scratch/')
            QTransport, QTarget, _, _ = self.calculate(X, nameRun=nameRun, folder=folder_run, evaluation_number=cont)

            cont += 1

            # Save state so that I can check initializations
            if folder_main is not None:
                if issubclass(self.TransportOptions["transport_evaluator"], TRANSPORTtools.power_transport):
                    self.save(folder / "powerstate.pkl")
                    shutil.copy2(folder_run / "input.gacode", folder)

            # ***************************************************************************************************************
            # Postprocess
            # ***************************************************************************************************************

            # Residual is the difference between the target and the transport
            yRes = (QTarget - QTransport).abs()
            # Metric is the mean of the absolute value of the residual
            yMetric = -yRes.mean(axis=-1)
            # Best in batch
            best_candidate = yMetric.argmax().item()
            # Only pass the best candidate
            yRes = yRes[best_candidate, :].detach()
            yMetric = yMetric[best_candidate].detach()
            Xpass = X[best_candidate, :].detach()

            # Store values
            if y_history is not None:      
                y_history.append(yRes)
            if x_history is not None:      
                x_history.append(Xpass)
            if metric_history is not None: 
                metric_history.append(yMetric)

            return QTransport, QTarget, yMetric

        # Concatenate the input gradients
        x0 = torch.Tensor().to(self.plasma["aLte"])
        for c, i in enumerate(self.ProfilesPredicted):
            x0 = torch.cat((x0, self.plasma[f"aL{i}"][:, 1:].detach()), dim=1)

        # Make sure is properly batched
        x0 = x0.view((self.plasma["rho"].shape[0],(self.plasma["rho"].shape[1] - 1) * len(self.ProfilesPredicted),))

        # Optimize
        _,Yopt, Xopt, metric_history = solver_fun(evaluator,x0, bounds=self.bounds_current,solver_options=solver_options)

        # For simplicity, return the trajectory of only the best candidate
        self.FluxMatch_Yopt, self.FluxMatch_Xopt = Yopt, Xopt

        print("**********************************************************************************************")
        print(f"\t- Flux matching of powerstate finished, and took {IOtools.getTimeDifference(timeBeginning)}\n")

        if debugYN:
            self.plot()
            embed()

    # ------------------------------------------------------------------
    # Plotting tools
    # ------------------------------------------------------------------

    def plot(self, axs=None, axsRes=None, axsMetrics=None, figs=None, fn=None,c="r", label="powerstate", batch_num=0, compare_to_state=None, c_orig = "b"):
        if axs is None:

            if fn is None:
                axsNotGiven = True
                from mitim_tools.misc_tools.GUItools import FigureNotebook

                fn = FigureNotebook("PowerState", geometry="1800x900")
            else:
                axsNotGiven = False

            # Powerstate
            figMain = fn.add_figure(label="PowerState", tab_color='r')
            # Optimization
            figOpt = fn.add_figure(label="Optimization", tab_color='r')
            grid = plt.GridSpec(2, 1+len(self.ProfilesPredicted), hspace=0.3, wspace=0.3)

            axsRes = [figOpt.add_subplot(grid[:, 0])]
            for i in range(len(self.ProfilesPredicted)):
                for j in range(2):
                    axsRes.append(figOpt.add_subplot(grid[j, i+1]))

            # Profiles
            figs = PROFILEStools.add_figures(fn, tab_color='b')

            axs, axsMetrics = add_axes_powerstate_plot(figMain, num_kp = len(self.ProfilesPredicted))
        
        else:
            axsNotGiven = False
            fn = None

        # Make sure tensors are detached
        self._detach_tensors()
        powers = [self]
        if compare_to_state is not None:
            compare_to_state._detach_tensors()
            powers.append(compare_to_state)

        POWERplot.plot(self, axs, axsRes, figs, c=c, label=label, batch_num=batch_num, compare_to_state=compare_to_state, c_orig = c_orig)

        if axsMetrics is not None:
            POWERplot.plot_metrics_powerstates(axsMetrics,powers[::-1])

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

    def _cpu_tensors(self):
        self._detach_tensors()
        self.plasma = {key: tensor.cpu() for key, tensor in self.plasma.items() if isinstance(tensor, torch.Tensor)}
        if self.plasma_fine is not None:
            self.plasma_fine = {key: tensor.cpu() for key, tensor in self.plasma_fine.items() if isinstance(tensor, torch.Tensor)}
        if hasattr(self, 'Xcurrent') and self.Xcurrent is not None and isinstance(self.Xcurrent, torch.Tensor):
            self.Xcurrent = self.Xcurrent.cpu()
        if hasattr(self, 'FluxMatch_Yopt') and self.FluxMatch_Yopt is not None and isinstance(self.FluxMatch_Yopt, torch.Tensor):
            self.FluxMatch_Yopt = self.FluxMatch_Yopt.cpu()
        if hasattr(self, 'profiles'):
            self.profiles.toNumpyArrays()

    def update_var(self, name, var=None, specific_profile_constructor=None):
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

        profile_constructor_choice = self.profile_constructors_coarse if specific_profile_constructor is None else specific_profile_constructor

        def _update_plasma_var(var_key, clamp_min=None, clamp_max=None):
            if var is not None:
                self.plasma[f"aL{var_key}"][: var.shape[0], :] = var[:, :]
            aLT_withZero = self.plasma[f"aL{var_key}"]
            _, varN = profile_constructor_choice[var_key](
                self.plasma["roa"], aLT_withZero)
            self.plasma[var_key] = varN.clamp(min=clamp_min, max=clamp_max) if ( (clamp_min is not None) or (clamp_max is not None) ) else varN
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
        if name == "ne":
            ne_0orig, ni_0orig = self.plasma["ne"].clone(), self.plasma["ni"].clone()
        
        if name == "w0":
            clamp_min = clamp_max = None
        else:
            clamp_min, clamp_max = 0, 200

        # UPDATE *******************************************************
        aLT_withZero = _update_plasma_var(name, clamp_min=clamp_min, clamp_max=clamp_max)
        # **************************************************************

        # Postprocessing (some require special treatment)
        if name == "ne":

            '''
            If ne is updated, then ni must be updated as well, but keeping the thermal ion concentrations constant
            '''
            if self.scaleIonDensities:

                # Keep the thermal ion concentrations constant, scale their densities
                self.plasma["ni"] = ni_0orig.clone()
                for i in range(self.plasma["ni"].shape[-1]):
                    self.plasma["ni"][..., i] = self.plasma["ne"] * (ni_0orig[..., i] / ne_0orig)

        if name == "nZ":

            '''
            If nZ is updated, change its position in the ions set
            '''
            self.plasma["ni"][..., self.impurityPosition] = self.plasma["nZ"]

        return aLT_withZero

    # ------------------------------------------------------------------
    # Toolset for calculation
    # ------------------------------------------------------------------

    def calculateProfileFunctions(self, calculateRotationQuantities=True, mref=2.01355):
        """
        Update the normalizations of the current state
        Notes:
            - By default, mref is the Deuterium mass, to be consistent with TGYRO always using this regardless of the first ion
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
            mref,
            self.plasma["B_unit"],
            self.plasma["a"].unsqueeze(-1),
        )

        # Collisionality
        self.plasma["nuei"] = PLASMAtools.xnue(self.plasma["te"], self.plasma["ne"] * 1e-1, self.plasma["a"].unsqueeze(-1), mref)

        # Gyro-radius
        self.plasma["rho_s"] = PLASMAtools.rho_s(self.plasma["te"], mref, self.plasma["B_unit"])
        self.plasma["c_s"] = PLASMAtools.c_s(self.plasma["te"], mref)

        # Other
        self.plasma["tite"] = self.plasma["ti"] / self.plasma["te"]
        self.plasma["fZ"] = self.plasma["nZ"] / self.plasma["ne"]
        self.plasma["beta_e"] = PLASMAtools.betae(self.plasma["te"], self.plasma["ne"] * 1e-1, self.plasma["B_unit"])
        
        aLni = [self.plasma[f"aLni{i}"] for i in range(self.plasma["ni"].shape[-1])]
        
        self.plasma["p_prime"] = PLASMAtools.p_prime(
                                    self.plasma["te"], self.plasma["ne"] * 1e-1, self.plasma["aLte"], self.plasma["aLne"],
                                    self.plasma["ti"], self.plasma["ni"] * 1e-1, self.plasma["aLti"], aLni,
                                    self.plasma["a"].unsqueeze(-1), self.plasma["B_unit"], self.plasma["q"], self.plasma["roa"]*self.plasma["a"].unsqueeze(-1))

        """
		Rotation stuff
		--------------
			Reason why I not always calculate this is because here I'm combining quantities derived (w0,cs)
			which can be accessed in a finer grid, with aLw0 which is a primitive one, that works in the coarse
			grid. Therefore, aLw0_n can only be calculated on the coarse grid.
		"""
        if calculateRotationQuantities:
            self.plasma["w0_n"] = self.plasma["w0"] / self.plasma["c_s"]
            self.plasma["aLw0_n"] = (self.plasma["aLw0"] * self.plasma["w0"] / self.plasma["c_s"])  # aLw0 * w0 = -a*dw0/dr; then aLw0_n = -dw0/dr * a/c_s

    def calculateTargets(self, relative_error_assumed=1.0):
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
            relative_error_assumed=relative_error_assumed,
            useConvectiveFluxes=self.useConvectiveFluxes,
            forceZeroParticleFlux=self.TransportOptions["ModelOptions"].get("forceZeroParticleFlux", False))

    def calculateTransport(
        self, nameRun="test", folder="~/scratch/", evaluation_number=0):
        """
        Update the transport of the current state.
        """
        folder = IOtools.expandPath(folder)

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

    def calculateMetrics(self):
        """
        Calculation of residual transport, dP/dt, ignoring the first (a zero)
        Define convergence metric of current state
        """

        # All fluxes in a single vector
        def _concatenate_flux(plasma, profile_key, flux_key):
            plasma["P"] = torch.cat((plasma["P"], plasma[profile_key][:, 1:]), dim=1).to(plasma["P"].device)
            plasma["P_tr"] = torch.cat((plasma["P_tr"], plasma[flux_key][:, 1:]), dim=1).to(plasma["P"].device)

        self.plasma["P"], self.plasma["P_tr"] = torch.Tensor().to(self.plasma["QeMWm2"]), torch.Tensor().to(self.plasma["QeMWm2"])

        for profile in self.ProfilesPredicted:
            _concatenate_flux(self.plasma, *self.profile_map[profile])
            
        self.plasma["S"] = self.plasma["P"] - self.plasma["P_tr"]
        self.plasma["residual"] = self.plasma["S"].abs().mean(axis=1, keepdim=True)

    def volume_integrate(self, var, force_dim=None):
        """
        If var in MW/m^3, this gives as output the MW/m^2 profile
        """

        if force_dim is None:
            return CALCtools.volume_integration(
                var, self.plasma["rmin"], self.plasma["volp"]
                ) / self.plasma["volp"]
        else:
            return CALCtools.volume_integration(
                var, self.plasma["rmin"][0,:].repeat(force_dim,1), self.plasma["volp"][0,:].repeat(force_dim,1)
                ) / self.plasma["volp"][0,:].repeat(force_dim,1)            

def add_axes_powerstate_plot(figMain, num_kp=3):

    numbers = [str(i) for i in range(4 * num_kp)]
    mosaic = []
    for row in range(4):
        first_cell = "A" if row < 2 else "B"        
        row_list = [first_cell]
        for col in range(num_kp):
            index = col * 4 + row
            row_list.append(numbers[index])
        
        mosaic.append(row_list)

    axsM = figMain.subplot_mosaic(mosaic)

    axs = []
    cont = 0
    for j in range(4):
        for i in range(num_kp):
            axs.append(axsM[f"{cont}"])
            cont += 1

    axsB = [axsM["A"], axsM["B"]]

    return axs, axsB

def read_saved_state(file):
    print(f"\t- Reading state file {IOtools.clipstr(file)}")
    state = IOtools.unpickle_mitim(file)

    return state
