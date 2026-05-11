"""
STATEedge.py
------------
Subclass of ``powerstate`` for edge / pedestal transport modelling.

Additional features vs core powerstate
---------------------------------------
- LCFS BC enforcement  : the reconstructed profile at the outermost radial point is pinned
                         to separatrix values computed by a boundary model (or user-provided)
                         after each modify() call.
- Boundary conditions  : two-point SOL/separatrix models (PeretSSF, SepOS, FixedInitial)
                         that compute n, T at the LCFS and their gradient scale lengths.
- Charge states        : Aurora-based steady-state charge-state distribution, radiation,
                         and neutral-source terms.

New keys consumed from ``evolution_options``
--------------------------------------------
domain_roa           : float | (float, float) or None
    Restrict the plasma radial grid to roa in [roa_min, 1]. Takes priority
    over domain_rho if both are provided. If a tuple is given, only the lower
    bound is used by ``powerstate_edge``.

domain_rho           : float | (float, float) or None
    Same restriction expressed in sqrt(toroidal flux) rho. If a tuple is given,
    only the lower bound is used by ``powerstate_edge``.

lcfs_bc              : dict or None
    Initial LCFS values overriding the gacode values.  If ``bc_model`` is set,
    the model output will replace this dict on every ``calculateBoundaryConditions``
    call.  Keys must match powerstate.plasma conventions (keV for te/ti, 1e19 m⁻³
    for ne, etc.).

bc_model             : str, default ``"FixedInitial"``
    Name of the boundary-condition model.  One of:
      "FixedInitial"              — freeze LCFS values at their initial gacode values
      "TwoFluid_PeretSSF"         — two-point model with Peret 2025 SSF scale lengths
      "TwoFluid_EichManz"         — two-point model with Eich–Manz SepOS scale lengths

bc_model_options     : dict, default {}
    Options forwarded to the BC model constructor (Zeff, Lpar, fmom, …).

charge_state_model   : str, default ``"Null"``
    Name of the charge-state solver:
      "Null"   — no-op zeros (for core-only testing)
      "Aurora" — Aurora-based steady-state solver

charge_state_model_options : dict, default {}
    Options forwarded to the charge-state model constructor (imp, D_z_cm2_s, …).
"""

import copy
import datetime
import shutil
import numpy as np
import torch
from types import MethodType
from mitim_modules.powertorch.STATEtools import powerstate
from mitim_modules.powertorch.utils import TRANSFORMedge, TRANSPORTtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.powertorch.physics_models import parameterizers as _parameterizers_mod, targets_analytic_edge
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.opt_tools.optimizers import multivariate_tools
from mitim_tools.misc_tools import PLASMAtools, IOtools
from IPython import embed
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function
import scipy as sp

class powerstate_edge(powerstate):

    # Edge-specific keys popped from evolution_options, with their defaults.
    # Shared by __init__ (pop) and _bind_edge_attrs (get) so both paths stay
    # in sync automatically.
    _EDGE_KEYS_DEFAULTS = [
        ("domain_roa",                  None),
        ("domain_rho",                  None),
        ("lcfs_bc",                     {}),
        ("bc_model",                    "FixedInitial"),
        ("bc_model_options",            {}),
        ("charge_state_model",          "Null"),
        ("charge_state_model_options",  {}),
        ("neutral_model",               "Null"),
        ("neutral_model_options",       {}),
        ("elm_model",                   "Null"),
        ("elm_model_options",           {}),
        ("defined_on",                  "y"),
        ("use_edge_targets",            True),
    ]

    def __init__(
        self,
        profiles_object,
        increase_profile_resol=True,
        evolution_options=None,
        transport_options=None,
        target_options=None,
        tensor_options=None,
        parameterizer_options=None,
    ):
        '''
        Inputs:
            - profiles_object: Object for gacode_state or others
            - evolution_options:
                - rhoPredicted: radial grid (MUST NOT CONTAIN ZERO, it will be added internally)
                - predicted_channels: list of profiles to predict
                - impurityPosition: int = position of the impurity in the ions set
            - transport_options: dictionary with transport_evaluator and transport_evaluator_options
            - target_options: dictionary with target_evaluator and target_evaluator_options
        '''

        if evolution_options is None:
            evolution_options = {}
            
        if transport_options is None:
            transport_options = {
            "evaluator": None,
            "options": {}
            }

        transport_options.setdefault("evaluator_instance_attributes", {})
        transport_options.setdefault("cold_start", False)

        # Target options defaults --------------------
        if target_options is None:
            target_options = {}
        if "options" not in target_options:
            target_options["options"] = {}

        target_options.setdefault("evaluator", targets_analytic_edge.analytical_model_edge)
        target_options["options"].setdefault("target_evaluator_method", "powerstate_edge")
        target_options["options"].setdefault("targets_evolve", ["qie", "qrad", "qfus"])
        target_options["options"].setdefault("force_zero_particle_flux", False)
        target_options["options"].setdefault("percent_error", 1.0)
        # ---------------------------------------------

        if tensor_options is None:
            tensor_options = {
                "dtype": torch.double,
                "device": torch.device("cpu"),
            }

        # Pop edge-specific keys
        # attribute assignment to _bind_edge_attrs so the upgrade path stays in sync.
        _edge_opts = {k: evolution_options.pop(k, d) for k, d in self._EDGE_KEYS_DEFAULTS}
        self._bind_edge_attrs(_edge_opts)

        # -------------------------------------------------------------------------------------
        # Check inputs
        # -------------------------------------------------------------------------------------

        print('>> Creating powerstate object...')

        self.transport_options = transport_options
        self.target_options = target_options
        self.targets_resolution = target_options["options"].get("targets_resolution", None)

        # Default options
        self.predicted_channels = evolution_options.get("ProfilePredicted", ["te", "ti", "ne"])
        self.impurityPosition = evolution_options.get("impurityPosition", 1)
        self.impurityPosition_transport = copy.deepcopy(self.impurityPosition)
        self.scaleIonDensities = evolution_options.get("scaleIonDensities", True)
        self.fImp_orig = evolution_options.get("fImp_orig", 1.0)
        rho_vec = evolution_options.get("rhoPredicted", [0.2, 0.4, 0.6, 0.8])
        roa_vec = np.interp(rho_vec, profiles_object.profiles["rho(-)"], profiles_object.derived["roa"])

        if rho_vec[0] == 0:
            raise ValueError("[MITIM] The radial grid must not contain the initial zero")
        
        if _edge_opts.get("use_edge_targets", True):
            self.target_options['evaluator'] = targets_analytic_edge.analytical_model_edge
            if self._neu_model_name != "Null" and "qpar" not in self.target_options["options"]["targets_evolve"]:
                self.target_options["options"]["targets_evolve"] += ["qpar"]
                self.target_options["options"]["targets_evolve"] += ["qiz"]
        else:
            # Keep legacy analytical target physics
            self.target_options["evaluator"] = targets_analytic_edge.analytical_model_legacy_edge_compat

        # Ensure that nZ is always after ne, because of how the scaling of ni rules are imposed
        def _ensure_ne_before_nz(lst):
            if "ne" in lst and "nZ" in lst:
                ne_index = lst.index("ne")
                nz_index = lst.index("nZ")
                if ne_index > nz_index:
                    # Swap "ne" and "nZ" positions
                    lst[ne_index], lst[nz_index] = lst[nz_index], lst[ne_index]
            return lst
        self.predicted_channels = _ensure_ne_before_nz(self.predicted_channels)

        # Build parameterizer options after channels and radial controls are known.
        parameterizer_name = evolution_options.get("parameterizer", "spline")
        parameterizer_options_input = evolution_options.get(
            "parameterizer_options", parameterizer_options
        )
        parameterizer_options_final = dict(parameterizer_options_input or {})
        parameterizer_options_final.setdefault(
            "predicted_profiles", self.predicted_channels
        )
        parameterizer_options_final.setdefault("knots", roa_vec)
        parameterizer_options_final.setdefault("include_zero_grad_on_axis", True)
        parameterizer_options_final.setdefault("sigma", 0.05)
        parameterizer_options_final.setdefault("bounds", None)
        parameterizer_options_final.setdefault("defined_on", self.defined_on)
        self._bind_parameterizer(parameterizer_name, parameterizer_options_final)

        # Default type and device tensor
        self.dfT = torch.randn((2, 2), **tensor_options)

        '''
        Potential profiles to evolve (aLX) and their corresponding flux matching
        ------------------------------------------------------------------------
            The order in the P and P_tr (and therefore the source S)
            tensors will be the same as in self.predicted_channels
        '''
        self.profile_map = {
            "te": ("QeMWm2", "QeMWm2_tr"),
            "ti": ("QiMWm2", "QiMWm2_tr"),
            "ne": ("Ce", "Ce_tr"),
            "nZ": ("CZ", "CZ_tr"),
            "w0": ("MtJm2", "MtJm2_tr")
        }

        self.labelsFluxes = {
            "te": "$Q_e$ ($MW/m^2$)",
            "ti": "$Q_i$ ($MW/m^2$)",
            "ne": "$Q_{conv}$ ($MW/m^2$)",
            "nZ": "$Q_{conv}$ $\\cdot f_{Z,0}$ ($MW/m^2$)",
            "w0": "$M_T$ ($J/m^2$)",
        }

        # Store control points

        self.rhoCP = rho_vec.to(self.dfT) if isinstance(rho_vec, torch.Tensor) else torch.from_numpy(rho_vec).to(self.dfT)

        # -------------------------------------------------------------------------------------
        # Populate plasma with radial grid
        # -------------------------------------------------------------------------------------

        # Refine grid
        self.plasma = {
            'rho' : torch.linspace(rho_vec[0]-0.01,torch.ones(1)[0],torch.ceil((1.0-rho_vec[0])/0.005).int()).to(self.dfT)
        }

        # Have the posibility of storing profiles in the class
        self.profiles_stored = []
        self.FluxMatch_Xopt, self.FluxMatch_Yopt = torch.Tensor().to(
            self.dfT
        ), torch.Tensor().to(self.dfT)

        self.labelsFM = []
        for profile in self.predicted_channels:
            self.labelsFM.append([f'aL{profile}', list(self.profile_map[profile])[0], list(self.profile_map[profile])[1]])

        # -------------------------------------------------------------------------------------
        # Object type (e.g. input.gacode)
        # -------------------------------------------------------------------------------------

        if isinstance(profiles_object, PROFILEStools.gacode_state):
            self.to_powerstate = TRANSFORMedge.gacode_to_powerstate
            self.from_powerstate = MethodType(TRANSFORMedge.to_gacode, self)

            # Use a copy because I'm deriving, it may be expensive and I don't want to carry that out outside of this class
            self.profiles = copy.deepcopy(profiles_object)
            if "derived" not in self.profiles.__dict__:
                self.profiles.derive_quantities()

        else:
            raise ValueError("[MITIM] The input profile object is not recognized, please use gacode_state")

        # -------------------------------------------------------------------------------------
        # Standard creation of plasma dictionary
        # -------------------------------------------------------------------------------------

        if increase_profile_resol:
            TRANSFORMedge.improve_resolution_profiles(self.profiles, rho_vec)

        # Convert to powerstate
        self.to_powerstate(self)

        # Convert into a batch so that always the quantities are (batch,dimX)
        self.batch_size = 0
        self._repeat_tensors(batch_size=1)

        self.Xcurrent = None

    def _bind_edge_attrs(self, opts: dict):
        """
        Assign all edge-specific instance attributes from a plain dict.

        Parameters
        ----------
        opts : dict
            Keys and defaults are defined by ``_EDGE_KEYS_DEFAULTS``; missing
            keys fall back to those defaults.
        """
        self._domain_roa           = opts.get("domain_roa", None)
        self._domain_rho           = opts.get("domain_rho", None)
        self._lcfs_bc              = opts.get("lcfs_bc", {}) or {}
        self._bc_model_name        = opts.get("bc_model", "FixedInitial")
        self._bc_model_options     = opts.get("bc_model_options", {})
        self._cs_model_name        = opts.get("charge_state_model", "Null")
        self._cs_model_options     = opts.get("charge_state_model_options", {})
        self._neu_model_name       = opts.get("neutral_model", "Null")
        self._neu_model_options    = opts.get("neutral_model_options", {})
        self._elm_model_name       = opts.get("elm_model", "Null")
        self._elm_model_options    = opts.get("elm_model_options", {})
        self._targets_scaled       = False
        self._bc_model_instance    = None
        self._cs_model_instance    = None
        self._neu_model_instance   = None
        self._elm_model_instance   = None
        self._channel_controls     = {}
        self._channel_models       = {}
        self.defined_on            = opts.get("defined_on", "y")
        self.bc_dict               = {k: [float(v), 1.0] for k, v in self._lcfs_bc.items()}
        self.bc_dict_batch         = None
        self.bc_tensors            = None

    def _bind_parameterizer(self, parameterizer_name, parameterizer_options):
        """Instantiate and bind the configured parameterizer model."""
        self._parameterizer_name = str(parameterizer_name or "spline")
        self._parameterizer_options = dict(parameterizer_options or {})

        if self._parameterizer_name == 'spline':
            if self._parameterizer_name == "akima":
                self._parameterizer_options.setdefault("spline_type", "akima")
            self.parameterizer = _parameterizers_mod.Spline(self._parameterizer_options)
        elif self._parameterizer_name == "mtanh":
            self.parameterizer = _parameterizers_mod.Mtanh(self._parameterizer_options)
        elif self._parameterizer_name in {"SplineMtanh", "spline_mtanh", "mtanh_spline", "MtanhSpline"}:
            self.parameterizer = _parameterizers_mod.SplineMtanh(self._parameterizer_options)
        else:
            raise ValueError(
                f"[powerstate_edge] Unknown parameterizer {self._parameterizer_name}"
            )


    # ------------------------------------------------------------------
    # Main tools
    # ------------------------------------------------------------------

    def _detach_tensors(self):
        
        # -------------------------------------------------------------------------------------
        # Detach plasma tensors
        # -------------------------------------------------------------------------------------

        self.plasma = {key: tensor.detach() if tensor.requires_grad else tensor for key, tensor in self.plasma.items()}

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
        if hasattr(self, 'Xcurrent') and self.Xcurrent is not None and isinstance(self.Xcurrent, torch.Tensor):
            self.Xcurrent = self.Xcurrent.cpu()
        if hasattr(self, 'FluxMatch_Yopt') and self.FluxMatch_Yopt is not None and isinstance(self.FluxMatch_Yopt, torch.Tensor):
            self.FluxMatch_Yopt = self.FluxMatch_Yopt.cpu()
        if hasattr(self, 'profiles'):
            self.profiles.toNumpyArrays()

    # ------------------------------------------------------------------
    # Override: modify()
    # ------------------------------------------------------------------

    def modify(self, X):
        """
        Edge-aware override of ``powerstate.modify()``.

        Supports control vectors for parameterizations whose active variables
        are not necessarily aLy-knotted splines (e.g. profile-value knots or
        mtanh global parameters).

                Accepted X formats
                ------------------
                - torch.Tensor with channel-concatenated active controls

        The active profile constructors already include the current LCFS BCs.
        """

        self.Xcurrent = X

        # Fast path: no updates requested, just rebuild profiles from current controls.
        if X is None:
            return


        batch_size = self.plasma["rho"].shape[0]
        ne_0orig, ni_0orig = self.plasma["ne"].clone(), self.plasma["ni"].clone()

        self.X_to_dict(X)
        use_tensor_bcs = isinstance(self.bc_tensors, dict) and len(self.bc_tensors) > 0

        if use_tensor_bcs:
            y, aLy, curv = self.parameterizer.update(
                self.X_dict,
                x_eval=self.plasma["roa"],
                bc_dict=self.bc_tensors,
            )
        else:
            y, aLy, curv = self.parameterizer.update(
                self.X_dict,
                x_eval=self.plasma["roa"],
                bc_dict=self.bc_dict,
            )

        for ch in self.predicted_channels:
            self.plasma[f"aL{ch}"] = torch.tensor(aLy[ch]).to(self.dfT)
            self.plasma[ch] = torch.tensor(y[ch]).to(self.dfT)
            self.plasma[f"curvature_{ch}"] = torch.tensor(curv[ch]).to(self.dfT)

            if ch == 'nZ':
                self.plasma["ni"][..., self.impurityPosition] = self.plasma["nZ"]


        if self.scaleIonDensities:

            # Keep thermal ion concentrations constant for non-impurity species.
            # If nZ is a predicted channel, impurity ni must follow nZ and should not
            # be overwritten by the ne-based rescaling below.
            for i in range(self.plasma["ni"].shape[-1]):
                if ("nZ" in self.predicted_channels) and (i == self.impurityPosition):
                    continue
                ratio = torch.where(
                    torch.abs(ne_0orig) > 1e-30,
                    ni_0orig[..., i] / ne_0orig,
                    torch.zeros_like(ne_0orig),
                )
                self.plasma["ni"][..., i] = torch.nan_to_num(
                    self.plasma["ne"] * ratio,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(min=0.0)

            if "nZ" in self.predicted_channels:
                self.plasma["ni"][..., self.impurityPosition] = torch.nan_to_num(
                    self.plasma["nZ"],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(min=0.0)

        # Legacy edge behavior: if nZ is not evolved directly, the tracked fully
        # ionized impurity is ni[..., impurityPosition]. Keep nZ mirrored to it
        # for compatibility with downstream consumers expecting plasma["nZ"].
        if (
            ("nZ" not in self.predicted_channels)
            and ("nZ" in self.plasma)
            and ("ni" in self.plasma)
            and (self.impurityPosition < self.plasma["ni"].shape[-1])
        ):
            self.plasma["nZ"] = torch.nan_to_num(
                self.plasma["ni"][..., self.impurityPosition],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(min=0.0)


        # Keep density scale lengths consistent with updated ni / nZ profiles and BCs.
        self._refresh_density_scale_lengths()

    def _refresh_density_scale_lengths(self):
        """
        Recompute density-gradient scale lengths from the current plasma state.

        Updates ``aLni*`` (for each ion species key present in ``self.plasma``)
        and ``aLnZ`` (fully-ionized impurity) when their corresponding density
        fields are available.
        """
        if "roa" not in self.plasma:
            return

        roa_np = self.plasma["roa"].detach().cpu().numpy()

        # Keep aLnZ synced with the tracked fully-ionized impurity channel.
        # Prefer ni[..., impurityPosition] and fall back to nZ for compatibility.
        tracked_imp = None
        if (
            "ni" in self.plasma
            and self.impurityPosition < self.plasma["ni"].shape[-1]
        ):
            tracked_imp = self.plasma["ni"][..., self.impurityPosition]
        elif "nZ" in self.plasma:
            tracked_imp = self.plasma["nZ"]

        if "aLnZ" in self.plasma and tracked_imp is not None:
            nz_np = torch.nan_to_num(
                tracked_imp,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(min=1e-30).detach().cpu().numpy()

            aLnZ_np = np.zeros_like(nz_np)
            for b in range(nz_np.shape[0]):
                aLnZ_np[b, :] = -np.gradient(np.log(nz_np[b, :]), roa_np[b, :])

            self.plasma["aLnZ"] = torch.nan_to_num(
                torch.from_numpy(aLnZ_np).to(self.dfT),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        # Keep ion scale lengths consistent with the updated ni profiles.
        if "ni" in self.plasma:
            ni_np = torch.nan_to_num(
                self.plasma["ni"],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(min=1e-30).detach().cpu().numpy()

            for j in range(self.plasma["ni"].shape[-1]):
                key = f"aLni{j}"
                if key not in self.plasma:
                    continue
                aLni_j = np.zeros_like(ni_np[..., j])
                for b in range(ni_np.shape[0]):
                    aLni_j[b, :] = -np.gradient(np.log(ni_np[b, :, j]), roa_np[b, :])
                self.plasma[key] = torch.nan_to_num(
                    torch.from_numpy(aLni_j).to(self.dfT),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )


    def update_var(self, name, var=None, specific_profile_constructor=None):
        """
        Update one channel in the active control vector and rebuild profiles.

        Returns
        -------
        torch.Tensor
            The channel control slice after update (batched).
        """
        if name not in self.predicted_channels:
            raise ValueError(f"[powerstate_edge] Channel '{name}' is not in predicted_channels")

        if specific_profile_constructor is not None:
            raise ValueError(
                "[powerstate_edge] specific_profile_constructor is no longer supported in update_var; "
                "use the parameterizer-driven path via modify()."
            )

        widths = self.parameterizer.n_params_per_profile
        if np.isscalar(widths):
            widths = [int(widths)] * len(self.predicted_channels)
        elif len(widths) == 1 and len(self.predicted_channels) > 1:
            widths = list(widths) * len(self.predicted_channels)
        else:
            widths = [int(w) for w in widths]

        if len(widths) != len(self.predicted_channels):
            raise ValueError("[powerstate_edge] Invalid parameterizer width metadata")

        offsets = [0]
        for w in widths:
            offsets.append(offsets[-1] + w)

        ch_idx = self.predicted_channels.index(name)
        i0, i1 = offsets[ch_idx], offsets[ch_idx + 1]
        n_total = offsets[-1]

        batch = self.plasma["rho"].shape[0]

        if isinstance(self.Xcurrent, torch.Tensor) and self.Xcurrent.shape == (batch, n_total):
            X_next = self.Xcurrent.detach().clone().to(self.dfT)
        else:
            X_next = torch.zeros((batch, n_total)).to(self.dfT)
            if var is None:
                raise ValueError(
                    f"[powerstate_edge] No active control vector available for update_var('{name}'); "
                    "provide var or call modify(X) first."
                )

        if var is not None:
            var_t = var if isinstance(var, torch.Tensor) else torch.as_tensor(var)
            var_t = var_t.to(self.dfT)
            if var_t.ndim == 1:
                var_t = var_t.unsqueeze(0)
            if var_t.shape[1] != (i1 - i0):
                raise ValueError(
                    f"[powerstate_edge] update_var('{name}') expected {i1 - i0} controls, got {var_t.shape[1]}"
                )
            X_next[: var_t.shape[0], i0:i1] = var_t

        self.modify(X_next)
        return X_next[:, i0:i1]
    

    def flux_match(self, algorithm="root", solver_options=None, bounds=None, debugYN=False):
        self.FluxMatch_plasma_orig = copy.deepcopy(self._slice_plasma_to_rhoCP())
        self.bounds_current = bounds

        print(f'\t- Flux matching of powerstate file ({self.plasma["rho"].shape[0]} parallel batches of {len(self.rhoCP)} radii) has been requested...')
        print("**********************************************************************************************")
        timeBeginning = datetime.datetime.now()

        if algorithm == "root":
            solver_fun = multivariate_tools.scipy_root
        elif algorithm == "simple_relax":
            solver_fun = multivariate_tools.simple_relaxation
        else:
            raise ValueError(f"[MITIM] Algorithm {algorithm} not recognized")
    
        if solver_options is None:
            solver_options = {}

        solver_options_use = dict(solver_options)

        folder_main = solver_options_use.get("folder", None)
        namingConvention = solver_options_use.get("namingConvention", "powerstate_sr_ev")

        eval_counter = {"cont": 0}
        def evaluator(X, y_history=None, x_history=None, metric_history=None):
            cont = eval_counter["cont"]
            nameRun = f"{namingConvention}_{cont}"

            if folder_main is not None:
                folder = IOtools.expandPath(folder_main) /  f"{namingConvention}_{cont}"
                if issubclass(self.transport_options["evaluator"], TRANSPORTtools.power_transport):
                    (folder / "transport_simulation_folder").mkdir(parents=True, exist_ok=True)

            # ***************************************************************************************************************
            # Calculate
            # ***************************************************************************************************************

            folder_run = folder / "transport_simulation_folder" if folder_main is not None else IOtools.expandPath('~/scratch/')
            QTransport, QTarget, _, _ = self.calculate(X, nameRun=nameRun, folder=folder_run, evaluation_number=cont)

            eval_counter["cont"] = cont + 1

            # Save state so that I can check initializations
            if folder_main is not None:
                if issubclass(self.transport_options["evaluator"], TRANSPORTtools.power_transport):
                    self.save(folder / "powerstate.pkl")
                    shutil.copy2(folder_run / "input.gacode", folder)

            # ***************************************************************************************************************
            # Postprocess
            # ***************************************************************************************************************

            # Residual is the difference between the target and the transport
            yRes = (QTarget - QTransport).abs()
            
            # Metric is the mean of the absolute value of the residual
            yMetric = -yRes.mean(axis=-1).detach()

            # Store values
            if y_history is not None:      
                y_history.append(yRes.detach())
            if x_history is not None:      
                x_history.append(self.Xcurrent.detach())
            if metric_history is not None: 
                metric_history.append(yMetric)

            return QTransport, QTarget, yMetric

        # Initialize optimization controls. For generalized parameterizers,
        # Xcurrent already carries the active control vector.
        if isinstance(self.Xcurrent, torch.Tensor) and self.Xcurrent.numel() > 0:
            self.modify(self.Xcurrent)  # Ensure profiles are consistent with Xcurrent
            x0 = self.Xcurrent.detach().clone()
            if x0.ndim == 1:
                x0 = x0.unsqueeze(0)
        else:
            # Backward-compatible fallback for legacy aL-profile controls.
            x0 = torch.Tensor().to(self.plasma["aLte"])
            for _, i in enumerate(self.predicted_channels):
                x0 = torch.cat(
                    (x0, self._interp_tensor_from_rho_to_rhoCP(self.plasma[f"aL{i}"]).detach()),
                    dim=1,
                )

            # Make sure is properly batched
            x0 = x0.view((
                self.plasma["rho"].shape[0],
                len(self.rhoCP) * len(self.predicted_channels),
            ))

        # Optimize
        x_best,Yopt, Xopt, metric_history = solver_fun(evaluator,x0, bounds=self.bounds_current,solver_options=solver_options_use)

        # For simplicity, return the trajectory of only the best candidate

        idx_flat = metric_history.argmax()
        index_best = divmod(idx_flat.item(), metric_history.shape[1])
        
        self.FluxMatch_Yopt, self.FluxMatch_Xopt = Yopt[:,index_best[1],:], Xopt[:,index_best[1],:]

        print("**********************************************************************************************")
        print(f"\t- Flux matching of powerstate finished, and took {IOtools.getTimeDifference(timeBeginning)}\n")

        if debugYN:
            self.plot()
            embed()
            
    # ------------------------------------------------------------------
    # Override: calculateProfileFunctions()
    # ------------------------------------------------------------------

    def calculateProfileFunctions(self, calculateRotationQuantities=True, **kwargs):
        """
        Extend base ``calculateProfileFunctions`` to add the diamagnetic
        toroidal rotation to ``w0`` before the rotation-derived quantities
        (``w0_n``, ``aLw0_n``) are computed.

        The base class computes ``p_prime`` (needed here) only during its
        first-pass execution, so we call it with
        ``calculateRotationQuantities=False`` first, compute the diamagnetic
        contribution, and then calculate the rotation normalizations by hand.

        Also computes ``plasma["E_rad"]`` — the normalized radial electric field
        in NEO's ``DPHI0DR`` convention for ``ROTATION_MODEL=1``.

        At the pedestal/edge, the diamagnetic flow dominates the toroidal ExB
        rotation, so `Er` is approximated by the radial force balance::

            Er [V/m] = -dp/dr / (Z_i * e * n_i)

        The pressure gradient `dp/dr [Pa/m]` is recovered from the normalised
        ``p_prime`` already computed in the base-class pass::

            p_prime = 1e-7 * q * a² / r / B_unit² * dp/dr
            → dp/dr = p_prime * B_unit² * roa * 1e7 / (q * a)

        NEO's ``DPHI0DR = d(phi0)/dr * a * e / Te[J]``.  Since
        ``Er = -d(phi0)/dr`` and ``Te[J] = te[keV]*1e3*e_SI``, e_SI cancels::

            DPHI0DR = -Er * a / (te[keV] * 1e3)

        The result is stored as ``plasma["E_rad"]`` and injected per-rho by
        ``transport_neo.evaluate_neoclassical()`` when the ``"edge"`` NEO model
        (``ROTATION_MODEL=1``) is active.
        """
        from scipy.constants import e as _q_e
        from scipy.interpolate import CubicSpline

        # First pass: everything except rotation-normalised quantities
        super().calculateProfileFunctions(calculateRotationQuantities=False, **kwargs)

        p = self.plasma

        # Diamagnetic toroidal rotation
        #   \omega_{dia} [rad/s] = -p' / (Z_D * e * n_D * B_unit)
        # where p' is the pressure gradient in SI already encoded in
        # plasma["p_prime"] and the remaining factors convert units.
        B_unit  = p["B_unit"]                              # (batch, rho)  T
        p_prime = p["p_prime"]                             # (batch, rho)  normalised
        a       = p["a"].unsqueeze(-1)                     # (batch, 1)    m
        n_main  = p["ni"][:, :, 0].clamp(min=1e-30)       # (batch, rho)  1e19 m⁻³
        n_main_m3 = n_main * 1e19                          # → m⁻³

        batch = p["roa"].shape[0]

        # Interpolate geometric factors to the active plasma grid and compute
        # dpidr from the batched main-ion pressure profile.
        if getattr(p,'B_p', None) is None and getattr(p,'R', None) is None:
            rho_source_key = "rho(-)" if "rho(-)" in self.profiles.profiles else "rho"
            rho_source = np.asarray(self.profiles.profiles[rho_source_key])
            rho_target = p["roa"][0, :].detach().cpu().numpy()

            R0 = float(np.asarray(self.profiles.profiles["rcentr(m)"]).reshape(-1)[0])
            rmin = np.asarray(self.profiles.profiles["rmin(m)"])
            R_source = R0 + rmin
            polflux = np.asarray(self.profiles.profiles["polflux(Wb/radian)"])
            Bp_source = np.gradient(polflux, rmin, edge_order=2) / np.maximum(R_source, 1e-10)

            R_interp = np.interp(rho_target, rho_source, R_source)
            Bp_interp = np.interp(rho_target, rho_source, Bp_source)

            R = torch.from_numpy(R_interp).to(self.dfT).unsqueeze(0).repeat(batch, 1)
            Bp = torch.from_numpy(Bp_interp).to(self.dfT).unsqueeze(0).repeat(batch, 1)
            p["R"] = R
            p["B_p"] = Bp
            p["B_T"] = float(np.asarray(self.profiles.profiles["bcentr(T)"]).reshape(-1)[0]) * R / R0  # Toroidal field from total field and geometry

        R  = p["R"]   # (batch, rho)
        Bp = p["B_p"] # (batch, rho)
        a = p['a'].view(-1)[0] # scalar, m

        # Main-ion pressure gradient on the radial coordinate r = a * roa.
        # p_i [Pa] = n_i [m^-3] * T_i [J] with T_i [J] = ti[keV] * 1e3 * e.
        pi_main_SI = n_main_m3 * p["ti"] * 1e3 * _q_e
        pi_np = pi_main_SI.detach().cpu().numpy()
        r_np = p['rmin'].detach().cpu().numpy()
        dpidr_np = np.zeros_like(pi_np)
        for b in range(batch):
            dpidr_np[b, :] = np.gradient(pi_np[b, :], r_np[b, :], edge_order=2)
        dpidr_SI = torch.from_numpy(dpidr_np).to(self.dfT)
        #p["dpidr_main_SI"] = dpidr_SI

        if "ions_set_Zi" in p:
            Zi = p["ions_set_Zi"]
            if Zi.dim() == 1:
                Zi = Zi.unsqueeze(0)
            Z_main = Zi[:, 0].unsqueeze(-1).clamp(min=1e-30)
        else:
            Z_main = torch.ones((batch, 1)).to(self.dfT)

        p['tau_norm'] = a / p['c_s']  # [m] / [m/s] = [s]
        w0_dia = -dpidr_SI / (Z_main * _q_e * n_main_m3 * R * Bp)

        Er = -R * Bp * p['w0']  # (batch, rho)  V/m
        p["E_rad"] = -Er * a / (p["te"] * 1e3)
        p['vexb'] = p["E_rad"] / p["B_T"]  # ExB velocity [m/s]

        p["w0"]     = w0_dia
        p["w0_n"]   = w0_dia / p["c_s"]
        p["aLw0_n"] = p["aLw0"] * w0_dia / p["c_s"]

        # GAMMA_E = r * d/dr(v_exb/r) [1/s], computed per batch element
        rmin_np    = p['rmin'].detach().cpu().numpy()   # (batch, rho)
        vexb_np    = p['vexb'].detach().cpu().numpy()   # (batch, rho)
        gamma_exb_np = np.zeros_like(rmin_np)
        for b in range(batch):
            r_b    = rmin_np[b, :]
            vexb_b = vexb_np[b, :]
            # Avoid division by zero at r=0; start spline from first positive radius
            i_start = 1 if r_b[0] == 0.0 else 0
            r_nz    = r_b[i_start:]
            spl     = CubicSpline(r_nz, vexb_b[i_start:] / r_nz, extrapolate=True)
            gamma_exb_np[b, i_start:] = r_nz * spl.derivative()(r_nz)
            if i_start > 0:
                gamma_exb_np[b, 0] = 0.0
        p["gamma_exb"] = torch.from_numpy(gamma_exb_np).to(self.dfT)  # [1/s]
        p['vexb_shear'] = p["gamma_exb"] * p['tau_norm']  # [-]

    # ------------------------------------------------------------------
    # Edge physics
    # ------------------------------------------------------------------

    def calculateBoundaryConditions(self):
        """
        Run the configured boundary-condition model for each batch element and
        update ``self.bc_dict`` with the resulting LCFS values.

        The BC model is instantiated lazily on the first call using the name and
        options stored at construction time (``bc_model`` key in evolution_options).

        After this method, ``self.bc_dict`` contains the current per-channel
        LCFS anchors and the active profile constructors are refreshed against them.
        """
        from mitim_tools.edge_tools.boundary import build_bc_model

        # Lazy instantiation of the BC model
        if self._bc_model_instance is None:
            self._bc_model_instance = build_bc_model(
                self._bc_model_name, self._bc_model_options
            )

        model = self._bc_model_instance
        batch  = self.plasma["te"].shape[0]

        # Solve for each batch element and keep distinct BC dictionaries.
        self.bc_dict_batch = []
        for b in range(batch):
            model.get_boundary_conditions(self, batch_idx=b)
            self.bc_dict_batch.append(copy.deepcopy(model.bc_dict))

        # Build tensorized BC representation: key -> {val, loc, mask} with shape (batch, n_bc_per_key).
        self.bc_tensors = {}
        if self.bc_dict_batch:
            keys = sorted({k for bcd in self.bc_dict_batch for k in bcd.keys()})
            for key in keys:
                vals = torch.zeros((batch, 1)).to(self.dfT)
                locs = torch.ones((batch, 1)).to(self.dfT)
                mask = torch.zeros((batch, 1), dtype=torch.bool, device=self.dfT.device)
                for b, bcd in enumerate(self.bc_dict_batch):
                    if key not in bcd:
                        continue
                    val_b, loc_b = bcd[key]
                    vals[b, 0] = float(val_b)
                    locs[b, 0] = float(loc_b)
                    mask[b, 0] = True
                self.bc_tensors[key] = {"val": vals, "loc": locs, "mask": mask}

        # Backward-compatibility scalar BC dict: keep first batch entry.
        if self.bc_dict_batch:
            self.bc_dict = copy.deepcopy(self.bc_dict_batch[0])
            for key, (val, _roa_loc) in self.bc_dict.items():
                self._lcfs_bc[key] = float(val)

    def calculateChargeStates(self):
        """
        Run the configured charge-state model for each batch element and store
        charge-state densities, neutral density, and Aurora radiation in
        ``self.plasma``.

        Populates
        ---------
        self.plasma['nz_all']        : (batch, rho, nZ+1)   [1e19 m⁻³]
        self.plasma['n0']            : (batch, rho)          [1e19 m⁻³]
        self.plasma['qrad_aurora']   : (batch, rho)          [W cm⁻³]

        These keys are then consulted by ``analytical_model_edge`` in
        ``calculateTargets()``.

        When ``charge_state_model`` is ``"Null"`` (the default), all arrays are
        set to zero and downstream edge-radiation terms are zero.
        """
        from mitim_tools.edge_tools.charge_states import build_charge_state_model

        if self._cs_model_instance is None:
            self._cs_model_instance = build_charge_state_model(
                self._cs_model_name, self._cs_model_options
            )

        model = self._cs_model_instance
        batch = self.plasma["te"].shape[0]
        for b in range(batch):
            model.solve(self, batch_idx=b)

        # Unpack charge-state distribution into tracked impurity and main-ion channels.
        # nz_all[..., -1] = fully ionized impurity; Σ_z z*nz_all gives impurity charge density.
        if "nz_all" in self.plasma and "ni" in self.plasma and "ne" in self.plasma:
            nz_all = self.plasma["nz_all"]  # (batch, rho, nZ+1)
            ni = self.plasma["ni"]
            ne = self.plasma["ne"]

            if self.impurityPosition < ni.shape[-1]:
                ni[..., self.impurityPosition] = torch.nan_to_num(
                    nz_all[..., -1],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(min=0.0)

            main_idx = int(getattr(model, "main_ion_species_index", 0))
            if 0 <= main_idx < ni.shape[-1]:
                Z_states = torch.arange(
                    nz_all.shape[-1], dtype=nz_all.dtype, device=nz_all.device
                )
                imp_charge = (nz_all * Z_states.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
                ni[..., main_idx] = torch.nan_to_num(
                    ne - imp_charge,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(min=0.0)

            # Keep nZ synchronized to the tracked fully ionized impurity channel.
            if "nZ" in self.plasma and self.impurityPosition < ni.shape[-1]:
                self.plasma["nZ"] = ni[..., self.impurityPosition]

        # Recompute Zeff from the final ni state (supports Zi as 1D or batched 2D).
        if (
            "ions_set_Zi" in self.plasma
            and "ne" in self.plasma
            and "ni" in self.plasma
        ):
            Zi = self.plasma["ions_set_Zi"]
            ne = self.plasma["ne"]
            ni = self.plasma["ni"]

            Zi_use = None
            if Zi.dim() == 1 and Zi.shape[0] == ni.shape[-1]:
                Zi_use = Zi.unsqueeze(0).expand(ni.shape[0], -1)
            elif Zi.dim() == 2 and Zi.shape[0] == ni.shape[0] and Zi.shape[1] == ni.shape[-1]:
                Zi_use = Zi

            if Zi_use is not None and ne.shape == ni[:, :, 0].shape:
                Zeff_computed = (ni * (Zi_use.unsqueeze(1) ** 2)).sum(dim=-1) / ne.clamp(min=1e-30)
                self.plasma["Zeff"] = torch.nan_to_num(
                    Zeff_computed,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

        # Keep impurity fraction consistent with updated impurity densities.
        if "ne" in self.plasma:
            if (
                "ni" in self.plasma
                and self.impurityPosition < self.plasma["ni"].shape[-1]
            ):
                nZ_track = self.plasma["ni"][..., self.impurityPosition]
            else:
                nZ_track = self.plasma.get("nZ", None)

            if nZ_track is not None:
                self.plasma["fZ"] = nZ_track / self.plasma["ne"].clamp(min=1e-30)
            

    def calculateNeutrals(self):
        """
        Run the configured main-ion neutral model for each batch element.

        Populates
        ---------
        self.plasma['n0']              : (batch, rho)  [1e19 m⁻³]
            Main-ion (D⁰) neutral density.
        self.plasma['S_ion_main']      : (batch, rho)  [1e19 m⁻³ s⁻¹]
            Volumetric ionisation source rate.
        self.plasma['nu_ioniz_main']   : (batch, rho)  [s⁻¹]
            ADAS-based ionisation frequency.
        self.plasma['tau_n0']          : (batch, rho)  [dimensionless]
            Optical depth integrated inward from the LCFS.

        These fields are consumed by ``_evaluate_particle_fluxes`` and
        ``_evaluate_ionization_loss`` inside ``calculateTargets()``.

        When ``neutral_model`` is ``"Null"`` (the default), all three arrays
        are set to zero and the corresponding physics channels are bypassed.
        """
        from mitim_tools.edge_tools.neutrals import build_neutral_model

        if self._neu_model_instance is None:
            self._neu_model_instance = build_neutral_model(
                self._neu_model_name, self._neu_model_options
            )

        model = self._neu_model_instance
        batch = self.plasma["te"].shape[0]
        for b in range(batch):
            model.solve(self, batch_idx=b)

    def calculateImpurities(self):
        """
        Alias for ``calculateChargeStates()`` — invoked after ``calculateNeutrals()``
        so that ``plasma['n0']`` is available for CXR in the Aurora namelist.
        """
        self.calculateChargeStates()

        # Legacy edge behavior: tracked fully-ionized impurity lives in
        # ni[..., impurityPosition] unless nZ is explicitly predicted.
        if (
            ("nZ" in self.plasma)
            and ("nZ" not in self.predicted_channels)
            and ("ni" in self.plasma)
            and (self.impurityPosition < self.plasma["ni"].shape[-1])
        ):
            self.plasma["nZ"] = torch.nan_to_num(
                self.plasma["ni"][..., self.impurityPosition],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(min=0.0)

        # Charge-state solvers can update ni / nZ; keep aLni* and aLnZ consistent.
        if str(self._cs_model_name).lower() not in ("null", "none"):
            self._refresh_density_scale_lengths()

    def calculateElm(self):
        """
        Evaluate the peeling-ballooning (ELM) stability criterion and, if the
        current pedestal profiles lie in the unstable region, artificially
        inflate the turbulent transport fluxes by a stiff penalty factor.

        The purpose of this step is to steer the transport solver *away* from
        ELM-territory: once the pressure gradient exceeds the ideal MHD
        peeling-ballooning boundary, enormous effective transport quickly erodes
        the excess gradient and returns the solution to the stable region.

        Model selection
        ---------------
        Set ``evolution_options['elm_model']`` to one of:

          ``'Null'``       — no-op; fluxes are unchanged (default).
          ``'AnalyticPB'`` — inline s-alpha peeling-ballooning criterion using
                            the Connor-Hastie first-stability boundary with
                            toroidicity and triangularity corrections.

        Pass model parameters via ``evolution_options['elm_model_options']``
        (see ``mitim_tools.edge_tools.elm.AnalyticPBElm`` for the full list).

        Turbulent flux keys inflated (if present in ``self.plasma``)
        -------------------------------------------------------------
        Per-species turbulent components (``*_tr_turb``) are scaled by the
        per-surface ``elm_factor`` tensor.  The combined total flux tensors
        (``*_tr`` = turbulent + neoclassical) and the convective particle
        fluxes (``Ce_tr``, ``CZ_tr``) are then recomputed for consistency.

        Diagnostic quantities stored in ``self.plasma``
        ------------------------------------------------
        ``plasma['elm_factor']``     : (batch, rho) — penalty multiplier (≥ 1).
        ``plasma['elm_alpha_MHD']``  : (batch, rho) — normalised pressure gradient.
        ``plasma['elm_alpha_crit']`` : (batch, rho) — stability boundary.
        ``plasma['elm_unstable']``   : (batch, rho) — float mask (1 = unstable).
        """
        from mitim_tools.edge_tools.elm import build_elm_model
        from mitim_tools.misc_tools.PLASMAtools import convective_flux

        # For EPED, default eped_folder to the active solver scratch folder
        # unless the user already provided one explicitly.
        if self._elm_model_name == "EPED":
            eped_folder = self._elm_model_options.get("eped_folder", None)
            if eped_folder is None:
                self._elm_model_options = copy.deepcopy(self._elm_model_options)
                self._elm_model_options["eped_folder"] = str(
                    getattr(self, "_solver_scratch_folder", IOtools.expandPath("~/scratch/"))
                )

        # Lazy instantiation
        if self._elm_model_instance is None:
            self._elm_model_instance = build_elm_model(
                self._elm_model_name, self._elm_model_options
            )

        model = self._elm_model_instance
        batch = self.plasma["te"].shape[0]
        n_rho = self.plasma["roa"].shape[-1]

        # Accumulate per-batch elm_factor; shape (batch, rho)
        elm_factor_batch = torch.ones(
            batch, n_rho,
            dtype=self.plasma["te"].dtype,
            device=self.plasma["te"].device,
        )
        alpha_MHD_batch  = torch.zeros_like(elm_factor_batch)
        alpha_crit_batch = torch.ones_like(elm_factor_batch)
        unstable_batch   = torch.zeros_like(elm_factor_batch)

        for b in range(batch):
            model.solve(self, batch_idx=b)
            elm_factor_batch[b, :]  = model.elm_factor.to(elm_factor_batch)
            alpha_MHD_batch[b, :]   = model.alpha_MHD.to(elm_factor_batch)
            alpha_crit_batch[b, :]  = model.alpha_crit.to(elm_factor_batch)
            unstable_batch[b, :]    = model.in_elm_region.to(elm_factor_batch)

        # Store diagnostics
        self.plasma["elm_factor"]     = elm_factor_batch
        self.plasma["elm_alpha_MHD"]  = alpha_MHD_batch
        self.plasma["elm_alpha_crit"] = alpha_crit_batch
        self.plasma["elm_unstable"]   = unstable_batch

        # If model is Null or no surface is unstable, nothing to do
        if (elm_factor_batch == 1.0).all():
            return

        f = elm_factor_batch   # (batch, rho)

        # ── Scale turbulent heat/momentum flux components ─────────────────────
        # Keys that map directly to (batch, rho) arrays
        _heat_momentum_turb = [
            "QeMWm2_tr_turb",
            "QiMWm2_tr_turb",
            "MtJm2_tr_turb",
        ]
        for key in _heat_momentum_turb:
            if key in self.plasma:
                self.plasma[key] = self.plasma[key] * f

        # ── Scale turbulent particle flux components ──────────────────────────
        _particle_turb = [
            "Ge1E20m2_tr_turb",
            "GZ1E20m2_tr_turb",
        ]
        for key in _particle_turb:
            if key in self.plasma:
                self.plasma[key] = self.plasma[key] * f

        # ── Recompute total fluxes (turb + neoc) ──────────────────────────────
        _total_map = [
            ("QeMWm2_tr_turb",   "QeMWm2_tr_neoc",   "QeMWm2_tr"),
            ("QiMWm2_tr_turb",   "QiMWm2_tr_neoc",   "QiMWm2_tr"),
            ("Ge1E20m2_tr_turb", "Ge1E20m2_tr_neoc", "Ge1E20m2_tr"),
            ("GZ1E20m2_tr_turb", "GZ1E20m2_tr_neoc", "GZ1E20m2_tr"),
            ("MtJm2_tr_turb",    "MtJm2_tr_neoc",    "MtJm2_tr"),
        ]
        for turb_key, neoc_key, total_key in _total_map:
            if turb_key in self.plasma and neoc_key in self.plasma:
                self.plasma[total_key] = (
                    self.plasma[turb_key] + self.plasma[neoc_key]
                )

        # ── Recompute convective particle flux wrappers ───────────────────────
        # Ce_tr  = (3/2) * Te[J] * Ge1E20m2_tr [1e20/m^2/s]  (in MW/m^2)
        # CZ_tr  = same but for the impurity, normalised by fImp_orig
        if "Ge1E20m2_tr" in self.plasma:
            self.plasma["Ce_tr"] = convective_flux(
                self.plasma["te"], self.plasma["Ge1E20m2_tr"]
            )
        if "Ge1E20m2_tr_turb" in self.plasma:
            self.plasma["Ce_tr_turb"] = convective_flux(
                self.plasma["te"], self.plasma["Ge1E20m2_tr_turb"]
            )

        if "GZ1E20m2_tr" in self.plasma:
            self.plasma["CZ_tr"] = convective_flux(
                self.plasma["te"], self.plasma["GZ1E20m2_tr"]
            ) / self.fImp_orig
        if "GZ1E20m2_tr_turb" in self.plasma:
            self.plasma["CZ_tr_turb"] = convective_flux(
                self.plasma["te"], self.plasma["GZ1E20m2_tr_turb"]
            ) / self.fImp_orig

    def calculateTransport(
        self, nameRun="test", folder="~/scratch/", evaluation_number=0):
        """
        Edge override of ``calculateTransport``.

        A shallow proxy powerstate is constructed whose ``plasma`` dict has every
        rho-shaped tensor interpolated from the fine grid down to the ``rhoCP``
        control-point grid.  The transport evaluator therefore only runs at the
        predicted-rho locations.  After evaluation, coarse-grid results are lifted
        back to the fine grid via cubic-spline interpolation/extrapolation so that downstream steps
        (ELM model, plotting, etc.) remain on the original grid.
        """
        folder = IOtools.expandPath(folder)

        # Build a proxy powerstate backed by the rhoCP-downsampled plasma.
        proxy = copy.copy(self)
        proxy.plasma = self._slice_plasma_to_rhoCP(pad_zero=True)

        # Instantiate and configure the transport evaluator on the proxy.
        if self.transport_options["evaluator"] is None:
            transport = TRANSPORTtools.power_transport(
                proxy, name=nameRun, folder=folder, evaluation_number=evaluation_number
            )
        else:
            transport = self.transport_options["evaluator"](
                proxy, name=nameRun, folder=folder, evaluation_number=evaluation_number
            )

        for key in self.transport_options["evaluator_instance_attributes"]:
            setattr(transport, key, self.transport_options["evaluator_instance_attributes"][key])

        transport.produce_profiles()
        transport.evaluate()

        self._lift_proxy_plasma_to_fine(proxy.plasma, key_filter="_tr")

        self.model_results = transport.model_results

    # ------------------------------------------------------------------
    # rhoCP downsampling helpers
    # ------------------------------------------------------------------

    def _lift_proxy_plasma_to_fine(self, proxy_plasma, key_filter=None):
        """
        Lift rhoCP-grid tensors from ``proxy_plasma`` back to the fine grid and
        write the interpolated values into ``self.plasma``.

        Parameters
        ----------
        proxy_plasma : dict
            Plasma-like dictionary whose tensors live on rhoCP (shape ``n_cp``)
            or rhoCP-with-axis padding (shape ``n_cp + 1``).
        key_filter : str or None, default ``"_tr"``
            Only keys containing this substring are updated. If ``None`` or an
            empty string, all eligible keys are updated.
        """
        n_cp = len(self.rhoCP)
        rho_cp = self.rhoCP.detach().cpu().numpy()
        rho_fine = self.plasma["rho"][0, :].detach().cpu().numpy()

        for key, val in proxy_plasma.items():
            if key_filter and key_filter not in key:
                continue
            if not isinstance(val, torch.Tensor):
                continue

            if val.dim() == 2 and val.shape[1] in (n_cp, n_cp + 1):
                batch = val.shape[0]
                v_np = val.detach().cpu().numpy()
                has_prepended_zero = val.shape[1] == (n_cp + 1)
                lifted = np.stack(
                    [
                        np.interp(
                            rho_fine,
                            rho_cp,
                            v_np[b, 1:] if has_prepended_zero else v_np[b, :],
                        )
                        # interpolation_function(
                        #     rho_fine,
                        #     rho_cp,
                        #     v_np[b, 1:] if has_prepended_zero else v_np[b, :],
                        # )
                        for b in range(batch)
                    ]
                )
                self.plasma[key] = torch.from_numpy(lifted).to(self.dfT)

            elif val.dim() == 3 and val.shape[1] in (n_cp, n_cp + 1):
                batch, _, d = val.shape
                v_np = val.detach().cpu().numpy()
                has_prepended_zero = val.shape[1] == (n_cp + 1)
                vals = v_np[:, 1:, :] if has_prepended_zero else v_np
                lifted = np.stack(
                    [
                        np.stack(
                            [np.interp(rho_fine, rho_cp, vals[b, :, i]) for i in range(d)],
                            axis=-1,
                        )
                        for b in range(batch)
                    ]
                )
                self.plasma[key] = torch.from_numpy(lifted).to(self.dfT)

    def _interp_tensor_from_rho_to_rhoCP(self, tensor):
        """
        Interpolate a rho-shaped tensor from the fine plasma grid to ``rhoCP``.

        Supported tensor layouts:
        - ``(batch, rho)``
        - ``(batch, rho, n_species_or_states)``
        """
        n_fine = self.plasma["rho"].shape[-1]
        if (
            (not isinstance(tensor, torch.Tensor))
            or tensor.dim() < 2
            or tensor.shape[1] != n_fine
        ):
            return tensor

        rho_fine = self.plasma["rho"][0, :].detach().cpu().numpy()
        rho_cp = self.rhoCP.detach().cpu().numpy()

        if tensor.dim() == 2:
            batch = tensor.shape[0]
            arr = tensor.detach().cpu().numpy()
            interp = np.stack([np.interp(rho_cp, rho_fine, arr[b]) for b in range(batch)])
            return torch.from_numpy(interp).to(tensor)

        if tensor.dim() == 3:
            batch, _, n_extra = tensor.shape
            arr = tensor.detach().cpu().numpy()
            interp = np.stack([
                np.stack([np.interp(rho_cp, rho_fine, arr[b, :, i]) for i in range(n_extra)], axis=-1)
                for b in range(batch)
            ])
            return torch.from_numpy(interp).to(tensor)

        return tensor

    def _slice_plasma_to_rhoCP(self, plasma_dict=None,pad_zero=True):
        """
        Return a new dict with every ``(batch, rho_fine [, d])`` tensor
        interpolated to ``rhoCP``. Scalars and tensors whose second dimension
        does not match the fine-grid length are passed through unchanged.
        """
        if plasma_dict is None:
            plasma_dict = self.plasma
        n_fine = self.plasma["rho"].shape[-1]
        out = {}
        for key, val in plasma_dict.items():
            if isinstance(val, torch.Tensor):
                if val.dim() in (2, 3) and val.shape[1] == n_fine:
                    out[key] = self._interp_tensor_from_rho_to_rhoCP(val)
                else:
                    out[key] = val
            else:
                out[key] = val

            # optional prepend zero for consistent shape with transport evaluator expectations (batch, n_cp)
            if pad_zero and isinstance(out[key], torch.Tensor) and out[key].dim() == 2 and out[key].shape[1] == len(self.rhoCP):
                out[key] = torch.cat((torch.zeros_like(out[key][:, :1]), out[key]), dim=1)

        return out

    def calculateMetrics(self):
        """
        Edge override: build ``P`` and ``P_tr`` using only the ``rhoCP``
        indices so that the flux-matching residual operates on the
        predicted-rho grid rather than the full fine plasma grid.
        """
        self.plasma["P"]    = torch.Tensor().to(self.plasma["QeMWm2"])
        self.plasma["P_tr"] = torch.Tensor().to(self.plasma["QeMWm2"])

        for profile in self.predicted_channels:
            profile_key, flux_key = self.profile_map[profile]
            profile_cp = self._interp_tensor_from_rho_to_rhoCP(self.plasma[profile_key])
            flux_cp = self._interp_tensor_from_rho_to_rhoCP(self.plasma[flux_key])
            self.plasma["P"] = torch.cat(
                (self.plasma["P"], profile_cp), dim=1
            ).to(self.plasma["P"].device)
            self.plasma["P_tr"] = torch.cat(
                (self.plasma["P_tr"], flux_cp), dim=1
            ).to(self.plasma["P"].device)

        self.plasma["S"]        = self.plasma["P"] - self.plasma["P_tr"]
        self.plasma["residual"] = self.plasma["S"].abs().mean(axis=1, keepdim=True)

    def calculateTargets(self, relative_error_assumed=1.0):
        """
        Update the targets of the current state
        """
        from mitim_modules.powertorch.utils import TARGETStools

        if (self.target_options["evaluator"] is None) or (
            self.target_options["options"]["target_evaluator_method"] == "tgyro"
        ):
            targets = TARGETStools.power_targets(self)
        else:
            targets = self.target_options["evaluator"](self)

        # Store *_edgetargets for use instead of fixed targets 
        # Ex:
        # Ge -> density_to_flux(interpolation_function(self.profiles.profiles["qpar_beam"])), Qe -> self.profiles.derived["qe_aux_MW"] / volp,
        #  Qi -> self.profiles.derived["qi_aux_MW"] / volp, GZ -> 0, Mt -> self.profiles.derived["mt_Jmiller"] / volp
        # Scale by target_multipliers if given in options and targets_scaled = False
        
        opts = self.target_options["options"]
        targets_scaled      = opts.get("targets_scaled", False)
        target_multipliers  = opts.get("target_multipliers", {}) or {}

        rho_vec = self.plasma['rho']
        rho_interp = rho_vec[0, :] if rho_vec.ndim > 1 else rho_vec 

        def _to_batch(arr_1d):
            arr = arr_1d.unsqueeze(0) if arr_1d.ndim == 1 else arr_1d
            if rho_vec.ndim > 1 and arr.shape[0] != rho_vec.shape[0]:
                arr = arr.repeat(rho_vec.shape[0], 1)
            return arr.to(rho_vec)

        def integrated_flux(P, r, dVdr):
            """
            Compute cumulative flux [#/m^2/s]
            from volumetric source density [#/m^3/s].
            """
            # cumulative enclosed source [#/s]
            I = sp.integrate.cumulative_trapezoid(P * dVdr, r, initial=0.0)
            # Use GACODE/MITIM flux normalization (surface_used = dV/dr = volp).
            # This matches powerstate.from_density_to_flux and ge_10E20m2.
            G = np.where(dVdr > 0, I / dVdr, 0.0)
            return G

        def integrateFS(P, r, dVdr):
            """
            Based on the idea that dVdr = dV/dr, whatever r is

            Ptot = int_V P*dV = int_r P*V'*dr

            """

            I = sp.integrate.cumulative_trapezoid(
                P * dVdr, r, initial=0.0)

            return I

        def _get_initial_power_flows(self, rho0=0.0):

            self.integrated_vars = {}
            v = self.integrated_vars
            dVdr = self.profiles.derived['volp_geo']
            r = self.profiles.profiles['rmin(m)']
            v['Gbeam_e'] = integrated_flux(self.profiles.profiles['qpar_beam(1/m^3/s)'], r, dVdr) / 1E20
            v['Gwall_e'] = integrated_flux(self.profiles.profiles['qpar_wall(1/m^3/s)'], r, dVdr) / 1E20

            v['Pohm_e'] = integrateFS(self.profiles.profiles['qohme(MW/m^3)'], r,  dVdr)
            v['Pbeam_e'] = integrateFS(self.profiles.profiles['qbeame(MW/m^3)'], r,  dVdr)
            v['Prf_e'] = integrateFS(self.profiles.profiles['qrfe(MW/m^3)'], r,  dVdr)
            v['Paux_e'] = v['Pohm_e'] + v['Pbeam_e'] + v['Prf_e']

            #v['Pohm_i'] = integrateFS(self.profiles.profiles['qohmi(MW/m^3)'], r,  dVdr)
            v['Pbeam_i'] = integrateFS(self.profiles.profiles['qbeami(MW/m^3)'], r,  dVdr)
            v['Prf_i'] = integrateFS(self.profiles.profiles['qrfi(MW/m^3)'], r,  dVdr)
            v['Paux_i'] = v['Pbeam_i'] + v['Prf_i'] #v['Pohm_i']

            v['Pei'] = integrateFS(self.profiles.profiles['qei(MW/m^3)'], r,  dVdr)

            v['Pbrem_e'] = integrateFS(self.profiles.profiles['qbrem(MW/m^3)'], r,  dVdr)
            #v['Pcx_i'] = integrateFS(self.profiles.profiles['qcxi(MW/m^3)'], r,  dVdr)
            v['Pfus_e'] = integrateFS(self.profiles.profiles['qfuse(MW/m^3)'], r,  dVdr)
            v['Pfus_i'] = integrateFS(self.profiles.profiles['qfusi(MW/m^3)'], r,  dVdr)
            v['Psync_e'] = integrateFS(self.profiles.profiles['qsync(MW/m^3)'], r,  dVdr)
            v['Pline_e'] = integrateFS(self.profiles.profiles['qline(MW/m^3)'], r,  dVdr)
            v['Pion_e'] = integrateFS(self.profiles.profiles['qione(MW/m^3)'], r,  dVdr)
            v['Pion_i'] = integrateFS(self.profiles.profiles['qioni(MW/m^3)'], r,  dVdr)
            v['Prad_e'] = -v['Psync_e'] - v['Pbrem_e'] - v['Pline_e'] - v['Pion_e']
            v['Prad_i'] = -v['Pion_i'] #+ v['Pcx_i']

            v['Ge'] = v['Gbeam_e']
            v['Gz'] = np.zeros_like(v['Ge'])
            v['Mt'] = self.profiles.derived['mt_Jmiller']
            v['Pe'] = v['Paux_e'] #- v['Pei'] + v['Pfus_e'] - v['Prad_e']
            v['Pi'] = v['Paux_i'] #+ v['Pei'] + v['Pfus_i'] - v['Prad_i']

            rho = self.profiles.profiles['rho(-)']
            dVdr_0 = np.interp(rho0, rho, dVdr)
            Prad_e_0 = np.interp(rho0, rho, v['Prad_e'])
            Prad_i_0 = np.interp(rho0, rho, v['Prad_i'])
            Pfus_e_0 = np.interp(rho0, rho, v['Pfus_e'])
            Pfus_i_0 = np.interp(rho0, rho, v['Pfus_i'])
            Pei_0 = np.interp(rho0, rho, v['Pei'])
            Gwall_e_0 = np.interp(rho0, rho, v['Gwall_e'])

            # For evolving targets, stop accumulating flux/flow beyond rho0 since those will be recalculated across iterations
            Prad_e_dist = Prad_e_0 * dVdr_0 / dVdr
            Prad_i_dist = Prad_i_0 * dVdr_0 / dVdr
            Pfus_e_dist = Pfus_e_0 * dVdr_0 / dVdr
            Pfus_i_dist = Pfus_i_0 * dVdr_0 / dVdr
            Pei_dist = Pei_0 * dVdr_0 / dVdr
            Gwall_e_dist = Gwall_e_0 * dVdr_0 / dVdr

            if 'qfus' not in self.target_options["options"]["targets_evolve"]:
                v['Pe'] += v['Pfus_e']
                v['Pi'] += v['Pfus_i']
            else:
                v['Pe'] += Pfus_e_dist
                v['Pi'] += Pfus_i_dist

            if 'qrad' not in self.target_options["options"]["targets_evolve"]:
                v['Pe'] -= v['Prad_e']
                v['Pi'] -= v['Prad_i']
            else:
                v['Pe'] -= Prad_e_dist
                v['Pi'] -= Prad_i_dist

            if 'qei' not in self.target_options["options"]["targets_evolve"]:
                v['Pe'] -= v['Pei']
                v['Pi'] += v['Pei']
            else:
                v['Pe'] -= Pei_dist
                v['Pi'] += Pei_dist

            if 'qpar' not in self.target_options["options"]["targets_evolve"]:
                v['Ge'] += v['Gwall_e']
            else:
                v['Ge'] += Gwall_e_dist

            # Apply target multipliers if given and targets are not already scaled
            if target_multipliers and not targets_scaled:
                for key in ['Ge', 'Gz', 'Pe', 'Pi', 'Mt']:
                    if key in target_multipliers:
                        v[key] *= target_multipliers[key]

            # Freeze baseline targets on the active plasma grid using the exact
            # keys consumed by analytical_model_edge postprocessing.
            rho_eval = self.plasma['rho'][0, :].detach().cpu().numpy()

            Ge_eval = np.interp(rho_eval, rho, v['Ge'])
            Gz_eval = np.interp(rho_eval, rho, v['Gz'])
            Qe_eval = np.interp(rho_eval, rho, v['Pe'])
            Qi_eval = np.interp(rho_eval, rho, v['Pi'])
            Mt_eval = np.interp(rho_eval, rho, v['Mt'])
            dVdr_eval = np.interp(rho_eval, rho, dVdr)

            self.plasma['Ge_edgetargets'] = _to_batch(torch.from_numpy(Ge_eval)) # 1E20/m^2
            self.plasma['GZ_edgetargets'] = _to_batch(torch.from_numpy(Gz_eval)) # 1E20/m^2
            self.plasma['Qe_edgetargets'] = _to_batch(torch.from_numpy(Qe_eval/dVdr_eval)) # MW
            self.plasma['Qi_edgetargets'] = _to_batch(torch.from_numpy(Qi_eval/dVdr_eval)) # MW
            self.plasma['Mt_edgetargets'] = _to_batch(torch.from_numpy(Mt_eval)) # J/m^2

        required_edge_targets = (
            'Qe_edgetargets',
            'Qi_edgetargets',
            'Ge_edgetargets',
            'GZ_edgetargets',
            'Mt_edgetargets',
        )
        if (not hasattr(self, "integrated_vars")) or any(k not in self.plasma for k in required_edge_targets):
            _get_initial_power_flows(self,rho0=self.rhoCP[0].item())

        # Evaluate local quantities
        targets.evaluate()

        # Integrate
        targets.flux_integrate()

        # Merge targets, calculate errors and normalize
        targets.postprocessing(
            relative_error_assumed=relative_error_assumed,
            force_zero_particle_flux=self.target_options["options"]["force_zero_particle_flux"]
            )
        
    def X_to_dict(self,X):
        widths = self.parameterizer.n_params_per_profile
        profs = self.predicted_channels
        if np.isscalar(widths):
            widths = [int(widths)] * len(profs)
        elif len(widths) == 1 and len(profs) > 1:
            widths = list(widths) * len(profs)
        i0 = 0
        self.X_dict = {}
        for ch, w in zip(profs, widths):
            self.X_dict[ch] = X[:, i0 : i0 + w]
            i0 += w

    # ------------------------------------------------------------------
    # Override: calculate()
    # ------------------------------------------------------------------

    def calculate(
        self, X=None, nameRun="test", folder="~/scratch/", evaluation_number=0
    ):
        """
        Extended ``calculate()`` for edge runs.

        Step sequence (additions relative to ``powerstate.calculate()`` marked >>>)
        ---------------------------------------------------------------------------
        >>> 1.  self.calculateBoundaryConditions()   ← updates _lcfs_bc from SOL model
        2.  self.modify(X)                           ← enforces LCFS BC via override above
        3.  self.calculateProfileFunctions()         ← includes diamagnetic w0
        >>> 3b. self.calculateNeutrals()             ← D⁰ density → writes plasma['n0']
        >>> 3c. self.calculateImpurities()           ← Aurora CS (uses plasma['n0'] for CXR)
        4.  self.calculateTargets(...)               ← uses analytical_model_edge
        5.  self.calculateTransport(...)
        >>> 6.  self.calculateElm()                  ← peeling-ballooning ELM penalty
        7.  self.calculateMetrics()
        """
        from mitim_tools.misc_tools import IOtools
        folder = IOtools.expandPath(folder)
        self._solver_scratch_folder = folder

        # 1. Separatrix boundary conditions → updates _lcfs_bc
        self.calculateBoundaryConditions()

        # 2. Reconstruct predicted profiles according to parameterization with LCFS BC enforcement
        self.modify(X)

        # 3. Profile-derived quantities (GB units, nuei, ρ_s, c_s, …)
        self.calculateProfileFunctions()

        # 3b. Main-ion neutral density (D⁰) → populates n0, S_ion_main, nu_ioniz_main, tau_n0
        self.calculateNeutrals()

        # 3c. Impurity charge-state distribution and radiation (uses plasma['n0'] for CXR)
        self.calculateImpurities()

        # 4. Sources and sinks
        relative_error_assumed = self.target_options["options"]["percent_error"]
        self.calculateTargets(relative_error_assumed=relative_error_assumed)

        # 5. Turbulent and neoclassical transport
        self.calculateTransport(
            nameRun=nameRun,
            folder=folder,
            evaluation_number=evaluation_number,
        )

        # 6. ELM peeling-ballooning penalty → inflates turbulent fluxes if unstable
        self.calculateElm() # EPED-NN?

        # 7. Residual powers
        self.calculateMetrics()

        return (
            self.plasma["P_tr"],
            self.plasma["P"],
            self.plasma["S"],
            self.plasma["residual"],
        )