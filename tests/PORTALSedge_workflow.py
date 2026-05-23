import os
from mitim_modules.portals.utils import PORTALSoptimization
import torch
from pathlib import Path
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals.PORTALSedge import portals_edge
from mitim_modules.portals.PORTALSmain import portals
from mitim_tools.gacode_tools import PROFILEStools

cold_start = True

# Inputs
inputgacode = Path("/home/smolesworth/projects/inputs/input.gacode_d3d128913.1500")
folderWork  = Path("/home/smolesworth/projects/shortfall/base_WideBounds")

if cold_start and folderWork.exists():
    os.system(f"rm -r {folderWork.resolve()}")

# ---------------------------------------------------------------------------------------------------------------------
# Optimization Class
# ---------------------------------------------------------------------------------------------------------------------

portals_fun = portals_edge(folderWork)
device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else "cpu")
portals_fun = portals_edge(folderWork, tensor_options={"dtype": torch.double, "device": device})

# -- Standard PORTALS options -------------------------------------------------

portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 10
portals_fun.optimization_options["initialization_options"]["initial_training"] = 10
portals_fun.optimization_options["initialization_options"]["initialization_fun"] = (
    PORTALSoptimization.initialization_simple_relax
)
portals_fun.optimization_options["initialization_options"]["simple_relax_options"] = {
    "relax": 0.2, # was ~0.2
    "dx_max": 0.4, # was ~0.2
    }
portals_fun.optimization_options["initialization_options"]["simple_relax_relax_jitter"] = 0.0

# -- Reduce SR optimizer iterations during acquisition function optimization
portals_fun.optimization_options["acquisition_options"]["optimizer_options"]["sr"]["maxiter"] = 200

#portals_fun.optimization_options["surrogate_options"]["test_combination_max_error_percent"] = 10.0
portals_fun.optimization_options["surrogate_options"]["test_combination_stop_on_failure"] = False
#portals_fun.optimization_options["surrogate_options"]["test_batch_max_error_percent"] = 10.0

portals_fun.optimization_options['convergence_options']['stopping_criteria_parameters']['maximum_value'] = 1e-4
portals_fun.optimization_options['convergence_options']['stopping_criteria_parameters']['maximum_value_is_rel'] = False

#portals_fun.optimization_options["evaluation_options"]["parallel_evaluations"] = 4
portals_fun.portals_parameters["transport"]["options"]["tglf"]["cores_per_tglf_instance"] = 4

portals_fun.portals_parameters["solution"]["turbulent_exchange_as_surrogate"] = True
portals_fun.portals_parameters["solution"]["predicted_roa"]      = [0.88, 0.91, 0.94, 0.97]
portals_fun.portals_parameters["solution"]["predicted_channels"] = ["te", "ti", "ne"]

# Profile parameterizer: "piecewise_linear" | "akima" | "mtanh" | "SplineMtanh"
portals_fun.portals_parameters["solution"]["parameterizer"] = "SplineMtanh"
portals_fun.portals_parameters["solution"]["parameterizer_options"] = {
    "knots": [0.88, 0.91, 0.94, 0.97], #in roa
    #"spline_type": "linear",
    "defined_on": "aLy",
    "residual_mode": True,
    "fit_max_rel_error": 0.1,
    }

#portals_fun.portals_parameters["solution"]["exploration_ranges"]["start_from_folder"] = Path("/home/smolesworth/projects/shortfall/portals_edge_test/")
# portals_fun.portals_parameters["solution"]["exploration_ranges"]["reevaluate_targets"] = 0
portals_fun.portals_parameters["solution"]["exploration_ranges"]["limits_are_relative"] = True

portals_fun.portals_parameters["solution"]["exploration_ranges"]["ymin"] = {
"te": [0.2, 0.2, 0.2, 0.2],
"ti": [0.2, 0.2, 0.2, 0.2],
"ne": [0.2, 0.2, 0.2, 0.2],
# "te": [-7.0, -5.0, -7.0, -20.0],
# "ti": [-7.0, -5.0, -7.0, -20.0],
# "ne": [-7.0, -5.0, -7.0, -20.0],
}
portals_fun.portals_parameters["solution"]["exploration_ranges"]["ymax"] = {
"te": [ 5.0, 5.0, 5.0, 5.0],
"ti": [ 5.0, 5.0, 5.0, 5.0],
"ne": [ 5.0, 5.0, 5.0, 5.0],
# "te": [ 7.0, 0.0, 5.0, 20.0],
# "ti": [ 7.0, 0.0, 5.0, 20.0],
# "ne": [ 7.0, 0.0, 5.0, 20.0],
}

# -- TGLF settings (edge-specific) -------------------------------------------

# "edge" preset: KYGRID_MODEL=4 (extended ky grid with ETG scales),
#                XNU_MODEL=4    (pitch-angle-scattering collisions, better for steep pedestal gradients),
#                ALPHA_P=0      (parallel velocity shear disabled, unreliable near separatrix)
# See templates/input.tglf.models.yaml for all available presets.
portals_fun.portals_parameters["transport"]["options"]["tglf"]["run"]["code_settings"] = "edge"
portals_fun.portals_parameters["transport"]["options"]["neo"]["run"]["code_settings"] = "edge"
portals_fun.portals_parameters["transport"]["options"]["tglf"]["use_scan_trick_for_stds"] = 0.02

# Additional overrides on top of the preset (uncomment as needed):
# portals_fun.portals_parameters["transport"]["options"]["tglf"]["run"]["extraOptions"] = {
#     # "ALPHA_MACH": 0,    # disable Mach-number correction (ExB handled separately via NEO DPHI0DR)
#     # "USE_BPAR":   False, # suppress transverse magnetic fluctuations (faster, often adequate in edge)
# }

# -- Edge-specific options ----------------------------------------------------

edge_options = {}

# Radial domain trimming (comment out both to use the full grid)
edge_options["domain_roa"] = [0.85, 1.0]   # restrict to r/a ∈ [roa_min, roa_max]
# edge_options["domain_rho"] = [0.90, 1.0]  # alternative: use sqrt-toroidal-flux ρ

# Initial LCFS boundary conditions enforced during every modify() call.
# Leave out any key to hold it fixed at the input.gacode value.
# edge_options["lcfs_bc"] = {
#     "te": 0.080,   # keV
#     "ti": 0.080,   # keV
#     "ne": 0.30,    # 1e19 m-3
# }

# Boundary-condition model: "Fixed" | "TwoFluid_PeretSSF" | "TwoFluid_EichManz"
edge_options["bc_model"] = "TwoFluid_PeretSSF"
edge_options["bc_model_options"] = {
    "ne_target": 3.0,    # divertor target density [1e19 m-3]
    "fmom": 0.5,  
    "fq_e": 0.1,
    "fq_i": 0.2,       
}

# Main-ion neutral model: "Null" | "Analytic"
edge_options["neutral_model"] = "Analytic"
edge_options["neutral_model_options"] = {
    "source_rate":  0.5e21,   # Main species LCFS source rate [s-1]
    "include_cx":   True,  # include CX contribution to neutral diffusivity
}

# Impurity charge-state model: "Null" | "Aurora"
edge_options["charge_state_model"] = "Aurora"
edge_options["charge_state_model_options"] = {
    "imp":         "C",    # element symbol
    "D_z_m2_s":  0.1,  # impurity diffusion coefficient [m2/s]
    "V0_m_s":    -1.0, # impurity convection velocity [m/s]
    "source_rate": 0.5e20,   # injection rate [s-1]
    "cxr_flag":    True,  # include CX recombination
}

# ELM stability and transport-penalty model: "Null" | "AnalyticPB" | "EPED"
edge_options["elm_model"] = "Null"
edge_options["elm_model_options"] = {
    "backend": "epednn",                 # force Julia EPEDNN backend (skip EPED binary)
    "warn_runtime_unavailable": True,      # print why runtime checks fail
    "verbose": True,                       # print backend and pedestal diagnostics
}

# Optional edge-UQ inflation (applied in powerstate_edge after impurities).
# Keep disabled unless a calibration exists in edge_uq_calib_dir.
edge_options["edge_uq_enable"] = False
edge_options["edge_uq_calib_dir"] = "./results_offline_edge_uq"
edge_options["edge_uq_calib_label"] = "baseline"
edge_options["edge_uq_scale_factor"] = 1.0
# edge_options["edge_uq_channel_mapping"] = {
#     "bc_ne": "ne",
#     "bc_te": "te",
#     "bc_ti": "ti",
# }

# Scaling target powers (1 on variables left empty)
edge_options["target_multipliers"] = {
    "Qe": 1.0,   # scale electron heat target by 85 %
    "Qi": 1.0,   # scale ion heat target by 90 %
    "Ge": 1.0,    # leave particle flux target unchanged
    # "Mt": 1.0,  # momentum flux (omit to keep default 1.0)
}

# Use analytical_model_edge (True) or the standard analytical_model (False)
edge_options["use_edge_targets"] = True

portals_fun.portals_parameters["edge_options"] = edge_options

# -- Prepare run --------------------------------------------------------------

plasma_state = PROFILEStools.gacode_state(inputgacode)
plasma_state.correct(options={
    "recalculate_ptot":   True,
    "remove_fast":        True,
    "quasineutrality":    True,
    "enforce_same_aLn":   True,
})

portals_fun.prep(plasma_state, cold_start=cold_start)

if portals_fun.powerstate.parameterizer.__class__.__name__.lower() == "mtanh":
    PORTALSoptimization.enable_mtanh_feasibility_constraint(portals_fun)

# ---------------------------------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------------------------------

mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=False)
mitim_bo.run()

#portals_fun.plot_optimization_results(analysis_level=1)

# Required if running in non-interactive mode
#portals_fun.fn.show()
