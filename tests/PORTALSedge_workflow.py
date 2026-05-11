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
inputgacode = Path("/home/smolesworth/projects/inputs/input_H_HighColl_179449_1500.gacode")
folderWork  = Path("/home/smolesworth/projects/H_HighColl/portals_edge_test")

if cold_start and folderWork.exists():
    os.system(f"rm -r {folderWork.resolve()}")

# ---------------------------------------------------------------------------------------------------------------------
# Optimization Class
# ---------------------------------------------------------------------------------------------------------------------

portals_fun = portals_edge(folderWork)
#device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else "cpu")
#portals_fun = portals_edge(folderWork, tensor_options={"dtype": torch.double, "device": device})

# -- Standard PORTALS options -------------------------------------------------

portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 3
portals_fun.optimization_options["initialization_options"]["initial_training"] = 10
portals_fun.optimization_options["initialization_options"]["initialization_fun"] = (
    PORTALSoptimization.initialization_simple_relax
)
portals_fun.optimization_options["initialization_options"]["simple_relax_options"] = {
    "relax": 0.2, # was ~0.2
    "dx_max": 0.4, # was ~0.2
    }
portals_fun.optimization_options["initialization_options"]["simple_relax_relax_jitter"] = 0.0

#portals_fun.optimization_options["surrogate_options"]["test_combination_max_error_percent"] = 10.0
portals_fun.optimization_options["surrogate_options"]["test_combination_stop_on_failure"] = False
#portals_fun.optimization_options["surrogate_options"]["test_batch_max_error_percent"] = 10.0

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
    }

#portals_fun.portals_parameters["solution"]["exploration_ranges"]["start_from_folder"] = Path("/home/smolesworth/projects/shortfall/portals_edge_test/")
# portals_fun.portals_parameters["solution"]["exploration_ranges"]["reevaluate_targets"] = 0
# portals_fun.portals_parameters["solution"]["exploration_ranges"]["limits_are_relative"] = True

# portals_fun.portals_parameters["solution"]["exploration_ranges"]["ymin"] = {
# "te": [0.2, 0.2, 0.2, 0.2],
# "ti": [0.2, 0.2, 0.2, 0.2],
# "ne": [0.2, 0.2, 0.2, 0.2],
# "te": [-7.0, -5.0, -7.0, -20.0],
# "ti": [-7.0, -5.0, -7.0, -20.0],
# "ne": [-7.0, -5.0, -7.0, -20.0],
# }
# portals_fun.portals_parameters["solution"]["exploration_ranges"]["ymax"] = {
# "te": [ 5.0, 5.0, 5.0, 5.0],
# "ti": [ 5.0, 5.0, 5.0, 5.0],
# "ne": [ 5.0, 5.0, 5.0, 5.0],
# "te": [ 7.0, 0.0, 5.0, 20.0],
# "ti": [ 7.0, 0.0, 5.0, 20.0],
# "ne": [ 7.0, 0.0, 5.0, 20.0],
# }

# -- TGLF settings (edge-specific) -------------------------------------------

# "edge" preset: KYGRID_MODEL=4 (extended ky grid with ETG scales),
#                XNU_MODEL=4    (pitch-angle-scattering collisions, better for steep pedestal gradients),
#                ALPHA_P=0      (parallel velocity shear disabled, unreliable near separatrix)
# See templates/input.tglf.models.yaml for all available presets.
portals_fun.portals_parameters["transport"]["options"]["tglf"]["run"]["code_settings"] = "edge"
portals_fun.portals_parameters["transport"]["options"]["neo"]["run"]["code_settings"] = "edge"
#portals_fun.portals_parameters["transport"]["options"]["tglf"]["use_scan_trick_for_stds"] = 0.02

# Additional overrides on top of the preset (uncomment as needed):
portals_fun.portals_parameters["transport"]["options"]["tglf"]["run"]["extraOptions"] = {
    # "ALPHA_MACH": 0,    # disable Mach-number correction (ExB handled separately via NEO DPHI0DR)
    "USE_BPAR":   False, # suppress transverse magnetic fluctuations (faster, often adequate in edge)
}

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
# edge_options["bc_model_options"] = {
#     "ne_target": 0.5,    # divertor target density [1e19 m-3]
#     "Te_target": 0.010,  # divertor target Te [keV]
#     "Lpar": 30.0,        # parallel connection length [m] (optional override)
# }

# Main-ion neutral model: "Null" | "Analytic"
edge_options["neutral_model"] = "Null"
edge_options["neutral_model_options"] = {
    "source_rate":  1e21,   # Main species LCFS source rate [s-1]
    "include_cx":   True,  # include CX contribution to neutral diffusivity
}

# Impurity charge-state model: "Null" | "Aurora"
edge_options["charge_state_model"] = "Null"
edge_options["charge_state_model_options"] = {
    "imp":         "C",    # element symbol
    "D_z_m2_s":  0.1,  # impurity diffusion coefficient [m2/s]
    "V0_m_s":    -0.5, # impurity convection velocity [m/s]
    "source_rate": 2e20,   # injection rate [s-1]
    "cxr_flag":    True,  # include CX recombination
}

# ELM stability and transport-penalty model: "Null" | "AnalyticPB" | "EPED"
edge_options["elm_model"] = "Null"
edge_options["elm_model_options"] = {
    "stiffness": 10.0,   # penalty stiffness above stability boundary
    "roa_min":   0.90,   # innermost r/a where the ELM penalty is applied
    # "eped_folder": "/path/to/eped_runs",  # required for EPED model only
}

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

portals_fun.prep(plasma_state, cold_start=True)

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
