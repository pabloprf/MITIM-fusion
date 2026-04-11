import os
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools import __mitimroot__

# ---------------------------------------------------------------------------
# PORTALS parallel simple-relax workflow test
#
# Exercises the multi-trajectory initialization path added in commit 1e4b4a2d:
#   optimization_options.initialization_options.initialization_params
#
# initial_training is split across N deterministic simple-relax trajectories,
# each with its own (relax, dx_max, dx_min_abs, ...). The total number of
# initial training points must be divisible by the number of trajectories.
# This test uses 4 initial points spread over 2 trajectories (2 steps each)
# with contrasting relax factors so the two trajectories visibly diverge in
# the Initialization/initialization_simple_relax/portals_sr_ev_* folder
# sequence.
# ---------------------------------------------------------------------------

cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

inputgacode = __mitimroot__ / "tests" / "data" / "input.gacode"
folderWork = __mitimroot__ / "tests" / "scratch" / "portals_parallel_SR_test"

if cold_start and folderWork.exists():
    os.system(f"rm -r {folderWork.resolve()}")

# ---------------------------------------------------------------------------
# PORTALS optimization class: start from the default namelist and override
# only the minimum to keep the test fast (few rhos, short BO loop).
# ---------------------------------------------------------------------------

portals_fun = PORTALSmain.portals(folderWork)

# Shorten the BO loop so the test is about the *initialization*, not the BO.
portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 1

# Total deterministic simple-relax points. Must be divisible by the number of
# trajectories below (2). 4 points = 2 trajectories x 2 steps each.
portals_fun.optimization_options["initialization_options"]["initial_training"] = 4

# Two deliberately contrasting trajectories so the initial training set covers
# a wide range of a/Lx states. Trajectory 0 is cautious (small dx_max, slower
# relax); trajectory 1 is aggressive (bigger steps, smaller min abs step).
portals_fun.optimization_options["initialization_options"]["initialization_params"] = [
    {
        "relax": 0.15,
        "dx_max": 0.15,
        "dx_min_abs": 0.10,
    },
    {
        "relax": 0.30,
        "dx_max": 0.30,
        "dx_min_abs": 0.05,
    },
]

portals_fun.portals_parameters["solution"]["turbulent_exchange_as_surrogate"] = True
portals_fun.portals_parameters["solution"]["predicted_rho"] = [0.35, 0.65, 0.85]
portals_fun.portals_parameters["solution"]["predicted_channels"] = ["te", "ti", "ne"]
portals_fun.portals_parameters["transport"]["options"]["tglf"]["run"]["code_settings"] = "SAT0"

# ---------------------------------------------------------------------------
# Prepare and run
# ---------------------------------------------------------------------------

plasma_state = PROFILEStools.gacode_state(inputgacode)
plasma_state.correct(options={
    "recalculate_ptot": True,
    "remove_fast": True,
    "quasineutrality": True,
    "enforce_same_aLn": True,
})

portals_fun.prep(plasma_state)

mitim_bo = STRATEGYtools.MITIM_BO(portals_fun, cold_start=cold_start, askQuestions=False)
mitim_bo.run()

# ---------------------------------------------------------------------------
# Assertions: after initialization the folder layout must reflect the
# step-major interleaving of the two trajectories:
#   portals_sr_ev_{s*n_traj + t}  for s in [0..steps_per_traj-1], t in [0..n_traj-1]
# For this test (initial_training=4, n_traj=2) that means folders 0..3 must
# all exist, plus Execution/Evaluation.0..3 copied from them.
# ---------------------------------------------------------------------------

init_dir = folderWork / "Initialization" / "initialization_simple_relax"
for i in range(4):
    expected = init_dir / f"portals_sr_ev_{i}"
    assert expected.exists(), f"Expected {expected} from parallel SR initialization"

exec_dir = folderWork / "Execution"
for i in range(4):
    expected = exec_dir / f"Evaluation.{i}" / "transport_simulation_folder"
    assert expected.exists(), f"Expected {expected} from parallel SR initialization"

print("\n[PORTALSparallel_SR_workflow] Four portals_sr_ev_* folders and "
      "matching Evaluation.{0..3} folders found. Step-major layout OK.")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

portals_fun.plot_optimization_results(analysis_level=2)

portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(folderWork)
portals_output.plotPORTALS(noshow=True)
portals_output.fn.save(folderWork / "final_portals_plots")

portals_fun.fn.show()
