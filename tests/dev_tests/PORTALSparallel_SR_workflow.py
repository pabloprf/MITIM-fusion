import os
from mitim_tools.opt_tools import STRATEGYtools
from mitim_modules.portals import PORTALSmain
from mitim_modules.portals.utils import PORTALSanalysis
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

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
    IOtools.shutil_rmtree(folderWork)

# ---------------------------------------------------------------------------
# PORTALS optimization class: start from the default namelist and override
# only the minimum to keep the test fast (few rhos, short BO loop).
# ---------------------------------------------------------------------------

portals_fun = PORTALSmain.portals(folderWork)

# Shorten the BO loop so the test is about the *initialization*, not the BO.
portals_fun.optimization_options["convergence_options"]["maximum_iterations"] = 2

# Total deterministic simple-relax points. Must be divisible by the number of
# trajectories below (2). 4 points = 2 trajectories x 2 steps each.
portals_fun.optimization_options["initialization_options"]["initial_training"] = 4

# Two deliberately contrasting trajectories so the initial training set covers
# a wide range of a/Lx states. Trajectory 0 is cautious (small dx_max, slower
# relax, starts from the unperturbed base); trajectory 1 is aggressive (bigger
# steps, smaller min abs step, gradients bumped by 10% via perturbation_base).
portals_fun.optimization_options["initialization_options"]["initialization_params"] = [
    {
        "relax": 0.15,
        "dx_max": 0.15,
        "dx_min_abs": 0.10,
        "perturbation_base": 0.0,
    },
    {
        "relax": 0.30,
        "dx_max": 0.30,
        "dx_min_abs": 0.05,
        "perturbation_base": 0.1,
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
# Assertions: after initialization the folder layout must reflect the new
# batched simple-relax shape:
#
#   portals_sr_ev_{s}                            for s in [0..steps_per_traj-1]
#     └── transport_simulation_folder/
#          ├── plasma_0/ (per-trajectory flux JSON + input.gacode)
#          └── plasma_1/
#
# and Execution/Evaluation.{s*n_traj+t} is populated from `plasma_{t}` for the
# step s, each carrying the single-plasma `fluxes_{turb,neoc}.json` pair that
# the BO-time cache gate in `power_transport.evaluate` will pick up for zero-
# work reuse.
#
# For this test (initial_training=4, n_traj=2) that means:
#   - portals_sr_ev_{0,1} exist, each with plasma_{0,1} sub-dirs containing
#     fluxes_turb.json + fluxes_neoc.json
#   - Evaluation.{0..3}/transport_simulation_folder exist and each contain the
#     single-plasma JSON pair at their top level.
# ---------------------------------------------------------------------------

N_TRAJ = 2
STEPS_PER_TRAJ = 2
N_POINTS = N_TRAJ * STEPS_PER_TRAJ  # = initial_training (4)

init_dir = folderWork / "Initialization" / "initialization_simple_relax"
for s in range(STEPS_PER_TRAJ):
    step_folder = init_dir / f"portals_sr_ev_{s}" / "transport_simulation_folder"
    assert step_folder.exists(), f"Expected {step_folder} from batched SR step {s}"
    for t in range(N_TRAJ):
        plasma_sub = step_folder / f"plasma_{t}"
        assert plasma_sub.exists(), f"Expected per-plasma sub-folder {plasma_sub}"
        assert (plasma_sub / "input.gacode").exists(), (
            f"Expected {plasma_sub/'input.gacode'} written by _evaluate_batched"
        )
        assert (plasma_sub / "fluxes_turb.json").exists(), (
            f"Expected single-plasma {plasma_sub/'fluxes_turb.json'}"
        )
        assert (plasma_sub / "fluxes_neoc.json").exists(), (
            f"Expected single-plasma {plasma_sub/'fluxes_neoc.json'}"
        )

exec_dir = folderWork / "Execution"
for i in range(N_POINTS):
    exec_tr = exec_dir / f"Evaluation.{i}" / "transport_simulation_folder"
    assert exec_tr.exists(), f"Expected {exec_tr} from parallel SR initialization"
    assert (exec_tr / "fluxes_turb.json").exists(), (
        f"Expected cached turbulence JSON at {exec_tr/'fluxes_turb.json'}"
    )
    assert (exec_tr / "fluxes_neoc.json").exists(), (
        f"Expected cached neoclassical JSON at {exec_tr/'fluxes_neoc.json'}"
    )

print(
    f"\n[PORTALSparallel_SR_workflow] {STEPS_PER_TRAJ} batched portals_sr_ev_* folders "
    f"each with {N_TRAJ} plasma_* sub-folders, matching {N_POINTS} Evaluation.* "
    "folders carrying single-plasma JSON pairs. Step-major batched layout OK."
)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

portals_fun.plot_optimization_results(analysis_level=2)

portals_output = PORTALSanalysis.PORTALSanalyzer.from_folder(folderWork)
portals_output.plotPORTALS(noshow=True)
portals_output.fn.save(folderWork / "final_portals_plots")

portals_fun.fn.show()
