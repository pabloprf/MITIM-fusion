import os
from mitim_tools.gacode_tools import CGYROtools
from mitim_tools.misc_tools import IOtools, GUItools
from mitim_tools import __mitimroot__

# ---------------------------------------------------------------------------
# Minimal CGYRO linear workflow test using preprocess_options.
#
# This test exercises the preprocess_options knob added in commit 848e408a
# (auto BOX_SIZE / N_RADIAL from local equilibrium quantities). Unlike the
# preprocess section in capability_tests/cgyro_2_nonlinear_run_from_inputgacode.py
# — which runs with run_type='prep' to only inspect the generated input files —
# this test invokes a full `run_type='normal'` linear run at one rho so the
# auto-computed BOX_SIZE, N_RADIAL and KY are exercised end-to-end through the
# submission pipeline.
# ---------------------------------------------------------------------------

cold_start = True
save_figures = False

gacode_file = __mitimroot__ / "tests" / "data" / "input.gacode"
folder = __mitimroot__ / "tests" / "scratch" / "cgyro_linear_preprocess_test"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

folder.mkdir(parents=True, exist_ok=True)

cgyro = CGYROtools.CGYRO(rhos=[0.5])

cgyro.prep(gacode_file, folder)

# ---------------------------------------------------------------------------
# One linear run at rho=0.5 in 'normal' mode. BOX_SIZE, N_RADIAL and KY are
# picked by _apply_cgyro_preprocessing from the local Q / s / r/a at rho and
# the caller-provided L_x / N_radial / ky_min. A short MAX_TIME keeps the
# actual CGYRO execution well under a few minutes.
# ---------------------------------------------------------------------------

cgyro.run(
    "linear_preprocessed",
    code_settings="Linear",
    extraOptions={
        "MAX_TIME": 10.0,
    },
    preprocess_options={
        "ky_min": 0.3,
        "L_x": 90.0,
        "N_radial": 256,
        "min_box_size": 100,
    },
    allocation={
        "resources_per_call": 16,
        "minutes": 10,
    },
    cold_start=cold_start,
    forceIfcold_start=True,
    run_type="normal",
)

# ---------------------------------------------------------------------------
# Sanity-check the generated input.cgyro file: the preprocess_options must
# have landed in the submitted input (BOX_SIZE, N_RADIAL, KY), and the
# standard CGYRO divisibility invariant N_RADIAL % BOX_SIZE == 0 must hold.
# ---------------------------------------------------------------------------

for rho in cgyro.rhos:
    input_file = cgyro.FolderSimLast / f"input.cgyro_{rho:.4f}"
    with open(input_file, "r") as f:
        txt = f.read()
    parsed = {}
    for line in txt.splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            parsed[k.strip()] = v.strip().split()[0]
    box_size = int(parsed["BOX_SIZE"])
    n_radial = int(parsed["N_RADIAL"])
    ky = float(parsed["KY"])
    assert n_radial % 2 == 0, f"N_RADIAL={n_radial} must be even"
    assert n_radial % box_size == 0, (
        f"N_RADIAL={n_radial} must be divisible by BOX_SIZE={box_size}"
    )
    print(
        f"[cgyro linear preprocess] rho={rho}: "
        f"BOX_SIZE={box_size} N_RADIAL={n_radial} KY={ky}"
    )

# Read results from the completed run.
cgyro.read(label="linear_preprocessed")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fn = GUItools.FigureNotebook("CGYRO linear preprocess", geometry="1600x1000", show=not save_figures)
cgyro.plot(labels=["linear_preprocessed"], fn=fn)

if not save_figures:
    cgyro.fn.show()
    cgyro.fn.close()
else:
    cgyro.fn.save(f"{folder}/figs_cgyro/")
