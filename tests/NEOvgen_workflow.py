import os
from mitim_tools.gacode_tools import NEOtools, PROFILEStools
from mitim_tools import __mitimroot__

cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

folder = __mitimroot__ / "tests" / "scratch" / "neovgen_test"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    os.system(f"rm -r {folder.resolve()}")

# Load the plasma state
plasma_state = PROFILEStools.gacode_state(input_gacode)

# --- Set up NEO for VGEN  (rhos=[] because VGEN runs on all flux surfaces)
neo = NEOtools.NEO(rhos=[])
neo.prep(input_gacode, folder)

# --- run_vgen: submit profiles_gen -vgen and wait
neo.run_vgen(
    subfolder="vgen1",
    rho_range=[0.8,1.0],
    vgenOptions={
        "er": 2,          # NEO weak rotation limit (recommended for zero Vtor)
        "vel": 1,         # NEO weak rotation limit
        "nth": "17,39",   # Min/max poloidal theta resolution
        "matched_ion": 1, # Ion species index to match (1-indexed)
    },
    cold_start=cold_start,
)

# --- read_vgen: parse ercomp, vel, and updated input.gacode
neo.read_vgen()

# --- plot_vgen: Er decomposition + w0/VEXB before & after
neo.plot_vgen()
neo.fn.show()
neo.fn.close()
