"""
CAPABILITY: Neoclassical radial electric field (E×B) with VGEN
--------------------------------------------------------------
This script teaches how to compute the neoclassical radial electric field Er
(and from it the E×B rotation and shearing rate) from a plasma state
(input.gacode), using `profiles_gen -vgen`, which wraps NEO over the flux
surfaces. The run happens on the machine configured for "profiles_gen" in
config_user.json.

Key teaching points:
    1. run_vgen() computes Er with the method selected in vgenOptions and
       writes an updated input.gacode where w0(rad/s) (the toroidal rotation
       consistent with that Er) is populated.
    2. The Er calculation is sensitive to kinks in the gradients of the input
       profiles: smooth_profiles=True smooths Te, Ti, ne, ni with a spline
       before the run (the original plasma state is never modified), and the
       plots then include a raw-vs-smoothed comparison.
    3. read_vgen() parses the Er component decomposition (out.vgen.ercomp),
       the velocity components (out.vgen.vel) and the updated input.gacode;
       plot_vgen() shows the Er decomposition and w0/VEXB_SHEAR before/after.
    4. Typical use of this capability: when toroidal rotation is negligible,
       the neoclassical VEXB_SHEAR computed here is what PORTALS can pass to
       TGLF (option `vgen_exb_shear` under transport.options.neo in the
       PORTALS namelist).
"""

from mitim_tools.gacode_tools import NEOtools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools

# cold_start=True starts from scratch (here, removing the previous folder); False reuses
# results already present in the folder instead of re-running
cold_start = True

(__mitimroot__ / "tests" / "scratch").mkdir(parents=True, exist_ok=True)

# Working folder of the run: the vgen subfolder with inputs and outputs lives in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_neo_vgen"
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)

# ---------------------------------------------------------------------------------------------------------------------
# 1. Prepare the NEO object from the plasma state
# ---------------------------------------------------------------------------------------------------------------------

# rhos=[] because VGEN does not run at user-selected radii: it sweeps the flux surfaces
# of the plasma state (optionally restricted with rho_range below)
neo = NEOtools.NEO(rhos=[])
neo.prep(input_gacode, folder)

# ---------------------------------------------------------------------------------------------------------------------
# 2. Run profiles_gen -vgen
# ---------------------------------------------------------------------------------------------------------------------

neo.run_vgen(
    # Name of the subfolder (inside the working folder) where this run lives
    subfolder="vgen1",
    # Restrict the calculation to this rho window (None runs the whole profile);
    # the edge region is where the neoclassical Er well matters most
    rho_range=[0.8, 1.0],
    vgenOptions={
        # Method to compute Er: 1 = force balance from the given omega0,
        # 2 = NEO weak rotation limit (recommended when toroidal rotation is zero
        # or negligible), 3 = NEO strong rotation limit, 4 = return the given omega0
        "er": 2,
        # Method to compute velocities: 1 = NEO weak rotation limit, 2 = strong rotation limit
        "vel": 1,
        # Min,max poloidal theta resolutions for the NEO solves
        "nth": "17,39",
        # Ion species (1-indexed) whose NEO and given velocities are matched
        "matched_ion": 1,
    },
    # Smooth the kinetic profiles before the run so that piecewise-linear kinks in the
    # gradients do not pollute the computed Er (the original plasma state is untouched)
    smooth_profiles=True,
    cold_start=cold_start,
)

# ---------------------------------------------------------------------------------------------------------------------
# 3. Read and plot
# ---------------------------------------------------------------------------------------------------------------------

# read_vgen() parses the updated input.gacode (w0 now populated from the neoclassical Er),
# the Er component decomposition and the velocity components
neo.read_vgen()

# plot_vgen() shows the Er decomposition, w0 and VEXB_SHEAR before/after the calculation,
# and the raw-vs-smoothed profile comparison (since smooth_profiles=True was used).
# All figures go into a multi-tab MITIM FigureNotebook (neo.fn); show() opens the GUI
neo.plot_vgen()
neo.fn.show()

