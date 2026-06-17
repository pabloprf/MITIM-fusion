"""
test_recompute_targets.py — exercises gacode_state.recompute_targets().

recompute_targets() re-derives the radiation, fusion alpha-heating and
electron-ion exchange power profiles from the kinetic profiles (Te, Ti, ne, ni,
geometry) and writes them back into the input.gacode columns. It is the single
entry point used by the MAESTRO confinement beat and RAPIDS.

Two cases on the bundled SPARC Primary Reference Discharge (a D-T burning
plasma):
    1. profiles as in the file  -> recompute reproduces the stored targets,
    2. Ti raised 20% throughout -> recompute responds (more fusion).

Run it interactively (recompute_targets(debug=True) pops the debug plots):

    python tests/dev_tests/test_recompute_targets.py
"""

from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools.LOGtools import printMsg as print

gacode_file = __mitimroot__ / "tests" / "data" / "input.gacode_SPARC_PRD"


def report(p, label):
    print(f"{label:>11}:  Pfus={p.derived['Pfus']:7.2f} MW   "
          f"Prad={p.derived['Prad']:7.3f} MW   Pei={p.derived['qe_exc_MW'][-1]:7.3f} MW")


def recompute_case(title, ti_scale=1.0):
    p = PROFILEStools.gacode_state(gacode_file)
    if ti_scale != 1.0:
        p.profiles["ti(keV)"] = p.profiles["ti(keV)"] * ti_scale
        p.derive_quantities(rederiveGeometry=False)
    print(f"\n--- {title} ---")
    report(p, "before")
    p.recompute_targets(debug=True)
    report(p, "recomputed")
    return p


pfus_in_file = PROFILEStools.gacode_state(gacode_file).derived["Pfus"]

p1 = recompute_case("profiles as in the file")
p2 = recompute_case("Ti raised 20% throughout", ti_scale=1.2)

assert abs(p1.derived["Pfus"] - pfus_in_file) < 0.05 * pfus_in_file, \
    "recompute should reproduce the file's fusion power for the unchanged profiles"
assert p2.derived["Pfus"] > p1.derived["Pfus"], \
    "raising Ti by 20% should increase the recomputed fusion power"
print("\nOK: recompute_targets reproduces the SPARC targets and responds to a Ti change.", typeMsg="i")
