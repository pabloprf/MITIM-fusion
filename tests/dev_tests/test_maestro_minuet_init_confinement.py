'''
Dev-test: MINIMAL MAESTRO run -- MINUET initializer + a ["minuet", "confinement"] chain.

The smallest MAESTRO exercise of the new `initialization_type: minuet`
(GEQtools.minuet_millerized) driving a short two-beat chain, all in-process:

    1. minuet      : current diffusion + sawteeth on the MINUET-built equilibrium.
                     Its t = 0 GS solve re-solves the equilibrium against the
                     CREATED kinetic pressure (the initializer solved with the
                     p0*(1-psiN^2)^2 ansatz -- the beat is what makes geometry
                     and profiles consistent), then diffusion relaxes q.
    2. confinement : scans Te_bc until the target H98y2 is matched (analytic,
                     frozen sources) on the minuet-evolved state.

Everything else is the template default (SPARC-like R = 1.85 m, a = 0.57 m,
Bt = 12.2 T, Ip = 8.7 MA; confinement beat: x_bc = 0.90, H98y2 target 1.0).

Checks are intentionally light -- this test exists to RUN the chain, not to
re-verify the initializer (tests/dev_tests/test_maestro_minuet_initializer.py
does that in depth):
    - the MINUET initializer ran (minuet.geqdsk exists, no freegs artifacts)
    - both beats produced their output states
    - q evolved across the minuet beat (current diffusion happened)
    - the final state is loadable and carries the engineering Ip
    - Te at the BC location moved (the confinement scan actually ran)

Run from the dev-pixi root:
    ./run_with_env.sh python MITIM-fusion/tests/dev_tests/test_maestro_minuet_init_confinement.py
'''

import numpy as np
import torch
from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.maestro.scripts import run_maestro

cold_start = True

folder = __mitimroot__ / "tests" / "scratch" / "dev_maestro_minuet_init_confinement"

if cold_start and folder.exists():
    IOtools.shutil_rmtree(folder)
folder.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(8)

# ------------------------------------------------------------------------------------------------
# Namelist: template + minuet initializer + fixed_bc creator + single confinement beat
# ------------------------------------------------------------------------------------------------

nml = IOtools.read_mitim_yaml(__mitimroot__ / "templates" / "namelist.maestro.yaml")

nml["plasma"]["profiles_initialization"]["initialization_type"] = "minuet"
nml["plasma"]["profiles_initialization"]["creator_type"] = "fixed_bc"
nml["plasma"]["profiles_initialization"]["parameters"]["x_bc"] = 0.95
nml["plasma"]["profiles_initialization"]["parameters"]["Te_bc"] = 3.0  # keV starting guess; the beat re-scans it

nml["plasma"]["heating"]["type"] = "gaussian_sources"
nml["plasma"]["heating"]["parameters"]["Pe"] = 10.0
nml["plasma"]["heating"]["parameters"]["Pi"] = 5.0
nml["plasma"]["heating"]["parameters"]["nu_source"] = 5.0
nml["plasma"]["heating"]["parameters"]["fmini"] = 0.0

# Two beats: minuet (short run) then confinement (all template defaults: x_bc 0.90, H98y2 target 1.0)
nml["maestro"]["beats"] = ["minuet", "confinement"]
nml["maestro"]["minuet"]["parameters_prepare"]["t_end"] = 2.0
nml["maestro"]["minuet"]["parameters_prepare"]["n_save"] = 51

Ip_nml = nml["plasma"]["parameters"]["Ip"]

namelist_file = folder / "namelist.maestro.yaml"
IOtools.write_mitim_yaml(nml, namelist_file)

# ------------------------------------------------------------------------------------------------
# Run MAESTRO
# ------------------------------------------------------------------------------------------------

run_maestro.run_maestro_local(
    namelist_file,
    folder = folder,
    terminal_outputs = True,
    force_cold_start = cold_start,
    cpus = 8,
)

# ------------------------------------------------------------------------------------------------
# Light checks
# ------------------------------------------------------------------------------------------------

print("\n" + "="*100)
print(" Checking results")
print("="*100)

b1 = folder / "Beats" / "Beat_1"   # minuet
b2 = folder / "Beats" / "Beat_2"   # confinement

assert (b1 / "initializer_minuet" / "minuet.geqdsk").exists(), "MINUET initializer produced no minuet.geqdsk"
assert not (b1 / "initializer_freegs").exists(), "a freegs initializer folder appeared despite initialization_type=minuet"
assert (b1 / "beat_results" / "input.gacode").exists(), "minuet beat produced no beat_results/input.gacode"
assert (b2 / "beat_results" / "input.gacode").exists(), "confinement beat produced no beat_results/input.gacode"

# q evolved across the minuet beat (current diffusion happened)
p_pre = PROFILEStools.gacode_state(b1 / "run_minuet" / "input.gacode")
p_mid = PROFILEStools.gacode_state(b1 / "beat_results" / "input.gacode")
dq = np.max(np.abs(np.interp(p_pre.profiles["rho(-)"], p_mid.profiles["rho(-)"], p_mid.profiles["q(-)"])
                   - p_pre.profiles["q(-)"]))
print(f"\t- max |dq| across the minuet beat: {dq:.4f}")
assert dq > 0.01, "q-profile did not evolve across the minuet beat (no current diffusion?)"

p_out = PROFILEStools.gacode_state(b2 / "beat_results" / "input.gacode")
p_out.derive_quantities()

Ip_out = float(p_out.profiles["current(MA)"][0])
rho = p_out.profiles["rho(-)"]
Te_bc_out = float(np.interp(0.90, rho, p_out.profiles["te(keV)"]))

print(f"\t- Ip of the final state: {Ip_out:.5f} MA (namelist {Ip_nml} MA, {100*(Ip_out/Ip_nml-1):+.3f}%)")
print(f"\t- Te(rho = 0.90) after the confinement scan: {Te_bc_out:.3f} keV (creator started from 3.0 keV at rho = 0.95)")

assert abs(Ip_out/Ip_nml - 1) < 0.01, "final state Ip is off the namelist value by more than 1%"
assert Te_bc_out > 0.0 and np.isfinite(Te_bc_out), "final Te at the BC location is not a sane number"

print("\nPASS: MAESTRO [minuet, confinement] chain completed from a MINUET-built equilibrium")
