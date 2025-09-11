"""
calculateTargets.py input.gacode run1
"""

import sys
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.powertorch import STATEtools
from mitim_modules.powertorch.physics_models import targets_analytic
from IPython import embed

def calculator(
    input_gacode,
    targets_evolve=["qie", "qrad", "qfus"],
    folder="~/scratch/",
    cold_start=True,
    file_name = "input.gacode.new.powerstate",
    rho_vec=np.linspace(0.01, 0.94, 50),
    profProvided=False,
    targets_resolution = None,
):
    profiles = input_gacode if profProvided else PROFILEStools.gacode_state(input_gacode)

    p = STATEtools.powerstate(
        profiles,
        evolution_options={
            "rhoPredicted": rho_vec,
        },
        target_options={
            "evaluator": targets_analytic.analytical_model,
            "options": {
                "targets_evolve": targets_evolve,
                "target_evaluator_method":  "powerstate",
                "targets_resolution": targets_resolution
                },
        },
        transport_options={
            "evaluator": None,
            "options": {}
        },
    )

    # Determine performance
    nameRun="test"
    folder=IOtools.expandPath(folder)

    if not folder.exists():
        folder.mkdir(parents=True)

    # ************************************
    # Calculate state
    # ************************************

    p.calculate(None, nameRun=nameRun, folder=folder)

    # ************************************
    # Postprocessing
    # ************************************

    p.plasma["Pfus"] = (
        p.from_density_to_flux(
            (p.plasma["qfuse"] + p.plasma["qfusi"]) * 5.0
        )
        * p.plasma["volp"]
    )[..., -1]
    p.plasma["Prad"] = (
        p.from_density_to_flux(p.plasma["qrad"]) * p.plasma["volp"]
    )[..., -1]

    p.profiles.derive_quantities()
    
    p.from_powerstate(
        write_input_gacode=folder / file_name,
        position_in_powerstate_batch=0,
        postprocess_input_gacode={
            "Tfast_ratio": False,
            "Ti_thermals": False,
            "ni_thermals": False,
            "recalculate_ptot": False,
            "force_mach": None,
        },
        insert_highres_powers=True,
        rederive_profiles=False,
    )

    p.plasma["Q"] = p.profiles.derived["Q"]
    p.plasma['Prad'] = p.profiles.derived['Prad']

    # ************************************
    # Print Info
    # ************************************

    print(f"Q = {p.plasma['Q'].item():.2f}")
    print(f"Prad = {p.plasma['Prad'].item():.2f}MW")

    return p


if __name__ == "__main__":
    input_gacode = IOtools.expandPath(sys.argv[1])
    folder = IOtools.expandPath(sys.argv[2])

    calculator(input_gacode, folder=folder)
