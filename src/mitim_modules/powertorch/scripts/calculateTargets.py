"""
calculateTargets.py input.gacode
"""

import sys
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.powertorch import STATEtools
from mitim_modules.powertorch.physics_models import targets_analytic, transport_tglfneo
from IPython import embed

def calculator(
    input_gacode,
    TypeTarget=3,
    folder="~/scratch/",
    cold_start=True,
    rho_vec=np.linspace(0.1, 0.9, 9),
    profProvided=False,
    fineTargetsResolution = None,
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
                "TypeTarget": TypeTarget,
                "target_evaluator_method":  "powerstate",
                "fineTargetsResolution": fineTargetsResolution
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
        write_input_gacode=folder / "input.gacode.new.powerstate",
        position_in_powerstate_batch=0,
        postprocess_input_gacode={
            "Tfast_ratio": False,
            "Ti_thermals": False,
            "ni_thermals": False,
            "recalculate_ptot": False,
            "ensureMachNumber": None,
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
