"""
calculateTargets.py input.gacode 1
"""

import sys
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.powertorch import STATEtools
from mitim_modules.powertorch.physics_models import targets_analytic, transport_tgyro
from IPython import embed

def calculator(
    input_gacode,
    typeCalculation=2,
    TypeTarget=3,
    folder="~/scratch/",
    cold_start=True,
    rho_vec=np.linspace(0.1, 0.9, 9),
    profProvided=False,
    fineTargetsResolution = None,
):
    profiles = (
        input_gacode if profProvided else PROFILEStools.PROFILES_GACODE(input_gacode)
    )

    # Calculate using TGYRO
    if typeCalculation == 1:
        p = STATEtools.powerstate(
            profiles,
            EvolutionOptions={
                "rhoPredicted": rho_vec,
                'fineTargetsResolution': fineTargetsResolution,
            },
            TargetOptions={
                "targets_evaluator": targets_analytic.analytical_model,
                "ModelOptions": {
                    "TypeTarget": TypeTarget,
                    "TargetCalc":  "tgyro"},
            },
            TransportOptions={
                "transport_evaluator": transport_tgyro.tgyro_model,
                "ModelOptions": {
                    "cold_start": cold_start,
                    "launchSlurm": True,
                    "MODELparameters": {
                        "Physics_options": {
                            "TypeTarget": 3,
                            "TurbulentExchange": 0,
                            "PtotType": 1,
                            "GradientsType": 0,
                            "InputType": 1,
                        },
                        "ProfilesPredicted": ["te", "ti", "ne"],
                        "RhoLocations": rho_vec,
                        "applyCorrections": {
                            "Tfast_ratio": False,
                            "Ti_thermals": True,
                            "ni_thermals": True,
                            "recompute_ptot": False,
                        },
                        "transport_model": {"TGLFsettings": 5, "extraOptionsTGLF": {}},
                    },
                    "includeFastInQi": False,
                },
            },
        )

    # Calculate using powerstate
    elif typeCalculation == 2:
        p = STATEtools.powerstate(
            profiles,
            EvolutionOptions={
                "rhoPredicted": rho_vec,
                'fineTargetsResolution': fineTargetsResolution,
            },
            TargetOptions={
                "targets_evaluator": targets_analytic.analytical_model,
                "ModelOptions": {
                    "TypeTarget": TypeTarget,
                    "TargetCalc":  "powerstate"},
            },
            TransportOptions={
                "transport_evaluator": None,
                "ModelOptions": {}
            },
        )

    # Determine performance
    nameRun="test"
    folder=IOtools.expandPath(folder)

    # ************************************
    # Calculate state
    # ************************************

    p.calculate(None, nameRun=nameRun, folder=folder)

    # ************************************
    # Postprocessing
    # ************************************

    p.plasma["Pfus"] = (
        p.volume_integrate(
            (p.plasma["qfuse"] + p.plasma["qfusi"]) * 5.0
        )
        * p.plasma["volp"]
    )[..., -1]
    p.plasma["Prad"] = (
        p.volume_integrate(p.plasma["qrad"]) * p.plasma["volp"]
    )[..., -1]

    p.profiles.derive_quantities()
    
    p.from_powerstate(
        write_input_gacode=folder / "input.gacode.new.powerstate",
        position_in_powerstate_batch=0,
        postprocess_input_gacode={
            "Tfast_ratio": False,
            "Ti_thermals": False,
            "ni_thermals": False,
            "recompute_ptot": False,
            "ensureMachNumber": None,
        },
        insert_highres_powers=True,
        rederive_profiles=False,
    )

    p.plasma["QiMWm2n"] = (
        (p.plasma["Paux_e"] + p.plasma["Paux_i"]) * p.plasma["volp"]
    )[..., -1]
    p.plasma["Q"] = p.plasma["Pfus"] / p.plasma["Pin"]

    # ************************************
    # Print Info
    # ************************************

    print(
        f"Q = {p.plasma['Q'].item():.2f} (Pfus = {p.plasma['Pfus'].item():.2f}MW, Pin = {p.plasma['Pin'].item():.2f}MW)"
    )

    print(f"Prad = {p.plasma['Prad'].item():.2f}MW")

    return p


if __name__ == "__main__":
    input_gacode = IOtools.expandPath(sys.argv[1])
    typeCalculation = int(sys.argv[2])

    calculator(input_gacode, typeCalculation=typeCalculation)
