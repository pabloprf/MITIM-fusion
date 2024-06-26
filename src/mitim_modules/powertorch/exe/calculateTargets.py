"""
calculateTargets.py input.gacode 1
"""

import sys
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.powertorch import STATEtools
from mitim_modules.portals.aux import PORTALSinit
from mitim_modules.powertorch.physics import TRANSPORTtools,TARGETStools
from IPython import embed

def calculator(
    input_gacode,
    typeCalculation=2,
    TypeTarget=3,
    folder="~/scratch/",
    restart=True,
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
                "targets_evaluator": TARGETStools.analytical_model,
                "ModelOptions": {
                    "TypeTarget": TypeTarget,
                    "TargetCalc":  "tgyro"},
            },
            TransportOptions={
                "transport_evaluator": TRANSPORTtools.tgyro_model,
                "ModelOptions": {
                    "restart": restart,
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
                        "transport_model": {"turbulence": 'TGLF',"TGLFsettings": 5, "extraOptionsTGLF": {}},
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
                "targets_evaluator": TARGETStools.analytical_model,
                "ModelOptions": {
                    "TypeTarget": TypeTarget,
                    "TargetCalc":  "powerstate"},
            },
            TransportOptions={
                "transport_evaluator": None,
                "ModelOptions": {}
            },
        )

    # p.profiles = p.insertProfiles(
    #     profiles, insert_highres_powers=True, rederive_profiles=True)
    p.determinePerformance(nameRun="test", folder=IOtools.expandPath(folder))

    return p


if __name__ == "__main__":
    input_gacode = IOtools.expandPath(sys.argv[1])
    typeCalculation = int(sys.argv[2])

    calculator(input_gacode, typeCalculation=typeCalculation)
