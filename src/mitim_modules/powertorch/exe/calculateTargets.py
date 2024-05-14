"""

calculateTargets.py input.gacode 1

"""

import sys
import numpy as np
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.powertorch import STATEtools
from mitim_modules.portals.aux import PORTALSinit


def calculator(
    input_gacode,
    typeCalculation=2,
    folder="~/scratch/",
    restart=True,
    rho_vec=np.linspace(0, 0.9, 10),
    profProvided=False,
):
    profiles = (
        input_gacode if profProvided else PROFILEStools.PROFILES_GACODE(input_gacode)
    )

    PORTALSinit.defineNewGridmitim(profiles, rho_vec[1:])

    # Calculate using TGYRO
    if typeCalculation == 1:
        p = STATEtools.powerstate(
            profiles,
            rho_vec,
            TargetOptions={"TypeTarget": 3, "TargetCalc": "tgyro"},
            TransportOptions={
                "TypeTransport": "tglf_neo-tgyro",
                "ModelOptions": {
                    "restart": restart,
                    "launchSlurm": True,
                    "MODELparameters": {
                        "Physics_options": {
                            "TargetType": 3,
                            "TurbulentExchange": 0,
                            "PtotType": 1,
                            "GradientsType": 0,
                            "InputType": 1,
                        },
                        "ProfilesPredicted": ["te", "ti", "ne"],
                        "RhoLocations": rho_vec[1:],
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
            rho_vec,
            TargetOptions={"TypeTarget": 3, "TargetCalc": "powerstate"},
            TransportOptions={"TypeTransport": None, "ModelOptions": {}},
        )

    p.profiles = p.insertProfiles(
        profiles, insertPowers=True, rederive_profiles=True, reRead=True
    )
    p.determinePerformance(nameRun="test", folder=IOtools.expandPath(folder))

    return p


if __name__ == "__main__":
    input_gacode = IOtools.expandPath(sys.argv[1])
    typeCalculation = int(sys.argv[2])

    calculator(input_gacode, typeCalculation=typeCalculation)
