import torch
import copy
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.powertorch import STATEtools
from mitim_modules.powertorch.physics import TRANSPORTtools
from mitim_tools import __mitimroot__

# Inputs
inputgacode = PROFILEStools.gacode_state(__mitimroot__ / "tests" / "data" / "input.gacode")
rho       = torch.from_numpy(np.linspace(0.1,0.9,9)).to(dtype=torch.double)

s = STATEtools.powerstate(inputgacode,
    evolution_options = { 'ProfilePredicted': ['te', 'ti'],
                         'rhoPredicted': rho
                        },
    transport_options = { 'transport_evaluator': TRANSPORTtools.diffusion_model,
                         'ModelOptions': {
                            'chi_e': torch.ones(rho.shape[0]).to(rho)*0.8,
                            'chi_i': torch.ones(rho.shape[0]).to(rho)*1.2
                            }
                        }
    )

# Calculate state for the original profiles
s_orig = copy.deepcopy(s)
s_orig.calculate()

# Find flux matching profiles
s.flux_match(algorithm='root')

# Plot two cases together
fn = s.plot(compare_to_state=s_orig)
