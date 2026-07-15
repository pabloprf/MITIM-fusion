from mitim_tools.plasmastate_tools.utils import calc_pulse_duration
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools import __mitimroot__

# Working folder of the run: prepared input files and run subfolders live in it
folder = __mitimroot__ / "tests" / "scratch" / "capability_pulse_duration"
folder.mkdir(parents=True, exist_ok=True)

# Read in sample input.gacode
input_gacode = __mitimroot__ / "tests" / "data" / "input.gacode_SPARC_PRD"
p = PROFILEStools.gacode_state(input_gacode)

# Calculate pulse duration when only cs_change_in_field and inboard_to_CS_distance are known - NOTE: the values chosen here are random and not representative of SPARC
max_duration, dataset = calc_pulse_duration.calc_flattop_time(p, cs_change_in_field=12, inboard_to_CS_distance=0.3, overwrite_flux=None)

# Calculate pulse duration when total flux from central solenoid is known
max_duration, dataset = calc_pulse_duration.calc_flattop_time(p, cs_change_in_field=None, overwrite_flux=35.0, inboard_to_CS_distance=None)