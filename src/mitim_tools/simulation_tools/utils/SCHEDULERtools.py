'''
Moved to mitim_tools.simulation_tools.gk_control.extra_points. This name is the same module object, so
existing imports, attribute access and monkeypatches keep working.
'''
import sys
from mitim_tools.simulation_tools.gk_control import extra_points as _moved

sys.modules[__name__] = _moved
