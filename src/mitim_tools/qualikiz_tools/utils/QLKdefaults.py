"""
Default "meta" (numerical/physics control) presets for QuaLiKiz, loaded from
``templates/input.qualikiz.models.yaml`` -- format and resolution (labels,
``base:`` inheritance) mirror ``templates/input.tglf.models.yaml`` and are
resolved through the same machinery as ``GACODEdefaults.addTGLFcontrol()``.

The "meta" block corresponds to QuaLiKiz's JSON parameters.json "meta"
section.
"""

from mitim_tools.gacode_tools.utils import GACODEdefaults
from mitim_tools.misc_tools.LOGtools import printMsg as print

# Passed to GACODEdefaults.resolve_preset() only so it derives the models
# file name (input.qualikiz.controls -> input.qualikiz.models.yaml); QuaLiKiz
# has no separate full-namelist ".controls" file of its own, unlike TGLF/NEO/CGYRO.
_CONTROLS_FILE = "input.qualikiz.controls"
_MODELS_FILE = "input.qualikiz.models.yaml"

# Default kthetarhos grid (binormal wavenumber spectrum, k_y*rho_s). Not part
# of the models.yaml presets since it is orthogonal to the "meta" block and
# passed to QuaLiKizXpoint separately (see QLKtools._build_plan_from_gacode).
_KTHETARHOS_STANDARD = [
    0.1,
    0.175,
    0.25,
    0.325,
    0.4,
    0.5,
    0.7,
    1.0,
    1.8,
    3.0,
    9.0,
    15.0,
    21.0,
    27.0,
    36.0,
    45.0,
]


def addQLKcontrol(code_settings=None, minimal=False, **kwargs):
    """
    Build the "meta" control dictionary for a QuaLiKiz run, resolved from
    ``templates/input.qualikiz.models.yaml`` (honors ``base:`` inheritance,
    label matched case-insensitively).

    code_settings : None or a preset label in that file, e.g. "FAST". Falls
                    back to "STANDARD" if None or not found.
    minimal       : if True, ignore code_settings and use the "MINIMAL" preset.
    """

    if minimal or code_settings == 0:
        preset = "MINIMAL"
    elif code_settings is None:
        preset = "STANDARD"
    else:
        preset = code_settings

    sett = GACODEdefaults.resolve_preset(preset, controls_file=_CONTROLS_FILE)
    if not sett or "controls" not in sett:
        print(f"\t- code_settings = {code_settings} not found in {_MODELS_FILE}, using STANDARD", typeMsg="w")
        sett = GACODEdefaults.resolve_preset("STANDARD", controls_file=_CONTROLS_FILE)

    return dict(sett["controls"])


def default_kthetarhos():
    return list(_KTHETARHOS_STANDARD)
