"""
Default "meta" (numerical/physics control) presets for QuaLiKiz, following the
same two-file convention as NEO/TGLF (``GACODEdefaults.addNEOcontrol`` /
``addTGLFcontrol``):

  * ``templates/input.qualikiz.controls``    -- full baseline "meta" namelist
    (the STANDARD values), parsed like any other ``*.controls`` file via
    ``IOtools.generateMITIMNamelist``.
  * ``templates/input.qualikiz.models.yaml`` -- named presets ("STANDARD",
    "MINIMAL", "FAST", "ROTATION", ...) layered as deltas on top of that
    baseline via ``GACODEdefaults.add_code_settings`` (same ``base:``
    inheritance machinery as ``input.tglf.models.yaml``).

The "meta" block corresponds to QuaLiKiz's JSON parameters.json "meta"
section.
"""

from mitim_tools import __mitimroot__
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools.utils import GACODEdefaults

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
    Build the "meta" control dictionary for a QuaLiKiz run: the baseline
    parsed from ``templates/input.qualikiz.controls``, with the named preset
    from ``templates/input.qualikiz.models.yaml`` layered on top (honors
    ``base:`` inheritance, label matched case-insensitively).

    code_settings : None or a preset label in that file, e.g. "FAST". Falls
                    back to "STANDARD" (the bare baseline) if None.
    minimal       : if True, ignore code_settings and use the "MINIMAL" preset.
    """

    if minimal or code_settings == 0:
        code_settings = "MINIMAL"
    elif code_settings is None:
        code_settings = "STANDARD"

    options = IOtools.generateMITIMNamelist(__mitimroot__ / "templates" / _CONTROLS_FILE, caseInsensitive=False)
    options = GACODEdefaults.add_code_settings(options, code_settings, models_file=_MODELS_FILE)

    return options


def default_kthetarhos():
    return list(_KTHETARHOS_STANDARD)
