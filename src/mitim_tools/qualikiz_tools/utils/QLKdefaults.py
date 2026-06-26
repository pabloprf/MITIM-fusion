"""
Default "meta" (numerical/physics control) blocks for QuaLiKiz, and a small
set of named presets, analogous in spirit to GACODEdefaults.addTGLFcontrol()
but self-contained (no dependency on templates/ files outside this package).

The "meta" block corresponds to QuaLiKiz's JSON parameters.json "meta"
section.
"""

# Minimum working set (fast, low-resolution, useful for quick checks)
_MINIMAL = {
    "phys_meth": 0,
    "coll_flag": 1,
    "rot_flag": 0,
    "verbose": 0,
    "separateflux": 0,
    "numsols": 2,
    "relacc1": 1e-2,
    "relacc2": 5e-2,
    "maxruns": 1,
    "maxpts": 5e4,
    "timeout": 30,
    "ETGmult": 1,
    "collmult": 1,
    "integration_routine": 1,
}

# Default/standard set
_STANDARD = {
    "phys_meth": 2,
    "coll_flag": 1,
    "rot_flag": 0,
    "verbose": 0,
    "separateflux": 1,
    "numsols": 3,
    "relacc1": 1e-3,
    "relacc2": 1e-2,
    "maxruns": 1,
    "maxpts": 1e6,
    "timeout": 60,
    "ETGmult": 1,
    "collmult": 1,
    "integration_routine": 1,
}

# Default kthetarhos grid (binormal wavenumber spectrum, k_y*rho_s)
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

# Named presets layered on top of _STANDARD. Mirrors the spirit of
# GACODEdefaults.addTGLFcontrol()'s code_settings presets, but there is no
# external *.models.yaml file here -- everything is a literal dict so this
# module has no dependency outside qualikiz_tools.
_PRESETS = {
    "STANDARD": {},
    "FAST": {
        "numsols": 2,
        "relacc1": 1e-2,
        "relacc2": 5e-2,
        "maxpts": 5e4,
        "timeout": 30,
    },
    "ROTATION": {
        "rot_flag": 2,
    },
}


def addQLKcontrol(code_settings=None, minimal=False, **kwargs):
    """
    Build the "meta" control dictionary for a QuaLiKiz run.

    code_settings : None or a key in _PRESETS (case-insensitive), e.g. "FAST".
    minimal       : if True, ignore code_settings and use the bare minimum set.
    """

    if minimal or code_settings == 0:
        options = dict(_MINIMAL)
        return options

    options = dict(_STANDARD)

    if code_settings is not None:
        preset_key = str(code_settings).upper()
        if preset_key in _PRESETS:
            options.update(_PRESETS[preset_key])
        else:
            from mitim_tools.misc_tools.LOGtools import printMsg as print
            print(f"\t- code_settings = {code_settings} not found in QLKdefaults presets, using STANDARD", typeMsg="w")

    return options


def default_kthetarhos():
    return list(_KTHETARHOS_STANDARD)
