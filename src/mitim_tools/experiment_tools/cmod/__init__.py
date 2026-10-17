"""Alcator C-Mod experimental-data retrieval (MDSplus via mdsthin). Plotting reuses
`experiment_tools.diiid.plotting` with `machine="cmod"` (or a CMODConnection).
`mdsthin` is an optional dependency: ``pip install mitim-fusion[mds]``. Import the
submodule directly (kept out of this __init__ so the package imports without mdsthin)::

    from mitim_tools.experiment_tools.cmod import retrieval
"""
