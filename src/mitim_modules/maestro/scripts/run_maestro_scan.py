"""One-call launcher for MAESTRO engineering-variable scans.

``launch_scan`` takes scan lists over (R, eps, Bt, fG, fLH), builds a
per-point namelist for each combination -- applying q*-preserving Ip,
Greenwald-consistent density, and Martin-2 LH-threshold heating -- and
submits the lot as a single SLURM job array.

Units: R, a [m]; Bt [T]; Ip [MA]; n [1e20 m^-3]; P [MW].

Example
-------
::

    from mitim_modules.maestro.scripts.run_maestro_scan import launch_scan

    def overrides(nm):
        # any project-specific knobs applied after the engineering math
        pp = nm['maestro']['portals']['parameters_prepare']
        pp['portals_parameters']['target']['options']['targets_evolve'] = ['qie']
        pp['portals_parameters']['solution']['predicted_channels']     = ['te', 'ti']
        pp['initialization_parameters']['zero_source_blocks']           = ['qrad', 'qfus', 'qohme']

    launch_scan(
        base_namelist = "/path/to/arc_V3Amiller.yaml",
        main_folder   = "/path/to/runs/tmarg_analysis/cases2",

        R   = [1.68, 3.5, 4.62],
        eps = [0.25541125541125537, 0.3, 0.352],
        Bt  = [2.0, 7.0, 11.4],
        fLH = [1.0, 2.0],
        fG  = [0.5, 1.0],

        slurm = dict(
            partition      = "sched_mit_psfc_r8",
            environment    = "source /path/to/dev-pixi/setup_env.sh",
            cpus = 16, hours = 8, memory = "100GB",
            max_concurrent = 16,                 # %N throttle on the array
        ),
        apply_overrides = overrides,             # optional
        # ped_vol_G      = 0.9,                  # fGped = fG * ped_vol_G
        # save           = True,                 # forwarded as --save to mitim_run_maestro
        # label_fmt      = None,                 # default 'case_R..._eps..._Bt..._fLH..._fG...'
    )

That submits one sbatch array of 3*3*3*2*2 = 108 tasks, each running
``mitim_run_maestro`` on its own per-point folder.
"""

import itertools
import re
from pathlib import Path

from mitim_tools.misc_tools import IOtools, PLASMAtools
from mitim_tools.opt_tools.scripts.slurm import run_slurm


def launch_scan(
    base_namelist,
    main_folder,
    *,
    R, eps, Bt, fG, fLH,
    slurm,
    apply_overrides=None,
    ped_vol_G=0.9,
    nu_ne=None,
    save=True,
    label_fmt=None,
):
    """Submit a job array over the cartesian product of engineering scan lists.

    For each (R, eps, Bt, fG, fLH) point, write
    ``<main_folder>/<label>/namelist.yaml`` whose engineering quantities
    are mutated from the base namelist as:

        - separatrix.R, separatrix.a = R*eps, Bt
        - Ip rescaled to preserve q* = (R*eps)**2 * Bt / R  (engineering
          kink-safety proxy; NOT a true q95 solve)
        - fGped = fG * ped_vol_G   (neped_20 nulled so the YAML stays self-consistent;
                                     fGped takes precedence inside MAESTRO regardless)
        - ne20  = fG * Greenwald(Ip, a)
        - heating type 'gaussian_sources', Pe = Pi = 0.5 * fLH *
          P_LH_Martin2(ne20, Bt, a, R)

    ``apply_overrides(nm)`` is an optional callable invoked after the
    engineering math, so the caller can set project-specific knobs
    (per-beat overrides, target options, etc.) without editing the base
    namelist.

    All points are submitted as one SLURM job array of length
    ``len(R)*len(eps)*len(Bt)*len(fG)*len(fLH)``; ``slurm['max_concurrent']``
    becomes the ``%N`` throttle.

    Parameters
    ----------
    base_namelist : str | Path
        Path to the base maestro namelist (YAML).
    main_folder : str | Path
        Folder under which all per-point folders are written.
    R, eps, Bt, fG, fLH : list[float]
        Engineering scan axes; cartesian product is taken.
    slurm : dict
        Required keys: partition, environment, cpus, hours, memory.
        Optional: max_concurrent.
    apply_overrides : callable | None
        ``apply_overrides(nm)`` called per point, after engineering math.
    ped_vol_G : float
        Coupling fGped = fG * ped_vol_G  (default 0.9).
    nu_ne : float | list[float] | None
        EPED-initializer density peaking factor
        (nm['plasma']['initialization']['parameters']['nu_ne']). If None
        (default), the value already in the base namelist is left untouched.
        Scalar overrides every point to the same value with no folder-name
        change. List makes nu_ne a scan axis and adds ``_nune{value:.3f}``
        to each per-point folder name.
    save : bool
        Pass --save to mitim_run_maestro (figures auto-saved).
    label_fmt : str | None
        Format string for per-point folder names; default
        ``"case_R{R:.3f}_eps{eps:.3f}_Bt{Bt:.3f}_fLH{fLH:.3f}_fG{fG:.3f}"``,
        with ``_nune{nu_ne:.3f}`` appended when ``nu_ne`` is a list.
    """
    base_namelist = Path(base_namelist).expanduser().resolve()
    main_folder = Path(main_folder).expanduser().resolve()
    main_folder.mkdir(parents=True, exist_ok=True)

    # Normalize nu_ne: None / scalar -> singleton, no folder suffix;
    # list -> scan axis, folder suffix added.
    nu_ne_is_scan = (nu_ne is not None) and (not isinstance(nu_ne, (int, float)))
    if nu_ne is None:
        nu_ne_list = [None]
    elif isinstance(nu_ne, (int, float)):
        nu_ne_list = [float(nu_ne)]
    else:
        nu_ne_list = [float(n) for n in nu_ne]

    default_fmt = "case_R{R:.3f}_eps{eps:.3f}_Bt{Bt:.3f}_fLH{fLH:.3f}_fG{fG:.3f}"
    if nu_ne_is_scan:
        default_fmt += "_nune{nu_ne:.3f}"
    fmt = label_fmt or default_fmt

    folders = []
    for R_i, eps_i, Bt_i, fLH_i, fG_i, nu_i in itertools.product(
            R, eps, Bt, fLH, fG, nu_ne_list):
        label = fmt.format(R=R_i, eps=eps_i, Bt=Bt_i,
                           fLH=fLH_i, fG=fG_i, nu_ne=nu_i)
        folder = main_folder / label
        folder.mkdir(parents=True, exist_ok=True)

        nm = IOtools.read_mitim_yaml(base_namelist)
        _apply_engineering_point(nm, R=R_i, eps=eps_i, Bt=Bt_i,
                                 fG=fG_i, fLH=fLH_i, ped_vol_G=ped_vol_G,
                                 nu_ne=nu_i)
        if apply_overrides is not None:
            apply_overrides(nm)
        IOtools.write_mitim_yaml(nm, folder / "namelist.yaml")
        folders.append(folder)

    _submit_array(folders, main_folder, slurm=slurm, save=save)


def _apply_engineering_point(nm, *, R, eps, Bt, fG, fLH, ped_vol_G, nu_ne=None):
    """Mutate a maestro namelist dict in-place for one engineering point."""
    if nu_ne is not None:
        nm['plasma']['profiles_initialization']['parameters']['nu_ne'] = nu_ne

    params = nm['plasma']['parameters']
    sep = params['separatrix']

    R_o = sep['R']
    eps_o = sep['a'] / R_o
    Bt_o = params['Bt']
    Ip_o = params['Ip']

    qstar_o = (R_o * eps_o) ** 2 * Bt_o / R_o
    qstar_n = (R   * eps  ) ** 2 * Bt   / R
    Ip = Ip_o * qstar_n / qstar_o
    a  = R * eps

    sep['R'] = R
    sep['a'] = a
    params['Bt'] = Bt
    params['Ip'] = Ip

    params['neped_20'] = None
    params['fGped'] = fG * ped_vol_G

    ne20 = PLASMAtools.Greenwald_density(Ip, a) * fG
    Ptot = PLASMAtools.LHthreshold_Martin2(ne20, Bt, a, R) * fLH

    heat = nm['plasma']['heating']
    heat['type'] = 'gaussian_sources'
    heat['parameters']['Pe'] = Ptot * 0.5
    heat['parameters']['Pi'] = Ptot * 0.5


def _submit_array(folders, main_folder, *, slurm, save):
    """Write scan_folders.txt and submit one sbatch array of len(folders) tasks."""
    listing = main_folder / 'scan_folders.txt'
    listing.write_text('\n'.join(str(f) for f in folders) + '\n')

    n = len(folders)
    mc = slurm.get('max_concurrent')
    job_array = f'0-{n-1}%{mc}' if mc else f'0-{n-1}'

    cpus = slurm['cpus']
    save_flag = '--save' if save else ''
    script = (
        f'F=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {listing}) && '
        f'mitim_run_maestro $F --namelist $F/namelist.yaml --cpus {cpus} {save_flag}'
    ).rstrip()

    run_slurm(script, main_folder, slurm['partition'], slurm['environment'],
              hours=slurm['hours'], n=cpus, mem=slurm['memory'],
              exclusive=False, are_n_threads=False, ntasks_per_node=cpus,
              job_array=job_array)

    _write_per_case_sbatch_stubs(main_folder, folders)


def _write_per_case_sbatch_stubs(main_folder, folders):
    """Mirror the main-folder sbatch log into each case folder as composite IDs.

    mitim_check_maestro discovers each case's SLURM job by reading
    <case>/sbatch_submission.log. With a job-array submission only the main
    folder gets that file from run_slurm; this writes a one-line stub per case
    of the form "Submitted batch job <arrayjob>_<task_id>", which squeue/check
    treats as the array task's composite ID.
    """
    main_log = main_folder / 'sbatch_submission.log'
    if not main_log.exists():
        return
    m = re.search(r'Submitted batch job (\S+)', main_log.read_text())
    if not m:
        return
    array_jobid = m.group(1)
    for task_id, folder in enumerate(folders):
        (folder / 'sbatch_submission.log').write_text(
            f'Submitted batch job {array_jobid}_{task_id}\n'
        )
