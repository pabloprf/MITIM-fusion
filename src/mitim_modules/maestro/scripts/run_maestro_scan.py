"""One-call launcher for MAESTRO engineering-variable scans.

``launch_scan_cartesian`` takes scan lists over (R, eps, Bt, fG, fLH, nu_ne)
plus optional separatrix-shape axes (kappa_sep, delta_sep) and builds every
combination; ``launch_scan`` is the underlying entry point that
takes an explicit list of point dicts (use it to run an arbitrary subset rather
than a full grid). Either way each point gets a per-point namelist -- applying
q*-preserving Ip, Greenwald-consistent density, and Martin-2 LH-threshold
heating -- and the lot is submitted as a single SLURM job array.

Units: R, a [m]; Bt [T]; Ip [MA]; n [1e20 m^-3]; P [MW].

Example (full cartesian grid)
-----------------------------
::

    from mitim_modules.maestro.scripts.run_maestro_scan import launch_scan_cartesian

    def overrides(nm):
        # any project-specific knobs applied after the engineering math
        pp = nm['maestro']['portals']['parameters_prepare']
        pp['portals_parameters']['target']['options']['targets_evolve'] = ['qie']
        pp['portals_parameters']['solution']['predicted_channels']     = ['te', 'ti']
        pp['initialization_parameters']['zero_source_blocks']           = ['qrad', 'qfus', 'qohme']

    launch_scan_cartesian(
        base_namelist = "/path/to/arc_V3Amiller.yaml",
        main_folder   = "/path/to/runs/tmarg_analysis/cases2",

        R   = [1.68, 3.5, 4.62],
        eps = [0.25541125541125537, 0.3, 0.352],
        Bt  = [2.0, 7.0, 11.4],
        fLH = [1.0, 2.0],
        fG  = [0.5, 1.0],
        nu_ne = [1.1, 1.4],                      # density peaking; [value] for a single point
        # kappa_sep = [1.8, 2.0],                # optional separatrix elongation scan (None -> base)
        # delta_sep = [0.4, 0.6],                # optional separatrix triangularity scan (None -> base)

        slurm = dict(
            partition      = "sched_mit_psfc_r8",
            environment    = "source /path/to/dev-pixi/setup_env.sh",
            cpus = 16, hours = 8, memory = "100GB",
            max_concurrent = 16,                 # %N throttle on the array
        ),
        apply_overrides = overrides,             # optional
        # ped_vol_G      = 0.9,                  # fGped = fG * ped_vol_G
        # save           = True,                 # forwarded as --save to mitim_run_maestro
        # label_fmt      = None,                 # default 'case_R..._eps..._Bt..._fLH..._fG..._nune...'
    )

That submits one sbatch array of 3*3*3*2*2*2 = 216 tasks, each running
``mitim_run_maestro`` on its own per-point folder.

To run an arbitrary subset instead of a full grid, call ``launch_scan``
directly with the exact points::

    from mitim_modules.maestro.scripts.run_maestro_scan import launch_scan

    launch_scan(
        base_namelist = "/path/to/arc_V3Amiller.yaml",
        main_folder   = "/path/to/runs/subset",
        combinations  = [
            dict(R=4.62, eps=0.30, Bt=7.0,  fG=0.5, fLH=2.0, nu_ne=1.1),
            dict(R=3.50, eps=0.32, Bt=11.4, fG=1.0, fLH=1.0, nu_ne=1.4),
        ],
        slurm = dict(partition="...", environment="...", cpus=16, hours=8, memory="100GB"),
    )
"""

import itertools
import re
from pathlib import Path

from mitim_tools.misc_tools import IOtools, PLASMAtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.opt_tools.scripts.slurm import run_slurm


def launch_scan(
    base_namelist,
    main_folder,
    *,
    combinations,
    slurm,
    apply_overrides=None,
    ped_vol_G=0.9,
    baseline_gacode=None,
    save=True,
    label_fmt=None,
    per_case_logs=True,
):
    """Submit a job array over an explicit list of engineering points.

    For each combination (a dict with keys R, eps, Bt, fG, fLH, nu_ne) write
    ``<main_folder>/<label>/namelist.yaml`` whose engineering quantities
    are mutated from the base namelist as:

        - separatrix.R = R, separatrix.a = R*eps, params.Bt = Bt
        - separatrix.kappa_sep / delta_sep set when scanned (else base kept)
        - Ip set by one of, in precedence order: an explicit ``Ip`` [MA]; else a
          target ``q95`` (taken as the q*_ITER target); else rescaled to preserve
          the baseline q*_ITER. q*_ITER is the Uckan-1990 (kappa/delta) + ITER eps
          correction cylindrical safety factor (PLASMAtools.evaluate_qstar), using
          95%-surface kappa95/delta95 obtained by scaling the separatrix kappa/delta
          with fixed ratios from ``baseline_gacode`` (or used directly, with a
          warning, when none is given); an engineering proxy, NOT a true q95 solve
        - nu_ne -> plasma.profiles_initialization.parameters.nu_ne
        - fGped = fG * ped_vol_G   (neped_20 nulled so the YAML stays self-consistent;
                                     fGped takes precedence inside MAESTRO regardless)
        - ne20  = fG * Greenwald(Ip, a)
        - heating type 'gaussian_sources', Pe = Pi = 0.5 * fLH *
          P_LH_Martin2(ne20, Bt, a, R)

    Pass exactly the points you want here. For the common case of every
    combination of per-axis lists, use ``launch_scan_cartesian``, which builds
    the cartesian product and forwards it to this function.

    ``apply_overrides(nm)`` is an optional callable invoked after the
    engineering math, so the caller can set project-specific knobs
    (per-beat overrides, target options, etc.) without editing the base
    namelist.

    All points are submitted as one SLURM job array of length
    ``len(combinations)``; ``slurm['max_concurrent']`` becomes the ``%N`` throttle.

    Parameters
    ----------
    base_namelist : str | Path
        Path to the base maestro namelist (YAML).
    main_folder : str | Path
        Folder under which all per-point folders are written.
    combinations : list[dict]
        Explicit engineering points. Each dict carries keys R, eps, Bt, fG, fLH
        and may also carry the optional knobs nu_ne (EPED-initializer density
        peaking, plasma.profiles_initialization.parameters.nu_ne), kappa_sep and
        delta_sep (separatrix elongation/triangularity,
        plasma.parameters.separatrix.{kappa_sep,delta_sep}), and q95 or Ip [MA]
        for the plasma current (explicit Ip wins; else q95 is used as the q*_ITER
        target; else Ip is rescaled to preserve the baseline q*_ITER -- give at
        most one of q95/Ip). Each optional knob may be None or omitted to leave
        the base value untouched, in which case its folder-name suffix (``_nune`` /
        ``_ksep`` / ``_dsep`` / ``_q95`` / ``_Ip``) is dropped. These are the same
        keys ``label_fmt`` formats with. A reserved ``label`` key, if present, is
        used verbatim as that point's folder name (overriding label_fmt / the
        default), which is handy for named scenarios (e.g. one point per machine).
    slurm : dict
        Required keys: partition, environment, cpus, hours, memory.
        Optional: max_concurrent.
    apply_overrides : callable | None
        ``apply_overrides(nm)`` called per point, after engineering math.
    ped_vol_G : float
        Coupling fGped = fG * ped_vol_G  (default 0.9).
    baseline_gacode : str | Path | None
        Baseline input.gacode used to map separatrix kappa/delta to the 95%-surface
        kappa95/delta95 (fixed kappa95/kappa_sep, delta95/delta_sep ratios) for the
        q*_ITER current rescale. If None, the separatrix values are used directly and
        a warning is printed.
    save : bool
        Pass --save to mitim_run_maestro (figures auto-saved).
    label_fmt : str | None
        Format string for per-point folder names; default
        ``"case_R{R:.3f}_eps{eps:.3f}_Bt{Bt:.3f}_fLH{fLH:.3f}_fG{fG:.3f}"`` with
        ``_nune{nu_ne:.3f}`` appended only when nu_ne is set (dropped when None).
    per_case_logs : bool
        If True (default), redirect each array task's stdout/stderr into its
        own case folder as ``slurm.out`` / ``slurm.err``, so a failing case's
        traceback lives next to its inputs. The array-level SLURM files in
        ``main_folder`` (``slurm_*_%A_%a.dat``) can't target the case folder
        -- the folder name is only resolved from scan_folders.txt at run time
        -- so they stay put and still capture sbatch-level (timeout/OOM/node)
        messages. Set False to keep only the array-level files.
    """
    base_namelist = Path(base_namelist).expanduser().resolve()
    main_folder = Path(main_folder).expanduser().resolve()
    main_folder.mkdir(parents=True, exist_ok=True)

    # Separatrix -> 95% kappa/delta ratios for the q*_ITER current rescale, read once from
    # the baseline equilibrium (None -> fall back to separatrix shaping, warned below).
    shape95_ratios = _shape95_ratios_from_gacode(baseline_gacode)
    if shape95_ratios is not None:
        print(f"\t- Separatrix->95% shape mapping from {Path(baseline_gacode).name}: "
              f"kappa95/kappa_sep={shape95_ratios[0]:.3f}, delta95/delta_sep={shape95_ratios[1]:.3f}")
    elif any(p.get('Ip') is None and p.get('q95') is None for p in combinations):
        # Only relevant when some point falls back to the q*_ITER rescale; points that set
        # Ip or q95 explicitly don't use the separatrix->95% shape mapping.
        print("\t- Warning: no baseline_gacode given; the q*_ITER current rescale will use "
              "the separatrix kappa/delta directly (kappa95/delta95 ~ kappa_sep/delta_sep), "
              "which overstates shaping. Pass baseline_gacode=<input.gacode> to map them.")

    default_fmt = "case_R{R:.3f}_eps{eps:.3f}_Bt{Bt:.3f}_fLH{fLH:.3f}_fG{fG:.3f}"

    folders = []
    for point in combinations:
        point = dict(point)                      # copy so the reserved 'label' can be popped
        label = point.pop('label', None)         # explicit per-point folder name (verbatim)
        if label is None:
            if label_fmt is not None:
                label = label_fmt.format(**point)
            else:
                # Optional engineering knobs (nu_ne, kappa_sep, delta_sep): a None / absent
                # value means "leave the base namelist untouched", so its folder-name tag is
                # appended only when the value is actually set.
                label = default_fmt.format(**point)
                for key, tag in (('nu_ne', 'nune'), ('kappa_sep', 'ksep'), ('delta_sep', 'dsep'),
                                 ('q95', 'q95'), ('Ip', 'Ip')):
                    if point.get(key) is not None:
                        label += f"_{tag}{point[key]:.3f}"
        folder = main_folder / label
        folder.mkdir(parents=True, exist_ok=True)

        nm = IOtools.read_mitim_yaml(base_namelist)
        _apply_engineering_point(nm, ped_vol_G=ped_vol_G, shape95_ratios=shape95_ratios, **point)
        if apply_overrides is not None:
            apply_overrides(nm)
        IOtools.write_mitim_yaml(nm, folder / "namelist.yaml")
        folders.append(folder)

    _submit_array(folders, main_folder, slurm=slurm, save=save,
                  per_case_logs=per_case_logs)


def launch_scan_cartesian(
    base_namelist,
    main_folder,
    *,
    R, eps, Bt, fG, fLH, nu_ne,
    kappa_sep=None, delta_sep=None, q95=None, Ip=None,
    **kwargs,
):
    """Cartesian-product convenience wrapper around ``launch_scan``.

    Builds every combination of the per-axis scan lists (R, eps, Bt, fG, fLH,
    nu_ne, plus the optional separatrix-shape axes kappa_sep, delta_sep and the
    optional current axes q95 or Ip [MA] -- at most one of q95/Ip) and forwards
    them to ``launch_scan`` as an explicit list of points. All other
    keyword arguments (slurm, apply_overrides, ped_vol_G, baseline_gacode, save,
    label_fmt, per_case_logs) are passed straight through.

    Submits the product over all axes. The required axes (R, eps, Bt, fG, fLH,
    nu_ne) must be lists (pass ``[value]`` for a single value). ``nu_ne``,
    ``kappa_sep``, ``delta_sep``, ``q95`` and ``Ip`` may also be ``None`` (or
    ``[None]``) to leave that base-namelist behaviour untouched and drop the
    suffix from folder names; kappa_sep/delta_sep/q95/Ip default to None.
    """
    if q95 is not None and Ip is not None:
        raise ValueError("launch_scan_cartesian: provide only one of q95 or Ip, not both")
    nu_ne_axis = [None] if nu_ne is None else nu_ne
    kappa_axis = [None] if kappa_sep is None else kappa_sep
    delta_axis = [None] if delta_sep is None else delta_sep
    q95_axis   = [None] if q95 is None else q95
    Ip_axis    = [None] if Ip is None else Ip
    combinations = [
        dict(R=R_i, eps=eps_i, Bt=Bt_i, fG=fG_i, fLH=fLH_i,
             nu_ne=nu_i, kappa_sep=k_i, delta_sep=d_i, q95=q_i, Ip=ip_i)
        for R_i, eps_i, Bt_i, fLH_i, fG_i, nu_i, k_i, d_i, q_i, ip_i in itertools.product(
            R, eps, Bt, fLH, fG, nu_ne_axis, kappa_axis, delta_axis, q95_axis, Ip_axis)
    ]
    launch_scan(base_namelist, main_folder, combinations=combinations, **kwargs)


def _apply_engineering_point(nm, *, R, eps, Bt, fG, fLH, ped_vol_G,
                             nu_ne=None, kappa_sep=None, delta_sep=None,
                             q95=None, Ip=None, shape95_ratios=None):
    """Mutate a maestro namelist dict in-place for one engineering point."""
    # nu_ne / kappa_sep / delta_sep = None (or absent) -> leave the base value untouched.
    if nu_ne is not None:
        nm['plasma']['profiles_initialization']['parameters']['nu_ne'] = nu_ne

    params = nm['plasma']['parameters']
    sep = params['separatrix']

    # --- Original geometry + separatrix shaping (read before any mutation) ---
    R_o     = sep['R']
    eps_o   = sep['a'] / R_o
    Bt_o    = params['Bt']
    Ip_o    = params['Ip']
    kappa_o = sep['kappa_sep']
    delta_o = sep['delta_sep']

    # New separatrix shaping: scanned value if given, otherwise keep the base.
    kappa_n = kappa_o if kappa_sep is None else kappa_sep
    delta_n = delta_o if delta_sep is None else delta_sep
    a = R * eps

    # --- Plasma current ---
    # Three ways to set Ip, in precedence order:
    #   1. explicit `Ip` (MA)                -> used directly.
    #   2. target `q95`                      -> Ip from the q*_ITER inversion below
    #                                           (q95 is taken AS the q*_ITER target: the
    #                                           shaping-aware cylindrical safety factor,
    #                                           an engineering proxy for q95, not a flux
    #                                           solve).
    #   3. neither (default)                 -> rescale to preserve the baseline q*_ITER.
    #
    # q*_ITER is the Uckan-1990 (kappa/delta) + ITER aspect-ratio correction safety factor
    # (PLASMAtools.evaluate_qstar, same as MITIMstate's derived['qstar_ITER']). It depends
    # on the 95%-surface kappa95/delta95, obtained here by scaling the separatrix kappa/delta
    # with the fixed kappa95/kappa_sep, delta95/delta_sep ratios from baseline_gacode
    # (shape95_ratios; default 1.0 = separatrix used directly, caller warned).
    if q95 is not None and Ip is not None:
        raise ValueError("_apply_engineering_point: provide only one of q95 or Ip, not both")
    rk, rd = shape95_ratios if shape95_ratios is not None else (1.0, 1.0)
    if Ip is not None:
        Ip_final = Ip
    elif q95 is not None:
        Ip_final = PLASMAtools.evaluate_qstar(
            q95, R, kappa_n * rk, Bt, eps, delta_n * rd,
            isInputIp=False, ITERcorrection=True, includeShaping=True)
    else:
        # When eps/kappa/delta are unchanged this reduces exactly to the old
        # Ip_o * (a^2 Bt/R)_n / (a^2 Bt/R)_o scaling.
        qstar_ITER = PLASMAtools.evaluate_qstar(
            Ip_o, R_o, kappa_o * rk, Bt_o, eps_o, delta_o * rd,
            isInputIp=True, ITERcorrection=True, includeShaping=True)
        Ip_final = PLASMAtools.evaluate_qstar(
            qstar_ITER, R, kappa_n * rk, Bt, eps, delta_n * rd,
            isInputIp=False, ITERcorrection=True, includeShaping=True)

    # --- Write the mutated geometry ---
    sep['R'] = R
    sep['a'] = a
    if kappa_sep is not None:
        sep['kappa_sep'] = kappa_sep
    if delta_sep is not None:
        sep['delta_sep'] = delta_sep
    params['Bt'] = Bt
    params['Ip'] = Ip_final

    params['neped_20'] = None
    params['fGped'] = fG * ped_vol_G

    ne20 = PLASMAtools.Greenwald_density(Ip_final, a) * fG
    # Evaluate the L-H threshold the SAME way MITIM's derived['LH_Martin2'] does, so
    # the power set here matches the achieved fLH = Psol/P_LH:
    #   - nmin low-density correction (LHthreshold_nmin): below n_min the Martin-2
    #     threshold rises again. Omitting it (the default nmin=[0]) set the power up
    #     to ~2x too LOW for low-density / low-fG plasmas -- e.g. JET fG=0.2 had
    #     ne<n_min, giving achieved fLH=0.37 for a requested 0.8.
    #   - (2/mbg)^1.11 isotope factor from the base namelist's fuel: ['D'] -> 2.0,
    #     ['D','T'] -> 2.5 (a D-T threshold is ~22% lower than a pure-D one).
    _ION_MASS = {'H': 1.0, 'D': 2.0, 'T': 3.0}
    fuel = nm['plasma']['species']['fuel']
    mbg = sum(_ION_MASS[s] for s in fuel) / len(fuel)
    nmin = PLASMAtools.LHthreshold_nmin(Ip_final, Bt, a, R)
    Ptot = (PLASMAtools.LHthreshold_Martin2(ne20, Bt, a, R, nmin=nmin)
            * (2.0 / mbg) ** 1.11 * fLH)

    heat = nm['plasma']['heating']
    heat['type'] = 'gaussian_sources'
    heat['parameters']['Pe'] = Ptot * 0.5
    heat['parameters']['Pi'] = Ptot * 0.5


def _shape95_ratios_from_gacode(baseline_gacode):
    """(kappa95/kappa_sep, delta95/delta_sep) read from a baseline input.gacode.

    These fixed ratios convert the (scanned) separatrix kappa/delta into the
    95%-flux-surface values that q*_ITER depends on, so the current rescale matches
    MITIMstate's qstar_ITER convention. Returns None when no baseline is given (caller
    then falls back to the separatrix values directly).
    """
    if baseline_gacode is None:
        return None
    p = PROFILEStools.gacode_state(IOtools.expandPath(baseline_gacode))
    p.derive_quantities()
    kappa_sep = float(p.profiles['kappa(-)'][-1])
    delta_sep = float(p.profiles['delta(-)'][-1])
    return (float(p.derived['kappa95']) / kappa_sep,
            float(p.derived['delta95']) / delta_sep)


def _submit_array(folders, main_folder, *, slurm, save, per_case_logs=True):
    """Write scan_folders.txt and submit one sbatch array of len(folders) tasks.

    When ``per_case_logs`` is True, each case folder gets slurm.out/slurm.err as
    symlinks into SLURM's live array logs (slurm_output/slurm_error_<%A>_<%a>.dat in
    the main folder). The case folder name is only known at run time (after
    sed-reading scan_folders.txt), so the link is created inside the script. We
    symlink rather than redirect so the logs (a) stream live and (b) are reachable
    from BOTH the case folder and the main folder; a redirect would move them out of
    the array logs, and a tee child can be reaped before flushing on a fast-exit
    traceback. mitim stays the last command, so its exit code propagates to SLURM.
    Caveat: the links point up into the main folder -- they dangle if a single case
    folder is copied away on its own.

    `slurm['exclude']` (and `slurm['qos']`) are forwarded to run_slurm when present.
    """
    listing = main_folder / 'scan_folders.txt'
    listing.write_text('\n'.join(str(f) for f in folders) + '\n')

    n = len(folders)
    mc = slurm.get('max_concurrent')
    job_array = f'0-{n-1}%{mc}' if mc else f'0-{n-1}'

    cpus = slurm['cpus']
    save_flag = '--save' if save else ''
    mitim_cmd = f'mitim_run_maestro $F --namelist $F/namelist.yaml --cpus {cpus} {save_flag}'.rstrip()
    if per_case_logs:
        suffix = '${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}'  # %A_%a, expanded at run time
        link_out = f'ln -sf "{main_folder}/slurm_output_{suffix}.dat" "$F/slurm.out"'
        link_err = f'ln -sf "{main_folder}/slurm_error_{suffix}.dat" "$F/slurm.err"'
        run_cmd = f'{link_out} && {link_err} && {mitim_cmd}'
    else:
        run_cmd = mitim_cmd
    script = f'F=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {listing}) && {run_cmd}'

    run_slurm(script, main_folder, slurm['partition'], slurm['environment'],
              hours=slurm['hours'], n=cpus, mem=slurm['memory'],
              exclude=slurm.get('exclude'), qos=slurm.get('qos'),
              max_hours=slurm.get('max_hours', 8),
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
