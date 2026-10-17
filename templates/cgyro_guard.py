"""
Cost guard of one running CGYRO radius (namelist transport.options.cgyro.run.cost_guard).

templates/cgyro_watchdog.sh runs it on the compute node as `python3 mitim_guard.py <radius folder>`,
next to the mitim_guard.json that CGYROtools.CostGuard staged (thresholds + turbulent targets of this
radius). Standard library only, Python >= 3.6. It prints ONE line and never raises:

    STOP t=.. cost=.. ratio=.. remaining=..    the radius is not worth its wall time
    WAIT ...                                   anything else, including "cannot tell"

Units and conventions (the same ones CGYROutils.CGYROoutput uses):
    t          [a/cs]   first column of out.cgyro.time
    cost       [s per a/cs] mean TOTAL of the out.cgyro.timing rows of the window, divided by the
               output spacing of t (so it does not assume one output per a/cs)
    fluxes     gyro-Bohm units as CGYRO writes them in bin.cgyro.ky_flux, summed over fields
               (phi, A_par, B_par) and ky; Qe is the LAST species, Qi the sum of all the others
    targets    turbulent targets in the same GB units: total target minus neoclassical
"""
import array
import json
import os
import sys

FILE = "mitim_guard.json"


def _numeric_rows(path, ncols=None):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or (ncols is not None and len(parts) != ncols):
                continue
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    return rows


def simulated_times(folder):
    return [r[0] for r in _numeric_rows(os.path.join(folder, "out.cgyro.time"))]


def timing_totals(folder):
    '''TOTAL column [s] of every "Run time" row of out.cgyro.timing, one row per output.'''
    with open(os.path.join(folder, "out.cgyro.timing")) as f:
        lines = f.read().splitlines()
    totals, i = [], 0
    while i < len(lines):
        if lines[i].strip().startswith("Run time") and i + 1 < len(lines):
            names = lines[i + 1].split()
            i += 2
            while i < len(lines):
                parts = lines[i].split()
                if len(parts) != len(names):
                    break
                try:
                    totals.append(float(parts[-1]))
                except ValueError:
                    break
                i += 1
            continue
        i += 1
    return totals


def end_time(folder):
    '''[a/cs] time this launch runs to (.mitim_t_end, written at launch by cgyro_requeue_trim.sh), or None.'''
    try:
        with open(os.path.join(folder, ".mitim_t_end")) as f:
            return float(f.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None


def _input_value(folder, key):
    '''Value of a key in input.cgyro.gen ("<value>  <KEY>" per line), or None.'''
    try:
        with open(os.path.join(folder, "input.cgyro.gen")) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1] == key:
                    return float(parts[0])
    except (OSError, ValueError):
        pass
    return None


def flux_file(folder, n_n):
    '''
    pygacode getflux(cflux='auto'): bin.cgyro.ky_cflux when gamma_e != 0 and n_n > 1, else
    bin.cgyro.ky_flux. gamma_e is read as GAMMA_E of input.cgyro.gen (PROFILE_MODEL=1, what MITIM runs).
    '''
    cflux = os.path.join(folder, "bin.cgyro.ky_cflux")
    if abs(_input_value(folder, "GAMMA_E") or 0.0) > 0.0 and n_n > 1 and os.path.isfile(cflux):
        return cflux
    return os.path.join(folder, "bin.cgyro.ky_flux")


def heat_fluxes(folder, nt, nwin):
    '''
    {"QeGB": mean, "QiGB": mean} over the last `nwin` records of the flux file.

    One record per output holds (n_species, n_moment, n_field, n_n) in Fortran order; moment 1 is
    the energy flux. Precision (4 or 8 bytes) and the number of moments (3, or 4 with the exchange)
    are the pair whose record count is nearest the `nt` rows of out.cgyro.time, as in
    CGYROoutput._reconcile_time_vector. A count more than 2 rows away from nt is not trusted.
    '''
    with open(os.path.join(folder, "out.cgyro.grids")) as f:
        n_n, n_species, n_field = [int(float(v)) for v in f.read().split()[:3]]
    path = flux_file(folder, n_n)
    size = os.path.getsize(path)
    per_moment = n_species * n_field * n_n

    nrec, nbytes, n_moment = min(((size // (b * per_moment * m), b, m) for b in (4, 8) for m in (4, 3)),
                                 key=lambda c: abs(c[0] - nt))
    if abs(nrec - nt) > 2 or nrec < 1:
        raise ValueError("flux records ({}) do not match out.cgyro.time rows ({})".format(nrec, nt))

    nwin = min(nwin, nrec)
    record = per_moment * n_moment
    data = array.array("f" if nbytes == 4 else "d")
    with open(path, "rb") as f:
        f.seek((nrec - nwin) * record * nbytes)
        data.frombytes(f.read(nwin * record * nbytes))

    q_species = [0.0] * n_species
    for k in range(nwin):
        for i_n in range(n_n):
            for i_field in range(n_field):
                base = k * record + n_species * (1 + n_moment * (i_field + n_field * i_n))
                for i_species in range(n_species):
                    q_species[i_species] += data[base + i_species]
    q_species = [q / nwin for q in q_species]
    return {"QeGB": q_species[-1], "QiGB": sum(q_species[:-1])}


def verdict(folder):
    with open(os.path.join(folder, FILE)) as f:
        g = json.load(f)

    t, totals = simulated_times(folder), timing_totals(folder)
    if not t or not totals:
        return "WAIT no output rows yet"
    t_now = t[-1]

    # Window: the outputs of the last `window` a/cs
    nwin = sum(1 for ti in t if ti > t_now - g["window"])
    if len(t) <= nwin or len(totals) < nwin:
        return "WAIT t={:g} trace shorter than the {:g} a/cs window".format(t_now, g["window"])
    dt_output = (t_now - t[-1 - nwin]) / nwin

    # A radius whose cost is coming back down is left alone: the last quarter must be slow too
    nrecent = max(1, nwin // 4)
    cost = min(sum(totals[-nwin:]) / nwin, sum(totals[-nrecent:]) / nrecent) / dt_output

    t_end = end_time(folder)
    remaining_h = None if t_end is None else max(t_end - t_now, 0.0) * cost / 3600.0
    status = "t={:g} cost={:.0f}s_per_acs remaining={}".format(
        t_now, cost, "NA" if remaining_h is None else "{:.1f}h".format(remaining_h))

    if cost < g["seconds_per_acs"]:
        return "WAIT " + status

    # Only a flux ABOVE its target counts: far below it at early times may just not have saturated
    fluxes = heat_fluxes(folder, len(t), nwin)
    ratios = {k: fluxes[k] / target for k, target in g["targets_GB"].items() if k in fluxes and target > 0.0}
    if not ratios:
        return "WAIT " + status + " no positive heat-flux target"
    channel = max(ratios, key=ratios.get)
    status += " ratio={:.1f}x_{}".format(ratios[channel], channel)

    waived = remaining_h is not None and remaining_h >= g["waive_min_time_hours"]
    stop = (ratios[channel] >= g["flux_ratio"]
            and t_now >= g["floor_time"]
            and (t_now >= g["min_time"] or waived))
    return ("STOP " if stop else "WAIT ") + status


if __name__ == "__main__":
    try:
        print(verdict(sys.argv[1]))
    except Exception as e:
        print("WAIT guard error: {}".format(e))
