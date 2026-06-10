import argparse
from mitim_tools.transp_tools import UFILEStools
from mitim_tools.misc_tools import GRAPHICStools
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from IPython import embed


def _curvature_kappa(R, Z):
    """Return pointwise curvature kappa(s) for a closed planar curve (R(s), Z(s)).

    Uses periodic finite differences with an arbitrary uniform parameterization.
    """
    R = np.asarray(R, dtype=float).copy()
    Z = np.asarray(Z, dtype=float).copy()

    if R.size < 4 or Z.size < 4:
        return np.full_like(R, np.nan, dtype=float)

    # If the curve is explicitly closed (last point == first), drop the duplicate.
    if np.isfinite(R[0]) and np.isfinite(Z[0]) and np.isfinite(R[-1]) and np.isfinite(Z[-1]):
        if np.isclose(R[0], R[-1]) and np.isclose(Z[0], Z[-1]):
            R = R[:-1]
            Z = Z[:-1]

    dR = 0.5 * (np.roll(R, -1) - np.roll(R, 1))
    dZ = 0.5 * (np.roll(Z, -1) - np.roll(Z, 1))
    ddR = np.roll(R, -1) - 2.0 * R + np.roll(R, 1)
    ddZ = np.roll(Z, -1) - 2.0 * Z + np.roll(Z, 1)

    denom = (dR**2 + dZ**2) ** 1.5
    numer = np.abs(dR * ddZ - dZ * ddR)
    kappa = np.full_like(R, np.nan, dtype=float)
    good = denom > 0
    kappa[good] = numer[good] / denom[good]
    return kappa


def _arc_length_param(R, Z):
    """Return normalized arc-length coordinate s in [0, 1) for a closed curve."""
    R = np.asarray(R, dtype=float).copy()
    Z = np.asarray(Z, dtype=float).copy()

    if R.size < 2 or Z.size < 2:
        return np.full_like(R, np.nan, dtype=float)

    if np.isfinite(R[0]) and np.isfinite(Z[0]) and np.isfinite(R[-1]) and np.isfinite(Z[-1]):
        if np.isclose(R[0], R[-1]) and np.isclose(Z[0], Z[-1]):
            R = R[:-1]
            Z = Z[:-1]

    dR = np.roll(R, -1) - R
    dZ = np.roll(Z, -1) - Z
    ds = np.sqrt(dR**2 + dZ**2)
    s = np.concatenate([[0.0], np.cumsum(ds[:-1])])
    L = np.sum(ds)
    if not np.isfinite(L) or L <= 0:
        return np.full_like(R, np.nan, dtype=float)
    return s / L


def _curvature_ratio_along_contour(R, Z, definition="mean"):
    """Return (s, ratio, kappa, kappa_ref) for a contour.

    `definition` controls the normalization used to form a dimensionless ratio
    `ratio(s) = kappa(s) / kappa_ref`.

    Supported `definition` values:
    - "mean": kappa_ref = mean(kappa)
    - "median": kappa_ref = median(kappa)
    - "pXX": percentile, e.g. "p50", "p5", "p95"
    """
    kappa = _curvature_kappa(R, Z)
    s = _arc_length_param(R, Z)

    # Align lengths in case one of the helpers trimmed a duplicated endpoint.
    if s.size != kappa.size:
        n = min(s.size, kappa.size)
        s = s[:n]
        kappa = kappa[:n]

    good = np.isfinite(kappa) & (kappa > 0)
    if np.count_nonzero(good) < 4:
        return s, np.full_like(kappa, np.nan, dtype=float), kappa, np.nan

    definition = str(definition).strip().lower()
    if definition == "mean":
        kappa_ref = float(np.nanmean(kappa[good]))
    elif definition == "median":
        kappa_ref = float(np.nanmedian(kappa[good]))
    elif definition.startswith("p"):
        try:
            pct = float(definition[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid curvature definition '{definition}'. Use 'mean', 'median', or 'pXX'.") from exc
        kappa_ref = float(np.nanpercentile(kappa[good], pct))
    else:
        raise ValueError(f"Invalid curvature definition '{definition}'. Use 'mean', 'median', or 'pXX'.")

    if not np.isfinite(kappa_ref) or kappa_ref <= 0:
        return s, np.full_like(kappa, np.nan, dtype=float), kappa, kappa_ref

    ratio = kappa / kappa_ref
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.nan)
    return s, ratio, kappa, kappa_ref

def plot_rfszfs(rfs_file, zfs_file, ax = None, time_pos = -1, surf_pos = -1, c='b'):
    
    print(f"Plotting RFS from {rfs_file} and ZFS from {zfs_file}")
    
    if ax is None:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6,6))
        
    # Load data (accept either file paths or pre-loaded UFILE objects)
    if isinstance(rfs_file, str):
        rfs = UFILEStools.UFILEtransp()
        rfs.readUFILE(rfs_file)
    else:
        rfs = rfs_file
    R = rfs.Variables['Z'][time_pos, :, surf_pos]

    if isinstance(zfs_file, str):
        zfs = UFILEStools.UFILEtransp()
        zfs.readUFILE(zfs_file)
    else:
        zfs = zfs_file
    Z = zfs.Variables['Z'][time_pos, :, surf_pos]
    
    # Plot
    ax.plot(R, Z, '-o', c=c, markersize=2, lw=0.5, label=f'Time {time_pos}, Surf {surf_pos}')
    ax.set_xlabel('R (m)')
    ax.set_ylabel('Z (m)')
    ax.axis('equal')
    ax.legend()
    GRAPHICStools.addDenseAxis(ax)
    
    return rfs, zfs

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("prefix", type=str)
    argparser.add_argument("--g", required=False, default=None)
    argparser.add_argument(
        "--curv-def",
        required=False,
        default="mean",
        help="Curvature ratio normalization: 'mean', 'median', or 'pXX' (e.g. p50, p5, p95).",
    )

    args = argparser.parse_args()
    rfs_file = args.prefix + ".RFS"
    zfs_file = args.prefix + ".ZFS"
    
    gfile = args.g
    curv_def = args.curv_def
    
    # Capture major and minor radius (geometric)
    rfs = UFILEStools.UFILEtransp()
    rfs.readUFILE(rfs_file)

    zfs = UFILEStools.UFILEtransp()
    zfs.readUFILE(zfs_file)
    
    time = rfs.Variables['X']
    
    R, a = [], []
    for time_pos in range(rfs.Variables['Z'].shape[0]):
    
        R.append( np.mean([rfs.Variables['Z'][time_pos,:,-1].max(),rfs.Variables['Z'][time_pos,:,-1].min()]))
        a.append( 0.5*(rfs.Variables['Z'][time_pos,:,-1].max() - rfs.Variables['Z'][time_pos,:,-1].min()) )
        
    fig, axs = plt.subplots(ncols=3, figsize=(18,6))
       
    if gfile is not None:
        from mitim_tools.gs_tools import GEQtools
        g = GEQtools.MITIMgeqdsk(gfile)
        g.plotFluxSurfaces(ax=axs[0], fluxes=[], color="b",lwB=2, plot1=True, label = '')
        
    cols = GRAPHICStools.listColors()
    cont = 0

    # Plot all contours (all times, all surfaces)
    for surf_pos in reversed(range(rfs.Variables['Z'].shape[2])):
        for time_pos in range(rfs.Variables['Z'].shape[0]):
        
            plot_rfszfs(rfs, zfs, ax=axs[0], time_pos=time_pos, surf_pos=surf_pos, c=cols[cont])
            
            if surf_pos == rfs.Variables['Z'].shape[2]-1:
                axs[1].plot(time[time_pos],R[time_pos], 'o-', c=cols[cont])
                axs[1].plot(time[time_pos],a[time_pos], 'o--', c=cols[cont])
            
            cont += 1   
            
    axs[1].plot(time, R, '-',label='R (m)',c=cols[0])
    axs[1].plot(time, a, '-',label='a (m)',c=cols[1])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('m'); axs[1].set_ylim([0, None])
    axs[1].legend()
    GRAPHICStools.addDenseAxis(axs[1])
    axs[1].set_title('Geometric major and minor radius evolution (Last surface)')

    # Curvature ratio along contour at the last time point
    time_last = -1
    n_surfaces = rfs.Variables['Z'].shape[2]
    ratio_min = np.inf
    ratio_max = -np.inf
    lc = None
    for surf_pos in reversed(range(n_surfaces)):
        R_curve = rfs.Variables['Z'][time_last, :, surf_pos]
        Z_curve = zfs.Variables['Z'][time_last, :, surf_pos]

        s, ratio, kappa, kappa_ref = _curvature_ratio_along_contour(R_curve, Z_curve, definition=curv_def)
        if not np.any(np.isfinite(ratio)):
            continue
        if np.any(np.isfinite(ratio)):
            ratio_min = min(ratio_min, np.nanmin(ratio))
            ratio_max = max(ratio_max, np.nanmax(ratio))
        col = cols[surf_pos % len(cols)]
        axs[2].plot(s, ratio, '-', lw=1.0, alpha=0.9, c=col)

        # Visual mapping: overlay last surface with an arc-length colormap on the R,Z plot
        if surf_pos == n_surfaces - 1 and s.size >= 4:
            # Build closed segments for LineCollection
            pts = np.column_stack([R_curve, Z_curve])
            if pts.shape[0] != s.size:
                pts = pts[: s.size]
            segs = np.concatenate([pts[:, None, :], np.roll(pts, -1, axis=0)[:, None, :]], axis=1)
            s_mid = 0.5 * (s + np.roll(s, -1))
            # Fix wrap-around midpoint for the last segment (s[-1] -> s[0]=0)
            s_mid[-1] = ((s[-1] + 1.0) / 2.0) % 1.0

            lc = LineCollection(segs, array=s_mid, cmap='twilight', linewidths=3.0, alpha=0.9, zorder=5)
            axs[0].add_collection(lc)

            # Add a few anchor markers so s positions are obvious
            s_marks = np.array([0.0, 0.25, 0.5, 0.75])
            for sm in s_marks:
                idx = int(np.nanargmin(np.abs(s - sm)))
                axs[0].plot(pts[idx, 0], pts[idx, 1], marker='x', ms=7, mew=2, c=lc.cmap(sm), zorder=6)
                axs[2].axvline(sm, color=lc.cmap(sm), lw=0.8, alpha=0.5)

    axs[2].set_xlabel('Normalized arc length $s$')
    axs[2].set_ylabel(rf'$\kappa(s)/\kappa_{{{curv_def}}}$')
    axs[2].set_yscale('log')
    if np.isfinite(ratio_min) and np.isfinite(ratio_max) and ratio_min > 0:
        lo = max(ratio_min * 0.8, 1e-4)
        hi = ratio_max * 1.2
        if hi > lo:
            axs[2].set_ylim([lo, hi])
    GRAPHICStools.addDenseAxis(axs[2])
    axs[2].set_title(f'Curvature ratio along contour (t={time[time_last]:.3f} s)')

    if lc is not None:
        cbar = fig.colorbar(lc, ax=axs[0], fraction=0.046, pad=0.04)
        cbar.set_label('Normalized arc length $s$')

    plt.tight_layout()
    plt.show()
    embed()
