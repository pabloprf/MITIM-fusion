'''
Statistics on a harvest database: which inputs matter for each flux, three ways.

    1. Spearman rank correlation (global, marginal): flux vs each input over the whole database. Captures
       any monotonic relation, but is confounded when inputs co-vary (e.g. collisionality, beta and q all
       follow the radius in a PORTALS run).
    2. Partial rank correlation coefficient, PRCC (global, conditional): the rank correlation of flux and
       input after removing, by least squares on ranks, the effect of all the other inputs. The standard
       global sensitivity measure for correlated, nonlinear-but-monotonic models. Undefined (NaN) for an
       input that the others determine almost completely (R^2 > collinear_r2 on ranks).
    3. Local sensitivities from one-at-a-time scans (TGLF std scan trick): records that are identical in
       every input except one input group (the ~+-2% perturbations around each base point) give a
       finite-difference derivative at that point, free of confounding:
            elasticity  S = d ln Q / d ln x = (dQ/dx) * x0 / Q0     (heat fluxes, where |Q0| is not ~0)
            dQ/d ln x   = (dQ/dx) * x0                             (all fluxes, in the flux's own units)
       with dQ/dx the least-squares slope over the cluster and (x0, Q0) its median-x record. A cluster counts as
       local only with at least 3 points (base and two members, as the scan trick makes) and a span of at most
       max_step_frac of EITHER |x0| OR the input's 5-95% range in the database (the first fails near zero, where
       the scan trick uses a small absolute step, e.g. a/Lne ~ 0; the second fails for narrowly explored inputs,
       e.g. Ti/Te). Others, such as two PORTALS evaluations that happen to differ in a single gradient, are
       secants and are discarded.

Inputs that are identical in every record (e.g. RLNS_1..4 under quasineutrality) are treated as one group,
named after its first member.
'''

import numpy as np
import pandas as pd
from mitim_tools.misc_tools.LOGtools import printMsg as print


class harvest_statistics:

    # Physics inputs analysed per code ('Te'/'Ti'/'ne' = electron / main-ion gradients resolved by charge);
    # geometry kept to a few non-degenerate descriptors (the rest are functions of the radius)
    _INPUTS = {
        'tglf':  ['Te', 'Ti', 'ne', 'TAUS_2', 'XNUE', 'BETAE', 'ZEFF', 'Q_LOC', 'Q_PRIME_LOC', 'RMIN_LOC', 'KAPPA_LOC', 'DELTA_LOC'],
        'neo':   ['Te', 'Ti', 'ne', 'TEMP_1', 'NU_1', 'RHO_STAR', 'Q', 'SHEAR', 'RMIN_OVER_A', 'KAPPA', 'DELTA'],
        'cgyro': ['Te', 'Ti', 'ne', 'NU_EE', 'BETAE_UNIT', 'Q', 'S', 'RMIN', 'KAPPA', 'DELTA',
                  'nu_ee', 'beta_star', 'q', 's', 'rmin', 'kappa', 'delta'],   # lowercase: records before schema 5
    }
    _FLUXES = ('Qe', 'Qi', 'Ge')

    def __init__(self, db, code, run=None, inputs=None, collinear_r2=0.995, min_records=30):
        self.db, self.code, self.run = db, code, run
        self.collinear_r2, self.min_records = collinear_r2, min_records
        self.df = db.load(code, run=run, with_run_info=False)
        self.fluxes = {k: db._first(self.df, 'out_', v) for k, v in db._FLUXES.items() if k in self._FLUXES}
        self.fluxes = {k: v for k, v in self.fluxes.items() if v is not None}
        self.inputs = self._resolve_inputs(inputs)      # {label: column}
        self._spearman = self._prcc = self._local = None

    # -------------------------------------------------------------------------- inputs
    def _resolve_inputs(self, names):
        df, db = self.df, self.db
        if len(df) == 0:
            return {}
        drives = db._drives(self.code, df)
        names = names or self._INPUTS.get(self.code)
        if names is None:
            cov = db.coverage(self.code, run=self.run)
            names = list(cov[(cov['n_distinct'] > 1) & (~cov['input'].str.upper().str.startswith('SHAPE'))]['input'])
        out = {}
        for n in names:
            col = drives.get(n) if n in ('Te', 'Ti', 'ne') else (f'in_{n}' if f'in_{n}' in df.columns else None)
            if col is None or col in out.values() or df[col].nunique() <= 1:
                continue
            label = f"{db._DRIVE_LABELS[n]} ({col[3:]})" if n in ('Te', 'Ti', 'ne') else col[3:]
            out[label] = col
        return out

    @property
    def enough(self):
        return len(self.df) >= self.min_records and len(self.inputs) >= 2 and len(self.fluxes) > 0

    # -------------------------------------------------------------------------- 1. Spearman
    def spearman(self):
        '''DataFrame (inputs x fluxes) of Spearman rank correlations'''
        if self._spearman is None:
            ranks = self.df[list(self.inputs.values()) + list(self.fluxes.values())].rank()
            corr = ranks.corr(method='pearson')   # Pearson on ranks = Spearman (ties averaged)
            self._spearman = pd.DataFrame({f: [corr.loc[c, fc] for c in self.inputs.values()] for f, fc in self.fluxes.items()},
                                          index=list(self.inputs))
        return self._spearman

    # -------------------------------------------------------------------------- 2. PRCC
    def prcc(self):
        '''DataFrame (inputs x fluxes) of partial rank correlations; NaN where the input is collinear with the others'''
        if self._prcc is None:
            cols = list(self.inputs.values())
            sub = self.df[cols + list(self.fluxes.values())].dropna()
            R = sub.rank().to_numpy(dtype=float)
            R = (R - R.mean(axis=0)) / np.where(R.std(axis=0) > 0, R.std(axis=0), 1.0)
            X, nx = R[:, :len(cols)], len(cols)
            out = {f: [] for f in self.fluxes}
            self.collinear = []
            for i in range(nx):
                others = np.column_stack([np.ones(len(X)), np.delete(X, i, axis=1)])
                beta_x, *_ = np.linalg.lstsq(others, X[:, i], rcond=None)
                res_x = X[:, i] - others @ beta_x
                r2 = 1 - res_x.var() / X[:, i].var() if X[:, i].var() > 0 else 1.0
                collinear = r2 > self.collinear_r2
                if collinear:
                    self.collinear.append(list(self.inputs)[i])
                for j, f in enumerate(self.fluxes):
                    y = R[:, nx + j]
                    if collinear or y.std() == 0:
                        out[f].append(np.nan)
                        continue
                    beta_y, *_ = np.linalg.lstsq(others, y, rcond=None)
                    res_y = y - others @ beta_y
                    den = np.sqrt((res_x ** 2).sum() * (res_y ** 2).sum())
                    out[f].append(float((res_x * res_y).sum() / den) if den > 0 else np.nan)
            self._prcc = pd.DataFrame(out, index=list(self.inputs))
        return self._prcc

    # -------------------------------------------------------------------------- 3. local sensitivities
    def local_sensitivities(self, rel_floor=1e-3, max_step_frac=0.1, min_points=3):
        '''
        One row per (one-at-a-time cluster, flux): input group scanned, x0, Q0, slope dQ/dx, dQ/dlnx and the
        elasticity dlnQ/dlnx (NaN for Ge, and where |Q0| < rel_floor * median|Q|). Clusters with fewer than
        min_points, or spanning more than max_step_frac of both |x0| and the input's 5-95% range, are discarded
        (counted in self.n_wide_clusters). Empty when the database has no one-at-a-time scans (e.g. NEO,
        CGYRO, or harvest.scan_trick_members: false).
        '''
        if self._local is not None:
            return self._local
        df = self.df
        varying = [c for c in df.columns if c.startswith('in_') and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
        # input groups that always move together (identical columns)
        groups, seen = [], {}
        for c in varying:
            key = np.round(np.nan_to_num(df[c].to_numpy(dtype=float), nan=np.inf), 12).tobytes()
            if key in seen:
                seen[key].append(c)
            else:
                seen[key] = [c]
                groups.append(seen[key])
        group_of = {g[0]: g for g in groups}
        label_of = {col: lab for lab, col in self.inputs.items()}
        rounded = df[varying].round(10)
        med = {f: np.nanmedian(np.abs(df[fc].to_numpy(dtype=float))) for f, fc in self.fluxes.items()}
        rows, self.n_wide_clusters = [], 0
        span = {rep: float(np.subtract(*np.nanpercentile(df[rep].to_numpy(dtype=float), [95, 5]))) for rep in group_of}
        for rep, members in group_of.items():
            rest = [c for c in varying if c not in members]
            if not rest:
                continue
            key = pd.util.hash_pandas_object(pd.concat([df['run'], rounded[rest]], axis=1), index=False)
            sizes = key.map(key.value_counts())
            idx = np.where(sizes.to_numpy() >= 2)[0]
            if len(idx) == 0:
                continue
            for _, grp in df.iloc[idx].groupby(key.iloc[idx]):
                x = grp[rep].to_numpy(dtype=float)
                if np.ptp(x) <= 1e-9 * max(np.abs(x).max(), 1e-12):
                    continue
                i0 = int(np.argsort(x)[len(x) // 2])
                x0 = x[i0]
                if len(x) < min_points or np.ptp(x) > max_step_frac * max(span[rep], abs(x0)):
                    self.n_wide_clusters += 1
                    continue
                for f, fc in self.fluxes.items():
                    q = grp[fc].to_numpy(dtype=float)
                    if not np.all(np.isfinite(q)):
                        continue
                    slope = np.polyfit(x, q, 1)[0]
                    q0 = q[i0]
                    elast = slope * x0 / q0 if (f != 'Ge' and abs(q0) > rel_floor * med[f]) else np.nan
                    rows.append({'input': label_of.get(rep, rep[3:]), 'column': rep, 'group': ','.join(m[3:] for m in members),
                                 'flux': f, 'n': len(x), 'x0': x0, 'Q0': q0, 'dQdx': slope, 'dQdlnx': slope * x0, 'elasticity': elast})
        self._local = pd.DataFrame(rows)
        return self._local

    # -------------------------------------------------------------------------- report
    def interpret(self, top=3):
        '''Text summary: strongest PRCC per flux, and median [IQR] local elasticities per scanned input'''
        if not self.enough:
            return f"  {self.code}: not enough records/varying inputs for statistics"
        lines = [f"Statistics ({len(self.df)} records, inputs: {', '.join(self.inputs)})"]
        pr, sp = self.prcc(), self.spearman()
        for f in self.fluxes:
            s = pr[f].dropna().sort_values(key=np.abs, ascending=False).head(top)
            lines.append(f"  {f}: strongest partial rank correlations: " + ", ".join(f"{k} {v:+.2f} (Spearman {sp.loc[k, f]:+.2f})" for k, v in s.items()))
        if getattr(self, 'collinear', None):
            lines.append(f"  PRCC undefined (collinear with the other inputs): {', '.join(self.collinear)}")
        loc = self.local_sensitivities()
        if len(loc):
            lines.append(f"  Local sensitivities from {loc.groupby(['column']).size().sum() // max(len(self.fluxes), 1)} one-at-a-time clusters "
                         f"(median [25%, 75%] of d ln Q / d ln x; Ge: d Ge / d ln x):")
            for inp, g in loc.groupby('input', sort=False):
                parts = []
                for f in self.fluxes:
                    v = g.loc[g.flux == f, 'dQdlnx' if f == 'Ge' else 'elasticity'].dropna()
                    if len(v):
                        parts.append(f"{f} {v.median():+.2g} [{v.quantile(.25):+.2g}, {v.quantile(.75):+.2g}]")
                lines.append(f"    {inp:22s} ({len(g) // max(len(self.fluxes), 1)} points): " + "; ".join(parts))
        else:
            lines.append("  No one-at-a-time scans in the database (local sensitivities need the TGLF scan-trick members)")
        if getattr(self, 'n_wide_clusters', 0):
            lines.append(f"  ({self.n_wide_clusters} one-at-a-time groups with < 3 points or spanning > 10% of both the input's value and range discarded: secants, not local derivatives)")
        return "\n".join(lines)

    # -------------------------------------------------------------------------- plots
    def plotImportance(self, fn=None):
        '''Heatmaps inputs x fluxes of Spearman and PRCC (color: -1 blue .. +1 red; gray = undefined)'''
        import matplotlib.pyplot as plt
        if not self.enough:
            return None
        fig = fn.add_figure(label=f'{self.code.upper()} stats') if fn is not None else plt.figure(figsize=(14, 8))
        axs = fig.subplots(1, 2)
        for ax, (title, tab) in zip(axs, (('Spearman rank correlation (marginal)', self.spearman()),
                                          ('Partial rank correlation, PRCC (others removed)', self.prcc()))):
            M = np.ma.masked_invalid(tab.to_numpy(dtype=float))
            cmap = plt.get_cmap('RdBu_r').copy()
            cmap.set_bad('lightgray')
            im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
            for (i, j), v in np.ndenumerate(tab.to_numpy(dtype=float)):
                ax.text(j, i, '--' if not np.isfinite(v) else f'{v:+.2f}', ha='center', va='center', fontsize=8,
                        color='w' if np.isfinite(v) and abs(v) > 0.6 else 'k')
            ax.set_xticks(range(tab.shape[1]))
            ax.set_xticklabels(tab.columns)
            ax.set_yticks(range(tab.shape[0]))
            ax.set_yticklabels(tab.index, fontsize=8)
            ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=axs, shrink=0.8, label='rank correlation')
        fig.text(0.01, 0.99, f"{self.code.upper()}, {len(self.df)} records. Spearman: flux vs input over the whole database "
                             f"(confounded by inputs that co-vary). PRCC: same after removing the other inputs' rank-linear effect; "
                             f"'--' = input collinear with the others (R^2 > {self.collinear_r2}).", fontsize=8, va='top', wrap=True)
        return fig

    def plotSensitivities(self, fn=None, color_by='XNUE', min_clusters=10):
        '''
        Local derivatives from one-at-a-time scans. Left: distribution of d ln Q / d ln x per scanned input for
        Qe and Qi (symlog). Middle: d Ge / d ln x. Right: elasticity vs the input itself for Qi-a/LTi and
        Qe-a/LTe (stiffness vs drive), colored by `color_by` when stored.
        '''
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        loc = self.local_sensitivities()
        if len(loc) // max(len(self.fluxes), 1) < min_clusters:
            return None
        fig = fn.add_figure(label=f'{self.code.upper()} sensitivities') if fn is not None else plt.figure(figsize=(18, 9))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1])
        ax_el, ax_ge = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
        inputs = list(dict.fromkeys(loc['input']))
        pos = np.arange(len(inputs))
        for k, (f, color) in enumerate((('Qe', 'tab:red'), ('Qi', 'tab:blue'))):
            data = [loc[(loc.input == i) & (loc.flux == f)]['elasticity'].dropna().to_numpy() for i in inputs]
            ok = [d if len(d) else np.array([np.nan]) for d in data]
            bp = ax_el.boxplot(ok, positions=pos + (k - 0.5) * 0.35, widths=0.3, vert=False, patch_artist=True, showfliers=False)
            for b in bp['boxes']:
                b.set_facecolor(color); b.set_alpha(0.6)
            ax_el.plot([], [], 's', color=color, alpha=0.6, label=f)
        ax_el.set_xscale('symlog', linthresh=1)
        ax_el.axvline(0, color='k', lw=0.5)
        ax_el.set_yticks(pos)
        ax_el.set_yticklabels(inputs, fontsize=8)
        ax_el.set_xlabel('d ln Q / d ln x  (symlog; box 25-75%, whiskers 1.5 IQR)')
        ax_el.set_title('Heat-flux elasticity per scanned input', fontsize=10)
        ax_el.legend(fontsize=8, loc='lower right')
        data = [loc[(loc.input == i) & (loc.flux == 'Ge')]['dQdlnx'].dropna().to_numpy() for i in inputs]
        bp = ax_ge.boxplot([d if len(d) else np.array([np.nan]) for d in data], positions=pos, widths=0.5, vert=False,
                           patch_artist=True, showfliers=False)
        for b in bp['boxes']:
            b.set_facecolor('tab:green'); b.set_alpha(0.6)
        ax_ge.axvline(0, color='k', lw=0.5)
        ax_ge.set_yticks(pos)
        ax_ge.set_yticklabels([])
        ax_ge.set_xlabel('d Ge / d ln x  (flux units)')
        ax_ge.set_title('Particle-flux sensitivity', fontsize=10)
        for ax in (ax_el, ax_ge):
            ax.grid(True, axis='x', alpha=0.3)
            ax.set_ylim(-0.7, len(inputs) - 0.3)
        # stiffness vs drive
        ccol = f'in_{color_by}' if f'in_{color_by}' in self.df.columns else None
        cvals = self.df[ccol] if ccol else None
        norm = LogNorm(vmin=cvals[cvals > 0].min(), vmax=cvals.max()) if cvals is not None and (cvals > 0).any() else None
        drives = self.db._drives(self.code, self.df)
        sc = None
        for row, (f, d) in enumerate((('Qi', 'Ti'), ('Qe', 'Te'))):
            ax = fig.add_subplot(gs[row, 2])
            col = drives.get(d)
            sub = loc[(loc.flux == f) & (loc.column == col)].dropna(subset=['elasticity']) if col else loc.iloc[0:0]
            if len(sub) == 0:
                ax.text(0.5, 0.5, f'no scans of {self.db._DRIVE_LABELS[d]}', ha='center', va='center', transform=ax.transAxes)
                continue
            if norm is not None:
                c = self.df.loc[self.df[col].isin(sub['x0']), ccol].groupby(self.df[col]).first().reindex(sub['x0']).to_numpy()
                sc = ax.scatter(sub['x0'], sub['elasticity'], c=c, norm=norm, cmap='viridis', s=8, alpha=0.8)
            else:
                ax.scatter(sub['x0'], sub['elasticity'], s=8, alpha=0.7)
            ax.set_yscale('symlog', linthresh=1)
            lo, hi = np.percentile(sub['elasticity'], [1, 99])   # a few points just above threshold (Q0 ~ 0) reach 1e4
            ax.set_ylim(min(lo, -1) * 1.5, max(hi, 1) * 1.5)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xlabel(f'{self.db._DRIVE_LABELS[d]} ({col[3:]})')
            ax.set_ylabel(f'd ln {f} / d ln {self.db._DRIVE_LABELS[d]}')
            ax.set_title(f'{f} stiffness vs drive', fontsize=10)
            ax.grid(True, alpha=0.3)
        nclust = len(loc) // max(len(self.fluxes), 1)
        fig.text(0.01, 0.99, f"{self.code.upper()}: {nclust} one-at-a-time clusters (records identical except one input group, "
                             f"e.g. the TGLF +-2% scan trick). Elasticity = d ln Q / d ln x from the least-squares slope over each "
                             f"cluster; Qi/Qe near zero (|Q0| < 1e-3 median) excluded; stiffness panels bounded to the 1-99th percentile.",
                 fontsize=8, va='top', wrap=True)
        fig.subplots_adjust(left=0.1, right=0.92, top=0.9, bottom=0.07, wspace=0.35, hspace=0.35)
        if sc is not None:
            fig.colorbar(sc, cax=fig.add_axes([0.94, 0.1, 0.01, 0.78]), label=f'{color_by} (log)')
        return fig
