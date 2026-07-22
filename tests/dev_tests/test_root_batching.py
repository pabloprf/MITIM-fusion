"""
Do PORTALS' acquisition inner-solvers -- ROOT (scipy LM) and SR (simple relaxation,
the cheap "first technique") -- work with batches, and does solving N starting points
in PARALLEL (one batched call) give the SAME result as SEQUENTIALLY (N independent calls)?

How each solver treats the batch
--------------------------------
(opt_tools/optimizers/multivariate_tools.py)

ROOT (`scipy_root`):
  Flattens the [N, dim] guesses to one [N*dim] vector and calls scipy.optimize.root
  (method='lm') ONCE. The Jacobian is block-diagonal (off-blocks zeroed), so the
  Newton/LM *direction* decouples per start -- BUT MINPACK shares ONE global damping
  lambda, ONE convergence test on the TOTAL residual sum-of-squares (ftol), and global
  scaling. -> batched solve can stop a slow member early, damp it differently, or (multi-
  root) route it into a different basin than it would reach alone.

SR (`simple_relaxation`):
  The update `_sr_step` (dx = relax*(QT-Q)/sqrt(Q^2+QT^2), capped, x += dx*|x|) and the
  dynamic-relax oscillation check are ENTIRELY element-wise over [N, channels] -- each
  start (and channel) relaxes independently, no cross-member term. The ONLY batch
  coupling is the STOP: the whole batch stops when the BEST member meets the (relative)
  tolerance `tol = tol_rel * M_initial_max` (anchored to the best STARTING point). So a
  slower member is simply TRUNCATED at the iteration the best one converges -- no basin
  jumping, just under-convergence relative to running alone.

PORTALS defaults (templates/namelist.optimization.yaml): root num_restarts=5, ftol=1e-4;
sr num_restarts=5, tol_rel=1e-3. So PORTALS runs both as "5 in parallel".

This script tests the solvers in isolation (analytic residuals, no GP). Flux-match
problems are in PORTALS convention (transport Q increasing in the gradient x, target QT),
so BOTH solvers apply. multiroot2d is an abstract F=0 system (ROOT only) for the basin demo.

Usage
-----
    run_with_env.sh python tests/dev_tests/test_root_batching.py            # one FigureNotebook window, a tab per problem
    run_with_env.sh python tests/dev_tests/test_root_batching.py --save     # headless: save the tabs to tests/scratch/root_batching/
"""

import io
import sys
import contextlib

import numpy as np
import torch
import matplotlib.pyplot as plt

from mitim_tools import __mitimroot__
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.opt_tools.optimizers import multivariate_tools

torch.set_default_dtype(torch.double)
DTYPE = torch.double
SP = torch.nn.functional.softplus

OUTDIR = __mitimroot__ / "tests" / "scratch" / "root_batching"
SAME_ROOT_TOL = 1e-4   # ||x_seq - x_par|| below this -> "same solution"

# Tolerances per solver: PORTALS default first, then a tightened one
FTOL_ROOT  = {"PORTALS (1e-4)": 1e-4, "tight (1e-10)": 1e-10}
TOLREL_SR  = {"PORTALS (1e-3)": 1e-3, "tight (1e-6)": 1e-6}
PORTALS_TOL = {"root": "PORTALS (1e-4)", "sr": "PORTALS (1e-3)"}


# ------------------------------------------------------------------------------------------------
# Evaluators in the interface the solvers expect: return (transport, target, metric)
# residual = target - transport ; metric M = -(1/N)||residual||_2  (PORTALS calculate_residuals)
# ------------------------------------------------------------------------------------------------

def make_flux_evaluator(transport, target):
    """PORTALS-convention flux-match: transport(x) increasing in x, target(x)."""
    def ev(X, y_history=None, x_history=None, metric_history=None):
        Q, QT = transport(X), target(X)
        M = -torch.linalg.norm(QT - Q, dim=-1) / QT.shape[-1]
        if metric_history is not None:
            metric_history.append(M.detach())
        if x_history is not None:
            x_history.append(X.detach())
        if y_history is not None:
            y_history.append((QT - Q).detach())
        return Q, QT, M
    return ev


def make_residual_evaluator(F):
    """Abstract F(x)=0 system (ROOT only): transport=0, target=F so residual=F."""
    def ev(X, y_history=None, x_history=None, metric_history=None):
        res = F(X)
        Q, QT = torch.zeros_like(res), res
        M = -torch.linalg.norm(res, dim=-1) / res.shape[-1]
        if metric_history is not None:
            metric_history.append(M.detach())
        if x_history is not None:
            x_history.append(X.detach())
        if y_history is not None:
            y_history.append(res.detach())
        return Q, QT, M
    return ev


# ------------------------------------------------------------------------------------------------
# Solver runners -> (x_sol [N,dim], paths [list of [n_iter,dim]], resid_traj [list of [n_iter]])
# ------------------------------------------------------------------------------------------------

def _split(x_sol, y_hist, x_hist, n):
    paths, traj = [], []
    for i in range(n):
        paths.append(x_hist[:, i, :].cpu().numpy())
        traj.append(np.linalg.norm(y_hist[:, i, :].cpu().numpy(), axis=-1))
    return x_sol.detach(), paths, traj


def run_root(ev, x0, ftol):
    opts = {"algorithm_options": {"maxiter": 1000, "ftol": ftol}, "solver": "lm", "write_trajectory": True}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        x_sol, y_hist, x_hist, *_ = multivariate_tools.scipy_root(ev, x0, bounds=None, solver_options=opts)
    return _split(x_sol, y_hist, x_hist, x0.shape[0])


def run_sr(ev, x0, tol_rel):
    opts = {"tol_rel": tol_rel, "maxiter": 2000, "relax": 0.1, "relax_dyn": True,
            "print_each": 100000, "write_trajectory": True}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        x_sol, y_hist, x_hist, *_ = multivariate_tools.simple_relaxation(ev, x0, bounds=None, solver_options=opts)
    return _split(x_sol, y_hist, x_hist, x0.shape[0])


RUNNERS = {"root": (run_root, FTOL_ROOT), "sr": (run_sr, TOLREL_SR)}


def resid_norm(ev, x):
    with torch.no_grad():
        Q, QT, _ = ev(x)
        return torch.linalg.norm(QT - Q, dim=-1).cpu().numpy()


def channel_residual(ev, x):
    with torch.no_grad():
        Q, QT, _ = ev(x)
        return (QT - Q).abs().cpu().numpy()   # [N, channels]


# ------------------------------------------------------------------------------------------------
# Problems
# ------------------------------------------------------------------------------------------------

def _fm_transport(x, beta, coupling=0.0):
    """gyroBohm-ish stiff transport: monotone increasing in the gradient x."""
    Q = beta * SP(x) ** 1.5
    if coupling:
        Q = Q + coupling * x.mean(dim=-1, keepdim=True)
    return Q


_U_T = torch.tensor([1.0, 1.4, 0.8, 1.2, 0.9, 1.1], dtype=DTYPE)
_U_B = torch.tensor([0.6, 0.5, 0.7, 0.55, 0.65, 0.6], dtype=DTYPE)

# multiscale: heat O(1), particle O(0.05), trace impurity O(2e-3) -> tests per-channel balance
_M_T = torch.tensor([1.0, 1.3, 0.05, 0.04, 2.0e-3, 1.5e-3], dtype=DTYPE)
_M_B = torch.tensor([0.6, 0.5, 0.05, 0.04, 2.0e-3, 1.5e-3], dtype=DTYPE)
_M_CH = ["Qe(heat)", "Qi(heat)", "Ge(part)", "Ge2(part)", "GZ(imp)", "GZ2(imp)"]


def _multiroot2d(X):
    x1, x2 = X[:, 0], X[:, 1]
    return torch.stack([x1 ** 2 + x2 ** 2 - 4.0, x1 * x2 - 1.0], dim=-1)


def _multiroot2d_roots():
    roots = []
    for u in (2 + np.sqrt(3), 2 - np.sqrt(3)):
        x1 = np.sqrt(u)
        for s in (+1, -1):
            roots.append((s * x1, 1.0 / (s * x1)))
    return np.array(roots)


def _starts(n, dim, lo, hi, seed):
    rng = np.random.RandomState(seed)
    return torch.tensor(rng.uniform(lo, hi, size=(n, dim)), dtype=DTYPE)


PROBLEMS = [
    dict(name="fm_uniform", dim=6, methods=["root", "sr"],
         make_ev=lambda: make_flux_evaluator(lambda x: _fm_transport(x, _U_B, coupling=0.05),
                                             lambda x: _U_T.expand_as(x)),
         # one near-root start + spread-out ones (exposes SR's global-stop truncation)
         x0=torch.cat([torch.full((1, 6), 1.0, dtype=DTYPE), _starts(4, 6, -0.5, 3.5, 11)], dim=0)),

    dict(name="fm_multiscale", dim=6, methods=["root", "sr"], chan=_M_CH,
         make_ev=lambda: make_flux_evaluator(lambda x: _fm_transport(x, _M_B),
                                             lambda x: _M_T.expand_as(x)),
         x0=_starts(5, 6, -0.5, 2.5, 12)),

    dict(name="fm_2d", dim=2, methods=["root", "sr"],
         make_ev=lambda: make_flux_evaluator(
             lambda x: _fm_transport(x, torch.tensor([0.6, 0.5], dtype=DTYPE)),
             lambda x: torch.tensor([1.0, 1.2], dtype=DTYPE).expand_as(x)),
         x0=_starts(5, 2, -1.0, 3.0, 13)),

    dict(name="multiroot2d", dim=2, methods=["root"], roots=_multiroot2d_roots(),
         make_ev=lambda: make_residual_evaluator(_multiroot2d),
         x0=torch.tensor([[1.8, 1.8], [-1.8, 1.8], [1.8, -1.8], [-1.8, -1.8], [0.3, 2.2]], dtype=DTYPE)),
]


# ------------------------------------------------------------------------------------------------
# Sequential vs parallel comparison
# ------------------------------------------------------------------------------------------------

def compare(prob, method, tol):
    runner = RUNNERS[method][0]
    ev, x0 = prob["make_ev"](), prob["x0"]
    n = x0.shape[0]

    x_seq = torch.empty_like(x0)
    paths_seq, traj_seq = [], []
    for i in range(n):
        xi, p, t = runner(ev, x0[i:i + 1], tol)
        x_seq[i] = xi[0]
        paths_seq.append(p[0]); traj_seq.append(t[0])

    x_par, paths_par, traj_par = runner(ev, x0, tol)

    return dict(
        x_seq=x_seq, x_par=x_par,
        delta=torch.linalg.norm(x_seq - x_par, dim=-1).cpu().numpy(),
        res_seq=resid_norm(ev, x_seq), res_par=resid_norm(ev, x_par),
        paths_seq=paths_seq, paths_par=paths_par, traj_seq=traj_seq, traj_par=traj_par,
    )


def diagnostics(prob, method, tol):
    runner = RUNNERS[method][0]
    ev, x0 = prob["make_ev"](), prob["x0"]
    same = x0[0:1].repeat(4, 1)
    x_same, _, _ = runner(ev, same, tol)
    spread = torch.linalg.norm(x_same - x_same[0:1], dim=-1).max().item()
    x_fwd, _, _ = runner(ev, x0, tol)
    x_rev, _, _ = runner(ev, torch.flip(x0, dims=[0]), tol)
    perm = torch.linalg.norm(x_fwd - torch.flip(x_rev, dims=[0]), dim=-1).max().item()
    return spread, perm


# ------------------------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------------------------

def plot_problem(prob, results, fig):
    """One row of 3 panels per method: solution mismatch, final residual, convergence."""
    name, n = prob["name"], prob["x0"].shape[0]
    methods = prob["methods"]
    starts = np.arange(n)

    # Each method gets a row of 3 uniquely-labelled panels (subplot_mosaic needs distinct chars)
    alphabet = "ABCDEFGHIJKL"
    panels = {m: alphabet[3 * k:3 * k + 3] for k, m in enumerate(methods)}
    axd = fig.subplot_mosaic("\n".join(panels[m] for m in methods))

    for m in methods:
        tols = list(RUNNERS[m][1].keys())
        lab0 = PORTALS_TOL[m]
        p = panels[m]

        # (1) ||x_seq - x_par|| per start, grouped by tolerance
        ax = axd[p[0]]
        width = 0.8 / len(tols)
        for j, lab in enumerate(tols):
            d = np.maximum(results[(m, lab)]["delta"], 1e-16)
            ax.bar(starts + j * width, d, width, label=lab)
        ax.axhline(SAME_ROOT_TOL, color="k", ls="--", lw=1)
        ax.set_yscale("log"); ax.set_ylabel(r"$\|x_{seq}-x_{par}\|$")
        ax.set_title(f"[{m.upper()}] seq vs parallel"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # (2) final residual, seq vs par (PORTALS tol)
        ax = axd[p[1]]
        r = results[(m, lab0)]
        ax.bar(starts - 0.2, np.maximum(r["res_seq"], 1e-16), 0.4, label="sequential")
        ax.bar(starts + 0.2, np.maximum(r["res_par"], 1e-16), 0.4, label="parallel")
        ax.set_yscale("log"); ax.set_ylabel(r"$\|$resid$\|$")
        ax.set_title(f"[{m.upper()}] final residual ({lab0})"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # (3) convergence (PORTALS tol): seq solid, par dashed
        ax = axd[p[2]]
        colors = plt.cm.viridis(np.linspace(0, 0.9, n))
        for i in range(n):
            ax.plot(np.maximum(r["traj_seq"][i], 1e-16), color=colors[i], lw=1.4)
            ax.plot(np.maximum(r["traj_par"][i], 1e-16), color=colors[i], lw=1.1, ls="--")
        ax.set_yscale("log"); ax.set_xlabel("residual evaluation #"); ax.set_ylabel(r"$\|$resid$\|$")
        ax.set_title(f"[{m.upper()}] convergence; solid=seq dashed=par"); ax.grid(True, alpha=0.3)

    fig.suptitle(f"{name}", fontsize=12)


def plot_channel_shares(prob, results, fig):
    """Per-channel residual at the solution -- shows which channels the global stop under-resolves."""
    ev = prob["make_ev"]()
    chan = prob["chan"]
    ax = fig.subplot_mosaic("A")["A"]
    x = np.arange(len(chan))
    width = 0.8 / len(prob["methods"])
    for j, m in enumerate(prob["methods"]):
        r = results[(m, PORTALS_TOL[m])]
        # worst (max over starts) per-channel residual at the parallel solution
        ch = channel_residual(ev, r["x_par"]).max(axis=0)
        ax.bar(x + j * width, np.maximum(ch, 1e-18), width, label=f"{m.upper()} (parallel)")
    ax.set_yscale("log"); ax.set_xticks(x + 0.4 - width / 2); ax.set_xticklabels(chan, rotation=30, ha="right")
    ax.set_ylabel("max over starts of |target - transport|")
    ax.set_title(f"{prob['name']}: per-channel residual at the solution (PORTALS tol)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)


def plot_paths_2d(prob, results, fig):
    """2D contour of log10||residual|| with ROOT and SR paths (seq solid, par dashed)."""
    ev = prob["make_ev"]()
    g = np.linspace(-3, 3, 250)
    GX, GY = np.meshgrid(g, g)
    pts = torch.tensor(np.stack([GX.ravel(), GY.ravel()], -1), dtype=DTYPE)
    with torch.no_grad():
        Q, QT, _ = ev(pts)
        Z = torch.linalg.norm(QT - Q, dim=-1).cpu().numpy().reshape(GX.shape)

    methods = prob["methods"]
    axd = fig.subplot_mosaic("".join(f"{m[0]}" for m in methods))
    n = prob["x0"].shape[0]
    colors = plt.cm.rainbow(np.linspace(0, 1, n))
    for m in methods:
        ax = axd[m[0]]
        ax.contourf(GX, GY, np.log10(Z + 1e-6), levels=25, cmap="Greys")
        if "roots" in prob:
            ax.plot(prob["roots"][:, 0], prob["roots"][:, 1], "*", color="gold", ms=16,
                    markeredgecolor="k", zorder=5)
        r = results[(m, PORTALS_TOL[m])]
        for i in range(n):
            ps, pp = r["paths_seq"][i], r["paths_par"][i]
            ax.plot(ps[:, 0], ps[:, 1], "-", color=colors[i], lw=1.5)
            ax.plot(pp[:, 0], pp[:, 1], "--", color=colors[i], lw=1.1)
            ax.plot(*prob["x0"][i].cpu().numpy(), "P", color=colors[i], ms=10, markeredgecolor="k")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_title(f"[{m.upper()}] solid=seq dashed=par +=start")
    fig.suptitle(f"{prob['name']} paths", fontsize=12)


def plot_summary(summary, fig):
    ax = fig.subplot_mosaic("A")["A"]
    keys = [f"{nm}\n[{m}]" for (nm, m) in dict.fromkeys((s[0], s[1]) for s in summary)]
    pairs = list(dict.fromkeys((s[0], s[1]) for s in summary))
    tol_kinds = ["PORTALS", "tight"]
    x = np.arange(len(pairs)); width = 0.4
    for j, kind in enumerate(tol_kinds):
        vals = []
        for nm, m in pairs:
            v = [s[3] for s in summary if s[0] == nm and s[1] == m and kind in s[2]]
            vals.append(max(v[0], 1e-16) if v else 1e-16)
        ax.bar(x + j * width, vals, width, label=kind)
    ax.axhline(SAME_ROOT_TOL, color="k", ls="--", lw=1, label=f"same-root tol ({SAME_ROOT_TOL:g})")
    ax.set_yscale("log"); ax.set_xticks(x + width / 2); ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylabel(r"max over starts of $\|x_{seq}-x_{par}\|$")
    ax.set_title("Sequential vs parallel: agreement by problem and solver")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

def main(show=True, save_folder=None):
    fn = FigureNotebook("ROOT/SR batching", geometry="1800x950", vertical=True, show=show)

    summary = []   # (name, method, tol_label, max_delta, n_disagree, n)
    for prob in PROBLEMS:
        name = prob["name"]
        print(f"==== {name} (dim={prob['dim']}, {prob['x0'].shape[0]} starts) ====")
        results = {}
        for m in prob["methods"]:
            for lab, tol in RUNNERS[m][1].items():
                res = compare(prob, m, tol)
                results[(m, lab)] = res
                n = prob["x0"].shape[0]
                n_dis = int((res["delta"] > SAME_ROOT_TOL).sum())
                summary.append((name, m, lab, float(res["delta"].max()), n_dis, n))
                print(f"  [{m:>4} {lab}] max||x_seq-x_par||={res['delta'].max():.2e} | "
                      f"disagree={n_dis}/{n} | max resid seq={res['res_seq'].max():.1e} par={res['res_par'].max():.1e}")
            sp, pm = diagnostics(prob, m, RUNNERS[m][1][PORTALS_TOL[m]])
            print(f"  [{m:>4}] diagnostics: identical-starts spread={sp:.1e} | permutation mismatch={pm:.1e}")

        plot_problem(prob, results, fn.add_figure(label=name))
        if "chan" in prob:
            plot_channel_shares(prob, results, fn.add_figure(label=f"{name} channels"))
        if prob["dim"] == 2:
            plot_paths_2d(prob, results, fn.add_figure(label=f"{name} paths"))
        print()

    plot_summary(summary, fn.add_figure(label="SUMMARY"))

    # ---- verdict ----
    print("================= VERDICT =================")
    print("Both ROOT and SR run with batches (PORTALS uses 5 starts in parallel for each).")
    print(f"  {'problem':<14}{'solver':<7}{'tolerance':<16}{'max dx':>11}{'disagree':>11}")
    for name, m, lab, mx, ndis, n in summary:
        flag = "  <-- DIFFER" if ndis > 0 else ""
        print(f"  {name:<14}{m:<7}{lab:<16}{mx:>11.2e}{f'{ndis}/{n}':>11}{flag}")
    print("-------------------------------------------")
    print("ROOT: batched LM shares a global damping + convergence + scaling -> can stop slow")
    print("      members early or (multi-root) land in a different basin than sequential.")
    print("SR:   per-member element-wise relaxation -> identical trajectories; the ONLY batch")
    print("      coupling is the global stop (batch halts when the BEST start meets tol),")
    print("      which TRUNCATES slower members -> under-converged vs running them alone.")

    if save_folder is not None:
        save_folder.mkdir(parents=True, exist_ok=True)
        fn.save(save_folder)
        print(f"\n* Saved notebook tabs to {save_folder}")
    if show:
        fn.show()


if __name__ == "__main__":
    save = "--save" in sys.argv
    main(show=not save, save_folder=OUTDIR if save else None)
