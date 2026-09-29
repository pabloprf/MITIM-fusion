'''
load_balance strategy 'extra_points': the perturbed CGYRO cases that run on nodes freed by
early radii, and their fluxes as extra surrogate training rows.

The scheduler lives in SIMtools/SCHEDULERtools and the hooks in CGYROtools; this module only
builds the perturbed input.cgyro and reads the finished cases back. See
templates/namelist.portals.yaml (`load_balance`) for the user-facing description.
'''

import json
from pathlib import Path

import numpy as np

from mitim_tools.simulation_tools import SIMtools
from mitim_tools.misc_tools.LOGtools import printMsg as print


class ExtraPointHarvester:
    '''
    One instance per gyrokinetic evaluation that asked for the strategy. `builder()` returns
    the callable CGYROtools invokes when a node frees up; `harvest()` reads back whatever the
    allocation managed to finish.
    '''

    # Surrogate x-vector at the perturbed radius; SURROGATEtools assembles from the model's x_names
    X_VARIABLES = ["aLte", "aLti", "aLne", "aLnZ", "aLw0_n", "nuei", "tite", "w0_n", "beta_e"]

    # channel -> (gradient variable, target key [MW/m2 or 1E20/m2/s], GB normalization key, flux prefix)
    CHANNELS = {
        "te": ("aLte", "QeMWm2", "Qgb", "Qe"),
        "ti": ("aLti", "QiMWm2", "Qgb", "Qi"),
        "ne": ("aLne", "Ge1E20m2", "Ggb", "Ge"),
    }

    # An extra case counts when it ran to MAX_TIME (CGYRO's EXIT line) or when the watchdog
    # stopped it past min_time (mitim_budget.tag)
    COMPLETION = SIMtools.CompletionSpec("out.cgyro.info", "EXIT", alt_file="mitim_budget.tag")

    def __init__(self, power_transport, code, rho_locations, run_kwargs, read_kwargs):
        self.power = power_transport
        self.code = code
        self.rho_locations = list(rho_locations)
        self.run_kwargs = run_kwargs
        self.read_kwargs = read_kwargs
        lb = run_kwargs.get("load_balance") or {}
        self.perturbation = float((lb.get("extra_points") or {}).get("perturbation", 0.15))

    @classmethod
    def usable(cls, folder):
        return cls.COMPLETION.finished(folder)[0]

    # ------------------------------------------------------------------

    def builder(self):
        '''
        builder(rho, finished_scratch_dir, local_dir) -> input.cgyro path (or None), called by
        CGYROtools when the scheduler frees a node.

        The extra case is the finished radius with the gradient of its largest-residual channel
        scaled by (1 -/+ perturbation), where the residual is (turbulent flux of the finished
        run + neoclassical flux of this evaluation - target) in GB units. The factor multiplies
        the SIGNED gradient, so it lowers the magnitude only where the gradient is positive; on
        a hollow profile (a/Ln < 0) it moves the gradient toward zero instead. Profiles are
        rebuilt from a copy of the powerstate, post-processed like the main call, and turned
        into a single-radius input.cgyro through the normal prep chain. The surrogate x-vector
        and the normalizations go to local_dir/mitim_extra_point.json.
        '''
        from mitim_tools.gacode_tools import CGYROtools
        from mitim_tools.gacode_tools.utils import CGYROutils

        postproc = self.power._resolve_postproc_fun(self.code)
        channels = [c for c in self.power.powerstate.predicted_channels if c in self.CHANNELS]

        def build(rho, scratch_dir, local_dir):
            if not channels:
                return None
            k = int(np.argmin([abs(r - rho) for r in self.rho_locations]))
            ir = k + 1
            out = CGYROutils.CGYROoutput(Path(scratch_dir), suffix=None, minimal=True, **self.read_kwargs)

            residual = {}
            for ch in channels:
                aL, tar_key, gb_key, flux = self.CHANNELS[ch]
                gb = float(self.power.powerstate.plasma[gb_key][0, ir])
                target = float(self.power.powerstate.plasma[tar_key][0, ir]) / gb
                neoc = float(getattr(self.power, f"{flux}GB_neoc", np.zeros(len(self.rho_locations)))[k])
                residual[ch] = (float(getattr(out, f"{flux}_mean")) + neoc - target, abs(target))
            # +1.0 puts GB heat (O(1-100)) and GB particle (O(0.1)) residuals on one scale
            ch = max(residual, key=lambda c: abs(residual[c][0]) / (residual[c][1] + 1.0))
            factor = (1.0 - self.perturbation) if residual[ch][0] > 0 else (1.0 + self.perturbation)
            aL = self.CHANNELS[ch][0]

            ps = self.power.powerstate.copy_state()
            ps.plasma[aL][:, ir] = ps.plasma[aL][:, ir] * factor
            ps.update_var(ch)
            ps.calculateProfileFunctions()
            x = {v: float(ps.plasma[v][0, ir]) for v in self.X_VARIABLES if v in ps.plasma}

            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            file_profs = local_dir / "input.gacode"
            ps.copy_state().from_powerstate(
                write_input_gacode=file_profs,
                postprocess_input_gacode=self.power.powerstate.transport_options["applyCorrections"],
                rederive_profiles=True, insert_highres_powers=True)
            if postproc is not None:
                postproc(file_profs)

            cg = CGYROtools.CGYRO(rhos=[rho])
            cg.prep(file_profs, local_dir)
            cg._preprocess_options = self.run_kwargs.get("preprocess_options")
            cg._run_prepare("base_cgyro", extraOptions=self.run_kwargs.get("extraOptions", {}),
                            multipliers=self.run_kwargs.get("multipliers", {}),
                            code_settings=self.run_kwargs.get("code_settings"),
                            allocation=self.run_kwargs.get("allocation"),
                            ApplyCorrections=self.run_kwargs.get("ApplyCorrections", True),
                            Quasineutral=self.run_kwargs.get("Quasineutral", False),
                            cold_start=True, forceIfcold_start=True, launchSlurm=False)
            input_cgyro = local_dir / "base_cgyro" / f"input.cgyro_{rho:.4f}"

            meta = {"rho": rho, "radius_index": ir, "channel": ch, "variable": aL, "factor": factor,
                    "residual_GB": residual[ch][0],
                    "parent_fluxes_GB": {c: float(getattr(out, f"{self.CHANNELS[c][3]}_mean")) for c in channels},
                    "x": x, "Qgb": float(ps.plasma["Qgb"][0, ir]), "Ggb": float(ps.plasma["Ggb"][0, ir]),
                    "Pgb": float(ps.plasma["Pgb"][0, ir]),
                    "evaluation_number": int(getattr(self.power, "evaluation_number", 0))}
            (local_dir / "mitim_extra_point.json").write_text(json.dumps(meta, indent=2))
            print(f"\t- [extra point] rho={rho:.4f}: {aL} x {factor:.3f} ({ch} residual {residual[ch][0]:+.2f} GB)", typeMsg="i")
            return input_cgyro

        return build

    def harvest(self):
        '''
        Read the accepted extra cases under <folder>/extra_cgyro/rho_*/ with the same averaging
        as the main radii and append one row per model (Qe/Qi/Ge_tr_turb_<k>) to
        Outputs/extra_points.csv, with named x columns. Each folder is harvested once: the
        marker goes down only after the CSV is on disk, so a failed write is retried next time
        instead of losing the points.
        '''
        import pandas as pd
        from mitim_tools.gacode_tools.utils import CGYROutils

        root = Path(self.power.folder) / "extra_cgyro"
        csv = Path(self.power.powerstate.transport_options["folder"]) / "Outputs" / "extra_points.csv"

        rows, harvested_folders = [], []
        for d in (sorted(root.glob("rho_*")) if root.is_dir() else []):
            meta_f, done = d / "mitim_extra_point.json", d / "mitim_harvested"
            if done.exists() or not (meta_f.exists() and self.usable(d) and (d / "out.cgyro.time").exists()):
                continue
            meta = json.loads(meta_f.read_text())
            try:
                out = CGYROutils.CGYROoutput(d, suffix=None, minimal=True, **self.read_kwargs)
            except Exception as e:
                print(f"\t- [extra point] {d.name} could not be read ({type(e).__name__}: {e}); skipped", typeMsg="w")
                continue
            base = {"x_names": repr(list(meta["x"].keys())), **meta["x"], "evaluation": meta["evaluation_number"],
                    "rho": meta["rho"], "channel": meta["channel"], "factor": meta["factor"],
                    "t_end": float(out.t[-1]) if hasattr(out, "t") else None, "source": str(d)}
            for flux in ("Qe", "Qi", "Ge"):
                mean, std = getattr(out, f"{flux}_mean", None), getattr(out, f"{flux}_std", None)
                if mean is not None:
                    rows.append({"Model": f"{flux}_tr_turb_{meta['radius_index']}", "y": float(mean), "yvar": float(std) ** 2, **base})
            harvested_folders.append(done)

        if not rows:
            return

        df = pd.DataFrame(rows)
        if csv.exists():
            df = pd.concat([pd.read_csv(csv), df], ignore_index=True)
        csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)
        for done in harvested_folders:
            done.write_text("harvested\n")
        print(f"\t- [extra point] {len(rows)} surrogate row(s) appended to {csv}", typeMsg="i")
