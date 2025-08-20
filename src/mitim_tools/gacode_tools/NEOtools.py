import os
from pathlib import Path
from mitim_tools import __version__ as mitim_version
from mitim_tools.misc_tools import FARMINGtools, IOtools
from mitim_tools.gacode_tools.utils import GACODErun, GACODEdefaults
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

from mitim_tools.misc_tools.PLASMAtools import md_u

class NEO(GACODErun.gacode_simulation):
    def __init__(
        self,
        rhos=[0.4, 0.6],  # rho locations of interest
    ):
        
        super().__init__(rhos=rhos)

        self.run_specifications = {
            'code': 'neo',
            'input_file': 'input.neo',
            'code_call': 'neo -e',
            'control_function': GACODEdefaults.addNEOcontrol
        }
        
        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t NEO class module")
        print("-----------------------------------------------------------------------------------------\n")

        self.ResultsFiles = self.ResultsFiles_minimal = ['out.neo.transport_flux']

    def prep_direct(
        self,
        mitim_state,    # A MITIM state class
        FolderGACODE,  # Main folder where all caculations happen (runs will be in subfolders)
        cold_start=False,  # If True, do not use what it potentially inside the folder, run again
        forceIfcold_start=False,  # Extra flag
        ):

        print("> Preparation of TGLF run from input.gacode (direct conversion)")
        
        cdf = self._prep_direct(
            mitim_state,
            FolderGACODE,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            state_converter='to_neo',
            input_class=NEOinput,
            input_file='input.neo'
        )

        return cdf

    def run(
        self,
        subFolderNEO,  # 'neo1/',
        NEOsettings=None,
        extraOptions={},
        multipliers={},
        minimum_delta_abs={},
        # runWaveForms=None,  # e.g. runWaveForms = [0.3,1.0]
        # forceClosestUnstableWF=True,  # Look at the growth rate spectrum and run exactly the ky of the closest unstable
        ApplyCorrections=True,  # Removing ions with too low density and that are fast species
        Quasineutral=False,  # Ensures quasineutrality. By default is False because I may want to run the file directly
        launchSlurm=True,
        cold_start=False,
        forceIfcold_start=False,
        extra_name="exe",
        slurm_setup={
            "cores": 4,
            "minutes": 5,
        },  # Cores per NEO call (so, when running nR radii -> nR*4)
        attempts_execution=1,
        only_minimal_files=False,
    ):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare inputs
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        neo_executor, neo_executor_full, folderlast = self.prep_run(
            subFolderNEO,
            neo_executor={},
            neo_executor_full={},
            NEOsettings=NEOsettings,
            extraOptions=extraOptions,
            multipliers=multipliers,
            minimum_delta_abs=minimum_delta_abs,
            # runWaveForms=runWaveForms,
            # forceClosestUnstableWF=forceClosestUnstableWF,
            ApplyCorrections=ApplyCorrections,
            Quasineutral=Quasineutral,
            launchSlurm=launchSlurm,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            extra_name=extra_name,
            slurm_setup=slurm_setup,
            attempts_execution=attempts_execution,
            only_minimal_files=only_minimal_files,
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Run NEO
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self._run(
            neo_executor,
            neo_executor_full=neo_executor_full,
            NEOsettings=NEOsettings,
            # runWaveForms=runWaveForms,
            # forceClosestUnstableWF=forceClosestUnstableWF,
            ApplyCorrections=ApplyCorrections,
            Quasineutral=Quasineutral,
            launchSlurm=launchSlurm,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            extra_name=extra_name,
            slurm_setup=slurm_setup,
            only_minimal_files=only_minimal_files,
        )

        self.FolderNEOlast = folderlast

    def prep_run(
        self,
        subFolder,
        neo_executor={},
        neo_executor_full={},
        NEOsettings=None,
        **kwargs
    ):

        return self._prep_run(
            subFolder,
            code_executor=neo_executor,
            code_executor_full=neo_executor_full,
            code_settings=NEOsettings,
            addControlFunction=self.run_specifications['control_function'],
            **kwargs
        )

    def _run(
            self,
            neo_executor,
            neo_executor_full={},
            **kwargs_NEOrun
            ):
        """
        extraOptions and multipliers are not being grabbed from kwargs_NEOrun, but from neo_executor for WF
        """

        print("\n> Run NEO")

        self._generic_run(
            neo_executor,
            self.run_specifications,
            **kwargs_NEOrun
        )
















    def prep(self, inputgacode, folder):
        self.inputgacode = inputgacode
        self.folder = IOtools.expandPath(folder)

        self.folder.mkdir(parents=True, exist_ok=True)



    def run_vgen(self, subfolder="vgen1", vgenOptions={}, cold_start=False):

        self.folder_vgen = self.folder / f"{subfolder}"

        # ---- Default options

        vgenOptions.setdefault("er", 2)
        vgenOptions.setdefault("vel", 1)
        vgenOptions.setdefault("numspecies", len(self.inputgacode.Species))
        vgenOptions.setdefault("matched_ion", 1)
        vgenOptions.setdefault("nth", "17,39")

        # ---- Prepare

        runThisCase = check_if_files_exist(
            self.folder_vgen,
            [
                ["vgen", "input.gacode"],
                ["vgen", "input.neo.gen"],
                ["out.vgen.neoequil00"],
                ["out.vgen.neoexpnorm00"],
                ["out.vgen.neontheta00"],
                ["vgen.dat"],
            ],
        )

        if (not runThisCase) and cold_start:
            runThisCase = print("\t- Files found in folder, but cold_start requested. Are you sure?",typeMsg="q",)

            if runThisCase:
                IOtools.askNewFolder(self.folder_vgen, force=True)

        self.inputgacode.write_state(file=(self.folder_vgen / f"input.gacode"))

        # ---- Run

        if runThisCase:
            file_new = GACODErun.runVGEN(
                self.folder_vgen, vgenOptions=vgenOptions, name_run=subfolder
            )
        else:
            print(f"\t- Required files found in {subfolder}, not running VGEN",typeMsg="i",)
            file_new = self.folder_vgen / f"vgen" / f"input.gacode"

        # ---- Postprocess

        from mitim_tools.gacode_tools import PROFILEStools
        self.inputgacode_vgen = PROFILEStools.gacode_state(file_new, derive_quantities=True, mi_ref=self.inputgacode.mi_ref)


def check_if_files_exist(folder, list_files):
    folder = IOtools.expandPath(folder)

    for file_parts in list_files:
        checkfile = folder
        for ii in range(len(file_parts)):
            checkfile = checkfile / f"{file_parts[ii]}"
        if not checkfile.exists():
            return False

    return True



class NEOinput:
    def __init__(self, file=None):
        self.file = IOtools.expandPath(file) if isinstance(file, (str, Path)) else None

        if self.file is not None and self.file.exists():
            with open(self.file, "r") as f:
                lines = f.readlines()
            file_txt = "".join(lines)
        else:
            file_txt = ""
        input_dict = GACODErun.buildDictFromInput(file_txt)

        self.process(input_dict)

    @classmethod
    def initialize_in_memory(cls, input_dict):
        instance = cls()
        instance.process(input_dict)
        return instance

    def process(self, input_dict):
        #TODO
        self.all = input_dict

        self.num_recorded = 6

    def anticipate_problems(self):
        pass

    def write_state(self, file=None):
        
        if file is None:
            file = self.file

        # Local formatter: floats -> 6 significant figures in exponential (uppercase),
        # ints stay as ints, bools as 0/1, sequences space-separated with same rule.
        def _fmt_num(x):
            import numpy as _np
            if isinstance(x, (bool, _np.bool_)):
                return "True" if x else "False"
            if isinstance(x, (_np.floating, float)):
                # 6 significant figures in exponential => 5 digits after decimal
                return f"{float(x):.5E}"
            if isinstance(x, (_np.integer, int)):
                return f"{int(x)}"
            return str(x)

        def _fmt_value(val):
            import numpy as _np
            if isinstance(val, (list, tuple, _np.ndarray)):
                # Flatten numpy arrays but keep ordering; join with spaces
                if isinstance(val, _np.ndarray):
                    flat = val.flatten().tolist()
                else:
                    flat = list(val)
                return " ".join(_fmt_num(v) for v in flat)
            return _fmt_num(val)
        
        with open(file, "w") as f:
            f.write("#-------------------------------------------------------------------------\n")
            f.write(f"# NEO input file modified by MITIM {mitim_version}\n")
            f.write("#-------------------------------------------------------------------------\n")

            for ikey in self.all:
                var = self.all[ikey]
                f.write(f"{ikey.ljust(23)} = {_fmt_value(var)}\n")