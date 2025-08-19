import os
from pathlib import Path
from mitim_tools import __version__ as mitim_version
from mitim_tools.misc_tools import FARMINGtools, IOtools
from mitim_tools.gacode_tools.utils import GACODErun
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

from mitim_tools.misc_tools.PLASMAtools import md_u

class NEO:
    def __init__(
        self,
        rhos=[0.4, 0.6],  # rho locations of interest
    ):
        
        self.rhos = rhos

    def prep(self, inputgacode, folder):
        self.inputgacode = inputgacode
        self.folder = IOtools.expandPath(folder)

        self.folder.mkdir(parents=True, exist_ok=True)


    def prep_direct(
        self,
        mitim_state,    # A MITIM state class
        FolderGACODE,  # Main folder where all caculations happen (runs will be in subfolders)
        cold_start=False,  # If True, do not use what it potentially inside the folder, run again
        forceIfcold_start=False,  # Extra flag
        ):

        print("> Preparation of NEO run from input.gacode (direct conversion)")

        self.FolderGACODE = IOtools.expandPath(FolderGACODE)
        
        if cold_start or not self.FolderGACODE.exists():
            IOtools.askNewFolder(self.FolderGACODE, force=forceIfcold_start)
            
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        self.profiles = mitim_state

        self.profiles.derive_quantities(mi_ref=md_u)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize from state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        self.inputsNEO = self.profiles.to_neo(r=self.rhos, r_is_rho = True)

        for rho in self.inputsNEO:
            
            # Initialize class
            self.inputsNEO[rho] = NEOinput.initialize_in_memory(self.inputsNEO[rho])
                
            # Write input.tglf file
            self.inputsNEO[rho].file = self.FolderGACODE / f'input.neo_{rho:.4f}'
            self.inputsNEO[rho].write_state()

    def run(
        self,
        subfolder,
        forceIfcold_start=False,
        ):

        # Create this run folder 
        
        subfolder = Path(subfolder)
        
        FolderNEO = self.FolderGACODE / subfolder
        IOtools.askNewFolder(FolderNEO, force=forceIfcold_start)

        folders, folders_red = [], []
        for rho in self.rhos:
            # Create subfolder for each rho
            FolderNEO_rho = FolderNEO / f"rho_{rho:.4f}"
            IOtools.askNewFolder(FolderNEO_rho, force=forceIfcold_start)
            
            # Copy the file
            os.system(f"cp {self.FolderGACODE / f'input.neo_{rho:.4f}'} {FolderNEO_rho / 'input.neo'}")

            folders.append(FolderNEO_rho)
            folders_red.append(str(subfolder / f"rho_{rho:.4f}"))

        # Run NEO
        
        neo_job = FARMINGtools.mitim_job(self.FolderGACODE)
        neo_job.define_machine_quick("neo",f"mitim_neo")
        
        NEOcommand = ""

        for folder in folders_red:
            NEOcommand += f"neo -e {folder} -p {neo_job.folderExecution} &\n"
        NEOcommand += "wait\n"
        
        neo_job.define_machine("neo",f"mitim_neo")

        neo_job.prep(
            NEOcommand,
            input_folders=[FolderNEO],
            output_folders=folders_red,
        )

        neo_job.run(
            removeScratchFolders=True,
            )

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

    def write_state(self, file=None):
        
        if file is None:
            file = self.file

        # Local formatter: floats -> 6 significant figures in exponential (uppercase),
        # ints stay as ints, bools as 0/1, sequences space-separated with same rule.
        def _fmt_num(x):
            import numpy as _np
            if isinstance(x, (bool, _np.bool_)):
                return "1" if x else "0"
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