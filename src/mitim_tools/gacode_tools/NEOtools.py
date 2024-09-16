import os
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gacode_tools.utils import GACODErun
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed


class NEO:
    def __init__(self):
        pass

    def prep(self, inputgacode, folder):
        self.inputgacode = inputgacode
        self.folder = IOtools.expandPath(folder)

        os.makedirs(self.folder, exist_ok=True)

    def run_vgen(self, subfolder="vgen1", vgenOptions={}, restart=False):
        while subfolder[-1] == "/":
            subfolder = subfolder[:-1]

        self.folder_vgen = f"{self.folder}/{subfolder}/"

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
                "vgen/input.gacode",
                "vgen/input.neo.gen",
                "out.vgen.neoequil00",
                "out.vgen.neoexpnorm00",
                "out.vgen.neontheta00",
                "vgen.dat",
            ],
        )

        if (not runThisCase) and restart:
            runThisCase = print(
                "\t- Files found in folder, but restart requested. Are you sure?",
                typeMsg="q",
            )

            if runThisCase:
                IOtools.askNewFolder(self.folder_vgen, force=True)

        self.inputgacode.writeCurrentStatus(file=f"{self.folder_vgen}/input.gacode")

        # ---- Run

        if runThisCase:
            file_new = GACODErun.runVGEN(
                self.folder_vgen, vgenOptions=vgenOptions, name_run=subfolder
            )
        else:
            print(
                f"\t- Required files found in {subfolder}, not running VGEN",
                typeMsg="i",
            )
            file_new = f"{self.folder_vgen}/vgen/input.gacode"

        # ---- Postprocess

        self.inputgacode_vgen = PROFILEStools.PROFILES_GACODE(
            file_new, calculateDerived=True, mi_ref=self.inputgacode.mi_ref
        )


def check_if_files_exist(folder, list_files):
    folder = IOtools.expandPath(folder)

    for file in list_files:
        if not os.path.exists(f"{folder}/{file}"):
            return False

    return True
