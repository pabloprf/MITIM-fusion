import os
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gacode_tools.aux import GACODErun

class NEO:
    def __init__(self):

        pass

    def prep(self,inputgacode,folder):

        self.inputgacode = inputgacode
        self.folder = IOtools.expandPath(folder)

        if not os.path.exists(self.folder): os.system(f'mkdir {self.folder}')

    def run_vgen(self, subfolder="vgen1",vgenOptions={}):
        
        while subfolder[-1] == "/":
            subfolder = subfolder[:-1]
        
        # ---- Default options
        
        vgenOptions.setdefault("er", 2)
        vgenOptions.setdefault("vel", 1)
        vgenOptions.setdefault("numspecies", len(self.inputgacode.Species))
        vgenOptions.setdefault("matched_ion", 1)
        vgenOptions.setdefault("nth", "17,39")

        # ---- Prepare

        self.folder_vgen = f'{self.folder}/{subfolder}/'

        if not os.path.exists(self.folder_vgen): os.system(f'mkdir {self.folder_vgen}')

        self.inputgacode.writeCurrentStatus(file=f"{self.folder_vgen}/input.gacode")

        # ---- Run

        file_new = GACODErun.runVGEN(self.folder_vgen, vgenOptions=vgenOptions, name_run=subfolder)

        # ---- Postprocess
        
        self.inputgacode_vgen = PROFILEStools.PROFILES_GACODE(file_new, calculateDerived=True, mi_ref=self.inputgacode.mi_ref)



