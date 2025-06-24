import vmecpp
import numpy as np
from pathlib import Path
from mitim_tools.misc_tools import IOtools
from mitim_tools.plasmastate_tools import MITIMstate
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class vmec_state(MITIMstate.mitim_state):
    '''
    Class to read and manipulate VMEC files
    '''

    # ------------------------------------------------------------------
    # Reading and interpreting
    # ------------------------------------------------------------------

    def __init__(self, file_vmec, file_profs, derive_quantities=True, mi_ref=None):

        # Initialize the base class and tell it the type of file
        super().__init__(type_file='vmec')

        # Read the input file and store the raw data
        self.files = [file_vmec, file_profs]
        if self.files is not None:
            self._read_vmec()
            # Derive (Depending on resolution, derived can be expensive, so I mmay not do it every time)
            self.derive_quantities(mi_ref=mi_ref, derive_quantities=derive_quantities)


    @IOtools.hook_method(after=MITIMstate.ensure_variables_existence)
    def _read_vmec(self):
        
        # Read VMEC file
        print("\t- Reading VMEC file")
        self.wout = vmecpp.VmecWOut.from_wout_file(Path(self.files[0]))
        
        # Produce variables
        
        self.profiles["torfluxa(Wb/radian)"] = [self.wout.phipf[-1]]
        
        
    def derive_geometry(self, **kwargs):
        
        rho = np.linspace(0, 1, self.wout.ns)

        ds = rho[1] - rho[0]
        half_grid_rho = rho - ds / 2

        d_volume_d_rho = (
            (2 * np.pi) ** 2
            * np.array(self.wout.vp)
            * 2
            * np.sqrt(half_grid_rho)
        )
        
        #self.derived["B_unit"] = self.profiles["torfluxa(Wb/radian)"] / (np.pi * self.wout.Aminor_p**2)

    def write_state(self,  file=None, **kwargs):
        pass
    
    def plot_geometry(self, axs, color="b", legYN=True, extralab="", lw=1, fs=6):
        pass