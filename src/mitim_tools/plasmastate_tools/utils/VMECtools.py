import vmecpp
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

    def __init__(self, file, derive_quantities=True, mi_ref=None):

        # Initialize the base class and tell it the type of file
        super().__init__(type_file='vmec')

        # Read the input file and store the raw data
        self.file = file
        self._read_inputgacocde()

        # Derive quantities if requested
        if self.file is not None:
            # Derive (Depending on resolution, derived can be expensive, so I mmay not do it every time)
            self.derive_quantities(mi_ref=mi_ref, derive_quantities=derive_quantities)


    @IOtools.hook_method(after=MITIMstate.ensure_variables_existence)
    def _read_vmec(self):
        pass
        
    def derive_geometry(self, **kwargs):
        pass

    def write_state(self,  file=None, **kwargs):
        pass
