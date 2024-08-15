
from mitim_tools.transp_tools.utils import TRANSPhelpers
from mitim_tools.gs_tools import FREEGStools

class populator_transp:
    def __init__(self, mitim_step_instance):
            
        self.mitim_step_instance = mitim_step_instance

    def from_freegs(self, runid, namelist_folder, namelist_file):

        self.mitim_step_instance.runid = runid
        self.mitim_step_instance.folder_transp  = self.mitim_step_instance.folder + "/transp"

        self.mitim_step_instance.transp = TRANSPhelpers.transp_run(self.mitim_step_instance.folder_transp, self.mitim_step_instance.shotnum, self.mitim_step_instance.runid)

        self.mitim_step_instance.transp.f = self.mitim_step_instance.f
        time = 0.0
        self.mitim_step_instance.transp._from_freegs_eq(time, ne0_20 = self.mitim_step_instance.ne0_20, Vsurf = self.mitim_step_instance.Vsurf, Zeff = self.mitim_step_instance.Zeff, PichT_MW = self.mitim_step_instance.PichT_MW)

        self.mitim_step_instance.transp.namelist_from(namelist_folder,namelist_file)
        self.mitim_step_instance.transp.write_ufiles(structures_position = -1, radial_position = 0)

class populator_portals:
    def __init__(self, mitim_step_instance):
            
        self.mitim_step_instance = mitim_step_instance

    def from_freegs(self, file):

        self.mitim_step_instance.f.to_profiles(file=file)


class mitim_step:

    def __init__(self, folder, shotnum='12345'):
        self.folder = folder
        self.shotnum = shotnum

        # Describe the time populators --------------
        self.populate_transp = populator_transp(self)
        self.populate_portals = populator_portals(self)
        # ------------------------------------------

    # --------------------------------------------------------------------------------------------
    # Initializations
    # --------------------------------------------------------------------------------------------

    def initialize_from_freegs(self, R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = 3.0, Vsurf = 0.0, Zeff = 1.5, PichT_MW = 11.0):
        
        # Create Miller FreeGS for the desired geometry
        self.f = FREEGStools.freegs_millerized( R, a, kappa_sep, delta_sep, zeta_sep, z0)
        self.f.prep(p0_MPa, Ip_MA, B_T)
        self.f.solve()
        self.f.derive()

        # Pass some parameters that are needed to define this step
        self.f.ne0_20 = ne0_20
        self.f.Vsurf = Vsurf
        self.f.Zeff = Zeff
        self.f.PichT_MW = PichT_MW

    # --------------------------------------------------------------------------------------------
    # Run TRANSP
    # --------------------------------------------------------------------------------------------

    def run_transp(self, tokamak = 'D3D', cores = 32, hours = 12, checkMin = 10):

        self.transp.run(tokamak, mpisettings={"trmpi": cores, "toricmpi": cores, "ptrmpi": 1}, minutesAllocation = 60*hours, case=self.runid, checkMin=checkMin, grabIntermediateEachMin=1E6)

    # --------------------------------------------------------------------------------------------
    # Run PORTALS
    # --------------------------------------------------------------------------------------------

    def run_portals(self):

        self.folder_portals = self.folder + "/portals"

