
from mitim_tools.transp_tools.utils import TRANSPhelpers
from mitim_tools.gs_tools import FREEGStools

class mitim_step:

    def __init__(self, folder):
        self.folder = folder

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

    def run_transp(self, shotnum, runid):

        self.folder_transp  = self.folder + "/transp"

        self.transp = TRANSPhelpers.transp_run(self.folder_transp, shotnum, runid)

        self.transp.f = self.f
        time = 0.0
        self.transp._from_freegs_eq(time, ne0_20 = self.ne0_20, Vsurf = self.Vsurf, Zeff = self.Zeff, PichT_MW = self.PichT_MW)

        self.transp.namelist_from(namelist_folder,"12345P15TR.DAT")
        self.transp.write_ufiles(structures_position = -1, radial_position = 0)

        cores = 32 #64 #32
        hours = 12 #8 #12
        self.transp.run('D3D', mpisettings={"trmpi": 1, "toricmpi": cores, "ptrmpi": 1}, minutesAllocation = 60*hours, case=runid, checkMin=None, grabIntermediateEachMin=60.0)

    def run_portals(self):

        self.folder_portals = self.folder + "/portals"

        