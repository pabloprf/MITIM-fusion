import os
import glob
from mitim_tools.gs_tools import FREEGStools
from mitim_tools.transp_tools import TRANSPtools

class transp_from_engineering_parameters:

    def __init__(self, R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = 3.3, folder = '~/scratch/', shot = '12345', runid = 'P01'):
        '''
        Desired geometry and GS parameters of plasma
        '''

        self.folder = folder
        if not os.path.exists(self.folder):
            os.system(f"mkdir -p {self.folder}")

        self.runid = runid 
        self.shot = shot

        # Create Miller FreeGS for the desired geometry
        self.f2 = FREEGStools.freegs_millerized( R, a, kappa_sep, delta_sep, zeta_sep, z0)
        self.f2.prep(p0_MPa, Ip_MA, B_T)
        self.f2.solve()
        self.f2.derive()

        self.ne0_20 = ne0_20

        # Initialize them as None, unless methods are called
        self.f1, self.t3_quantities = None, None 

    # --------------------------------------------------------------------------------------------
    # Optional methods to append before or after the TRANSP run
    # --------------------------------------------------------------------------------------------

    def initialize_from(self, R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T):
        '''
        To make TRANSP simulations easier to converge, we can initialize from a different plasma
        '''

        # Create Miller FreeGS for the initial geometry
        self.f1 = FREEGStools.freegs_millerized( R, a, kappa_sep, delta_sep, zeta_sep, z0)
        self.f1.prep(p0_MPa, Ip_MA, B_T)
        self.f1.solve()
        self.f1.derive()

    def finalize_to(self, R_m, Z_m, rho_tor, q, Te_keV, Ti_keV, ne_20, Ip_MA, RB_m):
        '''
        Once reached the desired geometry, we can transition once more to the desired realistic plasma
        '''

        self.t3_quantities = [R_m, Z_m, rho_tor, q, Te_keV, Ti_keV, ne_20, Ip_MA, RB_m]

    # --------------------------------------------------------------------------------------------
    
    def get_transp_inputs(self, folder_transp_inputs):

        os.system(f"cp {folder_transp_inputs}/* {self.folder}")
        oldTR = glob.glob(f"{self.folder}/*TR.DAT")[0].split('/')[-1]
        os.system(f"mv {self.folder}/{oldTR} {self.folder}/{self.shot}{self.runid}TR.DAT")

    def write_transp_inputs_from_freegs(self, times, plotYN = False):
        '''
        Write the TRANSP input files
        '''

        self.times = times

        if self.f1 is None:
            f1 = self.f2 
            VVfrom = 0
        else:
            f1 = self.f1
            f2 = self.f2
            VVfrom = 1

        FREEGStools.from_freegs_to_transp(f1,f2 = f2, t3_quantities=self.t3_quantities,
            folder = self.folder, ne0_20 =  self.ne0_20, times = times, VVfrom = VVfrom,plotYN = plotYN)

    def run_transp(self, tokamak, mpisettings={"trmpi": 32, "toricmpi": 32, "ptrmpi": 1}, minutesAllocation = 60*8, case='run1', checkMin=10.0, grabIntermediateEachMin=30.0):
        '''
        Run TRANSP
        '''

        self.t = TRANSPtools.TRANSP(self.folder, tokamak)

        self.t.defineRunParameters(
            self.shot + self.runid, self.shot,
            mpisettings = mpisettings,
            minutesAllocation = minutesAllocation)

        self.t.run()
        self.c = self.t.checkUntilFinished(label=case, checkMin=checkMin, grabIntermediateEachMin=grabIntermediateEachMin)

    def plot(self, case='run1'):

        self.t.plot(label=case)
