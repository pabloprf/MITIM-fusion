import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.gs_tools import FREEGStools
from mitim_tools.transp_tools import TRANSPtools, CDFtools, UFILEStools
from mitim_tools.im_tools.modules import EQmodule
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools import IOtools, MATHtools, PLASMAtools, GRAPHICStools
from IPython import embed

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
        self.self.f.prep(p0_MPa, Ip_MA, B_T)
        self.self.f.solve()
        self.self.f.derive()

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


class transp_from_transp:
    
    def __init__(self, cdf_original, nml_original, folder, shot, runid, time = -1):
        '''
        Initialize from a previous TRANSP run
        '''

        print(f"Initializing TRANSP run from {cdf_original}")

        # Read original TRANSP run
        self.nml_original = nml_original
        self.c_original = CDFtools.CDFreactor(cdf_original)

        self.folder = folder
        if not os.path.exists(self.folder):
            os.system(f"mkdir -p {self.folder}")

        self.time = time

        # Move original files
        self.shot = shot
        self.runid = runid
        self.nml = f"{self.folder}/{self.shot}{self.runid}TR.DAT"
        os.system(f"cp {self.nml_original} {self.nml}")
        

    def produce_new_ufiles(self, new_times = [0.0,1.0]):

        self.times = new_times

        # --------------------------------------------------------------
        # Populate profiles
        # --------------------------------------------------------------

        print("\t- Populating profiles")

        quantities = {
            'Ip': ['cur','CUR',None, 1E6],
            'RBt_vacuum': ['rbz','RBZ',None, 1E2],
            'Vsurf': ['vsf','VSF',None, 1.0],
            'q': ['qpr','QPR','xb', 1.0],
            'Te': ['ter','TEL','x', 1E3],
            'Ti': ['ti2','TIO','x', 1E3],
            'ne': ['ner','NEL','x', 1E20*1E-6],
            'Zeff': ['zf2','ZF2','x', 1.0],
            'PichT': ['rfp','RFP',[1], 1E6],
        }

        for var in quantities.keys():
            self._produce_quantity(var = var, ufile_id = quantities[var][0], ufile_name = quantities[var][1], coordinate = quantities[var][2], factor_from_transp_to_ufile = quantities[var][3])

        # --------------------------------------------------------------
        # Populate geometry
        # --------------------------------------------------------------

        self._produce_geometries()

    def _produce_quantity(self, var = 'Te', ufile_id = 'ter', ufile_name = 'TEL', coordinate = 'x', factor_from_transp_to_ufile = 1.0):

        print(f"\t\t* Populating {ufile_name} from {var}")

        # Grab quantity
        it = np.argmin(np.abs(self.c_original.t - self.time))
        z = self.c_original.__dict__[var][it] * factor_from_transp_to_ufile

        # Start ufile
        uf = UFILEStools.UFILEtransp(scratch=ufile_id)

        # Populate ufile
        if coordinate is None:
            uf.Variables['X'] =  self.times
            uf.Variables['Z'] = [z]*len( self.times)
        elif coordinate in ['x','xb']:
            uf.Variables['Y'] =  self.times
            uf.Variables['X'] = self.c_original.__dict__[coordinate][it]
            uf.Variables['Z'] = z
            uf.repeatProfile()
        else:
            uf.Variables['Y'] =  self.times
            uf.Variables['X'] = coordinate
            uf.Variables['Z'] = z
            uf.repeatProfile()

        # Write ufile
        uf.writeUFILE(f'{self.folder}/PRF12345.{ufile_name}')
    
    def _produce_geometries(self, plotYN = True):

        self._produce_MRY()
        self._produce_structures()

        if plotYN:
            fig, ax = plt.subplots()
            ax.plot(self.R_sep, self.Z_sep, '-o', c = 'b', markersize=3, label='Separatrix')
            ax.plot(self.rvv, self.zvv, '-o', c = 'k', markersize=3, label='VV')
            ax.plot(self.rvv_fit_cm*1E-2, self.zvv_fit_cm*1E-2, '--*', c = 'm', markersize=3, label='VV fit')

            R_ant, Z_ant = EQmodule.reconstructAntenna(
                self.antenna_R, self.antenna_r, self.antenna_t
            )
            ax.plot(R_ant, Z_ant, '-o', c = 'g', markersize=3, label='Antenna')

            ax.set_aspect('equal')
            ax.legend()
            ax.set_title('Geometry')
            ax.set_xlabel('R [m]')
            ax.set_ylabel('Z [m]')
            plt.show()

    def _produce_MRY(self):

        print("\t- Populating MRY from TRANSP boundary")

        R_sep, Z_sep = CDFtools.getFluxSurface(self.c_original.f, self.time, 1.0)
        self.R_sep, self.Z_sep = R_sep, Z_sep

        self.R_sep, self.Z_sep = MATHtools.downsampleCurve(
             self.R_sep, self.Z_sep, nsamp=100
        )
        self.R_sep = self.R_sep[:-1]
        self.Z_sep = self.Z_sep[:-1]

        tt = []
        for time in self.times:
            t = str(int(time*1E3)).zfill(5)
            EQmodule.writeBoundary(f'{self.folder}/BOUNDARY_123456_{t}.DAT', self.R_sep, self.Z_sep)
            tt.append(t)

        EQmodule.generateMRY(
            self.folder,
            tt,
            self.folder,
            12345)

    def _produce_structures(self, vv_relative_a = 0.25, antenna_a = 0.02):

        print(f"\t- Populating VV and Limiters with a circular structure with relative distance {vv_relative_a}*kappa*a")

        # --------------------------------------------------------------
        # Miller Structures
        # --------------------------------------------------------------

        it = np.argmin(np.abs(self.c_original.t - self.time))

        vv = GEQtools.mitim_flux_surfaces()
        thetas = np.linspace(0, 2*np.pi, 20, endpoint=True)
        a = self.c_original.a[it] * (1+vv_relative_a)
        vv.reconstruct_from_miller(
            self.c_original.Rmajor[it],
            a*self.c_original.kappa[it], 
            1.0, #self.c_original.kappa[it], 
            0.0, #self.c_original.Ymag[it], 
            0.0, #self.c_original.delta[it],
            0.0, #self.c_original.zeta[it],
            thetas = thetas)
        self.rvv , self.zvv= vv.R[0,:], vv.Z[0,:]

        # --------------------------------------------------------------
        # VV (namelist)
        # --------------------------------------------------------------

        print("\t\t* Populating VV in namelist")

        VVRmom, VVZmom, self.rvv_fit_cm, self.zvv_fit_cm = EQmodule.decomposeMoments(
            self.rvv*100.0 , self.zvv*100.0,
            r_ini = [self.c_original.Rmajor[it]*100.0, a*100.0, 3.0], z_ini = [0.0, a*self.c_original.kappa[it]*100.0, -3.0], verbose_level=5)

        VVRmom_str = ', '.join([f'{x:.8e}' for x in VVRmom])
        VVZmom_str = ', '.join([f'{x:.8e}' for x in VVZmom])

        #IOtools.changeValue(self.nml, "VVRmom", VVRmom_str, None, "=", MaintainComments=True)
        #IOtools.changeValue(self.nml, "VVZmom", VVZmom_str, None, "=", MaintainComments=True)

        # --------------------------------------------------------------
        # Limiters (ufile)
        # --------------------------------------------------------------

        print("\t\t* Populating Limiters in ufile")

        EQmodule.addLimiters_UF(f'{self.folder}/PRF12345.LIM', self.rvv, self.zvv, numLim=len(thetas))

        # --------------------------------------------------------------
        # Antenna (namelist)
        # --------------------------------------------------------------
        print(f"\t- Populating Antenna in namelist, with rmin = a+{antenna_a}m")
        self.antenna_R = self.c_original.Rmajor[it]
        self.antenna_r = self.c_original.a[it] + antenna_a
        self.antenna_t = 30.0
        IOtools.changeValue(self.nml, "rmjicha", 100.0*self.antenna_R, None, "=", MaintainComments=True)
        IOtools.changeValue(self.nml, "rmnicha", 100.0*self.antenna_r, None, "=", MaintainComments=True)
        IOtools.changeValue(self.nml, "thicha", self.antenna_t, None, "=", MaintainComments=True)

    def define_timings(self,tinit,ftime,curr_diff=None):

        if curr_diff is not None:
            curr_diff = tinit

        print(f"\t- Defining timings: tinit = {tinit}s, ftime = {ftime}s (and current diffusion from = {curr_diff}s)")

        IOtools.changeValue(self.nml, "tinit", tinit, None, "=", MaintainComments=True)
        IOtools.changeValue(self.nml, "ftime", ftime, None, "=", MaintainComments=True)
        IOtools.changeValue(self.nml, "tqmoda(1)", curr_diff, None, "=", MaintainComments=True)

    def run_transp(self, tokamak, mpisettings={"trmpi": 32, "toricmpi": 32, "ptrmpi": 1}, minutesAllocation = 60*8, case='run1', checkMin=10.0, grabIntermediateEachMin=30.0):
        '''
        Run TRANSP
        '''
        print("\t- Running TRANSP")

        self.t = TRANSPtools.TRANSP(self.folder, tokamak)

        self.t.defineRunParameters(
            self.shot + self.runid, self.shot,
            mpisettings = mpisettings,
            minutesAllocation = minutesAllocation)

        self.t.run()
        self.c = self.t.checkUntilFinished(label=case, checkMin=checkMin, grabIntermediateEachMin=grabIntermediateEachMin)

    def plot(self, case='run1'):

        self.t.plot(label=case)



class transp_run:
    def __init__(self, folder, shot, runid):

        self.folder = folder
        if not os.path.exists(self.folder):
            os.system(f"mkdir -p {self.folder}")

        self.shot = shot
        self.runid = runid

        self.variables = {}
        self.geometry = {}
        self.nml = None

        # Describe the time populator
        self.populate_time = transp_input_time(self)

        self.quantities = {
            'Ip': ['cur','CUR',None, 1E6],
            'RBt_vacuum': ['rbz','RBZ',None, 1E2],
            'Vsurf': ['vsf','VSF',None, 1.0],
            'q': ['qpr','QPR','xb', 1.0],
            'Te': ['ter','TEL','x', 1E3],
            'Ti': ['ti2','TIO','x', 1E3],
            'ne': ['ner','NEL','x', 1E20*1E-6],
            'Zeff': ['zf2','ZF2','x', 1.0],
            'PichT': ['rfp','RFP',[1], 1E6],
        }

    def namelist_from(self, folder_original,nml_original):

        self.nml = f"{self.folder}/{self.shot}{self.runid}TR.DAT"
        os.system(f"cp {folder_original}/{nml_original} {self.nml}")

        # Predictive namelists
        os.system(f"cp {folder_original}/*namelist.dat {self.folder}/.")

    def write_ufiles(self, structures_position = 0, radial_position = 0):

        self.times = np.sort(np.array(list(self.variables.keys())))

        # --------------------------------------------------------------------------------------------
        # Write ufiles of profiles
        # --------------------------------------------------------------------------------------------

        for quantity in self.quantities:

            # Initialize ufile
            uf = UFILEStools.UFILEtransp(scratch=self.quantities[quantity][0])

            # Grab variables
            t = []
            x = []
            z = []
            for time in self.times:
                if self.quantities[quantity][1] in self.variables[time].keys():
                    x0 = self.variables[time][self.quantities[quantity][1]]['x']
                    z0 = self.variables[time][self.quantities[quantity][1]]['z']
                    x.append(x0)
                    z.append(z0)
                    t.append(time)

            # If radial, interpolate to radial_position
            if self.quantities[quantity][2] in ['x','xb']:
                for i in range(len(z)):
                    z[i] = np.interp(x[radial_position], x[i], z[i])

            # Populate 1D time trace
            if self.quantities[quantity][2] is None:
                uf.Variables['X'] = t
                uf.Variables['Z'] = np.array(z)
            elif self.quantities[quantity][2] in ['x','xb']:
                uf.Variables['Y'] = t
                uf.Variables['X'] = x[radial_position]
                uf.Variables['Z'] = np.array(z).T
                uf.shiftVariables()
            else:
                uf.Variables['Y'] = t
                uf.Variables['X'] = self.quantities[quantity][2]
                uf.Variables['Z'] = np.array(z)

            # Write ufile
            uf.writeUFILE(f'{self.folder}/PRF12345.{self.quantities[quantity][1]}')

        # --------------------------------------------------------------------------------------------
        # Write MRY ufile
        # --------------------------------------------------------------------------------------------

        tt = []
        for time in self.times:
            if (time in self.geometry) and ('R_sep' in self.geometry[time]):
                t = str(int(time*1E3)).zfill(5)
                R, Z = self.geometry[time]['R_sep'], self.geometry[time]['Z_sep']
                EQmodule.writeBoundary(f'{self.folder}/BOUNDARY_123456_{t}.DAT', R, Z)
                tt.append(t)

        EQmodule.generateMRY(
            self.folder,
            tt,
            self.folder,
            self.shot)


        if structures_position is not None:

            # --------------------------------------------------------------------------------------------
            # Write VV in namelist
            # --------------------------------------------------------------------------------------------

            geos = []
            for time in self.times:    
                if (time in self.geometry) and ('VVRmom' in self.geometry[time]):    
                    geos.append(self.geometry[time])
            self.geometry_select = geos[structures_position]

            VVRmom_str = ', '.join([f'{x:.8e}' for x in self.geometry_select['VVRmom']])
            VVZmom_str = ', '.join([f'{x:.8e}' for x in self.geometry_select['VVZmom']])

            #IOtools.changeValue(self.nml, "VVRmom", VVRmom_str, None, "=", MaintainComments=True)
            #IOtools.changeValue(self.nml, "VVZmom", VVZmom_str, None, "=", MaintainComments=True)

            # --------------------------------------------------------------------------------------------
            # Write Limiters in ufile
            # --------------------------------------------------------------------------------------------

            EQmodule.addLimiters_UF(f'{self.folder}/PRF12345.LIM', self.geometry_select['R_lim'], self.geometry_select['Z_lim'], numLim=len(self.geometry_select['R_lim']))

            # --------------------------------------------------------------------------------------------
            # Write Antenna in namelist
            # --------------------------------------------------------------------------------------------

            IOtools.changeValue(self.nml, "rmjicha", 100.0*self.geometry_select['antenna_R'], None, "=", MaintainComments=True)
            IOtools.changeValue(self.nml, "rmnicha", 100.0*self.geometry_select['antenna_r'], None, "=", MaintainComments=True)
            IOtools.changeValue(self.nml, "thicha", self.geometry_select['antenna_t'], None, "=", MaintainComments=True)

        else:
            self.geometry_select = None

    def define_timings(self,tinit,ftime,curr_diff=None):

        if self.nml is None:
            print("Please read namelist first")
            return

        if curr_diff is None:
            curr_diff = tinit

        print(f"\t- Defining timings: tinit = {tinit}s, ftime = {ftime}s (and current diffusion from = {curr_diff}s)")

        IOtools.changeValue(self.nml, "tinit", tinit, None, "=", MaintainComments=True)
        IOtools.changeValue(self.nml, "ftime", ftime, None, "=", MaintainComments=True)
        IOtools.changeValue(self.nml, "tqmoda(1)", curr_diff, None, "=", MaintainComments=True)

    # --------------------------------------------------------------------------------------------
    # Utilities to populate specific times with something
    # --------------------------------------------------------------------------------------------

    def add_variable_time(self, time, value_x, value, variable='QPR'):

        if time not in self.variables.keys():
            self.variables[time] = {}

        self.variables[time][variable] = {
            'x': value_x,
            'z': value
        }

    def add_g_time(self, time, g_file_loc):

        g = GEQtools.MITIMgeqdsk(g_file_loc,fullLCFS=True)
        self._add_separatrix_time(time, g.Rb_prf, g.Yb_prf)

    def _add_separatrix_time(self, time, R_sep, Z_sep):

        if time not in self.geometry.keys():
            self.geometry[time] = {}

        self.geometry[time]['R_sep'] = R_sep
        self.geometry[time]['Z_sep'] = Z_sep

    # --------------------------------------------------------------------------------------------

    def run(self, tokamak, mpisettings={"trmpi": 32, "toricmpi": 32, "ptrmpi": 1}, minutesAllocation = 60*8, case='run1', checkMin=10.0, grabIntermediateEachMin=30.0):
        '''
        Run TRANSP
        '''
        print("\t- Running TRANSP")

        self.t = TRANSPtools.TRANSP(self.folder, tokamak)

        self.t.defineRunParameters(
            self.shot + self.runid, self.shot,
            mpisettings = mpisettings,
            minutesAllocation = minutesAllocation)

        self.t.run()
        self.c = self.t.checkUntilFinished(label=case, checkMin=checkMin, grabIntermediateEachMin=grabIntermediateEachMin)

    def plot(self, case='run1'):

        self.t.plot(label=case)

    def plot_inputs(self):

        colors = GRAPHICStools.listColors()

        fig = plt.figure(figsize=(15,7))
        axs = fig.subplot_mosaic(
            """
            ABC
            ADE
            """)

        ax = axs['A']
        for i,time in enumerate(self.geometry.keys()):
            ax.plot(self.geometry[time]['R_sep'], self.geometry[time]['Z_sep'], '-o', c=colors[i], markersize=2, label=f'Separatrix t={time}s')
        
        ax.plot(self.geometry_select['R_lim'], self.geometry_select['Z_lim'], '-o', c='k', markersize=2, label='Limiters')
        ax.plot(self.geometry_select['antenna_R'], self.geometry_select['antenna_r'], '-o', c='g', markersize=2, label='Antenna')

        ax.set_aspect('equal')
        ax.legend(loc='best',prop={'size': 6})
        ax.set_title('Geometry')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        GRAPHICStools.addDenseAxis(ax)

        ax = axs['B']
        for i,time in enumerate(self.variables.keys()):
            col = colors[i]
            if 'TEL' in self.variables[time].keys():
                ax.plot(self.variables[time]['TEL']['x'], self.variables[time]['TEL']['z']*1E-3, '-o',c=col, markersize=1, label=f'Te t={time}s')
            if 'TIO' in self.variables[time].keys():
                ax.plot(self.variables[time]['TIO']['x'], self.variables[time]['TIO']['z']*1E-3, '--s',c=col, markersize=1, label=f'Ti t={time}s')

        ax.legend(loc='best',prop={'size': 6})
        ax.set_xlabel('$\\rho_{tor}$')
        ax.set_ylabel('T [keV]')
        ax.set_xlim([0,1])
        ax.set_ylim(bottom=0)

        ax = axs['C']
        for i,time in enumerate(self.variables.keys()):
            col = colors[i]
            if 'NEL' in self.variables[time].keys():
                ax.plot(self.variables[time]['NEL']['x'], self.variables[time]['NEL']['z']*1E-20*1E6, '-o',c=col, markersize=1, label=f'ne t={time}s')

        ax.legend(loc='best',prop={'size': 6})
        ax.set_xlabel('$\\rho_{tor}$')
        ax.set_ylabel('$n_e$ [$10^{20}m^{-3}$]')
        ax.set_xlim([0,1])
        ax.set_ylim(bottom=0)

        ax = axs['D']
        for i,time in enumerate(self.variables.keys()):
            col = colors[i]
            if 'QPR' in self.variables[time].keys():
                ax.plot(self.variables[time]['QPR']['x'], self.variables[time]['QPR']['z'], '-o',c=col, markersize=1, label=f'q t={time}s')

        ax.legend(loc='best',prop={'size': 6})
        ax.set_xlabel('$\\rho_{tor}$')
        ax.set_ylabel('q profile')
        ax.set_xlim([0,1])
        ax.set_ylim(bottom=0)

        colsProfs = {'CUR':colors[0],'RBZ':colors[1]}

        ax = axs['E']
        for i,time in enumerate(self.variables.keys()):
            for var,factor in zip(['CUR','RBZ'],[1E-6,1E-2]):
                if 'CUR' in self.variables[time].keys():
                    ax.plot([time], [self.variables[time][var]['z']*factor], '-o',c=colsProfs[var], markersize=8, label=var if i == 0 else None)

        ax.legend(loc='best',prop={'size': 6})
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Current [MA], $R*B_t$ [m*T]')
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.show()

class transp_input_time:

    def __init__(self, transp_instance):
            
        self.transp_instance = transp_instance

    def _populate(self, time):
        
        self.transp_instance.variables[time] = self.variables
        self.transp_instance.geometry[time] = self.geometry

    def from_cdf(self, time, cdf_file, time_extraction = -1):

        self.time = time

        # Otherwise read from previous, do not load twice
        if cdf_file is not None:
            print(f"Populating time {time} from {cdf_file}")
            self.c_original = CDFtools.CDFreactor(cdf_file)
        else:
            print(f"Populating time {time} from previously opened CDF")
        
        if time_extraction is not None:
            self.time_extraction = time_extraction
            if self.time_extraction == -1:
                self.time_extraction = self.c_original.t[-1]
        else:
            print("Populating from same time as before")
        
        self.variables = {}
        for var in self.transp_instance.quantities.keys():
            self.variables[self.transp_instance.quantities[var][1]] = {}
            self.variables[self.transp_instance.quantities[var][1]]['x'],self.variables[self.transp_instance.quantities[var][1]]['z'] = self._produce_quantity_cdf(var = var, coordinate = self.transp_instance.quantities[var][2], factor_from_transp_to_ufile = self.transp_instance.quantities[var][3])

        self._produce_geometry_cdf()
        self._populate(time)

    def _produce_quantity_cdf(self, var = 'Te', coordinate = 'x', factor_from_transp_to_ufile = 1.0):

        # Grab quantity
        it = np.argmin(np.abs(self.c_original.t - self.time_extraction))
        z = self.c_original.__dict__[var][it] * factor_from_transp_to_ufile

        # Populate ufile
        if coordinate is None:
            x = None
        elif coordinate in ['x','xb']:
            x = self.c_original.__dict__[coordinate][it]
        else:
            x = coordinate
        
        return x,z

    def _produce_geometry_cdf(self):

        self.geometry = {}

        # --------------------------------------------------------------
        # Separatrix
        # --------------------------------------------------------------

        R_sep, Z_sep = CDFtools.getFluxSurface(self.c_original.f, self.time_extraction, 1.0)
        R_sep, Z_sep = MATHtools.downsampleCurve(R_sep, Z_sep, nsamp=100)
        self.geometry['R_sep'], self.geometry['Z_sep'] = R_sep[:-1], Z_sep[:-1]

        # --------------------------------------------------------------
        # VV
        # --------------------------------------------------------------

        it = np.argmin(np.abs(self.c_original.t - self.time_extraction))

        self._produce_structures_from_variables(
            self.c_original.Rmajor[it],
            self.c_original.a[it], 
            self.c_original.kappa[it], 
            self.c_original.Ymag[it], 
            self.c_original.delta[it],
            self.c_original.zeta[it],
            )

    def _produce_structures_from_variables(self, R, a_orig, kappa, delta, zeta, z0, vv_relative_a=0.25, antenna_a=0.02):

        # To fix
        kappa = 1.0
        zeta = 0.0
        delta - 0.0
    
        a = a_orig * kappa * (1+vv_relative_a)
        # -----

        vv = GEQtools.mitim_flux_surfaces()
        thetas = np.linspace(0, 2*np.pi, 20, endpoint=True)

        vv.reconstruct_from_miller(R, a, kappa, delta, zeta, z0, thetas = thetas)
        rvv , zvv= vv.R[0,:], vv.Z[0,:]

        self.geometry['VVRmom'], self.geometry['VVZmom'], rvv_fit_cm, zvv_fit_cm = EQmodule.decomposeMoments(
            rvv*100.0 , zvv*100.0,
            r_ini = [R*100.0, a*100.0, 3.0], z_ini = [0.0, a*kappa*100.0, -3.0], verbose_level=5)

        # --------------------------------------------------------------
        # Limiters
        # --------------------------------------------------------------

        self.geometry['R_lim'], self.geometry['Z_lim'] = rvv, zvv

        # --------------------------------------------------------------
        # Antenna (namelist)
        # --------------------------------------------------------------
        print(f"\t- Populating Antenna in namelist, with rmin = a+{antenna_a}m")
        self.antenna_R = R
        self.antenna_r = a_orig + antenna_a
        self.antenna_t = 30.0

        self.geometry['antenna_R'] = self.antenna_R
        self.geometry['antenna_r'] = self.antenna_r
        self.geometry['antenna_t'] = self.antenna_t

    def from_freegs(self, time, R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = 3.3, Vsurf = 0.0, Zeff = 1.5, PichT_MW = 11.0):

        self.variables = {}

        # Create Miller FreeGS for the desired geometry
        self.f = FREEGStools.freegs_millerized( R, a, kappa_sep, delta_sep, zeta_sep, z0)
        self.f.prep(p0_MPa, Ip_MA, B_T)
        self.f.solve()
        self.f.derive()

        self.ne0_20 = ne0_20
        
        psi = np.linspace(self.f.eq.psi_axis, self.f.eq.psi_bndry, 101, endpoint=True)
        psi_norm = (psi - self.f.eq.psi_axis) / (self.f.eq.psi_bndry - self.f.eq.psi_axis)
        rhotor = self.f.eq.rhotor(psi)

        # --------------------------------------------------------------
        # q profile
        # --------------------------------------------------------------

        q = self.f.eq.q(psinorm = psi_norm)

        self.variables['QPR'] = {
            'x': rhotor,
            'z':q
            }

        # --------------------------------------------------------------
        # pressure - temperature and density
        # --------------------------------------------------------------
        '''
        p_Pa = p_e + p_i = Te_eV * e_J * ne_20 * 1e20  + Ti_eV * e_J * ni_20 * 1e20

        if T=Te=Ti and ne=ni
        p_Pa = 2 * T_eV * e_J * ne_20 * 1e20

        T_eV = p_Pa / (2 * e_J * ne_20 * 1e20)

        '''

        pressure = self.f.eq.pressure(psinorm =psi_norm) # Pa

        _, ne_20 = PLASMAtools.parabolicProfile(
            Tbar=self.ne0_20/1.25,
            nu=1.25,
            rho=rhotor,
            Tedge=self.ne0_20/5,
        )

        T_eV = pressure / (2 * 1.60217662e-19 * ne_20 * 1e20)
        ne_cm = ne_20   * 1E20 * 1E-6

        self.variables['TEL'] = {
            'x': rhotor,
            'z':T_eV
            }

        self.variables['TIO'] = {
            'x': rhotor,
            'z':T_eV
            }

        self.variables['NEL'] = {
            'x': rhotor,
            'z':ne_cm
            }

        # --------------------------------------------------------------
        # Ip and RB
        # --------------------------------------------------------------

        self.variables['CUR'] = {
            'x': None,
            'z':Ip_MA * 1E6
            }

        self.variables['RBZ'] = {
            'x': None,
            'z': R * B_T * 1E2
            }

        # --------------------------------------------------------------
        # Quantities that do not come from FreeGS
        # --------------------------------------------------------------

        self.variables['VSF'] = {
            'x': None,
            'z': Vsurf
            }

        self.variables['ZF2'] = {
            'x': rhotor,
            'z': Zeff * np.ones(len(rhotor))
            }

        self.variables['RFP'] = {
            'x': [1],
            'z': PichT_MW * 1E6
            }

        # --------------------------------------------------------------
        # geometries
        # --------------------------------------------------------------

        RZ = self.f.eq.separatrix(npoints= 100)
        self.geometry = {
            'R_sep': RZ[:,0],
            'Z_sep': RZ[:,1]
            }

        self._produce_structures_from_variables(R, a, kappa_sep, delta_sep, zeta_sep, z0)

        # --------------------------------------------------------------
        # Populate
        # --------------------------------------------------------------
        
        self._populate(time)


