import os
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.gs_tools import FREEGStools
from mitim_tools.transp_tools import TRANSPtools, CDFtools, UFILEStools
from mitim_tools.im_tools.modules import EQmodule
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools import IOtools, MATHtools, PLASMAtools, GRAPHICStools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed
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

            if len(x) == 0:
                print('No data for ',quantity)
                continue

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

            if self.nml is not None:
                IOtools.changeValue(self.nml, "VVRmom", VVRmom_str, None, "=", MaintainComments=True)
                IOtools.changeValue(self.nml, "VVZmom", VVZmom_str, None, "=", MaintainComments=True)
            else:
                print("\t- Namelist not read, VV geometry not written", typeMsg='w')

            # --------------------------------------------------------------------------------------------
            # Write Limiters in ufile
            # --------------------------------------------------------------------------------------------

            EQmodule.addLimiters_UF(f'{self.folder}/PRF12345.LIM', self.geometry_select['R_lim'], self.geometry_select['Z_lim'], numLim=len(self.geometry_select['R_lim']))

            # --------------------------------------------------------------------------------------------
            # Write Antenna in namelist
            # --------------------------------------------------------------------------------------------

            if self.nml is not None:
                IOtools.changeValue(self.nml, "rmjicha", 100.0*self.geometry_select['antenna_R'], None, "=", MaintainComments=True)
                IOtools.changeValue(self.nml, "rmnicha", 100.0*self.geometry_select['antenna_r'], None, "=", MaintainComments=True)
                IOtools.changeValue(self.nml, "thicha", self.geometry_select['antenna_t'], None, "=", MaintainComments=True)
            else:
                print("\t- Namelist not read, Antenna geometry not written", typeMsg='w')

        else:
            self.geometry_select = None

    def define_timings(self,tinit,ftime,curr_diff=None, sawtooth = None):

        if self.nml is None:
            print("Please read namelist first", typeMsg='w')
            return

        if curr_diff is None:
            curr_diff = tinit

        if sawtooth is None:
            sawtooth = curr_diff

        print(f"\t- Defining timings: tinit = {tinit}s, ftime = {ftime}s (and current diffusion from = {curr_diff}s, sawtooth from = {sawtooth}s)")

        IOtools.changeValue(self.nml, "tinit", tinit, None, "=", MaintainComments=True)
        IOtools.changeValue(self.nml, "ftime", ftime, None, "=", MaintainComments=True)
        IOtools.changeValue(self.nml, "tqmoda(1)", curr_diff, None, "=", MaintainComments=True)
        IOtools.changeValue(self.nml, "t_sawtooth_on", sawtooth, None, "=", MaintainComments=True)

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
        print(f"\t- Populating Antenna in namelist, with rmin = {a_orig}+{antenna_a}m")
        self.antenna_R = R
        self.antenna_r = a_orig + antenna_a
        self.antenna_t = 30.0

        self.geometry['antenna_R'] = self.antenna_R
        self.geometry['antenna_r'] = self.antenna_r
        self.geometry['antenna_t'] = self.antenna_t

    def from_freegs(self, time, R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = 3.3, Vsurf = 0.0, Zeff = 1.5, PichT_MW = 11.0):

        # Create Miller FreeGS for the desired geometry
        self.f = FREEGStools.freegs_millerized( R, a, kappa_sep, delta_sep, zeta_sep, z0)
        self.f.prep(p0_MPa, Ip_MA, B_T)
        self.f.solve()
        self.f.derive()

        self._from_freegs_eq(time, ne0_20 = ne0_20, Vsurf = Vsurf, Zeff = Zeff, PichT_MW = PichT_MW)

    def _from_freegs_eq(self, time, f = None, ne0_20 = 3.3, Vsurf = None, Zeff = None, PichT_MW = None):

        self.variables = {}

        if f is not None:
            self.f = f

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
            'z':self.f.Ip_MA * 1E6
            }

        self.variables['RBZ'] = {
            'x': None,
            'z': self.f.R0 * self.f.B_T * 1E2
            }

        # --------------------------------------------------------------
        # Quantities that do not come from FreeGS
        # --------------------------------------------------------------

        if Vsurf is not None:
            self.variables['VSF'] = {
                'x': None,
                'z': Vsurf
                }

        if Zeff is not None:
            self.variables['ZF2'] = {
                'x': rhotor,
                'z': Zeff * np.ones(len(rhotor))
                }

        if PichT_MW is not None:
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

        self._produce_structures_from_variables(self.f.R0, self.f.a, self.f.kappa_sep, self.f.delta_sep, self.f.zeta_sep, self.f.Z0)

        # --------------------------------------------------------------
        # Populate
        # --------------------------------------------------------------
        
        self._populate(time)


