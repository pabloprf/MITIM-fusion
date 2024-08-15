import os
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.gs_tools import FREEGStools
from mitim_tools.transp_tools import TRANSPtools, CDFtools, UFILEStools, NMLtools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools import IOtools, MATHtools, PLASMAtools, GRAPHICStools, FARMINGtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

# ----------------------------------------------------------------------------------------------------------
# TRANSP object utilities
# ----------------------------------------------------------------------------------------------------------


class transp_run:
    def __init__(self, folder, shot, runid):

        self.shot, self.runid = shot, runid
        self.folder = folder
        if not os.path.exists(self.folder):
            os.system(f"mkdir -p {self.folder}")

        # Initialize variables
        self.variables, self.geometry = {}, {}
        self.nml = None

        # Describe the time populator --------------
        self.populate_time = transp_input_time(self)
        # ------------------------------------------

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

    # --------------------------------------------------------------------------------------------
    # Namelist
    # --------------------------------------------------------------------------------------------

    def namelist(
        self,
        timings = {},
        tokamak = 'SPARC',
        **transp_params
        ):

        t = NMLtools.transp_nml(shotnum=self.shot, inputdir=self.folder, timings=timings)
        t.define_machine(tokamak)
        t.populate(**transp_params)
        t.write(self.runid)

        self.nml = f"{self.folder}/{self.shot}{self.runid}TR.DAT"

    def namelist_from(self,
        folder_original,
        nml_original,
        tinit,ftime,curr_diff=None, sawtooth = None, writeAC = None
        ):

        self.nml = f"{self.folder}/{self.shot}{self.runid}TR.DAT"
        os.system(f"cp {folder_original}/{nml_original} {self.nml}")

        # Predictive namelists
        os.system(f"cp {folder_original}/*namelist.dat {self.folder}/.")

        # Define timings

        if self.nml is None:
            print("Please read namelist first", typeMsg='w')
            return

        if curr_diff is None:
            curr_diff = tinit

        if sawtooth is None:
            sawtooth = curr_diff

        print(f"\t- Defining timings: tinit = {tinit}s, ftime = {ftime}s (and current diffusion from = {curr_diff}s, sawtooth from = {sawtooth}s). Write AC: {writeAC}s")

        IOtools.changeValue(self.nml, "tinit", tinit, None, "=", MaintainComments=True)
        IOtools.changeValue(self.nml, "ftime", ftime, None, "=", MaintainComments=True)
        IOtools.changeValue(self.nml, "tqmoda(1)", curr_diff, None, "=", MaintainComments=True)
        IOtools.changeValue(self.nml, "t_sawtooth_on", sawtooth, None, "=", MaintainComments=True)

        if writeAC is not None:
            IOtools.changeValue(self.nml, "mthdavg", 2, None, "=", MaintainComments=True)

            avgtim = 0.05  # Average before
            IOtools.changeValue(self.nml, "avgtim", avgtim, None, "=", MaintainComments=True)

            for var in ['outtim','fi_outtim','fe_outtim']:
                IOtools.changeValue(self.nml, var, f"{writeAC:.3f}", None, "=", MaintainComments=True)

            IOtools.changeValue(self.nml, "nldep0_gather", "T", None, "=", MaintainComments=True)
        

    # --------------------------------------------------------------------------------------------
    # Ufiles
    # --------------------------------------------------------------------------------------------

    def ufiles_from(self, folder_original, ufiles):

        for ufile in ufiles:
            os.system(f"cp {folder_original}/PRF12345.{ufile} {self.folder}/.")

    def write_ufiles(self, structures_position = 0, radial_position = 0):

        self.times = np.sort(np.array(list(self.variables.keys())))

        # --------------------------------------------------------------------------------------------
        # Write ufiles of profiles
        # --------------------------------------------------------------------------------------------

        for quantity in self.quantities:

            # Initialize ufile
            uf = UFILEStools.UFILEtransp(scratch=self.quantities[quantity][0])

            # Grab variables
            t,x,z = [], [], []
            for time in self.times:
                if self.quantities[quantity][1] in self.variables[time].keys():
                    x0 = self.variables[time][self.quantities[quantity][1]]['x']
                    z0 = self.variables[time][self.quantities[quantity][1]]['z']
                    x.append(x0)
                    z.append(z0)
                    t.append(time)

            if len(x) == 0:
                print(f'\t\t\t- No data for {quantity}, not writing UFILE', typeMsg='w')
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
                writeBoundary(f'{self.folder}/BOUNDARY_123456_{t}.DAT', R, Z)
                tt.append(t)

        generateMRY(
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
                print("\t- Namelist not available in this transp instance, VV geometry not written", typeMsg='w')

            # --------------------------------------------------------------------------------------------
            # Write Limiters in ufile
            # --------------------------------------------------------------------------------------------

            addLimiters_UF(f'{self.folder}/PRF12345.LIM', self.geometry_select['R_lim'], self.geometry_select['Z_lim'], numLim=len(self.geometry_select['R_lim']))

            # --------------------------------------------------------------------------------------------
            # Write Antenna in namelist
            # --------------------------------------------------------------------------------------------

            if self.nml is not None:
                IOtools.changeValue(self.nml, "rmjicha", 100.0*self.geometry_select['antenna_R'], None, "=", MaintainComments=True)
                IOtools.changeValue(self.nml, "rmnicha", 100.0*self.geometry_select['antenna_r'], None, "=", MaintainComments=True)
                IOtools.changeValue(self.nml, "thicha", self.geometry_select['antenna_t'], None, "=", MaintainComments=True)
            else:
                print("\t- Namelist not available in this transp instance, Antenna geometry not written", typeMsg='w')

        else:
            self.geometry_select = None

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

    def icrf_on_time(self, time, power_MW, freq_MHz, ramp_time = 1E-3):

        for t in self.variables.keys():
            if t>time:
                self.add_variable_time(t, None, power_MW*1E6, variable='RFP')
        
        self.add_variable_time(time-ramp_time, None, 0.0, variable='RFP')
        self.add_variable_time(time, None, power_MW*1E6, variable='RFP')
        self.add_variable_time(1E3, None, power_MW*1E6, variable='RFP')

        # Antenna Frequency
        IOtools.changeValue(self.nml, "frqicha", freq_MHz*1E6, None, "=", MaintainComments=True)

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
        if checkMin is not None:
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
                self.time_extraction = self.c_original.t[self.c_original.ind_saw]
                print(f"Populating from time of last sawtooth: {self.time_extraction}")
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

    def _produce_structures_from_variables(self, R, a, kappa, delta, zeta, z0, vv_relative_a=0.25, antenna_a=0.02):

        # --------------------------------------------------------------
        # Antenna (namelist)
        # --------------------------------------------------------------
        print(f"\t- Populating Antenna in namelist, with rmin = {a}+{antenna_a}m")
        self.antenna_R = R
        self.antenna_r = a + antenna_a
        self.antenna_t = 30.0

        self.geometry['antenna_R'] = self.antenna_R
        self.geometry['antenna_r'] = self.antenna_r
        self.geometry['antenna_t'] = self.antenna_t

        # --------------------------------------------------------------
        # Limiters
        # --------------------------------------------------------------

        # To fix in the future---------------------------------
        R, a, kappa, delta, zeta, z0 = R, R, 1.0, 0.0, 0.0, 0.0
        # -----------------------------------------------------
        
        vv = GEQtools.mitim_flux_surfaces()
        vv.reconstruct_from_miller(R, a, kappa, delta, zeta, z0)
        rvv , zvv= vv.R[0,:], vv.Z[0,:]

        # --------------------------------------------------------------
        # VV moments
        # --------------------------------------------------------------

        self.geometry['VVRmom'], self.geometry['VVZmom'], rvv_fit_cm, zvv_fit_cm = decomposeMoments(
            rvv*100.0 , zvv*100.0,
            r_ini = [R*100.0, a*100.0, 3.0], z_ini = [0.0, a*kappa*100.0, -3.0], verbose_level =5)

        # --------------------------------------------------------------
        # Limiters
        # --------------------------------------------------------------

        self.geometry['R_lim'], self.geometry['Z_lim'] = rvv, zvv

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


# ----------------------------------------------------------------------------------------------------------
# Utilities (belonging original to EQmodule.py or namelist)
# ----------------------------------------------------------------------------------------------------------

def generateMRY(
    FolderEquilibrium,
    times,
    FolderMRY,
    nameBaseShot,
    momentsScruncher=12,
    BtSign=-1,
    IpSign=-1,
    name="",
    ):
    filesInput = [FolderEquilibrium + "/scrunch_in", FolderEquilibrium + "/ga.d"]

    if momentsScruncher > 12:
        print(
            "\t- SCRUNCHER may fail because the maximum number of moments is 12",
            typeMsg="w",
        )

    with open(FolderEquilibrium + "ga.d", "w") as f:
        for i in times:
            nam = f"BOUNDARY_123456_{i}.DAT"
            f.write(nam + "\n")
            filesInput.append(FolderEquilibrium + "/" + nam)

    # Generate answers to scruncher in file scrunch_in
    with open(FolderEquilibrium + "scrunch_in", "w") as f:
        f.write(f"g\nga.d\n{momentsScruncher}\na\nY\nN\nN\nY\nX")

    # Run scruncher
    print(
        f">> Running scruncher with {momentsScruncher} moments to create MRY file from boundary files..."
    )

    scruncher_job = FARMINGtools.mitim_job(FolderEquilibrium)

    scruncher_job.define_machine(
        "scruncher",
        f"tmp_scruncher_{name}/",
        launchSlurm=False,
    )

    scruncher_job.prep(
        "scruncher < scrunch_in",
        output_files=["M123456.MRY"],
        input_files=filesInput,
    )

    scruncher_job.run()

    fileUF = f"{FolderMRY}/PRF{nameBaseShot}.MRY"
    os.system(f"mv {FolderEquilibrium}/M123456.MRY {fileUF}")

    # Check if MRY file has the number of times expected
    UF = UFILEStools.UFILEtransp()
    UF.readUFILE(fileUF)
    if len(UF.Variables["X"]) != len(times):
        raise Exception(
            " There was a problem in scruncher, at least one boundary time not used"
        )

def addLimiters_NML(namelistPath, rs, zs, centerP, ax=None):
    numLim = 10

    x, y = MATHtools.downsampleCurve(rs, zs, nsamp=numLim + 1)
    x = x[:-1]
    y = y[:-1]

    t = []
    for i in range(numLim):
        t.append(
            -np.arctan((y[i] - centerP[1]) / (x[i] - centerP[0])) * 180 / np.pi + 90.0
        )

    alnlmr_str = IOtools.ArrayToString(x)
    alnlmy_str = IOtools.ArrayToString(y)
    alnlmt_str = IOtools.ArrayToString(t)
    IOtools.changeValue(
        namelistPath, "nlinlm", numLim, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "alnlmr", alnlmr_str, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "alnlmy", alnlmy_str, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "alnlmt", alnlmt_str, None, "=", MaintainComments=True
    )

    if ax is not None:
        ax.plot(
            x / 100.0, y / 100.0, 100, "-o", markersize=0.5, lw=0.5, c="k", label="lims"
        )

def addLimiters_UF(UFilePath, rs, zs, ax=None, numLim=100):
    # ----- ----- ----- ----- -----
    # Ensure that there are no repeats points
    rs, zs = IOtools.removeRepeatedPoints_2D(rs, zs)
    # ----- ----- ----- ----- -----

    x, y = rs, zs  # MATHtools.downsampleCurve(rs,zs,nsamp=numLim)

    # Write Ufile
    UF = UFILEStools.UFILEtransp(scratch="lim")
    UF.Variables["X"] = x
    UF.Variables["Z"] = y
    UF.writeUFILE(UFilePath)

    if ax is not None:
        ax.plot(x, y, "-o", markersize=0.5, lw=0.5, c="k", label="lims")

    print(
        f"\t- Limiters UFile created in ...{UFilePath[np.max([-40, -len(UFilePath)]):]}"
    )

def writeBoundary(nameFile, rs_orig, zs_orig):
    numpoints = len(rs_orig)

    closerInteg = int(numpoints / 10)

    if closerInteg * 10 < numpoints:
        extraneeded = True
        rs = np.reshape(rs_orig[: int(closerInteg * 10)], (int(closerInteg), 10))
        zs = np.reshape(zs_orig[: int(closerInteg * 10)], (int(closerInteg), 10))
    else:
        extraneeded = False
        rs = np.reshape(rs_orig, (int(closerInteg), 10))
        zs = np.reshape(zs_orig, (int(closerInteg), 10))

    # Write the file for scruncher
    f = open(nameFile, "w")
    f.write("Boundary description for timeslice 123456 1000 msec\n")
    f.write("Shot date: 29AUG-2018\n")
    f.write("UNITS ARE METERS\n")
    f.write(f"Number of points: {numpoints}\n")
    f.write(
        "Begin R-array ==================================================================\n"
    )
    f.close()

    f = open(nameFile, "a")
    np.savetxt(f, rs, fmt="%7.3f")
    if extraneeded:
        rs_extra = np.array([rs_orig[int(closerInteg * 10) :]])
        np.savetxt(f, rs_extra, fmt="%7.3f")
    f.write(
        "Begin z-array ==================================================================\n"
    )
    np.savetxt(f, zs, fmt="%7.3f")
    if extraneeded:
        zs_extra = np.array([zs_orig[int(closerInteg * 10) :]])
        np.savetxt(f, zs_extra, fmt="%7.3f")
    f.write("")
    f.close()

def readBoundary(RFS, ZFS, time=0.0, rho_index=-1):
    # Check if MRY file has the number of times expected
    UF = UFILEStools.UFILEtransp()
    UF.readUFILE(RFS)
    t = UF.Variables["X"]
    theta = UF.Variables["Y"]
    Rs = UF.Variables["Z"]
    UF.readUFILE(ZFS)
    Zs = UF.Variables["Z"]

    it = np.argmin(np.abs(t - time))
    R, Z = Rs[it, :, rho_index], Zs[it, :, rho_index]

    return R, Z

def reconstructVV(VVr, VVz, nth=1000):
    nfour = len(VVr)

    x = np.append(VVr, VVz)

    theta = np.zeros(nth)
    for i in range(0, nth):
        theta[i] = (i / float(nth - 1)) * 2 * np.pi

    cosvals = np.zeros((nfour, nth))
    sinvals = np.zeros((nfour, nth))
    r_eval = np.zeros(nth)
    z_eval = np.zeros(nth)

    for i in range(0, nfour):
        thetas = i * theta
        # print(x[i])
        for j in range(0, nth):
            cosvals[i, j] = x[i] * np.cos(thetas[j])

        # Evaluate the z value thetas
        for i in range(nfour, 2 * nfour):
            thetas = (i - nfour) * theta
            for j in range(0, nth):
                sinvals[i - nfour, j] = x[i] * np.sin(thetas[j])

            for i in range(0, nth):
                r_eval[i] = np.sum(cosvals[:, i])
                z_eval[i] = np.sum(sinvals[:, i])

    return r_eval, z_eval

def decomposeMoments(R, Z, nfour=5, r_ini = [180, 70, 3.0], z_ini = [0.0, 140, -3.0], verbose_level=0):
    nth = len(R)

    # Initial Guess for R and Z array
    r = r_ini #[180, 70, 3.0]  # [rmajor,anew,3.0,0,0]
    z = z_ini #[0.0, 140, -3.0]  # [0.0,anew*kappa,-3.0,0.0,0.0]

    for i in range(nfour - 3):
        r.append(0)
        z.append(0)

    x = np.concatenate([r, z])

    theta = np.zeros(nth)
    for i in range(0, nth):
        theta[i] = (i / float(nth - 1)) * 2 * np.pi

    def vv_func(x):
        cosvals = np.zeros((nfour, nth))
        sinvals = np.zeros((nfour, nth))
        r_eval = np.zeros(nth)
        z_eval = np.zeros(nth)

        # Evaluate the r value thetas
        for i in range(0, nfour):
            thetas = i * theta
            for j in range(0, nth):
                cosvals[i, j] = x[i] * np.cos(thetas[j])

        # Evaluate the z value thetas
        for i in range(nfour, 2 * nfour):
            thetas = (i - nfour) * theta
            for j in range(0, nth):
                sinvals[i - nfour, j] = x[i] * np.sin(thetas[j])

            for i in range(0, nth):
                r_eval[i] = np.sum(cosvals[:, i])
                z_eval[i] = np.sum(sinvals[:, i])

        rsum = np.sum(abs(r_eval - R))
        zsum = np.sum(abs(z_eval - Z))

        metric = rsum + zsum

        return metric

    # ----------------------------------------------------------------------------------
    # Perform optimization
    # ----------------------------------------------------------------------------------

    print(f">> Performing minimization to fit VV moments ({nfour})...")

    from scipy.optimize import minimize
    res = minimize(
        vv_func,
        x,
        method="nelder-mead",
        options={"xtol": 1e-4, "disp": verbose_level in [4, 5]},
    )

    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

    x = res.x

    rmom = x[0:nfour]
    zmom = x[nfour : 2 * nfour]

    # Convert back to see error
    cosvals = np.zeros((nfour, nth))
    sinvals = np.zeros((nfour, nth))
    r_eval = np.zeros(nth)
    z_eval = np.zeros(nth)

    # Evaluate the r value thetas
    for i in range(0, nfour):
        thetas = i * theta
        for j in range(0, nth):
            cosvals[i, j] = x[i] * np.cos(thetas[j])

        # Evaluate the z value thetas
        for i in range(nfour, 2 * nfour):
            thetas = (i - nfour) * theta
            for j in range(0, nth):
                sinvals[i - nfour, j] = x[i] * np.sin(thetas[j])

            for i in range(0, nth):
                r_eval[i] = np.sum(cosvals[:, i])
                z_eval[i] = np.sum(sinvals[:, i])

            rsum = np.sum(abs(r_eval - R))
            zsum = np.sum(abs(z_eval - Z))

    return rmom, zmom, r_eval, z_eval

def interpret_trdat(file):
    if not os.path.exists(file):
        print("TRDAT was not generated. It will likely fail!", typeMsg="q")
    else:
        with open(file, "r") as f:
            aux = f.readlines()

        file_plain = "".join(aux)

        errors = [pos for pos, char in enumerate(file_plain) if char == "?"]

        truly_error = False
        for cont, i in enumerate(errors):
            if (i == 0 or errors[cont - 1] < i - 1) and file_plain[
                i : i + 4
            ] != "????":  # Because that's an TOK error, it would be fine
                truly_error = True

        if truly_error:
            print(
                '\t- Detected "?" in TRDAT output, printing tr_dat around errors:',
                typeMsg="w",
            )
            print("-------------------------------", typeMsg="w")
            for i in range(len(aux)):
                if ("?" in aux[i]) and ("????" not in aux[i]):
                    print("".join(aux[np.max(i - 5, 0) : i + 2]), typeMsg="w")
                    print("-------------------------------", typeMsg="w")
            if not print(
                "Do you wish to continue? It will likely fail! (c)", typeMsg="q"
            ):
                embed()
        else:
            print("\t- TRDAT output did not show any error", typeMsg="i")

def reconstructAntenna(antrmaj, antrmin, polext):
    theta = np.linspace(-polext / 2.0, polext / 2.0, 100)
    R = []
    Z = []
    for i in theta:
        R.append(antrmaj + antrmin * np.cos(i * np.pi / 180.0))
        Z.append(antrmin * np.sin(i * np.pi / 180.0))

    return np.array(R), np.array(Z)



# ----------------------------------------------------------------------------------------------------------
# Utilities to run interpretive TRANSP
# ----------------------------------------------------------------------------------------------------------

def populateFromMDS(self, runidMDS):
    """
    This routine grabs NML and UFILES from MDS+ and puts them in the right folder
    """

    if self.tok == "CMOD":
        from mitim_tools.experiment_tools.CMODtools import getTRANSP_MDS
    else:
        raise Exception("Tokamak MDS+ not implemented")

    # shotNumber = self.runid[-3:]
    getTRANSP_MDS(
        runidMDS,
        self.runid,
        folderWork=self.FolderTRANSP,
        toric_mpi=self.mpisettings["toricmpi"],
        shotnumber=self.shotnumberReal,
    )


def defaultbasedMDS(self, outtims=[], PRFmodified=False):
    """
    This routine creates a default nml for the given tokamak, and modifies it according to an
    existing nml that, e.g. has come from MDS+

    PRFmodified = True doesn't care about original model settings, I use mine
    """

    if self.tok == "CMOD":
        from mitim_tools.experiment_tools.CMODtools import updateTRANSPfromNML
    else:
        raise Exception("Tokamak MDS+ not implemented")

    os.system("cp {0} {0}_old".format(self.nml_file))
    TRANSPnamelist_dict = updateTRANSPfromNML(
        self.nml_file + "_old",
        self.nml_file,
        self.FolderTRANSP,
        PRFmodified=PRFmodified,
    )

    # Write NML

    ntoric = int(self.mpisettings["toricmpi"] > 1)
    nbi = int(self.mpisettings["trmpi"] > 1)
    nptr = int(self.mpisettings["ptrmpi"] > 1)

    # Write generic namelist for this tokamak
    nml = NMLtools.default_nml(self.shotnumber, self.tok, pservers=[nbi, ntoric, nptr])
    nml.write(BaseFile=self.nml_file)
    nml.appendNMLs()

    # To allow the option of giving negative outtimes to reflect from the end
    outtims_new = []
    ftime = IOtools.findValue(self.nml_file, "ftime", "=")
    for i in outtims:
        if i < 0.0:
            outtims_new.append(ftime - i)
        else:
            outtims_new.append(i)

    # Modify according to TRANSPnamelist_dict
    changeNamelist(
        self.nml_file,
        self.shotnumber,
        TRANSPnamelist_dict,
        self.FolderTRANSP,
        outtims=outtims_new,
    )


def changeNamelist(
    namelistPath, nameBaseShot, TRANSPnamelist, FolderTRANSP, outtims=[]
    ):
    # Change shot number
    IOtools.changeValue(
        namelistPath, "nshot", nameBaseShot, None, "=", MaintainComments=True
    )

    # TRANSP fixed namelist + those parameters changed
    for itag in TRANSPnamelist:
        IOtools.changeValue(
            namelistPath, itag, TRANSPnamelist[itag], None, "=", MaintainComments=True
        )

    # Add inputdir to namelist
    with open(namelistPath, "a") as f:
        f.write("inputdir='" + os.path.abspath(FolderTRANSP) + "'\n")

    # Change PTR templates
    IOtools.changeValue(
        namelistPath,
        "pt_template",
        f'"{os.path.abspath(FolderTRANSP)}/ptsolver_namelist.dat"',
        None,
        "=",
        CommentChar=None,
    )
    IOtools.changeValue(
        namelistPath,
        "tglf_template",
        f'"{os.path.abspath(FolderTRANSP)}/tglf_namelist.dat"',
        None,
        "=",
        CommentChar=None,
    )
    IOtools.changeValue(
        namelistPath,
        "glf23_template",
        f'"{os.path.abspath(FolderTRANSP)}/glf23_namelist.dat"',
        None,
        "=",
        CommentChar=None,
    )

    # Add outtims
    if len(outtims) > 0:
        NMLtools.addOUTtimes(namelistPath, outtims)


