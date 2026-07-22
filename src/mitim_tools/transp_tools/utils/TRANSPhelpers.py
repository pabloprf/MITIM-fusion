import shutil
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.transp_tools import TRANSPtools, CDFtools, UFILEStools, NMLtools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools import IOtools, MATHtools, PLASMAtools, GRAPHICStools, FARMINGtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

# ----------------------------------------------------------------------------------------------------------
# TRANSP object utilities
# ----------------------------------------------------------------------------------------------------------

class transp_run:
    def __init__(self, folder, shot, runid):

        self.shot, self.runid = shot, runid
        self.folder = IOtools.expandPath(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

        # Initialize variables
        self.variables, self.geometry = {}, {}
        self.nml = None
        self.nml_object = None   # set by write_namelist; carries Pich (whether ICRF is modeled)
        self.namelist_variables = {}
        self.rotation_species = None   # [Z, A] of the species the 'omg' U-File rotation belongs to (set by to_transp)
        self.omg_zeroed = False        # if True, ship the 'omg' U-File as zeros (neoclassical_transp mode: weak-rotation input)
        self.compute_nclass_er = False # if True, compute the NCLASS neoclassical Er (nlvwnc=T); only neoclassical_transp keeps it (set by to_transp)

        # Describe the time populator --------------
        self.populate_time = transp_input_time(self)
        # ------------------------------------------

        # This helps as mapping between cdf, profs and ufiles
        # name_cdf : [uf type, uf name, coordinate, factor_cdf, name_profs, factor_profs]

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
            'PnbiT': ['nb2','NB2',[1], 1E6],
        }

    # --------------------------------------------------------------------------------------------
    # Namelist
    # --------------------------------------------------------------------------------------------

    def write_namelist(
        self,
        timings = {},
        tokamak_structures = None,
        **transp_params
        ):
        ''' 
        Create a namelist for TRANSP based on default parameters + transp_params
        '''

        # If a rotation U-File is being written (w0 present), have TRANSP read it
        # (UFrotation -> nlvphi=T + register the 'omg' U-File). An explicit
        # UFrotation passed by the caller is respected. Also declare the species the
        # rotation belongs to: w0 is the bulk plasma rotation, so to_transp sets
        # rotation_species to the MAIN ION (NGVTOR=0 + NVTOR_Z/XVTOR_A) rather than the
        # default lumped impurity.
        if 'w0' in self.quantities:
            transp_params.setdefault('UFrotation', True)
            # nlvwnc (NCLASS Er) is decoupled from rotation modeling (nlvphi): computed only
            # when the caller keeps it (neoclassical_transp). echo / neoclassical_portals
            # model the rotation but skip the (discarded) NCLASS Er.
            transp_params.setdefault('computeNCLASSpotential', self.compute_nclass_er)
            if self.rotation_species is not None:
                transp_params.setdefault('rotating_impurity', self.rotation_species)

        self.nml_object = NMLtools.transp_nml(shotnum=self.shot, inputdir=self.folder, timings=timings)
        self.nml_object.define_machine(tokamak_structures)
        self.nml_object.populate(**transp_params)
        self.nml_object.write(self.runid)

        self.nml = self.folder / f"{self.shot}{self.runid}TR.DAT"

        self._insert_parameters_namelist()

    def namelist_from(self,
        folder_original,
        nml_original,
        tinit,ftime,curr_diff=None, sawtooth = None, writeAC = None
        ):
        '''
        Copy namelist from folder_original and change timings
        '''

        self.nml = self.folder / f"{self.shot}{self.runid}TR.DAT"
        shutil.copy2(folder_original / f'{nml_original}', self.nml)

        # Predictive namelists
        for item in folder_original.glob('*namelist.dat'):
            shutil.copy2(item, self.folder)

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
        
        self._insert_parameters_namelist()

    # --------------------------------------------------------------------------------------------
    # Ufiles
    # --------------------------------------------------------------------------------------------

    def write_ufiles(
            self,
            structures_position = -1,
            radial_position = 0,
            use_mry_file = False,
            mxh_coeffs_smooth = 5,
            is_machine_fixed = False
            ):
        '''
        Write ufiles based on variables that were stored (e.g. from freegs or cdf)
        '''

        times = np.sort(np.array(list(self.variables.keys())))
        times_geometry = np.sort(np.array(list(self.geometry.keys())))

        self.times = np.unique(np.concatenate((times, times_geometry)))

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
            uf.writeUFILE(self.folder / f'MIT{self.shot}.{self.quantities[quantity][1]}')

        # --------------------------------------------------------------------------------------------
        # Write Boundary UFILE
        # --------------------------------------------------------------------------------------------

        if use_mry_file:
            # Use MRY file
            tt = []
            for time in self.times:
                if (time in self.geometry) and ('R_sep' in self.geometry[time]):
                    t = str(int(time*1E3)).zfill(5)
                    R, Z = self.geometry[time]['R_sep'], self.geometry[time]['Z_sep']

                    # Downsample seems to be necessary sometimes ----
                    R, Z = MATHtools.downsampleCurve(R, Z, nsamp=100)
                    R, Z = R[:-1], Z[:-1]
                    # -----------------------------------------------

                    writeBoundary(self.folder / f'BOUNDARY_123456_{t}.DAT', R, Z)
                    self.geometry[time]['R_sep_transp'], self.geometry[time]['Z_sep_transp'] = R, Z
                    tt.append(t)

            generateMRY(
                self.folder,
                tt,
                self.folder,
                self.shot,
                momentsScruncher=6, # To avoid curvature issues, let's smooth it
                )

        else:
            '''
            Use Boundary Files
            -------------------
            I'm only writing the rho=1 (what's important for the fixed boundary)
            and the rho=0 (because it's needed, but I'm only giving it at the center as a trick!)
            '''
            ts, Rs, Zs, R0, Z0 = [], [], [], [], []
            for time in self.times:
                if (time in self.geometry) and ('R_sep' in self.geometry[time]):

                    if mxh_coeffs_smooth is not None:
                        thetas, R, Z = prepare_RZsep_for_TRANSP(
                            self.geometry[time]['R_sep'],
                            self.geometry[time]['Z_sep'],
                            n_coeff=mxh_coeffs_smooth
                            )
                    else:
                        R, Z = self.geometry[time]['R_sep'], self.geometry[time]['Z_sep']

                    r0, z0 = (R.max()+R.min())/2, (Z.max()+Z.min())/2

                    # The exact curve handed to TRANSP (post-MXH-smoothing). Stored so a caller can
                    # freeze and reuse it verbatim instead of re-deriving (and re-smoothing) it.
                    self.geometry[time]['R_sep_transp'], self.geometry[time]['Z_sep_transp'] = R, Z

                    ts.append(time)
                    Rs.append(R)
                    Zs.append(Z)
                    R0.append(  r0 * np.ones(R.shape) )
                    Z0.append(  z0 * np.ones(R.shape) )

            Rs, Zs, R0, Z0 = np.array(Rs), np.array(Zs), np.array(R0), np.array(Z0)

            Zr = np.append(R0[np.newaxis,:,:],Rs[np.newaxis,:,:],axis=0)
            Zz = np.append(Z0[np.newaxis,:,:],Zs[np.newaxis,:,:],axis=0)

            '''
            The way my routines are written the UFILE RFS must be written:
                (t0, theta0, rho0), (t1, theta0, rho0), (t2, theta0, rho0), (t0, theta1, rho0), (t0, theta2, rho0), ...
            After trial and error, this transposition is what works
            '''
            Zr = np.transpose(Zr, (1, 2, 0))
            Zz = np.transpose(Zz, (1, 2, 0))
            # --------------------------------------------------------------------------------------------

            uf = UFILEStools.UFILEtransp(scratch='rfs')
            uf.Variables['X'] = ts
            uf.Variables['Q'] = [0.0,1.0]
            uf.Variables['Y'] = np.linspace(0, 2*np.pi, Rs.shape[-1], endpoint=True)
            uf.Variables['Z'] = Zr
            uf.writeUFILE(self.folder / f'MIT{self.shot}.RFS')

            uf = UFILEStools.UFILEtransp(scratch='zfs')
            uf.Variables['X'] = ts
            uf.Variables['Q'] = [0.0,1.0]
            uf.Variables['Y'] = np.linspace(0, 2*np.pi, Rs.shape[-1], endpoint=True)
            uf.Variables['Z'] = Zz
            uf.writeUFILE(self.folder / f'MIT{self.shot}.ZFS')

        if structures_position is not None:

            # --------------------------------------------------------------------------------------------
            # Write VV in namelist (defer to later)
            # --------------------------------------------------------------------------------------------

            geos = []
            for time in self.times:    
                if (time in self.geometry) and ('VVRmom' in self.geometry[time]):    
                    geos.append(self.geometry[time])
            self.geometry_select = geos[structures_position]

            self.namelist_variables['VVRmom'] = ', '.join([f'{x:.8e}' for x in self.geometry_select['VVRmom']])
            self.namelist_variables['VVZmom'] = ', '.join([f'{x:.8e}' for x in self.geometry_select['VVZmom']])

            # --------------------------------------------------------------------------------------------
            # Write Limiters in ufile
            # --------------------------------------------------------------------------------------------

            addLimiters_UF(self.folder / f'MIT{self.shot}.LIM', self.geometry_select['R_lim'], self.geometry_select['Z_lim'], numLim=len(self.geometry_select['R_lim']))

            # --------------------------------------------------------------------------------------------
            # Write Antenna in namelist
            # --------------------------------------------------------------------------------------------

            # The auto-generated antenna geometry only belongs in the namelist when ICRF is
            # actually modeled: TRANSP turns the TORIC solver ON from the mere presence of
            # rmjicha/rmnicha/thicha (DATCKICH "old style ICH antenna geometry converted"),
            # which then crashes when no TORIC grid exists (addICRF is skipped for Pich=False).
            # nml_object is None during the to_transp-time call (which writes nothing to the
            # namelist anyway); the post-write_namelist call carries the real Pich.
            icrf_on = self.nml_object is not None and self.nml_object.Pich
            if (not is_machine_fixed) and icrf_on:

                self.namelist_variables['rmjicha'] = 100.0*self.geometry_select['antenna_R']
                self.namelist_variables['rmnicha'] = 100.0*self.geometry_select['antenna_r']
                self.namelist_variables['thicha'] = self.geometry_select['antenna_t']

        else:
            self.geometry_select = None

        self._insert_parameters_namelist()

    def _insert_parameters_namelist(self):

        if self.nml is not None:
            for var in self.namelist_variables.keys():
                IOtools.changeValue(self.nml, var, self.namelist_variables[var], None, "=", MaintainComments=True)
            print("\t- Namelist updated with new parameters of VV and antenna", typeMsg='i')
        else:
            print("\t- Namelist not available in this transp instance yet, defering writing VV and antenna to later", typeMsg='i')

    def ufiles_from(self, folder_original, ufiles):
        '''
        Copy ufiles from folder_original
        '''

        for ufile in ufiles:
            shutil.copy2(folder_original / f'MIT12345.{ufile}', self.folder)

    # --------------------------------------------------------------------------------------------
    # Utilities to populate specific times with something
    # --------------------------------------------------------------------------------------------

    def add_g_time(self, time, g_file_loc):

        g = GEQtools.MITIMgeqdsk(g_file_loc)
        self._add_separatrix_time(time, g.Rb, g.Yb)

    def _add_separatrix_time(self, time, R_sep, Z_sep):

        if time not in self.geometry.keys():
            self.geometry[time] = {}

        self.geometry[time]['R_sep'] = R_sep
        self.geometry[time]['Z_sep'] = Z_sep


    def power_on_time(self, time, power_MW, ramp_time = 1E-3, nchannels = 1, n_channels_active = 1, variable_name = 'RFP'):

        channels_array = np.arange(nchannels) +1

        def add_variable_time(time, value):

            if time not in self.variables.keys():
                self.variables[time] = {}
            value = value * np.ones(nchannels) / n_channels_active

            self.variables[time][variable_name] = {
                'x': channels_array,
                'z': value
            }
        
        for t in self.variables.keys():
            if t>time:
                add_variable_time(t, power_MW*1E6)
            else:
                add_variable_time(t, 0.0)
        
        time_prev = round(time - ramp_time, 10)
        add_variable_time(time_prev, 0.0)
        add_variable_time(time, power_MW*1E6)
        add_variable_time(1E3, power_MW*1E6)

    def icrf_on_time(self, time, power_MW, freq_MHz, ramp_time = 1E-3, nicha = 1):

        self.power_on_time(time, power_MW, ramp_time = ramp_time, nchannels = nicha, n_channels_active = nicha, variable_name = 'RFP')  

        # Antenna Frequency
        if freq_MHz is not None:
            IOtools.changeValue(self.nml, "frqicha", freq_MHz*1E6, None, "=", MaintainComments=True)

        self.quantities['PichT'][2] = 'x' #channels_array

    def nbi_on_time(self, time, power_MW, ramp_time = 1E-3, nbeams = 1, nbeams_active = 1):

        self.power_on_time(time, power_MW, ramp_time = ramp_time, nchannels = nbeams, n_channels_active = nbeams_active, variable_name = 'NB2')  

        self.quantities['PnbiT'][2] = 'x' #channels_array

    # --------------------------------------------------------------------------------------------

    def run(self, tokamakTRANSP, tokamak_name = None, mpisettings={"trmpi": 32, "toricmpi": 32, "ptrmpi": 1}, minutesAllocation = 60*8, case='run1', checkMin=10.0, grabIntermediateEachMin=1E6, retrieveAC=False, cold_start=True):
        '''
        Run TRANSP

        cold_start:
            True  (default) -> always (re)run, preserving the historical behaviour.
            False -> if the output CDF ({shot}{runid}.CDF) already exists in the run
                     folder, reuse it instead of re-staging + re-submitting to SLURM
                     (so a re-run just reads the existing result). The standalone run
                     path has no idempotency on its own, hence this guard.
        '''

        # Skip-if-done: reuse an existing finished CDF when a cold start was not requested
        cdf_file = self.folder / f"{self.shot}{self.runid}.CDF"
        if (not cold_start) and cdf_file.exists():
            print(f"\t- Found existing TRANSP output {IOtools.clipstr(cdf_file)}; reusing it (cold_start=False), not submitting", typeMsg="i")
            return

        print("\t- Running TRANSP")

        if tokamak_name is None:
            tokamak_name = tokamakTRANSP

        self.t = TRANSPtools.TRANSP(self.folder, tokamakTRANSP)

        self.t.defineRunParameters(
            self.shot + self.runid, self.shot,
            mpisettings = mpisettings,
            minutesAllocation = minutesAllocation,
            tokamak_name = tokamak_name)

        self.t.run()

        # Check until finished if it's a slurm job or globus
        from mitim_tools.transp_tools.src.TRANSPglobus import TRANSPglobus
        is_this_worth_waiting = isinstance(self.t, TRANSPglobus) or (len(self.t.job.machineSettings['slurm']) > 0)
        if is_this_worth_waiting and (checkMin is not None):
            self.c = self.t.checkUntilFinished(label=case, checkMin=checkMin, grabIntermediateEachMin=grabIntermediateEachMin, retrieveAC=retrieveAC)

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

        GRAPHICStools.adjust_figure_layout(fig)
        plt.show()

def prepare_RZsep_for_TRANSP(Ro, Zo, n_coeff=5, thetas = np.linspace(0, 2*np.pi, 101, endpoint=True), plotYN = False):
    '''
    TRANSP tends to give troubles with kinks, curvatures and loops in the boundary files.
    This method developed in MITIM helps to smooth the boundary and avoid these issues.
    It converts the boundary to MXH and then back to boundary, using a number of coefficients.
    '''

    surfaces = GEQtools.mitim_flux_surfaces()
    surfaces.reconstruct_from_RZ(Ro,Zo)
    surfaces._to_mxh(n_coeff=n_coeff)
    surfaces._from_mxh(thetas = thetas)

    if plotYN:
        fig, ax = plt.subplots()
        ax.plot(Ro, Zo, '-o', ms=5, label='Original')
        ax.plot(surfaces.R[0], surfaces.Z[0], '-s', ms=5, label=f'Smoothed n_coeff={n_coeff}')
        ax.legend(loc='best')
        ax.set_aspect('equal')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        plt.show()
        embed()

    return thetas, surfaces.R[0], surfaces.Z[0]

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
            if isinstance(cdf_file, str):
                print(f"Populating time {time} from {cdf_file}")
                self.c_original = CDFtools.transp_output(cdf_file)
            else:
                print(f"Populating time {time} from opened CDF")
                self.c_original = cdf_file
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
        self.geometry['R_sep'], self.geometry['Z_sep'] = R_sep, Z_sep

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
        print(f"\t- Populating Antenna in namelist, with rmin = {a:.3f}+{antenna_a:.3f}m")
        self.antenna_R = R
        self.antenna_r = a + antenna_a
        self.antenna_t = 30.0

        self.geometry['antenna_R'] = self.antenna_R
        self.geometry['antenna_r'] = self.antenna_r
        self.geometry['antenna_t'] = self.antenna_t

        # --------------------------------------------------------------
        # Limiters
        # --------------------------------------------------------------

        #TODO: Fix-----------------------------------------------------
        R, a, kappa, delta, zeta, z0 = R*2, R*2, 1.0, 0.0, 0.0, 0.0
        # -------------------------------------------------------------
        
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

    def from_freegs(self, time, R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = 3.3, Vsurf = 0.0, Zeff = 1.5, Paux_MW = 11.0):

        # Create Miller FreeGS for the desired geometry
        self.f = GEQtools.freegs_millerized( R, a, kappa_sep, delta_sep, zeta_sep, z0)
        self.f.prep(p0_MPa, Ip_MA, B_T, constraint_miller_squareness_point= True) #TODO: Not sure why, but C-Mod TRANSP initialization case works better with this
        self.f.solve()
        self.f.derive()
        #self.f.check(plotYN=True)

        self._from_freegs_eq(time, ne0_20 = ne0_20, Vsurf = Vsurf, Zeff = Zeff, Paux_MW = Paux_MW)

    def _from_freegs_eq(self, time, freegs_eq_object = None, ne0_20 = 3.3, Vsurf = 0.0, Zeff = 1.5, Paux_MW = 11.0):

        if freegs_eq_object is None:
            freegs_eq_object = self.f.eq

        # Rho Tor 
        psi = np.linspace(freegs_eq_object.psi_axis, freegs_eq_object.psi_bndry, 101, endpoint=True)
        psi_norm = (psi - freegs_eq_object.psi_axis) / (freegs_eq_object.psi_bndry - freegs_eq_object.psi_axis)
        rhotor = freegs_eq_object.rhotor(psi)

        q = freegs_eq_object.q(psinorm = psi_norm)
        pressure = freegs_eq_object.pressure(psinorm =psi_norm) # Pa

        Ip = freegs_eq_object._profiles.Ip  # A
        RB = freegs_eq_object._profiles._fvac* 1E2 
        RZ = freegs_eq_object.separatrix(npoints= 100)

        self._from_eq_quantities(time, rhotor, q, pressure, Ip, RB, RZ, ne0_20 = ne0_20, Vsurf = Vsurf, Zeff = Zeff, Paux_MW = Paux_MW)

    def from_geqdsk(self, time, geqdsk_object, ne0_20 = 3.3, Vsurf = 0.0, Zeff = 1.5, Paux_MW = 11.0):
        

        rhotor = geqdsk_object.g['RHOVN']
        q = geqdsk_object.g['QPSI']
        pressure = geqdsk_object.g['PRES']

        Ip = geqdsk_object.g['CURRENT']  # A
        RB = geqdsk_object.g['RCENTR']*geqdsk_object.g['BCENTR'] * 1E2 
        RZ = np.array([geqdsk_object.Rb,geqdsk_object.Yb]).T

        self._from_eq_quantities(time, rhotor, q, pressure, Ip, RB, RZ, ne0_20 = ne0_20, Vsurf = Vsurf, Zeff = Zeff, Paux_MW = Paux_MW)

    def _from_eq_quantities(self, time, rhotor, q, pressure, Ip, RB, RZ, ne0_20 = 3.3, Vsurf = 0.0, Zeff = 1.5, Paux_MW = 11.0):

        self.variables = {}
        self.ne0_20 = ne0_20

        # --------------------------------------------------------------
        # q profile
        # --------------------------------------------------------------

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
            'z': Ip
            }

        self.variables['RBZ'] = {
            'x': None,
            'z': RB
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

        if Paux_MW is not None:
            self.variables['RFP'] = {
                'x': [1],
                'z': Paux_MW * 1E6
                }
            self.variables['NB2'] = {
                'x': [1],
                'z': Paux_MW * 1E6
                }

        # --------------------------------------------------------------
        # geometries
        # --------------------------------------------------------------

        Rsep, Zsep = RZ[:,0], RZ[:,1]
        self.geometry = {
            'R_sep': Rsep,
            'Z_sep': Zsep
            }

        surfaces = GEQtools.mitim_flux_surfaces()
        surfaces.reconstruct_from_RZ(Rsep, Zsep)
        surfaces._to_miller()
        self._produce_structures_from_variables(surfaces.R0[0], surfaces.a[0], surfaces.kappa[0], surfaces.delta[0], surfaces.zeta[0], surfaces.Z0[0])

        # --------------------------------------------------------------
        # Populate
        # --------------------------------------------------------------
        
        self._populate(time)

    def from_profiles(self, time, profiles_file, Vsurf = 0.0, boundary_surface_psin = 1.0, boundary_override = None):

        self.time = time

        if isinstance(profiles_file, str):
            from mitim_tools.gacode_tools import PROFILEStools
            self.p = PROFILEStools.gacode_state(profiles_file)
        else:
            self.p = profiles_file

        self.variables = {}
        for var in self.transp_instance.quantities.keys():
            self.variables[self.transp_instance.quantities[var][1]] = {}

            if var in ['Ip','RBt_vacuum','q','Te','Ti','ne','Zeff','PichT','PnbiT','w0']:
                self.variables[self.transp_instance.quantities[var][1]]['x'],self.variables[self.transp_instance.quantities[var][1]]['z'] = self._produce_quantity_profiles(var = var)
            
            # --------------------------------------------------------------
            # Quantities that do not come from profiles
            # --------------------------------------------------------------

            elif (var == 'Vsurf') and (Vsurf is not None):
                self.variables[self.transp_instance.quantities[var][1]]['x'] = None
                self.variables[self.transp_instance.quantities[var][1]]['z'] = Vsurf

        self._produce_geometry_profiles(boundary_surface_psin = boundary_surface_psin, boundary_override = boundary_override)
        self._populate(time)

    def _produce_quantity_profiles(self, var = 'Te', Vsurf = None):

        if var == 'Ip':
            x = None
            z = self.p.profiles['current(MA)'][0] * 1E6
        elif var == 'RBt_vacuum':
            x = None
            z = self.p.profiles["bcentr(T)"][0]* self.p.profiles["rcentr(m)"][0] * 1E2
        elif var == 'q':
            x = self.p.profiles['rho(-)']
            z = self.p.profiles['q(-)']
        elif var == 'Te':
            x = self.p.profiles['rho(-)']
            z = self.p.profiles['te(keV)']*1E3
        elif var == 'Ti':
            x = self.p.profiles['rho(-)']
            z = self.p.profiles['ti(keV)'][:,0]*1E3
        elif var == 'ne':
            x = self.p.profiles['rho(-)']
            z = self.p.profiles['ne(10^19/m^3)']*1E19*1E-6
        elif var == 'w0':
            # Toroidal angular rotation, rad/s -> 'omg' U-File (rad/sec); same quantity,
            # factor 1.0. Flux function (no major-radius dependence). In neoclassical_transp mode
            # the omg U-File is shipped as zeros (weak-rotation input, V_phi=0).
            x = self.p.profiles['rho(-)']
            z = self.p.profiles['w0(rad/s)'] * (0.0 if self.transp_instance.omg_zeroed else 1.0)
        elif var == 'Zeff':
            x = self.p.profiles['rho(-)']
            z = self.p.derived['Zeff']
        elif var == 'PichT':
            x = [1]
            z = self.p.derived['qRF_MW'][-1]*1E6
        elif var == 'PnbiT':
            x = [1]
            z = self.p.derived['qBEAM_MW'][-1]*1E6

        return x,z

    def _produce_geometry_profiles(self, boundary_surface_psin = 1.0, boundary_override = None):

        self.geometry = {}

        # --------------------------------------------------------------
        # Boundary: the separatrix (last surface) by default, or -- when boundary_surface_psin < 1 --
        # a flux surface just inside it, interpolated at that normalized poloidal flux. Backing the
        # TRANSP fixed boundary off a sharp separatrix (rounder interior surface) avoids TRANSP's
        # boundary curvature-ratio abort while preserving the true shape.
        #
        # boundary_override, when given, is an explicit (R,Z) curve used verbatim: no psi_N
        # interpolation at all. MAESTRO uses it to hand every TRANSP beat after the first the SAME
        # fixed boundary that beat 1 built (see TRANSPbeat._fixed_boundary_for_transp). Re-deriving it
        # per beat would compound: TRANSP is fixed-boundary, so its output psi_N=1 surface IS the
        # boundary it was given, and taking psi_N=boundary_surface_psin of that again shrinks the
        # plasma by that factor every beat (0.995 -> 0.995^2 -> ...).
        # --------------------------------------------------------------

        if boundary_override is not None:
            self.geometry['R_sep'] = np.array(boundary_override[0])
            self.geometry['Z_sep'] = np.array(boundary_override[1])
            self._produce_structures_from_variables(
                self.p.profiles['rcentr(m)'][0],
                self.p.derived['a'],
                self.p.profiles['kappa(-)'][-1],
                self.p.profiles['zmag(m)'][0],
                self.p.profiles['delta(-)'][-1],
                self.p.profiles['zeta(-)'][-1],
                )
            return

        Rsurf, Zsurf = self.p.derived["R_surface"][0], self.p.derived["Z_surface"][0]   # [nrad, ntheta]
        if boundary_surface_psin is None or boundary_surface_psin >= 1.0:
            R_sep, Z_sep = Rsurf[-1, :], Zsurf[-1, :]
        else:
            psi = self.p.profiles['polflux(Wb/radian)']
            psin = (psi - psi[0]) / (psi[-1] - psi[0])     # 0 at axis, 1 at separatrix
            R_sep = np.array([np.interp(boundary_surface_psin, psin, Rsurf[:, j]) for j in range(Rsurf.shape[1])])
            Z_sep = np.array([np.interp(boundary_surface_psin, psin, Zsurf[:, j]) for j in range(Zsurf.shape[1])])

        self.geometry['R_sep'], self.geometry['Z_sep'] = R_sep, Z_sep

        # --------------------------------------------------------------
        # VV
        # --------------------------------------------------------------

        self._produce_structures_from_variables(
            self.p.profiles['rcentr(m)'][0],
            self.p.derived['a'],
            self.p.profiles['kappa(-)'][-1],
            self.p.profiles['zmag(m)'][0],
            self.p.profiles['delta(-)'][-1],
            self.p.profiles['zeta(-)'][-1],
            )

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
    FolderEquilibrium = IOtools.expandPath(FolderEquilibrium)
    FolderMRY = IOtools.expandPath(FolderMRY)
    filesInput = [FolderEquilibrium / "scrunch_in", FolderEquilibrium / "ga.d"]

    if momentsScruncher > 12:
        print(
            "\t- SCRUNCHER may fail because the maximum number of moments is 12",
            typeMsg="w",
        )

    with open(FolderEquilibrium / "ga.d", "w") as f:
        for i in times:
            nam = f"BOUNDARY_123456_{i}.DAT"
            f.write(nam + "\n")
            filesInput.append(FolderEquilibrium / nam)

    # Generate answers to scruncher in file scrunch_in
    with open(FolderEquilibrium / "scrunch_in", "w") as f:
        f.write(f"g\nga.d\n{momentsScruncher}\na\nY\nN\nN\nY\nX")

    # Run scruncher
    print(
        f">> Running scruncher with {momentsScruncher} moments to create MRY file from boundary files..."
    )

    scruncher_job = FARMINGtools.mitim_job(FolderEquilibrium)

    scruncher_job.define_machine(
        "scruncher",
        f"tmp_scruncher_{name}",
        launchSlurm=False,
    )

    scruncher_job.prep(
        "scruncher < scrunch_in",
        output_files=["M123456.MRY"],
        input_files=filesInput,
    )

    scruncher_job.run()

    fileUF = FolderMRY / f"MIT{nameBaseShot}.MRY"
    (FolderEquilibrium / 'M123456.MRY').replace(fileUF)

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

    x, y = rs, zs

    # Write Ufile
    UF = UFILEStools.UFILEtransp(scratch="lim")
    UF.Variables["X"] = x
    UF.Variables["Z"] = y
    UF.writeUFILE(UFilePath)

    if ax is not None:
        ax.plot(x, y, "-o", markersize=0.5, lw=0.5, c="k", label="lims")

    print(f"\t- Limiters UFile created in ...{IOtools.clipstr(UFilePath)}")

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
        options={ "disp": verbose_level in [4, 5]},
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

# Apptainer/singularity messages indicating the container itself could not start (e.g. node image
# with unprivileged user namespaces disabled and no apptainer-suid), so TRANSP never ran at all
CONTAINER_LAUNCH_ERRORS = [
    "Failed to create user namespace",
    "max_user_namespaces",
]

def interpret_trdat(file):
    file = IOtools.expandPath(file)
    if not file.exists():
        print("TRDAT was not generated. It will likely fail!", typeMsg="q")
    else:
        with open(file, "r") as f:
            aux = f.readlines()

        file_plain = "".join(aux)

        if any(err in file_plain for err in CONTAINER_LAUNCH_ERRORS):
            raise RuntimeError(
                "[MITIM] TRDAT could not run: the container failed to launch on this node"
                " (user namespace creation denied, check user.max_user_namespaces)."
                " TRANSP cannot run here, aborting now instead of waiting for the job time limit"
            )

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
# Utilities to run interpretive TRANSP (#TODO: review and fix)
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

def defaultbasedMDS(self, outtims=None, MITIMmodified=False):
    """
    This routine creates a default nml for the given tokamak, and modifies it according to an
    existing nml that, e.g. has come from MDS+

    MITIMmodified = True doesn't care about original model settings, I use mine
    """

    if outtims is None:
        outtims = []

    if self.tok == "CMOD":
        from mitim_tools.experiment_tools.CMODtools import updateTRANSPfromNML
    else:
        raise Exception("Tokamak MDS+ not implemented")

    old_nml_file = self.nml_file.with_name(f"{self.nml_file.name}_old")
    shutil.copy2(self.nml_file, old_nml_file)
    TRANSPnamelist_dict = updateTRANSPfromNML(
        old_nml_file,
        self.nml_file,
        self.FolderTRANSP,
        MITIMmodified=MITIMmodified,
    )

    # Write NML

    ntoric = int(self.mpisettings["toricmpi"] > 1)
    nbi = int(self.mpisettings["trmpi"] > 1)
    nptr = int(self.mpisettings["ptrmpi"] > 1)

    # Write generic namelist for this tokamak
    default_nml(self.shotnumber, self.tok, self.nml_file, pservers=[nbi, ntoric, nptr])

    # To allow the option of giving negative outtimes to reflect from the end
    outtims_new = []
    ftime = IOtools.findValue(self.nml_file, "ftime", "=")
    for i in outtims:
        if i < 0.0:
            outtims_new.append(ftime - i)
        else:
            outtims_new.append(i)

    # -----------------------------------------
    # Modify according to TRANSPnamelist_dict
    # -----------------------------------------

    # Change shot number
    IOtools.changeValue(
        self.nml_file, "nshot", self.shotnumber, None, "=", MaintainComments=True
    )

    # TRANSP fixed namelist + those parameters changed
    for itag in TRANSPnamelist_dict:
        IOtools.changeValue(
            self.nml_file, itag, TRANSPnamelist_dict[itag], None, "=", MaintainComments=True
        )

    # Add inputdir to namelist
    with open(self.nml_file, "a") as f:
        f.write(f"inputdir='{self.FolderTRANSP}'\n")

    # Change PTR templates
    IOtools.changeValue(
        self.nml_file,
        "pt_template",
        #f'"{os.path.abspath(self.FolderTRANSP)}/ptsolver_namelist.dat"',
        f'"{self.FolderTRANSP}/ptsolver_namelist.dat"',
        None,
        "=",
        CommentChar=None,
    )
    IOtools.changeValue(
        self.nml_file,
        "tglf_template",
        #f'"{os.path.abspath(self.FolderTRANSP)}/tglf_namelist.dat"',
        f'"{self.FolderTRANSP}/tglf_namelist.dat"',
        None,
        "=",
        CommentChar=None,
    )
    IOtools.changeValue(
        self.nml_file,
        "glf23_template",
        #f'"{os.path.abspath(self.FolderTRANSP)}/glf23_namelist.dat"',
        f'"{self.FolderTRANSP}/glf23_namelist.dat"',
        None,
        "=",
        CommentChar=None,
    )

    # Add outtims
    if len(outtims_new) > 0:

        differenceBetween=0.0

        strNBI, strTOR, strECH = "", "", ""
        for time in outtims_new:
            print(f"\t- Adding time (t={time:.3f}s) to output intermediate files (NUBEAM)")
            strNBI += f"{time:.3f},"

            print(f"\t- Adding time (t={time:.3f}s) to output intermediate files (TORIC)")
            strTOR += f"{time - differenceBetween:.3f},"

            print(f"\t- Adding time (t={time:.3f}s) to output intermediate files (TORBEAM)")
            strECH += f"{time - differenceBetween:.3f},"

        strNBI = strNBI[:-1]
        strTOR = strTOR[:-1]
        strECH = strECH[:-1]

        avgtim = 0.05  # Average before

        IOtools.changeValue(self.nml_file, "mthdavg", 2, None, "=", MaintainComments=True)
        IOtools.changeValue(
            self.nml_file, "avgtim", avgtim, None, "=", MaintainComments=True
        )
        IOtools.changeValue(
            self.nml_file, "outtim", strNBI, None, "=", MaintainComments=True
        )
        IOtools.changeValue(
            self.nml_file, "fi_outtim", strTOR, None, "=", MaintainComments=True
        )
        IOtools.changeValue(
            self.nml_file, "fe_outtim", strECH, None, "=", MaintainComments=True
        )

        # To get birth deposition:
        IOtools.changeValue(
            self.nml_file, "nldep0_gather", "T", None, "=", MaintainComments=True
        )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Defaults for some machines
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def default_nml(
    shotnum,
    tok,
    nml_file,
    PlasmaFeatures={
        "ICH": True,
        "ECH": False,
        "NBI": False,
        "ASH": False,
        "Fuel": 2.5,
    },
    pservers=[1, 1, 0],
    code_settings=5,
    useMMX = False,
    isolver = False,
    grTGLF = False  # Disable by default because it takes disk space and time... enable for 2nd preditive outside of this routine
    ):
    
    Pich, Pech, Pnbi, AddHe4ifDT, Fuel = (
        PlasmaFeatures["ICH"],
        PlasmaFeatures["ECH"],
        PlasmaFeatures["NBI"],
        PlasmaFeatures["ASH"],
        PlasmaFeatures["Fuel"],
    )

    transp_params = {"Ufiles": ["lim", "qpr", "cur", "ter", "ti2", "ner", "zf2", "gfd"]}

    if useMMX:  transp_params["Ufiles"].append("mmx")
    else:       transp_params["Ufiles"].append("mry")

    # ----------------------------------------------------------------------------------------------------------
    # Differences between tokamaks (different than defaults in NMLtools)
    # ----------------------------------------------------------------------------------------------------------

    if tok == "SPARC" or tok == "ARC":
        transp_params["Ufiles"].append("df4")
        transp_params["Ufiles"].append("vc4")

        transp_params["dtHeating_ms"] = 10.0 
        transp_params["nteq_mode"] = 2

    if tok == "AUG":
        transp_params["dtHeating_ms"] = 10.0
        transp_params["nteq_mode"] = 2
        transp_params['dtOut_ms'] = 0.1
        transp_params['UFrotation'] = True

    if tok == "CMOD":
        transp_params["taupD"] = 30e-3
        transp_params["taupZ"] = 20e-3
        transp_params["taupmin"] = 1e6
        transp_params['UFrotation'] = True
        transp_params['coeffsSaw'] = [1.0,2.0,1.0,0.4] # If not, 15 condition triggered too much

    # ----------------------------------------------------------------------------------------------------------
    # Namelist
    # ----------------------------------------------------------------------------------------------------------

    nml = NMLtools.transp_nml(shotnum=shotnum,inputdir=None,pservers=pservers)

    nml.define_machine(tok)

    nml.populate(
        Pich=Pich,
        Pech=Pech,
        Pnbi=Pnbi,
        DTplasma=Fuel == 2,
        AddHe4ifDT=AddHe4ifDT,
        isolver=isolver,
        PTsolver=True,
        code_settings=code_settings,
        grTGLF=grTGLF,
        **transp_params
    )

    nml.write(file=nml_file)
