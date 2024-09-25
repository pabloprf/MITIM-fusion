import os
import copy
from mitim_tools.transp_tools import CDFtools
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat
from IPython import embed

class transp_beat(beat):

    def __init__(self, maestro_instance):            
        super().__init__(maestro_instance, beat_name = 'transp')

    def prepare(
        self,
        letter = None,
        shot = None, 
        flattop_window      = 0.20,  # To allow for steady-state in heating and current diffusion
        transition_window   = 0.10,  # To prevent equilibrium crashes
        freq_ICH            = None,  # Frequency of ICRF heating (if None, find optimal)
        extractAC           = False,  # To extract AC quantities
        **transp_namelist
        ):
        '''
        Using some smart defaults to avoid repeating TRANSP runid
            shot will be 5 digits that depend on the last subfolder
                e.g. run_cmod1 -> '94351', run_cmod2 -> '94352', run_d3d1 -> '72821', etc
            letter will depend on username in this machine, if it can be found
                e.g. pablorf -> 'P"
        '''

        # Define timings
        currentheating_window = 0.001
        self.time_init = 0.0                                                # Start with a TRANSP machine equilibrium
        self.time_transition = self.time_init+ transition_window            # Transition to new equilibrium (and profiles), also defined at 100.0
        self.time_diffusion = self.time_transition + currentheating_window  # Current diffusion and ICRF on
        self.time_end = self.time_diffusion + flattop_window                # End
        self.timeAC = self.time_end - 0.001 if extractAC else None          # Time to extract TORIC and NUBEAM files

        if shot is None:
            folder_last = os.path.basename(os.path.normpath(self.maestro_instance.folder))
            shot = IOtools.string_to_sequential_number(folder_last, num_digits=5)

        if letter is None:
            username = IOtools.expandPath('$USER')
            letter = username[0].upper()
            if letter == '$':
                letter = 'A'
        # ---------------------------------------------------------

        # Define run parameters
        self.shot = shot
        self.runid = letter + str(self.maestro_instance.counter_current).zfill(2)

        # Write TRANSP from profiles
        times = [self.time_transition,self.time_end+1.0]
        self.transp = self.profiles_current.to_transp(
            folder = self.folder,
            shot = self.shot, runid = self.runid, times = times,
            Vsurf = self.profiles_current.Vsurf)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generatic TRANSP operation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        transp_namelist_mod = copy.deepcopy(transp_namelist)

        if 'timings' in transp_namelist_mod:
            raise ValueError('Cannot define timings in MAESTRO transp_namelist')
        else:
            transp_namelist_mod['timings'] = {
                "time_start": self.time_init,
                "time_current_diffusion": self.time_diffusion,
                "time_end": self.time_end,
                "time_extraction": self.timeAC,
            }

        if 'Ufiles' in transp_namelist_mod:
            raise ValueError('Cannot define UFILES in MAESTRO transp_namelist')
        else:
            transp_namelist_mod['Ufiles'] = ["qpr","cur","vsf","ter","ti2","ner","rbz","lim","zf2", "rfs", "zfs"]

        # Write namelist
        self.transp.write_namelist(**transp_namelist_mod)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Additional operations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self._additional_operations_add_initialization()

        # ICRF on
        PichT_MW    = self.profiles_current.derived['qRF_MWmiller'][-1]
        
        if freq_ICH is None:

            B_T         = self.profiles_current.profiles['bcentr(T)'][0]

            '''
            Best resonance condition for minority ions
            ------------------------------------------
            B = (Fich * 2 * np.pi) / qm 
            Fich_MHz = B * qm / (2 * np.pi) * 1e-6
            qm ~ q/m * 1E8
            Fich_MHz = B * q/m * 1E8  / (2 * np.pi) * 1e-6 ~ B * q/m * 15.0
                e.g. He3 in SPARC: F = 12 * 2/3 * 15 = 120 MHz
            '''

            qm_minority = self.transp.nml_object.Minorities[0]/self.transp.nml_object.Minorities[1]
            factor_to_account_for_Bplasma = 1.0 #1.05
            freq_ICH = B_T * qm_minority * 15.0 * factor_to_account_for_Bplasma

        self.transp.icrf_on_time(self.time_diffusion, power_MW = PichT_MW, freq_MHz = freq_ICH)

        # Write Ufiles
        self.transp.write_ufiles()

    def run(self, **kwargs):

        self.transp.run(
            self.machine_run,
            mpisettings = kwargs.get("mpisettings",{"trmpi": 32, "toricmpi": 32, "ptrmpi": 1}),
            minutesAllocation = 60*kwargs.get("hours_allocation",8),
            case = self.transp.runid,
            tokamak_name = kwargs.get("tokamak_name",None),
            checkMin = kwargs.get("checkMin",3),
            retrieveAC = self.timeAC is not None,
            )

        self.transp.c = CDFtools.transp_output(f"{self.folder}/{self.shot}{self.runid}.CDF")

    def finalize(self, force_auxiliary_heating_at_output = {'Pe': None, 'Pi': None}, **kwargs):

        # Copy to outputs
        os.system(f'cp {self.folder}/{self.shot}{self.runid}TR.DAT {self.folder_output}/.')
        os.system(f'cp {self.folder}/{self.shot}{self.runid}.CDF {self.folder_output}/.')
        os.system(f'cp {self.folder}/{self.shot}{self.runid}tr.log {self.folder_output}/.')

        # Prepare final beat's input.gacode, extracting profiles at time_extraction
        time_extraction = self.transp.c.t[self.transp.c.ind_saw -1] # Since the time is coarse in MAESTRO TRANSP runs, make I'm not extracting with profiles sawtoothing
        self.profiles_output = self.transp.c.to_profiles(time_extraction=time_extraction)

        # Potentially force auxiliary
        self._add_heating_profiles(force_auxiliary_heating_at_output)

        # Write profiles
        self.profiles_output.writeCurrentStatus(file=f"{self.folder_output}/input.gacode")

    def _add_heating_profiles(self, force_auxiliary_heating_at_output = {'Pe': None, 'Pi': None}):
        '''
        force_auxiliary_heating_at_output['Pe'] has the shaping function (takes rho) and the integrated value
        '''

        for key, pkey, ikey in zip(['Pe','Pi'], ['qrfe(MW/m^3)', 'qrfi(MW/m^3)'], ['qRFe_MWmiller', 'qRFi_MWmiller']):

            if force_auxiliary_heating_at_output[key] is not None:
                self.profiles_output.profiles[pkey] = force_auxiliary_heating_at_output[key][0](self.profiles_output.profiles['rho(-)'])
                self.profiles_output.deriveQuantities()
                self.profiles_output.profiles[pkey] = self.profiles_output.profiles[pkey] *  force_auxiliary_heating_at_output[key][1]/self.profiles_output.derived[ikey][-1]

    def merge_parameters(self):
        '''
        The goal of the TRANSP beat is to produce:
            - Internal GS equilibrium
            - q-profile
            - Power deposition profiles of high quality (auxiliary heating, but also dynamic targets)
            - Species and fast ions
        However, TRANSP is not modifying the kinetic profiles and therefore I should use the profiles that were frozen before, to
        avoid "grid leaks", i.e. from beat to beat, the coarse grid interpolates to point to point.
        So, this merge:
            - Brings back the resolution of the frozen profiles
            - Inserts kinetic profiles from frozen
            - Inserts engineering parameters (Ip, Bt) from frozen
            - Scales power deposition profiles to match the frozen power deposition which I treat as an engineering parameter (Pin)
        '''

        # Write the pre-merge input.gacode before modifying it
        profiles_output_pre_merge = copy.deepcopy(self.profiles_output)
        profiles_output_pre_merge.writeCurrentStatus(file=f"{self.folder_output}/input.gacode_pre_merge")

        # First, bring back to the resolution of the frozen
        p_frozen = self.maestro_instance.profiles_with_engineering_parameters
        self.profiles_output.changeResolution(rho_new = p_frozen.profiles['rho(-)'])

        # --------------------------------------------------------------------------------------------
        # Insert relevant quantities
        # --------------------------------------------------------------------------------------------

        # Insert kinetic profiles from frozen
        self.profiles_output.profiles['ne(10^19/m^3)'] = p_frozen.profiles['ne(10^19/m^3)']
        self.profiles_output.profiles['te(keV)'] = p_frozen.profiles['te(keV)']
        self.profiles_output.profiles['ti(keV)'][:,0] = p_frozen.profiles['ti(keV)'][:,0]

        self.profiles_output.makeAllThermalIonsHaveSameTemp()
        profiles_output_pre_merge.changeResolution(rho_new = p_frozen.profiles['rho(-)'])
        self.profiles_output.scaleAllThermalDensities(scaleFactor = self.profiles_output.profiles['ne(10^19/m^3)']/profiles_output_pre_merge.profiles['ne(10^19/m^3)'])

        # Insert engineering parameters (except shape)
        for key in ['current(MA)', 'bcentr(T)']:
            self.profiles_output.profiles[key] = p_frozen.profiles[key]

        # Power scale
        self.profiles_output.profiles['qrfe(MW/m^3)'] *= p_frozen.derived['qRF_MWmiller'][-1] / self.profiles_output.derived['qRF_MWmiller'][-1]
        self.profiles_output.profiles['qrfi(MW/m^3)'] *= p_frozen.derived['qRF_MWmiller'][-1] / self.profiles_output.derived['qRF_MWmiller'][-1]

        # --------------------------------------------------------------------------------------------

        # Write to final input.gacode
        self.profiles_output.deriveQuantities()
        self.profiles_output.writeCurrentStatus(file=f"{self.folder_output}/input.gacode")

    def grab_output(self):

        isitfinished = self.maestro_instance.check(beat_check=self)

        if isitfinished:
            c = CDFtools.transp_output(self.folder_output)
            profiles = PROFILEStools.PROFILES_GACODE(f'{self.folder_output}/input.gacode')
        else:
            # Trying to see if there's an intermediate CDF in folder
            print('\t\t- Searching for intermediate CDF in folder')
            try:
                c = CDFtools.transp_output(self.folder)
            except ValueError:
                c = None
            profiles = None

        return c, profiles

    def plot(self,  fn = None, counter = 0, **kwargs):

        c, _ = self.grab_output()
        
        if c is None:
            return '\t\t- Cannot plot because the TRANSP beat has not finished yet'
        
        c.plot(fn = fn, tab_color = counter) 

        return '\t\t- Plotting of TRANSP beat done'

    def finalize_maestro(self):

        cdf = CDFtools.transp_output(f"{self.folder}/{self.transp.shot}{self.transp.runid}.CDF")
        self.maestro_instance.final_p = cdf.to_profiles()
        
        final_file = f'{self.maestro_instance.folder_output}/input.gacode_final'
        self.maestro_instance.final_p.writeCurrentStatus(file=final_file)
        print(f'\t\t- Final input.gacode saved to {IOtools.clipstr(final_file)}')

    # --------------------------------------------------------------------------------------------
    # Additional TRANSP utilities
    # --------------------------------------------------------------------------------------------

    def _additional_operations_add_initialization(self, machine_initialization = 'CMOD'):
        '''
        ----------------------------------------------------------------------------------------------------------------------
        TRANSP must be initialized with a specific machine, so here I use the trick of changing the equilibrium and parameters
        in time, to make a smooth transition and avoid equilibrium crashes (e.g. quval error)
        ----------------------------------------------------------------------------------------------------------------------
        '''
        self.machine_run = machine_initialization

        if self.machine_run == 'D3D':
            R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = 1.67, 0.6, 1.75, 0.38, 0.0, 0.0, 0.074, 1.6, 2.0, 1.0
        elif self.machine_run == 'CMOD':
            R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = 0.68, 0.22, 1.5, 0.46, 0.0, 0.0, 0.3, 1.0, 5.4, 1.0
        
        self.transp.populate_time.from_freegs(self.time_init, R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = ne0_20)
