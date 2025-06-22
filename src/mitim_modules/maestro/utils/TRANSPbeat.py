import os
import shutil
import copy
from mitim_tools.transp_tools import CDFtools
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_modules.maestro.utils.MAESTRObeat import beat
from IPython import embed

class transp_beat(beat):

    def __init__(
        self,
        maestro_instance,
        letter              = None,
        shot                = None, 
        extract_last_instead_of_sawtooth = False,   # To extract last time instead of sawtooth
        ):   

        super().__init__(maestro_instance, beat_name = 'transp')

        # Decide now the shot and runid and how to extract (need to do this now and not in prepare because of restart options, that do not run prepare)

        if shot is None:
            folder_last = self.maestro_instance.folder.resolve().name
            shot = IOtools.string_to_sequential_number(folder_last, num_digits=5)

        if letter is None:
            username = os.environ['USER']
            letter = username[0].upper()
            if letter == '$':
                letter = 'A'

        self.shot = shot
        self.runid = letter + str(self.maestro_instance.counter_current).zfill(2)

        self.extract_last_instead_of_sawtooth = extract_last_instead_of_sawtooth

    def prepare(
        self,
        flattop_window      = 0.20,                 # To allow for steady-state in heating and current diffusion
        freq_ICH            = None,                 # Frequency of ICRF heating (if None, find optimal)
        extractAC           = False,                # To extract AC quantities
        
        **transp_namelist
        ):
        '''
        - For letter and shot:
            Using some smart defaults to avoid repeating TRANSP runid
                shot will be 5 digits that depend on the last subfolder
                    e.g. run_cmod1 -> '94351', run_cmod2 -> '94352', run_d3d1 -> '72821', etc
                letter will depend on username in this machine, if it can be found
                    e.g. pablorf -> 'P"
        - transp_namelist is a dictionary with the keys that I want to be different from the defaults
            (mitim_tools/transp_tools/NMLtools.py: _default_params())
        '''

        # Define timings
        transition_window     = 0.1     # To prevent equilibrium crashes
        currentheating_window = 0.001
        self.time_init = 0.0                                                # Start with a TRANSP machine equilibrium
        self.time_transition = self.time_init+ transition_window            # Transition to new equilibrium (and profiles), also defined at 100.0
        self.time_diffusion = self.time_transition + currentheating_window  # Current diffusion and ICRF on
        self.time_end = self.time_diffusion + flattop_window                # End
        self.timeAC = self.time_end - 0.001 if extractAC else None          # Time to extract TORIC and NUBEAM files


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
            raise ValueError('[MITIM] You cannot define timings in a MAESTRO transp_namelist!')
        else:
            transp_namelist_mod['timings'] = {
                "time_start": self.time_init,
                "time_current_diffusion": self.time_diffusion,
                "time_end": self.time_end,
                "time_extraction": self.timeAC,
            }

        if 'Ufiles' in transp_namelist_mod:
            raise ValueError('[MITIM] You cannot define UFILES in a MAESTRO transp_namelist')
        else:
            transp_namelist_mod['Ufiles'] = ["qpr","cur","vsf","ter","ti2","ner","rbz","lim","zf2", "rfs", "zfs"]

        # Write namelist
        self.transp.write_namelist(**transp_namelist_mod)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Additional operations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self._additional_operations_add_initialization()

        # ICRF on
        PichT_MW    = self.profiles_current.derived['qRF_MW'][-1]
        
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

        mpi_settings = kwargs.get("mpisettings",{"trmpi": 32, "toricmpi": 32, "ptrmpi": 1})

        print('\t\t- Running TRANSP beat with MPI settings: ',mpi_settings)

        self.transp.run(
            self.machine_run,
            mpisettings = mpi_settings,
            minutesAllocation = 60*kwargs.get("hours_allocation",8),
            case = self.transp.runid,
            tokamak_name = kwargs.get("tokamak_name",None),
            checkMin = kwargs.get("checkMin",3),
            retrieveAC = self.timeAC is not None,
            )

    def finalize(self, force_auxiliary_heating_at_output = {'Pe': None, 'Pi': None}, **kwargs):

        # Copy to outputs
        try:
            shutil.copy2(self.folder / f"{self.shot}{self.runid}TR.DAT", self.folder_output)
            shutil.copy2(self.folder / f"{self.shot}{self.runid}.CDF", self.folder_output)
            shutil.copy2(self.folder / f"{self.shot}{self.runid}tr.log", self.folder_output)
        except FileNotFoundError:
            print('\t\t- No TRANSP files in beat folder, assuming they may exist in the output folder (MAESTRO restart case)', typeMsg='w')
            
            # Find CDF name
            files = [f for f in self.folder.iterdir() if f.is_file()]
            cdf_prefix = next(
                (file.stem                           
                for file in files
                if file.suffix.lower() == ".cdf"    # keep only .cdf files …
                    and not file.name.lower().endswith("ph.cdf")),  # … but skip *.ph.cdf
                None
            )

            shutil.copy2(self.folder / f"{cdf_prefix}TR.DAT", self.folder_output / f"{self.shot}{self.runid}TR.DAT")
            shutil.copy2(self.folder / f"{cdf_prefix}.CDF", self.folder_output / f"{self.shot}{self.runid}.CDF")
            shutil.copy2(self.folder / f"{cdf_prefix}tr.log", self.folder_output / f"{self.shot}{self.runid}tr.log")

        # Extract output
        cdf_results = CDFtools.transp_output(self.folder_output / f"{self.shot}{self.runid}.CDF")

        # Prepare final beat's input.gacode, extracting profiles at time_extraction
        it_extract = cdf_results.ind_saw -1 if not self.extract_last_instead_of_sawtooth else -1 # Since the time is coarse in MAESTRO TRANSP runs, make I'm not extracting with profiles sawtoothing
        time_extraction = cdf_results.t[it_extract] 
        self.profiles_output = cdf_results.to_profiles(time_extraction=time_extraction)

        # Potentially force auxiliary
        self._add_heating_profiles(force_auxiliary_heating_at_output)

        # Write profiles
        self.profiles_output.write_state(file=self.folder_output / "input.gacode")

    def _add_heating_profiles(self, force_auxiliary_heating_at_output = {'Pe': None, 'Pi': None}):
        '''
        force_auxiliary_heating_at_output['Pe'] has the shaping function (takes rho) and the integrated value
        '''

        for key, pkey, ikey in zip(['Pe','Pi'], ['qrfe(MW/m^3)', 'qrfi(MW/m^3)'], ['qRFe_MW', 'qRFi_MW']):

            if force_auxiliary_heating_at_output[key] is not None:
                self.profiles_output.profiles[pkey] = force_auxiliary_heating_at_output[key][0](self.profiles_output.profiles['rho(-)'])
                self.profiles_output.derive_quantities()
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
        profiles_output_pre_merge.write_state(file=self.folder_output / 'input.gacode_pre_merge')

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
        self.profiles_output.profiles['qrfe(MW/m^3)'] *= p_frozen.derived['qRF_MW'][-1] / self.profiles_output.derived['qRF_MW'][-1]
        self.profiles_output.profiles['qrfi(MW/m^3)'] *= p_frozen.derived['qRF_MW'][-1] / self.profiles_output.derived['qRF_MW'][-1]

        # --------------------------------------------------------------------------------------------

        # Write to final input.gacode
        self.profiles_output.derive_quantities()
        self.profiles_output.write_state(file=self.folder_output / 'input.gacode')

    def grab_output(self):

        isitfinished = self.maestro_instance.check(beat_check=self)

        if isitfinished:
            c = CDFtools.transp_output(self.folder_output)
            profiles = PROFILEStools.gacode_state(self.folder_output / 'input.gacode')
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

# -----------------------------------------------------------------------------------------------------------------------
# Defaults to help MAESTRO
# -----------------------------------------------------------------------------------------------------------------------

def transp_beat_default_nml(parameters_engineering, parameters_mix, only_current_diffusion = False):

    duration_s   = 1.0
    time_step_s = duration_s * 1E-2

    transp_namelist = {
        'flattop_window': 2.5,       
        'extractAC': False,      
        'dtOut_ms' : time_step_s*1E3,
        'dtIn_ms' : time_step_s*1E3,
        'nzones' : 60,
        'nzones_energetic' : 20, 
        'nzones_distfun' : 10,     
        'MCparticles' : 1e4,
        'toric_ntheta' : 64,   
        'toric_nrho' : 128, 
        'Pich': parameters_engineering['PichT_MW']>0.0,
        'DTplasma': parameters_mix['DTplasma'],
        'useNUBEAMforAlphas': True,
        'Minorities': parameters_mix['minority'],
        "zlump" :[  [74.0, 184.0, 0.1*parameters_mix['impurity_ratio_WtoZ']],
                    [parameters_mix['lowZ_impurity'], parameters_mix['lowZ_impurity']*2, 0.1] ],
        }

    if only_current_diffusion:

        duration_s   = 20.0
        time_step_s  = 0.1

        transp_namelist['flattop_window'] = duration_s
        transp_namelist['dtEquilMax_ms'] = time_step_s*1E3
        transp_namelist['dtHeating_ms'] = time_step_s*1E3
        transp_namelist['dtCurrentDiffusion_ms'] = time_step_s*1E3
        transp_namelist['dtOut_ms'] = time_step_s*1E3
        transp_namelist['dtIn_ms'] = time_step_s*1E3
        
        transp_namelist['useNUBEAMforAlphas'] = False
        transp_namelist['Pich'] = False

    return transp_namelist

