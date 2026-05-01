import os
from pdb import run
import shutil
import copy
import numpy as np
from typing import OrderedDict
from mitim_tools.transp_tools import CDFtools
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import PLASMAtools
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
        transition_window   = 0.1,                  # Transition (in seconds) to move from guess TRANSP equilibrium to actual. To prevent equilibrium crashes
        currentheating_window = 0.1,                # Increased to hopefully avoid RuntimeError
        time_before_end     = 0.001,
        machine_initialization = 'CMOD',
        machine_initialization_match_target = False,
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
        self.transition_window     = transition_window 
        self.time_init = 0.0                                                # Start with a TRANSP machine equilibrium
        self.time_transition = self.time_init+ self.transition_window       # Transition to new equilibrium (and profiles), also defined at 100.0
        self.time_diffusion = self.time_transition + currentheating_window  # Current diffusion and ICRF on
        self.time_end = self.time_diffusion + flattop_window                # End
        self.timeAC = self.time_end - time_before_end if extractAC else None          # Time to extract TORIC and NUBEAM files

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
        
        if machine_initialization_match_target:
            modify_Ip_to_match_qstar = self.profiles_current.derived['qstar_ITER']
            modify_p_to_match_pB2 = self.profiles_current.derived['pthr_manual'][0]/self.profiles_current.derived["B0"]**2
        else:
            modify_Ip_to_match_qstar = None
            modify_p_to_match_pB2 = None
        
        self._additional_operations_add_initialization(
            machine_initialization = machine_initialization,
            modify_Ip_to_match_qstar=modify_Ip_to_match_qstar,
            modify_p_to_match_pB2=modify_p_to_match_pB2,
            )
        
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
        
        # Check if run went as expected. Sometimes it may come back "just fine" but in reality it
        # didn't run until the specified time
        print('\t\t- Checking TRANSP run completeness')
        cdf_results = CDFtools.transp_output(self.folder / f"{self.shot}{self.runid}.CDF")
        last_time_simulated = cdf_results.t[-1]
        seconds_check = 0.1
        if last_time_simulated < self.time_end - seconds_check:   
            raise RuntimeError(f"[MITIM] TRANSP run did not complete until the expected time_end = {self.time_end:.4f} s. Last time simulated was {last_time_simulated:.4f} s.")

    def finalize(self, force_auxiliary_heating_at_output = None, **kwargs):

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

        # Remove any existing files in the output folder (to avoid multiple CDFs)
        for cdf_file in self.folder_output.glob("*.CDF"):
            if cdf_file.name != f"{self.shot}{self.runid}.CDF":
                os.remove(cdf_file)
        for trlog_file in self.folder_output.glob("*tr.log"):
            if trlog_file.name != f"{self.shot}{self.runid}tr.log":
                os.remove(trlog_file)
        for trdat_file in self.folder_output.glob("*TR.DAT"):
            if trdat_file.name != f"{self.shot}{self.runid}TR.DAT":
                os.remove(trdat_file)

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
        p = PROFILEStools.gacode_state(self.folder_output / "input.gacode")
        print(f' printing p right now: {p}')
        

    def _add_heating_profiles(self, force_auxiliary_heating_at_output = None):
        '''
        force_auxiliary_heating_at_output['Pe'] has the shaping function (takes rho) and the integrated value
        '''
        if force_auxiliary_heating_at_output is None:
            force_auxiliary_heating_at_output = {'Pe': None, 'Pi': None, 'Ge': None}
        
        for key, pkey, ikey in zip(['Pe','Pi', 'Ge'], ['qrfe(MW/m^3)', 'qrfi(MW/m^3)', 'qpar_beam(1/m^3/s)'], ['qRFe_MW', 'qRFi_MW', 'ge_10E20']):
            if force_auxiliary_heating_at_output[key] is not None:
                unit = "MW of power" if key in ('Pe', 'Pi') else "* 1e20 particles/(m^3 s)"
                print(f'************************************************************')
                print(f'\t\t- Adding {key} = {force_auxiliary_heating_at_output[key][1]} {unit}')
                self.profiles_output.profiles[pkey] = force_auxiliary_heating_at_output[key][0](self.profiles_output.profiles['rho(-)'])
                print(f'************************************************************')
                print(f'first step:')
                print(self.profiles_output.profiles[pkey])
                self.profiles_output.derive_quantities()
                print(f'second step:')
                print(self.profiles_output.profiles[pkey])
                self.profiles_output.profiles[pkey] = self.profiles_output.profiles[pkey] *  force_auxiliary_heating_at_output[key][1]/self.profiles_output.derived[ikey][-1]
                print(f'third step:')
                print(self.profiles_output.profiles[pkey])
                print(f'fourth step:')
                print(self.profiles_output.derived[ikey])
                self.profiles_output.derive_quantities()
                print(f'four point five step:')
                print(self.profiles_output.derived[ikey])
                print(f'fifth step:')
                print(self.profiles_output.derived["qe_auxONLY_MW"])
                print(self.profiles_output.derived["qi_auxONLY_MW"])

            else:
                print(f'\t\t- Keeping auxiliary power from TRANSP output')

    
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
        print('\t\t\t* Bringing resolution of frozen plasma state to new plasma state')
        self.profiles_output.changeResolution(rho_new = p_frozen.profiles['rho(-)'])

        # --------------------------------------------------------------------------------------------
        # Insert relevant quantities
        # --------------------------------------------------------------------------------------------

        # Insert kinetic profiles from frozen
        print('\t\t\t* Bringing kinetic profiles of frozen plasma state to new plasma state')
        self.profiles_output.profiles['ne(10^19/m^3)'] = p_frozen.profiles['ne(10^19/m^3)']
        self.profiles_output.profiles['te(keV)'] = p_frozen.profiles['te(keV)']
        self.profiles_output.profiles['ti(keV)'][:,0] = p_frozen.profiles['ti(keV)'][:,0]

        self.profiles_output.makeAllThermalIonsHaveSameTemp()
        profiles_output_pre_merge.changeResolution(rho_new = p_frozen.profiles['rho(-)'])
        self.profiles_output.scaleAllThermalDensities(scaleFactor = self.profiles_output.profiles['ne(10^19/m^3)']/profiles_output_pre_merge.profiles['ne(10^19/m^3)'])

        # Insert engineering parameters (except shape)
        print('\t\t\t* Bringing Bt and Ip of frozen plasma state to new plasma state')
        for key in ['current(MA)', 'bcentr(T)']:
            self.profiles_output.profiles[key] = p_frozen.profiles[key]

        ## ********************CHANGE HERE******************** ##
        # Power scale
        if self.maestro_instance.counter_current == 1:
            print('\t\t\t* NOT Bringing total power of frozen plasma state to new plasma state (NO rescaling the profile)')
        else:
            print('\t\t\t* Bringing total power of frozen plasma state to new plasma state (scaling the profile)')
            self.profiles_output.profiles['qrfe(MW/m^3)'] *= p_frozen.derived['qRF_MW'][-1] / self.profiles_output.derived['qRF_MW'][-1]
            self.profiles_output.profiles['qrfi(MW/m^3)'] *= p_frozen.derived['qRF_MW'][-1] / self.profiles_output.derived['qRF_MW'][-1]
            if self.profiles_output.derived['qRF_MW'][-1] < 0 or abs(self.profiles_output.derived['qRF_MW'][-1]) <= 5e-6:  # to prevent NaNs
                    profiles_output.profiles['qrfi(MW/m^3)'] = 0
                    profiles_output.profiles['qrfe(MW/m^3)'] = 0    
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

    def _additional_operations_add_initialization(self, machine_initialization = 'CMOD', modify_Ip_to_match_qstar = None, modify_p_to_match_pB2=None):
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
        elif self.machine_run == 'NSTX':
            R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = 0.89, 0.61, 2.5, 0.46, 0.0, 0.0, 0.4, 1.0, 0.5, 1.0
            # says it has no psi-bndry

        if modify_Ip_to_match_qstar is not None:
            qstar_now = PLASMAtools.evaluate_qstar(
                Ip_MA,
                R,
                kappa_sep*0.95,
                B_T,
                a/R,
                delta_sep*0.95,
                isInputIp=True,
                ITERcorrection=True,
                includeShaping=True,
            )
            factor_Ip = qstar_now / modify_Ip_to_match_qstar
            
            print(f'\t\t- Modifying Ip of initialization machine from {Ip_MA:.3f} MA to {Ip_MA*factor_Ip:.3f} MA to match target qstar = {modify_Ip_to_match_qstar:.3f} (original qstar was {qstar_now:.3f})')
            Ip_MA = Ip_MA * factor_Ip
            
        if modify_p_to_match_pB2 is not None:
            beta_now = p0_MPa / B_T**2
            factor_p = beta_now / modify_p_to_match_pB2
            
            print(f'\t\t- Modifying p0 of initialization machine from {p0_MPa:.3f} MPa to {p0_MPa/factor_p:.3f} MPa to match target p/B^2 = {modify_p_to_match_pB2:.3f} (original p/B^2 was {beta_now:.3f})')
            p0_MPa = p0_MPa / factor_p

        self.transp.populate_time.from_freegs(self.time_init, R, a, kappa_sep, delta_sep, zeta_sep, z0,  p0_MPa, Ip_MA, B_T, ne0_20 = ne0_20)

    # -----------------------------------------------------------------------------------------------------------------------
    # MAESTRO interface
    # -----------------------------------------------------------------------------------------------------------------------
    def _inform_save(self, *args, **kwargs):
        
        c, _ = self.grab_output()
        
        # Grab the oder of user-specified impuritites in the TRANSP ions list
        
        transp_impurities = c.nZs.keys()
        profiles_species = [i['N'] for i in self.profiles_output.Species]
        
        impurity_order_transp = OrderedDict()
        for z in transp_impurities:
            for i,spec in enumerate(profiles_species):
                if spec == z:
                    impurity_order_transp[spec] = i
                    break
        
        self.maestro_instance.parameters_trans_beat['impurity_order_transp'] = impurity_order_transp

        # If I have run TRANSP, I cannot reuse surrogate data #TODO: Maybe not always true?
        
        self.maestro_instance.parameters_trans_beat['portals_surrogate_data_file'] = None 

# -----------------------------------------------------------------------------------------------------------------------
# Defaults to help MAESTRO
# -----------------------------------------------------------------------------------------------------------------------

def preprocess_prepare_transp(transp_namelist,maestro_namelist, preprocess_prepare_parameters):
    
    print('\t- Preprocessing settings for TRANSP beat')
    
    # Minority
    Zmini = maestro_namelist["plasma"]["heating"]["parameters"]["minority"][0]
    Amini = maestro_namelist["plasma"]["heating"]["parameters"]["minority"][1]
    fmini = maestro_namelist["plasma"]["heating"]["parameters"]["fmini"]

    # Only correct Pich from the maestro namelist if it's not already False    
    if transp_namelist['Pich']:
        transp_namelist['Pich'] =   maestro_namelist["plasma"]['heating']['type'] == 'ICRH' and \
                                    maestro_namelist["plasma"]['heating']['parameters']['P_icrh'] > 0.0
    
    if transp_namelist['Pich']:
        transp_namelist['Minorities'] = [ Zmini, Amini, fmini ]
        transp_namelist['freq_ICH'] = maestro_namelist["plasma"]['heating']['parameters']['freq_ICH']

    # Grab Z and A of high-Z
    import periodictable as pt
    e = pt.elements.symbol(maestro_namelist["plasma"]["species"]["mix"]["highZ"])
    highZ = e.number
    highA = e.mass    
    # ------ 

    LowZ, Wratio = PLASMAtools.estimateLowZ(
        maestro_namelist["plasma"]["species"]["mix"]["fmain"],
        maestro_namelist["plasma"]["species"]["Zeff"],
        Zmini,
        fmini,
        maestro_namelist["plasma"]["species"]["mix"]["CShighZ_estimate"],
        maestro_namelist["plasma"]["species"]["mix"]["fhighZ"] )
    
    lowA = 2*LowZ   # Approximation
    
    transp_namelist["zlump"] =[  [highZ, highA, 0.1*Wratio],
                                 [ LowZ,  lowA, 0.1       ] ]

    transp_namelist['DTplasma'] = maestro_namelist["plasma"]["species"]['fuel'] == ['D', 'T']   #TODO: generalize TRANSP module
    
    return transp_namelist

def preprocess_run_transp(run_namelist, maestro_namelist, cpus, cold_start):
    
    toric = maestro_namelist["maestro"]["transp"]["parameters_prepare"]["Pich"]
    nubeam = maestro_namelist["maestro"]["transp"]["parameters_prepare"]["useNUBEAMforAlphas"]
    cpus_toric = maestro_namelist["maestro"]["transp"]["preprocess_prepare_parameters"]["cpus_toric"]
    cpus_nubeam = maestro_namelist["maestro"]["transp"]["preprocess_prepare_parameters"]["cpus_nubeam"]
    
    if toric:
        toricmpi = cpus_toric if cpus_toric is not None else cpus
    else:
        toricmpi = 1
        
    if nubeam:
        trmpi = cpus_nubeam if cpus_nubeam is not None else cpus
    else:
        trmpi = 1
    
    # Force auxiliary heating at output
    if maestro_namelist["plasma"]["heating"]["type"] == 'gaussian_sources':

        print('\t- Gaussian sources specified, adding to run_namelist of TRANSP beat')
        
        Pe = maestro_namelist["plasma"]["heating"]["parameters"]["Pe"]
        Pi = maestro_namelist["plasma"]["heating"]["parameters"]["Pi"]
        nu_source = maestro_namelist["plasma"]["heating"]["parameters"]["nu_source"]
        particles_source = maestro_namelist["plasma"]["heating"]["parameters"]["particles_source"]    # in units of 1e20 particles/(m^3 s)

        def P_auxiliary(rhotor):
            _, y = PLASMAtools.parabolicProfile(Tbar=1.0,nu=nu_source,rho=rhotor,Tedge=0.0)
            return y

        def G_auxiliary(rhotor):
            _, y = PLASMAtools.parabolicProfile(Tbar=1.0,nu=nu_source,rho=rhotor,Tedge=0.0)
            return y
    
        force_auxiliary_heating_at_output = {
            'Pe': [P_auxiliary, Pe],
            'Pi': [P_auxiliary, Pi],
            'Ge': [G_auxiliary, particles_source]
            }
   
    else:
        force_auxiliary_heating_at_output = {'Pe': None, 'Pi': None, 'Ge': None}

    run_namelist['mpisettings'] = {
        "trmpi": trmpi, 
        "toricmpi": toricmpi,
        "ptrmpi": 1
        }
    run_namelist['force_auxiliary_heating_at_output'] = force_auxiliary_heating_at_output
    
    return run_namelist