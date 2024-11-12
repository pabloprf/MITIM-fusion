import copy
import os
import shutil
import torch
import numpy as np
from mitim_tools.misc_tools import PLASMAtools, IOtools
from mitim_tools.gacode_tools import TGYROtools
from mitim_modules.portals.utils import PORTALScgyro
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class power_transport:
    '''
    Default class for power transport models, change "evaluate" method to implement a new model and produce_profiles if the model requires written input.gacode written

    Notes:
        - After evaluation, the self.model_results attribute will contain the results of the model, which can be used for plotting and analysis
        - model results can have .plot() method that can grab kwargs or be similar to TGYRO plot

    '''
    def __init__(self, powerstate, name = "test", folder = "~/scratch/", evaluation_number = 0):

        self.name = name
        self.folder = IOtools.expandPath(folder)
        self.evaluation_number = evaluation_number
        self.powerstate = powerstate

        # Allowed fluxes in powerstate so far
        self.quantities = ['Pe', 'Pi', 'Ce', 'CZ', 'Mt']

        # Each flux has a turbulent and neoclassical component
        self.variables = [f'{i}_tr_turb' for i in self.quantities] + [f'{i}_tr_neo' for i in self.quantities]

        # Each flux component has a standard deviation
        self.variables += [f'{i}_stds' for i in self.variables]

        # There is also turbulent exchange
        self.variables += ['PexchTurb', 'PexchTurb_stds']

        # And total transport flux
        self.variables += [f'{i}_tr' for i in self.quantities]

        # Model results is None by default, but can be assigned in evaluate
        self.model_results = None

        # Assign zeros to transport ones if not evaluated
        for i in self.variables:
            self.powerstate.plasma[i] = self.powerstate.plasma["te"] * 0.0

        # There is also target components
        self.variables += [f'{i}' for i in self.quantities] + [f'{i}_stds' for i in self.quantities]

        # ----------------------------------------------------------------------------------------
        # labels for plotting
        # ----------------------------------------------------------------------------------------

        self.powerstate.labelsFluxes = {
            "te": "$Q_e$ ($MW/m^2$)",
            "ti": "$Q_i$ ($MW/m^2$)",
            "ne": (
                "$Q_{conv}$ ($MW/m^2$)"
                if self.powerstate.TransportOptions["ModelOptions"].get("useConvectiveFluxes", True)
                else "$\\Gamma_e$ ($10^{20}/s/m^2$)"
            ),
            "nZ": (
                "$Q_{conv}$ $\\cdot f_{Z,0}$ ($MW/m^2$)"
                if self.powerstate.TransportOptions["ModelOptions"].get("useConvectiveFluxes", True)
                else "$\\Gamma_Z$ $\\cdot f_{Z,0}$ ($10^{20}/s/m^2$)"
            ),
            "w0": "$M_T$ ($J/m^2$)",
        }

    def _produce_profiles(self,deriveQuantities=True):

        self.applyCorrections = (
            self.powerstate.TransportOptions["ModelOptions"]
            .get("MODELparameters", {})
            .get("applyCorrections", {})
        )

        # Write this updated profiles class (with parameterized profiles and target powers)
        self.file_profs = self.folder / "input.gacode"
        self.powerstate.profiles = self.powerstate.to_gacode(
            write_input_gacode=self.file_profs,
            postprocess_input_gacode=self.applyCorrections,
            rederive_profiles = deriveQuantities,        # Derive quantities so that it's ready for analysis and plotting later
            insert_highres_powers = deriveQuantities,    # Insert powers so that Q, Pfus and all that it's consistent when read later
        )

    def produce_profiles(self):
        pass

    # ----------------------------------------------------------------------------------------------------
    # EVALUATE (custom part)
    # ----------------------------------------------------------------------------------------------------
    def evaluate(self):
        print(">> No transport fluxes to evaluate", typeMsg="w")
        pass

# ----------------------------------------------------------------------------------------------------
# FULL TGYRO
# ----------------------------------------------------------------------------------------------------

class tgyro_model(power_transport):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

    def produce_profiles(self):
        self._produce_profiles()

    def evaluate(self):

        # After producing the profiles, copy for future modifications
        self.file_profs_mod = self.file_profs.parent / f"{self.file_profs.name}_modified"
        shutil.copy2(self.file_profs, self.file_profs_mod)

        # ------------------------------------------------------------------------------------------------------------------------
        # Model Options
        # ------------------------------------------------------------------------------------------------------------------------

        ModelOptions = self.powerstate.TransportOptions["ModelOptions"]

        MODELparameters = ModelOptions.get("MODELparameters",None)
        includeFast = ModelOptions.get("includeFastInQi",False)
        impurityPosition = ModelOptions.get("impurityPosition", 1)
        useConvectiveFluxes = ModelOptions.get("useConvectiveFluxes", True)
        UseFineGridTargets = ModelOptions.get("UseFineGridTargets", False)
        launchMODELviaSlurm = ModelOptions.get("launchMODELviaSlurm", False)
        cold_start = ModelOptions.get("cold_start", False)
        provideTurbulentExchange = ModelOptions.get("TurbulentExchange", False)
        profiles_postprocessing_fun = ModelOptions.get("profiles_postprocessing_fun", None)
        OriginalFimp = ModelOptions.get("OriginalFimp", 1.0)
        forceZeroParticleFlux = ModelOptions.get("forceZeroParticleFlux", False)
        percentError = ModelOptions.get("percentError", [5, 1, 0.5])
        use_tglf_scan_trick = ModelOptions.get("use_tglf_scan_trick", None)

        # ------------------------------------------------------------------------------------------------------------------------
        # 1. tglf_neo_original: Run TGYRO workflow - TGLF + NEO in subfolder tglf_neo_original (original as in... without stds or merging)
        # ------------------------------------------------------------------------------------------------------------------------

        RadiisToRun = [
            self.powerstate.plasma["rho"][0, 1:][i].item()
            for i in range(len(self.powerstate.plasma["rho"][0, 1:]))
        ]

        tgyro = TGYROtools.TGYRO(cdf=dummyCDF(self.folder, self.folder))
        tgyro.prep(self.folder, profilesclass_custom=self.powerstate.profiles)

        if launchMODELviaSlurm:
            print("\t- Launching TGYRO evaluation as a batch job")
        else:
            print("\t- Launching TGYRO evaluation as a terminal job")

        tgyro.run(
            subFolderTGYRO="tglf_neo_original",
            cold_start=cold_start,
            forceIfcold_start=True,
            special_radii=RadiisToRun,
            iterations=0,
            PredictionSet=[
                int("te" in self.powerstate.ProfilesPredicted),
                int("ti" in self.powerstate.ProfilesPredicted),
                int("ne" in self.powerstate.ProfilesPredicted),
            ],
            TGLFsettings=MODELparameters["transport_model"]["TGLFsettings"],
            extraOptionsTGLF=MODELparameters["transport_model"]["extraOptionsTGLF"],
            TGYRO_physics_options=MODELparameters["Physics_options"],
            launchSlurm=launchMODELviaSlurm,
            minutesJob=5,
            forcedName=self.name,
        )

        tgyro.read(label="tglf_neo_original")

        # Copy one with evaluated targets
        self.file_profs_targets = tgyro.FolderTGYRO / "input.gacode.new"

        # ------------------------------------------------------------------------------------------------------------------------
        # 2. tglf_neo: Write TGLF, NEO and TARGET errors in tgyro files as well
        # ------------------------------------------------------------------------------------------------------------------------

        # Copy original TGYRO folder
        if (self.folder / "tglf_neo").exists():
            shutil.rmtree(self.folder / "tglf_neo")
        shutil.copytree(self.folder / "tglf_neo_original", self.folder / "tglf_neo")

        # Add errors and merge fluxes as we would do if this was a CGYRO run
        curateTGYROfiles(
            tgyro,
            "tglf_neo_original",
            RadiisToRun,
            self.powerstate.ProfilesPredicted,
            self.folder / "tglf_neo",
            percentError,
            impurityPosition=impurityPosition,
            includeFast=includeFast,
            provideTurbulentExchange=provideTurbulentExchange,
            use_tglf_scan_trick = use_tglf_scan_trick,
            cold_start=cold_start,
            extra_name = self.name,
        )

        # Read again to capture errors
        tgyro.read(label="tglf_neo", folder=self.folder / "tglf_neo")

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # Run TGLF standalone --> In preparation for the transition
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # from mitim_tools.gacode_tools import TGLFtools
        # tglf = TGLFtools.TGLF(rhos=RadiisToRun)
        # _ = tglf.prep(
        #     self.folder / 'stds',
        #     inputgacode=self.file_profs,
        #     recalculatePTOT=False, # Use what's in the input.gacode, which is what PORTALS TGYRO does
        #     cold_start=cold_start)

        # tglf.run(
        #     subFolderTGLF="tglf_neo_original",
        #     TGLFsettings=MODELparameters["transport_model"]["TGLFsettings"],
        #     cold_start=cold_start,
        #     forceIfcold_start=True,
        #     extraOptions=MODELparameters["transport_model"]["extraOptionsTGLF"],
        #     launchSlurm=launchMODELviaSlurm,
        #     slurm_setup={"cores": 4, "minutes": 1},
        # )

        # tglf.read(label="tglf_neo_original")

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # ------------------------------------------------------------------------------------------------------------------------
        # 3. tglf_neo: Populate powerstate with the TGYRO results
        # ------------------------------------------------------------------------------------------------------------------------

        # Produce right quantities (TGYRO -> powerstate.plasma)
        self.powerstate = tgyro.results["tglf_neo"].TGYROmodeledVariables(
            self.powerstate,
            useConvectiveFluxes=useConvectiveFluxes,
            includeFast=includeFast,
            impurityPosition=impurityPosition,
            UseFineGridTargets=UseFineGridTargets,
            OriginalFimp=OriginalFimp,
            forceZeroParticleFlux=forceZeroParticleFlux,
            provideTurbulentExchange=provideTurbulentExchange,
            provideTargets=self.powerstate.TargetOptions['ModelOptions']['TargetCalc'] == "tgyro",
        )

        # ------------------------------------------------------------------------------------------------------------------------
        # 4. cgyro_neo: Trick to fake a tgyro output to reflect CGYRO
        # ------------------------------------------------------------------------------------------------------------------------

        if MODELparameters['transport_model']['turbulence'] == 'CGYRO':

            print(
                "\t- Checking whether cgyro_neo folder exists and it was written correctly via cgyro_trick..."
            )

            correctly_run = (self.folder / "cgyro_neo").exists()
            if correctly_run:
                print("\t\t- Folder exists, but was cgyro_trick run?")
                with open(self.folder / "cgyro_neo" / "mitim_flag", "r") as f:
                    correctly_run = bool(float(f.readline()))

            if correctly_run:
                print("\t\t\t* Yes, it was", typeMsg="w")
            else:
                print("\t\t\t* No, it was not, repating process", typeMsg="i")

                # Copy tglf_neo results
                shutil.copytree(self.folder / "tglf_neo", self.folder / "cgyro_neo")

                # CGYRO writter
                cgyro_trick(
                    self,
                    self.folder / "cgyro_neo",
                    profiles_postprocessing_fun=profiles_postprocessing_fun,
                    name=self.name,
                )

            # Read TGYRO files and construct portals variables

            tgyro.read(label="cgyro_neo", folder=self.folder / "cgyro_neo") 

            powerstate_orig = copy.deepcopy(self.powerstate)

            self.powerstate = tgyro.results["cgyro_neo"].TGYROmodeledVariables(
                self.powerstate,
                useConvectiveFluxes=useConvectiveFluxes,
                includeFast=includeFast,
                impurityPosition=impurityPosition,
                UseFineGridTargets=UseFineGridTargets,
                OriginalFimp=OriginalFimp,
                forceZeroParticleFlux=forceZeroParticleFlux,
                provideTurbulentExchange=provideTurbulentExchange,
                provideTargets=self.powerstate.TargetOptions['ModelOptions']['TargetCalc'] == "tgyro",
            )

            print("\t- Checking model modifications:")
            for r in ["Pe_tr_turb", "Pi_tr_turb", "Ce_tr_turb", "CZ_tr_turb", "Mt_tr_turb"]: #, "PexchTurb"]: #TODO: FIX
                print(
                    f"\t\t{r}(tglf)  = {'  '.join([f'{k:.1e} (+-{ke:.1e})' for k,ke in zip(powerstate_orig.plasma[r][0][1:],powerstate_orig.plasma[r+'_stds'][0][1:]) ])}"
                )
                print(
                    f"\t\t{r}(cgyro) = {'  '.join([f'{k:.1e} (+-{ke:.1e})' for k,ke in zip(self.powerstate.plasma[r][0][1:],self.powerstate.plasma[r+'_stds'][0][1:]) ])}"
                )

            # **
            tgyro.results["use"] = tgyro.results["cgyro_neo"]

        else:
            # copy profiles too!
            profilesToShare(self)

            # **
            tgyro.results["use"] = tgyro.results["tglf_neo"]

        # ------------------------------------------------------------------------------------------------------------------------
        # Results class that can be used for further plotting and analysis in PORTALS
        # ------------------------------------------------------------------------------------------------------------------------

        self.model_results = copy.deepcopy(tgyro.results["use"]) # Pass the TGYRO results class that should be use for plotting and analysis

        self.model_results.extra_analysis = {}
        for ikey in tgyro.results:
            if ikey != "use":
                self.model_results.extra_analysis[ikey] = tgyro.results[ikey]

def tglf_scan_trick(
    fluxesTGYRO, 
    tgyro, 
    label, 
    RadiisToRun, 
    profiles, 
    impurityPosition=1, includeFast=False,  
    delta=0.02, 
    cold_start=False, 
    check_coincidence_thr=1E-2, 
    extra_name="", 
    remove_folders_out = False
    ):

    print(f"\t- Running TGLF standalone scans ({delta = }) to determine relative errors")

    # Grab fluxes from TGYRO
    Qe_tgyro, Qi_tgyro, Ge_tgyro, GZ_tgyro, Mt_tgyro, Pexch_tgyro = fluxesTGYRO

    # ------------------------------------------------------------------------------------------------------------------------
    # TGLF scans
    # ------------------------------------------------------------------------------------------------------------------------

    # Prepare scan 

    tglf = tgyro.grab_tglf_objects(fromlabel=label, subfolder = 'tglf_explorations')

    variables_to_scan = []
    for i in profiles:
        if i == 'te': variables_to_scan.append('RLTS_1')
        if i == 'ti': variables_to_scan.append('RLTS_2')
        if i == 'ne': variables_to_scan.append('RLNS_1')
        if i == 'nZ': variables_to_scan.append(f'RLNS_{impurityPosition+1}')
        if i == 'w0': 
            raise ValueError("[mitim] Mt not implemented yet in TGLF scans")

    #TODO: Only if that parameter is changing at that location
    if 'te' in profiles or 'ti' in profiles:
        variables_to_scan.append('TAUS_2')
    if 'te' in profiles or 'ne' in profiles:
        variables_to_scan.append('XNUE')
    if 'te' in profiles or 'ne' in profiles:
        variables_to_scan.append('BETAE')
    
    relative_scan = [1-delta, 1+delta]

    name = 'turb_drives'

    tglf.rhos = RadiisToRun # To avoid the case in which TGYRO was run with an extra rho point

    tglf.runScanTurbulenceDrives(	
                    subFolderTGLF = name,
                    variablesDrives = variables_to_scan,
                    varUpDown     = relative_scan,
                    TGLFsettings = None,
                    ApplyCorrections = False,
                    add_baseline_to = 'first',
                    cold_start=cold_start,
                    forceIfcold_start=True,
                    slurm_setup={"cores": 1}, # 1 core per radius, since this is going to launch ~ Nr=5 x (Nv=3 x Nd=2 + 1) = 35 TGLFs at once
                    extra_name = f'{extra_name}_{name}',
                    )

    # Remove folders because they are heavy to carry many throughout
    if remove_folders_out:
        shutil.rmtree(tglf.FolderGACODE)

    Qe = np.zeros((len(RadiisToRun), len(variables_to_scan)*len(relative_scan)+1 ))
    Qi = np.zeros((len(RadiisToRun), len(variables_to_scan)*len(relative_scan)+1 ))
    Ge = np.zeros((len(RadiisToRun), len(variables_to_scan)*len(relative_scan)+1 ))
    GZ = np.zeros((len(RadiisToRun), len(variables_to_scan)*len(relative_scan)+1 ))

    cont = 0
    for vari in variables_to_scan:
        jump = tglf.scans[f'{name}_{vari}']['Qe'].shape[-1]

        Qe[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Qe']
        Qi[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Qi']
        Ge[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Ge']
        GZ[:,cont:cont+jump] = tglf.scans[f'{name}_{vari}']['Gi']
        cont += jump

    # ----------------------------------------------------
    # Do a check that TGLF scans are consistent with TGYRO
    Qe_err = np.abs( (Qe[:,0] - Qe_tgyro) / Qe_tgyro )
    Qi_err = np.abs( (Qi[:,0] - Qi_tgyro) / Qi_tgyro )
    Ge_err = np.abs( (Ge[:,0] - Ge_tgyro) / Ge_tgyro )
    GZ_err = np.abs( (GZ[:,0] - GZ_tgyro) / GZ_tgyro )

    F_err = np.concatenate((Qe_err, Qi_err, Ge_err, GZ_err))
    if F_err.max() > check_coincidence_thr:
        print(f"\t- WARNING: TGLF scans are not consistent with TGYRO, maximum error = {F_err.max()*100:.2f}%",typeMsg="w")
    else:
        print(f"\t- TGLF scans are consistent with TGYRO, maximum error = {F_err.max()*100:.2f}%")
    # ----------------------------------------------------

    # Calculate the standard deviation of the scans, that's going to be the reported stds

    def calculate_mean_std(Q):
        # Assumes Q is [radii, points], with [radii, 0] being the baseline

        Qm = Q[:,0]
        Qstd = np.std(Q, axis=1)

        # Qstd    = ( Q.max(axis=1)-Q.min(axis=1) )/2 /2  # Such that the range is 2*std
        # Qm      = Q.min(axis=1) + Qstd*2                # Mean is at the middle of the range

        return  Qm, Qstd

    Qe_point, Qe_std = calculate_mean_std(Qe)
    Qi_point, Qi_std = calculate_mean_std(Qi)
    Ge_point, Ge_std = calculate_mean_std(Ge)
    GZ_point, GZ_std = calculate_mean_std(GZ)

    #TODO: Implement Mt and Pexch
    Mt_point, Pexch_point = Mt_tgyro, Pexch_tgyro
    Mt_std, Pexch_std = abs(Mt_point) * 0.1, abs(Pexch_point) * 0.1

    #TODO: Careful with fast particles

    return Qe_point, Qi_point, Ge_point, GZ_point, Mt_point, Pexch_point, Qe_std, Qi_std, Ge_std, GZ_std, Mt_std, Pexch_std


# ------------------------------------------------------------------
# SIMPLE Diffusion
# ------------------------------------------------------------------

class diffusion_model(power_transport):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

        # Ensure that the provided diffusivities include the zero location
        self.chi_e = self.powerstate.TransportOptions["ModelOptions"]["chi_e"]
        self.chi_i = self.powerstate.TransportOptions["ModelOptions"]["chi_i"]

        if self.chi_e.shape[0] < self.powerstate.plasma['rho'].shape[-1]:
            self.chi_e = torch.cat((torch.zeros(1), self.chi_e))

        if self.chi_i.shape[0] < self.powerstate.plasma['rho'].shape[-1]:
            self.chi_i = torch.cat((torch.zeros(1), self.chi_i))

    def produce_profiles(self):
        pass

    def evaluate(self):

        # Make sure the chis are applied to all the points in the batch
        Pe_tr = PLASMAtools.conduction(
            self.powerstate.plasma["ne"],
            self.powerstate.plasma["te"],
            self.chi_e.repeat(self.powerstate.plasma['rho'].shape[0],1),
            self.powerstate.plasma["aLte"],
            self.powerstate.plasma["a"].unsqueeze(-1),
        )
        Pi_tr = PLASMAtools.conduction(
            self.powerstate.plasma["ni"].sum(axis=-1),
            self.powerstate.plasma["ti"],
            self.chi_i.repeat(self.powerstate.plasma['rho'].shape[0],1),
            self.powerstate.plasma["aLti"],
            self.powerstate.plasma["a"].unsqueeze(-1),
        )

        self.powerstate.plasma["Pe_tr_turb"] = Pe_tr * 2 / 3
        self.powerstate.plasma["Pi_tr_turb"] = Pi_tr * 2 / 3

        self.powerstate.plasma["Pe_tr_neo"] = Pe_tr * 1 / 3
        self.powerstate.plasma["Pi_tr_neo"] = Pi_tr * 1 / 3

        self.powerstate.plasma["Pe_tr"] = self.powerstate.plasma["Pe_tr_turb"] + self.powerstate.plasma["Pe_tr_neo"]
        self.powerstate.plasma["Pi_tr"] = self.powerstate.plasma["Pi_tr_turb"] + self.powerstate.plasma["Pi_tr_neo"]

# ------------------------------------------------------------------
# SURROGATE
# ------------------------------------------------------------------

class surrogate_model(power_transport):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

    def produce_profiles(self):
        pass

    def evaluate(self):

        """
        flux_fun as given in ModelOptions must produce Q and Qtargets in order of te,ti,ne
        """

        X = torch.cat((self.powerstate.plasma['aLte'][:,1:],self.powerstate.plasma['aLti'][:,1:],self.powerstate.plasma['aLne'][:,1:]),axis=1)

        _, Q, _, _ = self.powerstate.TransportOptions["ModelOptions"]["flux_fun"](X) #self.Xcurrent[0])

        numeach = self.powerstate.plasma["rho"].shape[1] - 1

        quantities = {
            "te": "Pe",
            "ti": "Pi",
            "ne": "Ce",
            "nZ": "CZ",
            "w0": "Mt",
        }

        for c, i in enumerate(self.powerstate.ProfilesPredicted):
            self.powerstate.plasma[f"{quantities[i]}_tr"] = torch.cat((torch.tensor([[0.0]]),Q[:, numeach * c : numeach * (c + 1)]),dim=1)

# **************************************************************************************************
# Functions
# **************************************************************************************************

def curateTGYROfiles(
    tgyroObject,
    label,
    RadiisToRun,
    ProfilesPredicted,
    folder,
    percentError,
    provideTurbulentExchange=False,
    impurityPosition=1,
    includeFast=False,
    use_tglf_scan_trick=None,
    cold_start=False,
    extra_name="",
    ):

    tgyro = tgyroObject.results[label]
    
    # Determine NEO and Target errors
    relativeErrorNEO = percentError[1] / 100.0
    relativeErrorTAR = percentError[2] / 100.0

    # **************************************************************************************************************************
    # TGLF
    # **************************************************************************************************************************
    
    # Grab fluxes
    Qe = tgyro.Qe_sim_turb[0, 1:]
    Qi = tgyro.QiIons_sim_turb[0, 1:] if includeFast else tgyro.QiIons_sim_turb_thr[0, 1:]
    Ge = tgyro.Ge_sim_turb[0, 1:]
    GZ = tgyro.Gi_sim_turb[impurityPosition - 1, 0, 1:]
    Mt = tgyro.Mt_sim_turb[0, 1:]
    Pexch = tgyro.EXe_sim_turb[0, 1:]
    
    # Determine TGLF standard deviations
    if use_tglf_scan_trick is not None:

        if provideTurbulentExchange:
            raise ValueError("[mitim] Turbulent exchange not implemented yet in TGLF scans")

        # --------------------------------------------------------------
        # If using the scan trick
        # --------------------------------------------------------------

        Qe, Qi, Ge, GZ, Mt, Pexch, QeE, QiE, GeE, GZE, MtE, PexchE = tglf_scan_trick(
            [Qe, Qi, Ge, GZ, Mt, Pexch],
            tgyroObject,
            label, 
            RadiisToRun, 
            ProfilesPredicted, 
            impurityPosition=impurityPosition, 
            includeFast=includeFast, 
            delta = use_tglf_scan_trick,
            cold_start=cold_start,
            extra_name=extra_name
            )

        min_relative_error = 0.01 # To avoid problems with gpytorch, 1% error minimum

        QeE = QeE.clip(abs(Qe)*min_relative_error)
        QiE = QiE.clip(abs(Qi)*min_relative_error)
        GeE = GeE.clip(abs(Ge)*min_relative_error)
        GZE = GZE.clip(abs(GZ)*min_relative_error)
        MtE = MtE.clip(abs(Mt)*min_relative_error)
        PexchE = PexchE.clip(abs(Pexch)*min_relative_error)

    else:

        # --------------------------------------------------------------
        # If simply a percentage error provided
        # --------------------------------------------------------------

        relativeErrorTGLF = [percentError[0] / 100.0]*len(RadiisToRun)
    
        QeE = abs(Qe) * relativeErrorTGLF
        QiE = abs(Qi) * relativeErrorTGLF
        GeE = abs(Ge) * relativeErrorTGLF
        GZE = abs(GZ) * relativeErrorTGLF
        MtE = abs(Mt) * relativeErrorTGLF
        PexchE = abs(Pexch) * relativeErrorTGLF

    # **************************************************************************************************************************
    # Neo
    # **************************************************************************************************************************

    QeNeo = tgyro.Qe_sim_neo[0, 1:]
    if includeFast:
        QiNeo = tgyro.QiIons_sim_neo[0, 1:]
    else:
        QiNeo = tgyro.QiIons_sim_neo_thr[0, 1:]
    GeNeo = tgyro.Ge_sim_neo[0, 1:]
    GZNeo = tgyro.Gi_sim_neo[impurityPosition - 1, 0, 1:]
    MtNeo = tgyro.Mt_sim_neo[0, 1:]

    QeNeoE = abs(tgyro.Qe_sim_neo[0, 1:]) * relativeErrorNEO
    if includeFast:
        QiNeoE = abs(tgyro.QiIons_sim_neo[0, 1:]) * relativeErrorNEO
    else:
        QiNeoE = abs(tgyro.QiIons_sim_neo_thr[0, 1:]) * relativeErrorNEO
    GeNeoE = abs(tgyro.Ge_sim_neo[0, 1:]) * relativeErrorNEO
    GZNeoE = abs(tgyro.Gi_sim_neo[impurityPosition - 1, 0, 1:]) * relativeErrorNEO
    MtNeoE = abs(tgyro.Mt_sim_neo[0, 1:]) * relativeErrorNEO

    # Merge

    PORTALScgyro.modifyFLUX(
        tgyro,
        folder,
        Qe,
        Qi,
        Ge,
        GZ,
        Mt,
        Pexch,
        QeNeo=QeNeo,
        QiNeo=QiNeo,
        GeNeo=GeNeo,
        GZNeo=GZNeo,
        MtNeo=MtNeo,
        impurityPosition=impurityPosition,
    )

    PORTALScgyro.modifyFLUX(
        tgyro,
        folder,
        QeE,
        QiE,
        GeE,
        GZE,
        MtE,
        PexchE,
        QeNeo=QeNeoE,
        QiNeo=QiNeoE,
        GeNeo=GeNeoE,
        GZNeo=GZNeoE,
        MtNeo=MtNeoE,
        impurityPosition=impurityPosition,
        special_label="_stds",
    )

    # **************************************************************************************************************************
    # Targets
    # **************************************************************************************************************************

    QeTargetE = abs(tgyro.Qe_tar[0, 1:]) * relativeErrorTAR
    QiTargetE = abs(tgyro.Qi_tar[0, 1:]) * relativeErrorTAR
    GeTargetE = abs(tgyro.Ge_tar[0, 1:]) * relativeErrorTAR
    GZTargetE = GeTargetE * 0.0
    MtTargetE = abs(tgyro.Mt_tar[0, 1:]) * relativeErrorTAR

    PORTALScgyro.modifyEVO(
        tgyro,
        folder,
        QeTargetE * 0.0,
        QiTargetE * 0.0,
        GeTargetE * 0.0,
        GZTargetE * 0.0,
        MtTargetE * 0.0,
        impurityPosition=impurityPosition,
        positionMod=1,
        special_label="_stds",
    )
    PORTALScgyro.modifyEVO(
        tgyro,
        folder,
        QeTargetE,
        QiTargetE,
        GeTargetE,
        GZTargetE,
        MtTargetE,
        impurityPosition=impurityPosition,
        positionMod=2,
        special_label="_stds",
    )


def profilesToShare(self):
    if "extra_params" in self.powerstate.TransportOptions["ModelOptions"] and "folder" in self.powerstate.TransportOptions["ModelOptions"]["extra_params"]:
        whereFolder = IOtools.expandPath(
            self.powerstate.TransportOptions["ModelOptions"]["extra_params"]["folder"] / "Outputs" / "portals_profiles"
        )
        if not whereFolder.exists():
            IOtools.askNewFolder(whereFolder)

        fil = whereFolder / f"input.gacode.{self.evaluation_number}"
        shutil.copy2(self.file_profs_mod, fil)
        shutil.copy2(self.file_profs, fil.parent / f"{fil.name}_unmodified")
        shutil.copy2(self.file_profs_targets, fil.parent / f"{fil.name}_unmodified.new")
        print(f"\t- Copied profiles to {IOtools.clipstr(fil)}")
    else:
        print("\t- Could not move files", typeMsg="w")


def cgyro_trick(
    self,
    FolderEvaluation_TGYRO,
    profiles_postprocessing_fun=None,
    name="",
):

    with open(FolderEvaluation_TGYRO / "mitim_flag", "w") as f:
        f.write("0")

    # **************************************************************************************************************************
    # Print Information
    # **************************************************************************************************************************

    txt = "\nFluxes to be matched by CGYRO ( TARGETS - NEO ):"

    for var, varn in zip(
        ["r/a  ", "rho  ", "a/LTe", "a/LTi", "a/Lne", "a/LnZ", "a/Lw0"],
        ["roa", "rho", "aLte", "aLti", "aLne", "aLnZ", "aLw0"],
    ):
        txt += f"\n{var}        = "
        for j in range(self.powerstate.plasma["rho"].shape[1] - 1):
            txt += f"{self.powerstate.plasma[varn][0,j+1]:.6f}   "

    for var, varn in zip(
        ["Qe (MW/m^2)", "Qi (MW/m^2)", "Ce (MW/m^2)", "CZ (MW/m^2)", "Mt (J/m^2) "],
        ["Pe", "Pi", "Ce", "CZ", "Mt"],
    ):
        txt += f"\n{var}  = "
        for j in range(self.powerstate.plasma["rho"].shape[1] - 1):
            txt += f"{self.powerstate.plasma[varn][0,j+1]-self.powerstate.plasma[f'{varn}_tr_neo'][0,j+1]:.4e}   "

    print(txt)

    # **************************************************************************************************************************
    # Modification to input.gacode (e.g. lump impurities)
    # **************************************************************************************************************************

    if profiles_postprocessing_fun is not None:
        print(
            f"\t- Modifying input.gacode.modified to run transport calculations based on {profiles_postprocessing_fun}",
            typeMsg="i",
        )
        profiles = profiles_postprocessing_fun(self.file_profs_mod)

    # Copy profiles so that later it is easy to grab all the input.gacodes that were evaluated
    profilesToShare(self)

    # **************************************************************************************************************************
    # Evaluate CGYRO
    # **************************************************************************************************************************

    PORTALScgyro.evaluateCGYRO(
        self.powerstate.TransportOptions["ModelOptions"]["extra_params"]["PORTALSparameters"],
        self.powerstate.TransportOptions["ModelOptions"]["extra_params"]["folder"],
        self.evaluation_number,
        FolderEvaluation_TGYRO,
        self.file_profs,
        self.powerstate.plasma["roa"][0,1:],
    )

    # **************************************************************************************************************************
    # EXTRA
    # **************************************************************************************************************************

    # Make tensors
    for i in ["Pe_tr_turb", "Pi_tr_turb", "Ce_tr_turb", "CZ_tr_turb", "Mt_tr_turb"]:
        try:
            self.powerstate.plasma[i] = torch.from_numpy(self.powerstate.plasma[i]).to(self.powerstate.dfT).unsqueeze(0)
        except:
            pass

    # Write a flag indicating this was performed, to avoid an issue that... the script crashes when it has copied tglf_neo, without cgyro_trick modification
    with open(FolderEvaluation_TGYRO / "mitim_flag", "w") as f:
        f.write("1")

def dummyCDF(GeneralFolder, FolderEvaluation):
    """
    This routine creates path to a dummy CDF file in FolderEvaluation, with the name "simulation_evaluation.CDF"

    GeneralFolder, e.g.    ~/runs_portals/run10/
    FolderEvaluation, e.g. ~/runs_portals/run10000/Execution/Evaluation.0/model_complete/
    """

    # ------- Name construction for scratch folders in parallel ----------------

    GeneralFolder = IOtools.expandPath(GeneralFolder, ensurePathValid=True)

    a, subname = IOtools.reducePathLevel(GeneralFolder, level=1, isItFile=False)

    FolderEvaluation = IOtools.expandPath(FolderEvaluation)

    name = FolderEvaluation.name.split(".")[-1]  # 0   (evaluation #)

    if name == "":
        name = "0"

    cdf = FolderEvaluation / f"{subname}_ev{name}.CDF"

    return cdf
