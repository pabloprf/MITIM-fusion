from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gacode_tools.utils import (
    NORMtools,
    GACODEinterpret,
    #GACODEdefaults,
    #GACODEplotting,
    GACODErun,
    )
from mitim_tools.misc_tools.LOGtools import printMsg as print

from IPython import embed
from pathlib import Path
import numpy as np
import copy
import scipy.constants as cnt
import sys

# Useful constants
#mi_D = 2.01355
mass_ref = 2.0  # NOTE: Assumed reference mass, see (https://github.com/gafusion/gacode/issues/398)

class NEO:
    '''
    NEO class that manages the run and the results.

    The philosophy of this is that a single 'neo' class will handle the neo simulation and results
    at one time slice but with possibility of several radii at once.

    It can also handle different NEO settings, running them one by one, storing results in folders and then
    grabbing them.

    Scans can also be run. At several radii at once if wanted.

    '''

    #######################################################################
    #
    #           Main
    #
    #######################################################################

    ### --- Default I/O --- ###
    def __init__(
        self,
        rhos=[0.4, 0.6],  # rho locations of interest
        cdf=None,  # Option1: Start from CDF file (TRANSP) - Path to file
        time=100.0,  # 		   Time to extract CDF file
        avTime=0.0,  # 		   Averaging window to extract CDF file
        alreadyRun=None,  # Option2: Do more stuff with a class that has already been created and store
        ):
        # Init
        print(
            "\n-----------------------------------------------------------------------------------------"
        )
        print("\t\t\t NEO class module")
        print(
            "-----------------------------------------------------------------------------------------\n"
        )

        # Reload previous case
        if alreadyRun is not None:
            # For the case in which I have run NEO somewhere else, not using to plot and modify the class
            self.__class__ = alreadyRun.__class__
            self.__dict__ = alreadyRun.__dict__
            print("* Readying previously-run NEO class", typeMsg="i")
        
        # Prepare new case
        else:
            # List of output files
            # Description under: https://gafusion.github.io/doc/neo/outputs.html
            self.ResultsFiles = [
                "out.neo.run",              # Run settings info
                "out.neo.equil",            # eq/geom data
                "out.neo.f",                # 1st-order distribution function
                "out.neo.grid",             # Numerical grid of (species, energy, pitch, poloidal angle, radius)
                "out.neo.phi",              # Poloidal variation of 1st-order ES potential perturbation
                "out.neo.theory",           # Fluxes from analytic theory
                "out.neo.species",          # masses/charges of species
                #"out.neo.theory_nclass",    # Fluxes from NCLASS code, requires SIM_MODEL=1 (not default)
                "out.neo.transport",        # Numerical normlized DKE fluxes
                "out.neo.transport_flux",   # GyroBohm normalized all fluxes
                "out.neo.transport_gv",     # Fluxes from gyroviscosity (add to DKE solution)
                "out.neo.vel",              # 1st-order poloidal variation of the neoclassical toroidal flow
                "out.neo.vel_fourier",      # Fourier components for poloidal variation of neoclassical flow vector
                #"out.neo.transport_exp",    # Fluxes in real units
                #"out.neo.exp_norm",         # Profile normalization parameters
                "out.neo.rotation",         # Polodial asymmetry from strong rotation, requres ROTATION_MODEL=2 (not GA, but is MITIM default)
                #"out.neo_diagnostic_geo",   # ????
                #"out.neo.diagnostic_geo2",  # ????
                #"out.neo.diagnositc_rot",   # ????
                #"out.neo.gxi",              # ????
                #"out.neo.gxi_t",            # ????
                #"out.neo.gxi_x",            # ????
                #"out.neo.localdump",        # Seems to list contents of input.neo
                #"out.neo.prec",             # ????
                #"out.neo.spitzer",          # solutions to Spitzer problem, requires SPITZER_MODEL=1 (not default)
                "out.neo.version",          # Version control
            ]

            # Prepares run metadata
            self.LocationCDF = cdf
            if self.LocationCDF is not None:
                _, self.nameRunid = IOtools.getLocInfo(self.LocationCDF)
            else:
                self.nameRunid = "0"
            self.time, self.avTime = time, avTime
            self.rhos = np.array(rhos)

            # Init attrs for outputs
            (
                self.results,
                self.scans,
            ) = ({}, {})

            # Init normalizations
            self.NormalizationSets = {
                "TRANSP": None,
                "PROFILES": None,
                "TGYRO": None,
                "EXP": None,
                "input_gacode": None,
                "SELECTED": None,
            }

    ### --- Prepares input.neo files --- ###
    def prep(
        self, 
        folder,
        inputgacode = None,
        cdf_open = None,
        cold_start = False, 
        specificInputs = None,
        ):
        print("> Preparation of NEO run")

        # Loads profiles from input.gacode file
        self.profiles = (
            PROFILEStools.PROFILES_GACODE(inputgacode, mi_ref=mass_ref)
            if inputgacode is not None
            else None
        )

        # Create output folders
        self.folder = IOtools.expandPath(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

        # Loads in NEO controls
        if specificInputs is None:
            # Loads settings as dictionary per rho
            self.inputsNEO = self.profiles.to_neo(rhos=self.rhos)
        
        # If user curated their own input dictionary already
        else:
            self.inputsNEO = specificInputs

        # Init filing
        for rho in self.inputsNEO:
            self.inputsNEO[rho] = NEOinput.initialize_in_memory(self.inputsNEO[rho])

        # Writes the input files
        for rho in self.inputsNEO:
            self.inputsNEO[rho].file = self.folder / f'input.neo_{rho:.4f}'
            self.inputsNEO[rho].writeCurrentStatus()

        print("> Setting up normalizations")
        print(
            "\t- Using mass of deuterium to normalize things (not necesarily the first ion)",
            typeMsg="w",
        )
        self.profiles.deriveQuantities(mi_ref=mass_ref)

        # Sets normalization data
        self.tgyro_results = None
        self.NormalizationSets, cdf = NORMtools.normalizations(
            self.profiles,
            LocationCDF=self.LocationCDF,
            time=self.time,
            avTime=self.avTime,
            cdf_open=cdf_open,
            tgyro=self.tgyro_results,
            )

        # Output
        return cdf

    ### --- Runs standalone NEO case for serveral radii --- ###
    def run(
        self,
        subFolderNEO,
        NEOsettings = None,     # ????, Move to .prep? 
        extraOptions = None,    # ????, Move to .prep?
        launchSlurm = True,
        cold_start = False,
        forceIfcold_start=False,
        extra_name="exe",
        #anticipate_problems=True,
        slurm_setup={
            "cores": 4,
            "minutes": 5,
            },  # Cores per TGLF call (so, when running nR radii -> nR*4)
        ):

        # Prepares inputs
        neo_executor, neo_executor_full, folderlast = self._prepare_run_radii(
            subFolderNEO,
            launchSlurm=launchSlurm,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            extra_name=extra_name,
            slurm_setup=slurm_setup,
            )

        # Runs NEO
        self._run(
            neo_executor,
            neo_executor_full=neo_executor_full,
            #TGLFsettings=TGLFsettings,
            #runWaveForms=runWaveForms,
            #forceClosestUnstableWF=forceClosestUnstableWF,
            #ApplyCorrections=ApplyCorrections,
            #Quasineutral=Quasineutral,
            launchSlurm=launchSlurm,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            extra_name=extra_name,
            slurm_setup=slurm_setup,
            )

        self.FolderNEOlast = folderlast

    # Prepares executables for run
    def _prepare_run_radii(
        self,
        subFolderNEO,
        rhos = None,
        neo_executor={},
        neo_executor_full={},
        multipliers={},
        launchSlurm = True,
        cold_start = None,
        extra_name = None,
        slurm_setup = None,
        forceIfcold_start=False,
        ):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare for run
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Init
        if rhos is None:
            rhos = self.rhos
        inputs = copy.deepcopy(self.inputsNEO)
        FolderNEO = self.folder / subFolderNEO

        # Error check???
        ResultsFiles_new = []
        for ii in self.ResultsFiles:
            if "mitim.out" not in ii:
                ResultsFiles_new.append(ii)
        self.ResultsFiles = ResultsFiles_new

        # Do I need to run all radii?
        rhosEvaluate = cold_start_checker(
            rhos,
            self.ResultsFiles,
            FolderNEO,
            cold_start=cold_start,
            )

        if len(rhosEvaluate) == len(rhos):
            # All radii need to be evaluated
            IOtools.askNewFolder(FolderNEO, force=forceIfcold_start)

        # Applies any changes to the profile inputs and write inputs
        (
            latest_inputsFileNEO,
            latest_inputsFileNEODict,
            ) = changeANDwrite_NEO(
                rhos,
                inputs,
                FolderNEO,
                multipliers=multipliers
                )

        # Organizes execution data
        neo_executor_full[subFolderNEO] = {}
        neo_executor[subFolderNEO] = {}
        for irho in self.rhos:
            neo_executor_full[subFolderNEO][irho] = {
                "folder": FolderNEO,
                "dictionary": latest_inputsFileNEODict[irho],
                "inputs": latest_inputsFileNEO[irho],
                #"extraOptions": extraOptions,
                "multipliers": multipliers,
            }
            if irho in rhosEvaluate:
                neo_executor[subFolderNEO][irho] = neo_executor_full[subFolderNEO][
                    irho
                ]

        # Output
        return neo_executor, neo_executor_full, FolderNEO

    # Manages submitting the run
    def _run(
        self, neo_executor, neo_executor_full={}, **kwargs_NEOrun
        ):
        print("\n> Run NEO")

        c = 0
        for subFolderNEO in neo_executor:
            c += len(neo_executor[subFolderNEO])

        if c > 0:
            GACODErun.runNEO(
                self.folder,
                neo_executor,
                filesToRetrieve=self.ResultsFiles,
                minutes=(
                    kwargs_NEOrun["slurm_setup"]["minutes"]
                    if "slurm_setup" in kwargs_NEOrun
                    and "minutes" in kwargs_NEOrun["slurm_setup"]
                    else 5
                ),
                cores_neo=(
                    kwargs_NEOrun["slurm_setup"]["cores"]
                    if "slurm_setup" in kwargs_NEOrun
                    and "cores" in kwargs_NEOrun["slurm_setup"]
                    else 4
                ),
                name=f"neo_{self.nameRunid}{kwargs_NEOrun['extra_name'] if 'extra_name' in kwargs_NEOrun else ''}",
                launchSlurm=(
                    kwargs_NEOrun["launchSlurm"]
                    if "launchSlurm" in kwargs_NEOrun
                    else True
                ),
            )
        else:
            print(
                "\t- NEO not run because all results files found (please ensure consistency!)",
                typeMsg="i",
            )

    ### --- Reads NEO results --- ###
    def read(
        self,
        label="neo1",
        folder=None,  # If None, search in the previously run folder
        suffix=None,  # If None, search with my standard _0.55 suffixes corresponding to rho of this NEO class
        ):
        print("> Reading NEO results")

        # If no specified folder, check the last one
        if folder is None:
            folder = self.FolderNEOlast

        # Reads NEO data
        self.results[label] = readNEOresults(
            folder,
            self.NormalizationSets,
            self.rhos,
            suffix=suffix,
            inputs=self.inputsNEO
            )

        # Store the input.gacode used
        self.results[label]["profiles"] = (
            self.NormalizationSets["input_gacode"]
            if (self.NormalizationSets["input_gacode"] is not None)
            else None
            )

    #######################################################################
    #
    #           Scanning Utilities
    #
    #######################################################################

    # Manager for certain types of scans
    def runAnalysis(
        self,
        subFolderNEO="analysis1",
        label="analysis1",
        analysisType="Z",
        trace=[16.0, 40.0],
        **kwargs_NEOrun,
        ):

        # ------------------------------------------
        # Impurity D and V
        # ------------------------------------------
        if analysisType == "Z":
            #if ("ApplyCorrections" not in kwargs_NEOrun) or (
            #    kwargs_NEOrun["ApplyCorrections"]
            #    ):
            #    print(
            #        "\t- Forcing ApplyCorrections=False because otherwise the species ordering in NEO file might be messed up",
            #        typeMsg="w",
            #    )
            #    kwargs_NEOrun["ApplyCorrections"] = False

            # Amount to vary trace impurity gradient
            varUpDown = np.linspace(0.5, 1.5, 3)

            # Defines trace impurity to add
            fimp, Z, A = 1e-6, trace[0], trace[1]
            print(
                f"*** Running D and V analysis for trace ({fimp:.1e}) specie with Z={trace[0]:.1f}, A={trace[1]:.1f}"
                )

            self.inputsNEO_orig = copy.deepcopy(self.inputsNEO)

            # Adds trace impurity to NEO input files
            for irho in self.inputsNEO:
                position = self.inputsNEO[irho].addTraceSpecie(Z, A, AS=fimp)

            # Runs gradient scan for this specie
            self.variable = f"DLNNDR_{position}"

            self.runScan(
                subFolderNEO,
                varUpDown=varUpDown,
                variable=self.variable,
                **kwargs_NEOrun,
                )
            '''
            # Reads scan data
            self.readScan(label=label, variable=self.variable, positionIon=position)

            # Init
            x = self.scans[label]["xV"]
            yV = self.scans[label]["pflux"]
            self.variable_y = "pflux"

            self.scans[label]["DZ"] = []
            self.scans[label]["VZ"] = []
            self.scans[label]["VoD"] = []
            self.scans[label]["x_grid"] = []
            self.scans[label]["y_grid"] = []

            # Calculates D, V profiles from flux-gradient relationship
            for irho_cont in range(len(self.rhos)):
                irho = np.where(self.scans[label]["x"] == self.rhos[irho_cont])[0][0]

                rho = self.NormalizationSets["SELECTED"]["rho"]
                a_m = self.NormalizationSets["SELECTED"]["rmin"][-1]
                ni_prof = self.NormalizationSets["SELECTED"]["ne_20"] * fimp

                ni_20 = ni_prof[np.argmin(np.abs(rho - self.rhos[irho_cont]))]

                D, V, x_grid, y_grid = PLASMAtools.DVanalysis(
                    x[irho], yV[irho], a_m, ni_20
                )
                self.scans[label]["DZ"].append(D)
                self.scans[label]["VZ"].append(V)

                self.scans[label]["var_x"] = self.variable
                self.scans[label]["var_y"] = self.variable_y
                self.scans[label]["x_grid"].append(x_grid)
                self.scans[label]["y_grid"].append(y_grid)

                self.scans[label]["VoD"].append(V / D)
            '''
            # Back to original (not trace)
            self.inputsNEO = self.inputsNEO_orig
            
        # Error check
        else:
            print('Analysis Type requested not implemented yet!')

    # Runs NEO scanning over a variable
    def runScan(
        self,
        subFolderNEO,  # 'scan1',
        multipliers={},
        variable="DLNNDR_1",
        varUpDown=[0.5, 1.0, 1.5],
        relativeChanges=True,
        **kwargs_NEOrun,
        ):

        # Error check; add baseline if needed
        if (1.0 not in varUpDown) and relativeChanges:
            print(
                "\n* Since variations vector did not include base case, I am adding it",
                typeMsg="i",
            )
            varUpDown_new = []
            added = False
            for i in varUpDown:
                if i > 1.0 and not added:
                    varUpDown_new.append(1.0)
                    added = True
                varUpDown_new.append(i)
        else:
            varUpDown_new = varUpDown

        # Prepares input files for scan
        neo_executor, neo_executor_full, folders, varUpDown_new = self._prepare_scan(
            subFolderNEO,
            multipliers=multipliers,
            variable=variable,
            varUpDown=varUpDown_new,
            relativeChanges=relativeChanges,
            **kwargs_NEOrun,
        )

        # Run them all
        self._run(
            neo_executor,
            tglf_executor_full=neo_executor_full,
            **kwargs_NEOrun,
        )

        # Read results
        for cont_mult, mult in enumerate(varUpDown_new):
            name = f"{variable}_{mult}"
            self.read(
                label=f"{self.subFolderNEO_scan}_{name}", 
                folder=folders[cont_mult], 
                )

    # Prepares exectuables for scans
    def _prepare_scan(
        self,
        subFolderNEO,  # 'scan1',
        multipliers={},
        variable="DLNNDR_1",
        varUpDown=[0.5, 1.0, 1.5],
        relativeChanges=True,
        **kwargs_NEOrun,
    ):
        """
        Multipliers will be modified by adding the scaning variables, but I don't want to modify the original
        multipliers, as they may be passed to the next scan

        Set relativeChanges=False if varUpDown contains the exact values to change, not multipleiers
        """
        multipliers_mod = copy.deepcopy(multipliers)

        self.subFolderNEO_scan = subFolderNEO

        if relativeChanges:
            for i in range(len(varUpDown)):
                varUpDown[i] = round(varUpDown[i], 6)

        print(f"\n- Proceeding to scan {variable}:")
        neo_executor = {}
        neo_executor_full = {}
        folders = []
        for cont_mult, mult in enumerate(varUpDown):
            mult = round(mult, 6)

            if relativeChanges:
                print(
                    f"\n + Multiplier: {mult} -----------------------------------------------------------------------------------------------------------"
                )
            else:
                print(
                    f"\n + Value: {mult} ----------------------------------------------------------------------------------------------------------------"
                )

            multipliers_mod[variable] = mult
            name = f"{variable}_{mult}"

            neo_executor, neo_executor_full, folderlast = self._prepare_run_radii(
                f"{self.subFolderNEO_scan}_{name}",
                neo_executor=neo_executor,
                neo_executor_full=neo_executor_full,
                multipliers=multipliers_mod,
                **kwargs_NEOrun,
            )

            folders.append(copy.deepcopy(folderlast))

        return neo_executor, neo_executor_full, folders, varUpDown

    #######################################################################
    #
    #           Extra
    #
    #######################################################################

    def run_vgen(self, subfolder="vgen1", vgenOptions={}, cold_start=False):

        self.folder_vgen = self.folder / f"{subfolder}"

        # ---- Default options

        vgenOptions.setdefault("er", 2)
        vgenOptions.setdefault("vel", 1)
        vgenOptions.setdefault("numspecies", len(self.inputgacode.Species))
        vgenOptions.setdefault("matched_ion", 1)
        vgenOptions.setdefault("nth", "17,39")

        # ---- Prepare

        runThisCase = check_if_files_exist(
            self.folder_vgen,
            [
                ["vgen", "input.gacode"],
                ["vgen", "input.neo.gen"],
                ["out.vgen.neoequil00"],
                ["out.vgen.neoexpnorm00"],
                ["out.vgen.neontheta00"],
                ["vgen.dat"],
            ],
        )

        if (not runThisCase) and cold_start:
            runThisCase = print(
                "\t- Files found in folder, but cold_start requested. Are you sure?",
                typeMsg="q",
            )

            if runThisCase:
                IOtools.askNewFolder(self.folder_vgen, force=True)

        self.inputgacode.writeCurrentStatus(file=(self.folder_vgen / f"input.gacode"))

        # ---- Run

        if runThisCase:
            file_new = GACODErun.runVGEN(
                self.folder_vgen, vgenOptions=vgenOptions, name_run=subfolder
            )
        else:
            print(
                f"\t- Required files found in {subfolder}, not running VGEN",
                typeMsg="i",
            )
            file_new = self.folder_vgen / f"vgen" / f"input.gacode"

        # ---- Postprocess

        self.inputgacode_vgen = PROFILEStools.PROFILES_GACODE(
            file_new, calculateDerived=True, mi_ref=self.inputgacode.mi_ref
        )

#######################################################################
#
#           Input Utilities
#
#######################################################################

# Function to prepare NEO input dictionary and allow changes over scans
def changeANDwrite_NEO(
    rhos,
    inputs0,
    FolderNEO,
    multipliers={},
    ):
    # Init
    inputs = copy.deepcopy(inputs0)
    modInputNEO = {}

    # Loop over rhos
    for ii, rho in enumerate(rhos):
        # Init
        inputNEO_rho = inputs[rho]

        # Loop over modifier key
        for kk in multipliers:
            tmp = float(inputNEO_rho.plasma[kk])*multipliers[kk]
            inputNEO_rho.plasma[kk] = '%0.5E'%(tmp)

        # Writes new input
        inputNEO_rho.file = FolderNEO / f'input.neo_{rho:.4f}'
        inputNEO_rho.writeCurrentStatus()
        modInputNEO[rho] = inputNEO_rho

    # Convert back to a string because that's how ruTGLFproduction operates
    inputFileNEO = inputToVariable(FolderNEO, rhos)

    # Output
    return inputFileNEO, modInputNEO

def check_if_files_exist(folder, list_files):
    folder = IOtools.expandPath(folder)

    for file_parts in list_files:
        checkfile = folder
        for ii in range(len(file_parts)):
            checkfile = checkfile / f"{file_parts[ii]}"
        if not checkfile.exists():
            return False

    return True

# Function to check if any rhos have already been run
def cold_start_checker(
    rhos,
    ResultsFiles,
    FolderNEO,
    cold_start=False,
    print_each_time=False,
):
    """
    This function checks if the TGLF inputs are already in the folder. If they are, it returns True
    """
    cont_each = 0
    if cold_start:
        rhosEvaluate = rhos
    else:
        rhosEvaluate = []
        for ir in rhos:
            existsRho = True
            for j in ResultsFiles:
                ffi = FolderNEO / f"{j}_{ir:.4f}"
                existsThis = ffi.exists()
                existsRho = existsRho and existsThis
                if not existsThis:
                    if print_each_time:
                        print(f"\t* {ffi} does not exist")
                    else:
                        cont_each += 1
            if not existsRho:
                rhosEvaluate.append(ir)

    if not print_each_time and cont_each > 0:
        print(f'\t* {cont_each} files from expected set are missing')

    if len(rhosEvaluate) < len(rhos) and len(rhosEvaluate) > 0:
        print(
            "~ Not all radii are found, but not removing folder and running only those that are needed",
            typeMsg="i",
        )

    return rhosEvaluate

# From file to dict
def inputToVariable(finalFolder, rhos):
    """
    Entire text file to variable
    """

    inputFilesNEO= {}
    for cont, rho in enumerate(rhos):
        fileN = finalFolder / f"input.neo_{rho:.4f}"

        with open(fileN, "r") as f:
            lines = f.readlines()
        inputFilesNEO[rho] = "".join(lines)

    return inputFilesNEO

# Prepares to write input.neo files for NEOtools.prep()
class NEOinput:
    def __init__(self, file=None):
        self.file = IOtools.expandPath(file) if isinstance(file, (str, Path)) else None

        if self.file is not None and self.file.exists():
            with open(self.file, "r") as f:
                lines = f.readlines()
            file_txt = "".join(lines)
        else:
            file_txt = ""
        input_dict = GACODErun.buildDictFromInput(file_txt)

        self.process(input_dict)

    @classmethod
    def initialize_in_memory(cls, input_dict):
        instance = cls()
        instance.process(input_dict)
        return instance

    # Prepares dictionary of NEO inputs for file writting structure
    def process(self, input_dict):
        # Avoid errors at __init__
        if 'N_SPECIES' not in input_dict.keys():
            return
        
        # Numerical resolution parameters
        kres = ['N_RADIAL', 'N_THETA', 'N_XI', 'N_ENERGY']
        self.controls = {}
        for kk in kres:
            # Error check
            if kk not in input_dict.keys():
                print('Using default value for parameter: %s'%(kk))
            else:
                self.controls[kk] = '%i'%(input_dict[kk])

        # Plasma equilibrium/geometry paramters
        kgeom = [
            'EQUILIBRIUM_MODEL',
            'RMIN_OVER_A', 'RMAJ_OVER_A',
            'KAPPA', 'S_KAPPA', 'DELTA', 'S_DELTA', 'ZETA', 'S_ZETA',
            'SHIFT', 'ZMAG_OVER_A', 'S_ZMAG', 'Q', 'SHEAR', 
            'BETA_STAR', 'IPCCW', 'BTCCW', 'RHO_STAR',
            'DPHI0DR', 'EPAR0', 'EPAR0_SPITZER'
            ]
        for ii in range(3, 6+1):
            kgeom.append('SHAPE_SIN%i'%(ii))
            kgeom.append('SHAPE_S_SIN%i'%(ii))
        for ii in range(0, 6+1):
            kgeom.append('SHAPE_COS%i'%(ii))
            kgeom.append('SHAPE_S_COS%i'%(ii))
        self.geom = {}
        for kk in kgeom:
            # Error check
            if kk not in input_dict.keys():
                print('Using default value for parameter: %s'%(kk))
            else:
                if kk in ['EQUILIBRIUM_MODEL', 'IPCCW', 'BTCCW']:
                    self.geom[kk] = '%i'%(input_dict[kk])
                else:
                    self.geom[kk] = '%0.5E'%(input_dict[kk])

        # General model parameters
        kmod = ['SILENT_FLAG', 'SIM_MODEL', 'SPITZER_MODEL', 'COLLISION_MODEL']
        self.models = {}
        for kk in kmod:
            # Error check
            if kk not in input_dict.keys():
                print('Using default value for parameter: %s'%(kk))
            else:
                self.models[kk] = '%i'%(input_dict[kk])

        # Rotation parameters
        krot = ['ROTATION_MODEL', 'OMEGA_ROT', 'OMEGA_ROT_DERIV']
        self.rot = {}
        for kk in krot:
            # Error check
            if kk not in input_dict.keys():
                print('Using default value for parameter: %s'%(kk))
            else:
                if kk in ['ROTATION_MODEL']:
                    self.rot[kk] = '%i'%(input_dict[kk])
                else:
                    self.rot[kk] = '%0.5E'%(input_dict[kk])

        # Species-dependent parameters
        kspe = [
            'NU', 'Z', 'MASS', 'DENS', 'TEMP', 'DLNNDR', 'DLNTDR',
            'ANISO_MODEL', 'TEMP_PARA', 'DLNTDR_PARA', 'TEMP_PERP', 'DLNTDR_PERP'
            ]
        self.plasma = {}
        self.plasma['N_SPECIES'] = '%i'%(input_dict['N_SPECIES'])
        # Loop over species
        for ii in range(0, input_dict['N_SPECIES']):
            # Loop over keys
            for kk in kspe:
                lab = '%s_%i'%(kk, ii+1) # Full key label
                # Error check
                if (kk == 'NU') and (ii > 0):
                    continue
                elif (
                    (kk in ['TEMP_PARA', 'DLNTDR_PARA', 'TEMP_PERP', 'DLNTDR_PERP'])
                    and (input_dict['ANISO_MODEL_%i'%(ii+1)] == 1)
                    ):
                    continue
                elif lab not in input_dict.keys():
                    print('Using default value for parameter: %s'%(kk))

                else:
                    if kk in ['ANISO_MODEL']:
                        self.plasma[lab] = '%i'%(input_dict[lab])
                    else:
                        self.plasma[lab] = '%0.5E'%(input_dict[lab])

    # Function to write the input.neo file
    def writeCurrentStatus(self, file=None):
        print("\t- Writting NEO input file")

        self.maxSpeciesNEO = 11  # NEO cannot handle more than 11 species

        if file is None:
            file = self.file

        with open(file, "w") as f:
            f.write(
                "#-------------------------------------------------------------------------\n"
            )
            f.write(
                "# NEO input file modified by MITIM framework (Rodriguez-Fernandez, 2020)\n"
            )
            f.write(
                "#-------------------------------------------------------------------------\n"
            )

            f.write("\n# Numerical resolution:\n")
            f.write("# ------------------\n")
            for ikey in self.controls:
                var = self.controls[ikey]
                f.write(f"{ikey}={var}\n")

            f.write("\n# Plasma equilibrium/geometry:\n")
            f.write("# ------------------\n")
            for ikey in self.geom:
                var = self.geom[ikey]
                f.write(f"{ikey}={var}\n")

            f.write("\n# General models:\n")
            f.write("# ------------------\n")
            for ikey in self.models:
                var = self.models[ikey]
                f.write(f"{ikey}={var}\n")

            f.write("\n# Rotation physics:\n")
            f.write("# ------------------\n")
            for ikey in self.rot:
                var = self.rot[ikey]
                f.write(f"{ikey}={var}\n")

            f.write("\n# Species-dependent parameters:\n")
            f.write("# ------------------\n")
            for ikey in self.plasma:
                if ikey == "N_SPECIES":
                    var = '%i'%(np.min([int(self.plasma[ikey]), self.maxSpeciesNEO]))
                else:
                    # Error check
                    if int(ikey.split('_')[-1]) > self.maxSpeciesNEO:
                        print(
                            "\t- Maximum number of species in NEO reached, not considering after {0} species".format(
                                maxSpeciesNEO
                            ),
                            typeMsg="w",
                        )
                        break
                    var = self.plasma[ikey]
                f.write(f"{ikey}={var}\n")

        print(f"\t\t~ File {IOtools.clipstr(file)} written")

    # Adds trace species for impurity D, V scan
    def addTraceSpecie(self, ZS, MASS, AS=1e-6, pos_elec = 1, pos_main_ion = 2):
        # Adds species index
        position = int(self.plasma['N_SPECIES']) + 1

        # Error check
        if position > self.maxSpeciesNEO:
            print(
                "\t- Maximum number of species, {0}, in NEO reached. Exiting".format(
                    maxSpeciesNEO
                    ),
                typeMsg="w",
                )
            sys.exit(1)

        # Adds species
        ### NOTE: Here, we assume user intends the first species index to be electrons
        ### and the second species index the main ion (i.e. D) like used in NEOtools.prep
        self.plasma['N_SPECIES'] = '%i'%(position)
        self.plasma['Z_%i'%(position)] = '%0.5E'%(ZS)
        self.plasma['MASS_%i'%(position)] = '%0.5E'%(MASS/mass_ref)

        self.plasma['DENS_%i'%(position)] = '%0.5E'%(
            AS * float(self.plasma['DENS_%i'%(pos_elec)])
            )

        kpl = ['TEMP', 'DLNNDR', 'DLNTDR']
        for kk in kpl:
            self.plasma['%s_%i'%(kk, position)] = self.plasma['%s_%i'%(kk, pos_main_ion)]

        self.plasma['ANISO_MODEL_%i'%(position)] = '1'

        # Output
        return position 

#######################################################################
#
#           Output Utilities
#
#######################################################################

# Handles reading results v. rho
def readNEOresults(
    FolderGACODE_tmp,
    NormalizationSets,
    rhos,
    suffix=None,
    inputs=None,
    ):
    # Init
    NEOstd_NEOout, inputclasses, parsed = [], [], []

    # Loop over radii
    for rho in rhos:
        # Read full folder
        NEOout = NEOoutput(
            FolderGACODE_tmp, 
            suffix=f"_{rho:.4f}" if suffix is None else suffix,
            dcontrols=inputs[rho].controls,
            dplasma=inputs[rho].plasma
            )

        # Unnormalize
        NEOout.unnormalize(
            NormalizationSets["SELECTED"],
            rho=rho,
            )

        NEOstd_NEOout.append(NEOout)
        inputclasses.append(NEOout.inputclass)

        parse = GACODErun.buildDictFromInput(NEOout.inputFileNEO)
        parsed.append(parse)

    results = {
        "inputclasses": inputclasses,
        "parsed": parsed,
        "NEOout": NEOstd_NEOout,
        "x": np.array(rhos),
        }

    # Output
    return results

class NEOoutput:
    def __init__(self, FolderGACODE, suffix="", dcontrols=None, dplasma=None):
        # Numerical resolution controls
        self.n_radial = int(dcontrols['N_RADIAL'])
        self.n_theta = int(dcontrols['N_THETA'])
        self.n_xi = int(dcontrols['N_XI'])
        self.n_energy = int(dcontrols['N_ENERGY'])

        # Plasma parameters
        self.n_species = int(dplasma['N_SPECIES'])
        self.dens_FSA = np.zeros(self.n_species) # [1e20/m^3/n_norm], dim(n_species,)
        for ii in np.arange(self.n_species):
            self.dens_FSA[ii] = float(dplasma['DENS_%i'%(ii+1)])

        # File management
        self.FolderGACODE, self.suffix = FolderGACODE, suffix

        if suffix == "":
            print(
                f"\t- Reading results from folder {IOtools.clipstr(FolderGACODE)} without suffix"
            )
        else:
            print(
                f"\t- Reading results from folder {IOtools.clipstr(FolderGACODE)} with suffix {suffix}"
            )
        self.inputclass = NEOinput(file=self.FolderGACODE / f"input.neo{self.suffix}")
        self.roa = self.inputclass.geom["RMIN_OVER_A"]

        # Reads the files
        self.read()

        # Post-processes the files
        #self.postprocess()

        print(
            f"\t- NEO was run with {self.n_species} species, {self.n_theta} poloidal mesh points",
            )

    ### --- Function to read NEO file --- ###
    def read(self):

        # Gets outputted transport fluxes in gyroBohm units, [GB]
        data_flux = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.neo.transport_flux" + self.suffix),
            blocks = 3,
            columns = 4,
            numky = None
            )

        # Init
        self.Z = np.zeros(self.n_species)       # [charge], dim(n_species,)
        self.pflux = np.zeros(self.n_species)   # [GB], dim(n_species,)
        self.eflux = np.zeros(self.n_species)   # [GB], dim(n_species,)
        self.mflux = np.zeros(self.n_species)   # [GB], dim(n_species,)

        # Loop over species
        for ii in np.arange(self.n_species):
            # Saves species charge to keep it straight
            self.Z[ii] = data_flux[0][ii][0]

            # Loop over flux contributions (drift-kinetic and gyroviscosity)
            for kk in np.arange(2):
                self.pflux[ii] += data_flux[kk][ii][1]
                self.eflux[ii] += data_flux[kk][ii][2]
                self.mflux[ii] += data_flux[kk][ii][3]

        # Gets numerical mesh data
        data_grid = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.neo.grid" + self.suffix),
            blocks = 1,
            columns = 1,
            numky = None
            )

        # Saves (rho, theta) grid
        self.rho = data_grid[0][-1][0]      # [], dim(1,)
        self.theta = data_grid[0][:-1][:,0] # [rad], dim(n_theta,)

        # Gets parallel flow data
        data_vel = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.neo.vel" + self.suffix),
            blocks = 1,
            columns = 1,
            numky = None
            )

        # Poloidal variation of field-aligned flow
        self.u_para = data_vel[0][:,0].reshape(self.n_theta, self.n_species).T # [m/s/vnorm], dim(n_species, n_theta)

        # Gets poloidal asymmetry data
        data_rot = GACODEinterpret.TGLFreader(
            self.FolderGACODE / ("out.neo.rotation" + self.suffix),
            blocks = 1,
            columns = 1,
            numky = None
            )

        # Init
        self.pol_dens_overMid = np.zeros((self.n_species, self.n_theta)) # dim(n_species, n_theta)
        strt = 2 + 2*self.n_species # starting idex

        # Ratio between the outboard midplane density to flux-surface-averaged values (assume latter is profile data)
        self.mid_dens_overFSA = data_rot[0][2:2*self.n_species+2:2,0] # dim(n_species)

        # Difference btw potential and outboard midplane value
        self.pol_pot = data_rot[0][strt:strt+self.n_theta,0] # [keV/Tnorm], dim(n_species, n_theta)

        # Loop over species
        for ii in np.arange(self.n_species):
            # Local density per the outboard midplane value, 
            self.pol_dens_overMid[ii,:] = data_rot[0][strt+(ii+1)*self.n_theta:strt+(ii+2)*self.n_theta, 0]

        # NEO input file
        with open(self.FolderGACODE / ("input.neo" + self.suffix), "r") as fi:
            lines = fi.readlines()
        self.inputFileNEO = "".join(lines)

    # Function to unnormalize quantities
    def unnormalize(self, normalization, rho=None):
        # Init interpolation
        from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function

        # Define interpolator at this rho
        def interpolator(y):
            return interpolation_function(rho, normalization['rho'],y).item()

        # Init normalization values
        Qgb = interpolator(normalization['q_gb'])
        Ggb = interpolator(normalization['g_gb'])
        n_norm = interpolator(normalization['ne_20'])
        T_norm = interpolator(normalization['Ti_keV']) # [keV]

        v_norm = np.sqrt(
            T_norm*1e3*cnt.e
            /(mass_ref*cnt.m_u)
            ) # [m/s]

        # Calc unnormalized values
        self.pflux_unn = self.pflux*Ggb # [1e20/m^2/s], dim(n_species,)
        self.eflux_unn = self.eflux*Qgb # [MW/m^2], dim(n_species,)

        self.u_para_unn = self.u_para*v_norm # [m/s], dim(n_species,n_theta)

        self.pol_pot_unn = self.pol_pot*T_norm*1e3 # [eV], dim(n_theta,)

        self.pol_dens_unn = (
            self.pol_dens_overMid
            * self.mid_dens_overFSA[:,None]
            * self.dens_FSA[:,None]
            * n_norm
            ) # [1e20/m^3], dim(n_species, n_theta)

