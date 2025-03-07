from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gacode_tools.utils import (
    NORMtools,
    #GACODEinterpret,
    #GACODEdefaults,
    #GACODEplotting,
    GACODErun,
    )
from mitim_tools.misc_tools.LOGtools import printMsg as print

from IPython import embed
from pathlib import Path
import numpy as np

# Useful constants
mi_D = 2.01355

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
                self.tgyro,
                self.ky_single,
            ) = ({}, {}, None, None)

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
            PROFILEStools.PROFILES_GACODE(inputgacode)
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
        self.profiles.deriveQuantities(mi_ref=mi_D)

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
        # Slurm controls
        launchSlurm=True,
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
            #tglf_executor={},
            #tglf_executor_full={},
            #NEOsettings=NEOsettings,
            #extraOptions=extraOptions,
            #multipliers=multipliers,
            #runWaveForms=runWaveForms,
            #forceClosestUnstableWF=forceClosestUnstableWF,
            #ApplyCorrections=ApplyCorrections,
            #Quasineutral=Quasineutral,
            launchSlurm=launchSlurm,
            cold_start=cold_start,
            forceIfcold_start=forceIfcold_start,
            extra_name=extra_name,
            slurm_setup=slurm_setup,
            #anticipate_problems=anticipate_problems,
        )

    # Prepares executables for run
    def _prepare_run_radii(
        subFolderNEO,
        launchSlurm = None,
        cold_start = None,
        extra_name = None,
        slurm_setup = None,
        forceIfcold_start=False,
        ):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare for run
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Init
        inputs = copy.deepcopy(self.inputsNEO)
        FolderNEO = self.fodler / subFolderNEO

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

        # Convert back to a string because that's how runTGLFproduction operates
        inputFileNEO = inputToVariable(FolderNEO, rhos)

        # Organizes execution data
        neo_executor_full[subFolderNEO] = {}
        neo_executor[subFolderNEO] = {}
        for irho in self.rhos:
            neo_executor_full[subFolderNEO][irho] = {
                "folder": FolderNEO,
                "dictionary": self.inputsNEO[irho],
                "inputs": inputFileNEO[irho],
                #"extraOptions": extraOptions,
                #"multipliers": multipliers,
            }
            if irho in rhosEvaluate:
                neo_executor[subFolderNEO][irho] = neo_executor_full[subFolderNEO][
                    irho
                ]

        # Output
        return neo_executor, neo_executor_full, FolderNEO

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
#           Utilities
#
#######################################################################

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
                if kk in ['EQUILIBRIUM_MODEL']:
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

        maxSpeciesNEO = 11  # NEO cannot handle more than 11 species

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
                    var = '%i'%(np.min([int(self.plasma[ikey]), maxSpeciesNEO]))
                else:
                    # Error check
                    if int(ikey.split('_')[-1]) > maxSpeciesNEO:
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

