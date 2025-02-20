from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gacode_tools.utils import GACODErun
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed


class NEO:
    '''
    NEO class that manages the run and the results.

    The philosophy of this is that a single 'neo' class will handle the neo simulation and results
    at one time slice but with possibility of several radii at once.

    It can also handle different NEO settings, running them one by one, storing results in folders and then
    grabbing them.

    Scans can also be run. At several radii at once if wanted.

    '''
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
                #"out.neo.theory_nclass",    # Fluxes from NCLASS code
                "out.neo.transport",        # Numerical normlized DKE fluxes
                "out.neo.transport_flux",   # GyroBohm normalized all fluxes
                "out.neo.transport_gv",     # Fluxes from gyroviscosity (add to DKE solution)
                "out.neo.vel",              # 1st-order poloidal variation of the neoclassical toroidal flow
                "out.neo.vel_fourier",      # Fourier components for poloidal variation of neoclassical flow vector
                #"out.neo.transport_exp",    # Fluxes in real units
                #"out.neo.exp_norm",         # Profile normalization parameters
                "out.neo.rotation",         # Polodial asymmetry from strong rotation
                #"out.neo_diagnostic_geo",   # ????
                #"out.neo.diagnostic_geo2",  # ????
                #"out.neo.diagnositc_rot",   # ????
                #"out.neo.gxi",              # ????
                #"out.neo.gxi_t",            # ????
                #"out.neo.gxi_x",            # ????
                #"out.neo.localdump",        # Seems to list contents of input.neo
                #"out.neo.prec",             # ????
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

    def prep(self, inputgacode, folder):
        self.inputgacode = inputgacode
        self.folder = IOtools.expandPath(folder)

        self.folder.mkdir(parents=True, exist_ok=True)

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


def check_if_files_exist(folder, list_files):
    folder = IOtools.expandPath(folder)

    for file_parts in list_files:
        checkfile = folder
        for ii in range(len(file_parts)):
            checkfile = checkfile / f"{file_parts[ii]}"
        if not checkfile.exists():
            return False

    return True
