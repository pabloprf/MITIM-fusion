import copy
import shutil
import torch
from mitim_tools.misc_tools import IOtools
from mitim_modules.portals.utils import PORTALScgyro
from mitim_modules.powertorch.physics_models import transport_tgyro
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class cgyro_model(transport_tgyro.tgyro_model):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

    def produce_profiles(self):
        super().produce_profiles()

    def evaluate(self):

        # Run original evaluator
        tgyro = self._evaluate_tglf_neo()

        # Run CGYRO trick
        powerstate_orig = self._trick_cgyro(tgyro)

        # Process results
        self._postprocess_results(tgyro, "cgyro_neo")

        # Some checks
        print("\t- Checking model modifications:")
        for r in ["Pe_tr_turb", "Pi_tr_turb", "Ce_tr_turb", "CZ_tr_turb", "Mt_tr_turb"]: #, "PexchTurb"]: #TODO: FIX
            print(f"\t\t{r}(tglf)  = {'  '.join([f'{k:.1e} (+-{ke:.1e})' for k,ke in zip(powerstate_orig.plasma[r][0][1:],powerstate_orig.plasma[r+'_stds'][0][1:]) ])}")
            print(f"\t\t{r}(cgyro) = {'  '.join([f'{k:.1e} (+-{ke:.1e})' for k,ke in zip(self.powerstate.plasma[r][0][1:],self.powerstate.plasma[r+'_stds'][0][1:]) ])}")

    # ************************************************************************************
    # Private functions for CGYRO evaluation
    # ************************************************************************************

    def _trick_cgyro(self, tgyro):

        print("\t- Checking whether cgyro_neo folder exists and it was written correctly via cgyro_trick...")

        correctly_run = (self.folder / "cgyro_neo").exists()
        if correctly_run:
            print("\t\t- Folder exists, but was cgyro_trick run?")
            with open(self.folder / "cgyro_neo" / "mitim_flag", "r") as f:
                correctly_run = bool(float(f.readline()))

        if correctly_run:
            print("\t\t\t* Yes, it was", typeMsg="w")
        else:
            print("\t\t\t* No, it was not, repating process", typeMsg="i")

            # Remove cgyro_neo folder
            if (self.folder / "cgyro_neo").exists():
                IOtools.shutil_rmtree(self.folder / "cgyro_neo")

            # Copy tglf_neo results
            shutil.copytree(self.folder / "tglf_neo", self.folder / "cgyro_neo")

            # CGYRO writter
            cgyro_trick(self,self.folder / "cgyro_neo")

        # Read TGYRO files and construct portals variables

        tgyro.read(label="cgyro_neo", folder=self.folder / "cgyro_neo") 

        powerstate_orig = copy.deepcopy(self.powerstate)
        
        return powerstate_orig

def cgyro_trick(self,FolderEvaluation_TGYRO):

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

    # Copy profiles so that later it is easy to grab all the input.gacodes that were evaluated
    self._profiles_to_store()

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
        self.powerstate.ProfilesPredicted,
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
