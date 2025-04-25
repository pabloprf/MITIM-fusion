import copy
import shutil
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
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
        self.quantities = ['QeMWm2', 'QiMWm2', 'Ce', 'CZ', 'MtJm2']

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

    def produce_profiles(self):
        # Only add self._produce_profiles() if it's needed (e.g. full TGLF), otherwise this is somewhat expensive
        # (e.g. for flux matching of analytical models)
        pass

    def _produce_profiles(self,deriveQuantities=True):

        self.applyCorrections = self.powerstate.TransportOptions["ModelOptions"].get("MODELparameters", {}).get("applyCorrections", {})

        # Write this updated profiles class (with parameterized profiles and target powers)
        self.file_profs = self.folder / "input.gacode"

        powerstate_detached = self.powerstate.copy_state()

        self.powerstate.profiles = powerstate_detached.from_powerstate(
            write_input_gacode=self.file_profs,
            postprocess_input_gacode=self.applyCorrections,
            rederive_profiles = deriveQuantities,        # Derive quantities so that it's ready for analysis and plotting later
            insert_highres_powers = deriveQuantities,    # Insert powers so that Q, Pfus and all that it's consistent when read later
        )

        self.profiles_transport = copy.deepcopy(self.powerstate.profiles)

        self._modify_profiles()

    def _modify_profiles(self):
        '''
        Modify the profiles (e.g. lumping) before running the transport model 
        '''

        # After producing the profiles, copy for future modifications
        self.file_profs_unmod = self.file_profs.parent / f"{self.file_profs.name}_unmodified"
        shutil.copy2(self.file_profs, self.file_profs_unmod)

        profiles_postprocessing_fun = self.powerstate.TransportOptions["ModelOptions"].get("profiles_postprocessing_fun", None)

        if profiles_postprocessing_fun is not None:
            print(f"\t- Modifying input.gacode to run transport calculations based on {profiles_postprocessing_fun}",typeMsg="i")
            self.profiles_transport = profiles_postprocessing_fun(self.file_profs)

        # Position of impurity ion may have changed
        p_old = PROFILEStools.PROFILES_GACODE(self.file_profs_unmod)
        p_new = PROFILEStools.PROFILES_GACODE(self.file_profs)

        impurity_of_interest = p_old.Species[self.powerstate.impurityPosition]

        try:
            impurityPosition_new = p_new.Species.index(impurity_of_interest)

        except ValueError:
            print(f"\t- Impurity {impurity_of_interest} not found in new profiles, keeping position {self.powerstate.impurityPosition}",typeMsg="w")
            impurityPosition_new = self.powerstate.impurityPosition

        if impurityPosition_new != self.powerstate.impurityPosition:
            print(f"\t- Impurity position has changed from {self.powerstate.impurityPosition} to {impurityPosition_new}",typeMsg="w")
            self.powerstate.impurityPosition_transport = p_new.Species.index(impurity_of_interest)

    # ----------------------------------------------------------------------------------------------------
    # EVALUATE (custom part)
    # ----------------------------------------------------------------------------------------------------
    def evaluate(self):
        '''
        This needs to populate the following in self.powerstate.plasma
            - Pe, Pe_tr, Pe_tr_turb, Pe_tr_neo -> MW/m^2
            - Pi, Pi_tr, Pi_tr_turb, Pi_tr_neo -> MW/m^2
            - Ce, Ce_tr, Ce_tr_turb, Ce_tr_neo -> MW/m^2
                * Ge1E20sm2, Ge1E20sm2_tr, Ge1E20sm2_tr_turb, Ge1E20sm2_tr_neo -> 10^20/s/m^2
            - CZ, CZ_tr, CZ_tr_turb, CZ_tr_neo -> MW/m^2 (but modified as needed, for example dividing by fZ0)
                * CZ_raw, CZ_raw_tr, CZ_raw_tr_turb, CZ_raw_tr_neo -> 10^20/s/m^2  (NOT modified)
            - Mt, Mt_tr, Mt_tr_turb, Mt_tr_neo -> J/m^2
            - PexchTurb -> MW/m^3
        and their respective standard deviations
        '''

        print(">> No transport fluxes to evaluate", typeMsg="w")
        pass

