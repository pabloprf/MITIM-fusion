import copy
import torch
import datetime
import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
from mitim_tools.misc_tools import PLASMAtools, IOtools
from mitim_modules.powertorch.aux import TRANSFORMtools, POWERplot
from mitim_modules.powertorch.iteration import ITtools
from mitim_modules.powertorch.physics import TARGETStools, TRANSPORTtools, CALCtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

UseCUDAifAvailable = True


def read_saved_state(file):
    print(f"\t- Reading state file {IOtools.clipstr(file)}")
    with open(file, "rb") as handle:
        state = pickle.load(handle)
    return state


# ------------------------------------------------------------------
# POWERSTATE Class
# ------------------------------------------------------------------


class powerstate:
    def __init__(
        self,
        input_gacode_orig,
        rho_vec,
        ProfilesPredicted=["te", "ti", "ne"],
        TransportOptions={"TypeTransport": "chis", "ModelOptions": {}},
        TargetOptions={"TypeTarget": 3, "TargetCalc": "powerstate"},
        useConvectiveFluxes=True,
        impurityPosition=1,
        fineTargetsResolution=None,
    ):
        self.dfT = torch.randn(
            (2, 2),
            dtype=torch.double,
            device=torch.device(
                "cpu"
                if ((not UseCUDAifAvailable) or (not torch.cuda.is_available()))
                else "cuda"
            ),
        )

        self.plasma = {}

        # -------------------------------------------------------------------------------------
        # Inputs
        # -------------------------------------------------------------------------------------

        self.ProfilesPredicted = ProfilesPredicted
        self.TargetType = TargetOptions["TypeTarget"]
        self.TargetCalc = TargetOptions["TargetCalc"]
        self.useConvectiveFluxes = useConvectiveFluxes
        self.impurityPosition = impurityPosition
        self.TransportOptions = TransportOptions

        # -------------------------------------------------------------------------------------
        # Populate plama with radial grid
        # -------------------------------------------------------------------------------------

        # If it's given as an array, convert to double tensor (it may be slow to use double precision)
        self.plasma["rho"] = (
            rho_vec if (type(rho_vec) == torch.Tensor) else torch.from_numpy(rho_vec)
        )
        self.plasma["rho"] = self.plasma["rho"].to(self.dfT)

        # -------------------------------------------------------------------------------------
        # Define how the plasma dictionary will be populated, to repeat/unrepeat/detach
        # -------------------------------------------------------------------------------------

        self.keys1D = {
            "roa": 1,
            "rho": 1,
            "rmin": 1,
            "volp": 1,
            "B_unit": 1,
            "B_ref": 1,
            "te": 1,
            "ti": 1,
            "ne": 1,
            "nZ": 1,
            "w0": 1,
            "PauxE": 1,
            "PauxI": 1,
            "GauxE": 1,
            "GauxZ": 1,
            "MauxT": 1,
        }
        self.keys2D = {"ni": 1}
        self.keys0D = {}
        self.keys1D_derived = (
            {}
        )  # Here I will store variables later. Only repeat if requested

        self.keys1D["aLte"] = self.keys1D["aLti"] = self.keys1D["aLne"] = self.keys1D[
            "aLnZ"
        ] = self.keys1D["aLw0"] = 1

        # -------------------------------------------------------------------------------------
        # input.gacode
        # -------------------------------------------------------------------------------------

        # Use a copy because I'm deriving, it may be expensive and I don't want to carry that out outside of this class
        input_gacode = copy.deepcopy(input_gacode_orig)
        if "derived" not in input_gacode.__dict__:
            input_gacode.deriveQuantities()

        """
		------------------------------------------------------------------------------------------------------------------------------
		Fine targets procedure:
			Create a plasma_fine dictionary that will contain high resolution profiles, to be used only in calculateTargets.
			For the rest, keep using simply the coarse grid
		------------------------------------------------------------------------------------------------------------------------------
		"""

        if fineTargetsResolution is None:
            self.plasma_fine, self.positions_targets = None, None
        else:
            # Copy coarse plasma to insert back later
            plasma_copy = copy.deepcopy(self.plasma)

            # *******************
            # High resolution rho
            # *******************
            rho_new = torch.linspace(
                self.plasma["rho"][0], self.plasma["rho"][-1], fineTargetsResolution
            ).to(self.plasma["rho"])
            for i in self.plasma["rho"]:
                if not torch.isclose(
                    rho_new, torch.Tensor([i]).to(self.dfT), atol=1e-10
                ).any():
                    rho_new = torch.cat((rho_new, i.unsqueeze(0).to(self.dfT)))
            rho_new = torch.sort(rho_new)[0]

            # Recalculate with higher resolution
            TRANSFORMtools.fromGacodeToPower(self, input_gacode, rho_new)

            # Insert back
            self.plasma_fine = copy.deepcopy(self.plasma)
            self.plasma = plasma_copy
            self.positions_targets = []
            for i in self.plasma["rho"]:
                self.positions_targets.append(
                    (torch.isclose(rho_new, torch.Tensor([i]).to(self.dfT), atol=1e-10))
                    .nonzero()[0][0]
                    .item()
                )

        # ------------------------------------------------------------------------------------------------------------------------------

        # -------------------------------------------------------------------------------------
        # Standard creation of plasma dictionary
        # -------------------------------------------------------------------------------------

        TRANSFORMtools.fromGacodeToPower(self, input_gacode, self.plasma["rho"])

        # -------------------------------------------------------------------------------------
        # Postprocessing operations
        # -------------------------------------------------------------------------------------

        # Convert into a batch so that always the quantities are (batch,dimX)
        self.repeat(batch_size=1)

        # Have the posibility of storing profiles in the class
        self.profiles_stored = []
        self.FluxMatch_Xopt, self.FluxMatch_Yopt = torch.Tensor().to(
            self.dfT
        ), torch.Tensor().to(self.dfT)

    def insertProfiles(
        self,
        profiles,
        writeFile=None,
        PositionInBatch=0,
        applyCorrections={},
        insertPowers=False,
        rederive_profiles=True,
        reRead=True,
    ):
        print(">> Inserting powerstate profiles into input.gacode")

        profiles = TRANSFORMtools.fromPowerToGacode(
            self,
            profiles,
            PositionInBatch=PositionInBatch,
            options=applyCorrections,
            insertPowers=insertPowers,
            rederive=rederive_profiles,
            ProfilesPredicted=self.ProfilesPredicted,
        )

        if writeFile is not None:
            print(f"\t- Writing input.gacode file: {IOtools.clipstr(writeFile)}")
            profiles.writeCurrentStatus(file=writeFile)

        # If corrections modify the ions set... it's better to re-read, otherwise powerstate will be confused
        if reRead:
            TRANSFORMtools.defineIons(
                self, profiles, self.plasma["rho"][0, :], self.dfT
            )

            # Repeat, that's how it's done earlier
            self.plasma["ni"] = self.plasma["ni"].repeat(
                self.plasma["rho"].shape[0], 1, 1
            )
            if self.plasma_fine is not None:
                self.plasma_fine["ni"] = self.plasma_fine["ni"].repeat(
                    self.plasma["rho"].shape[0], 1, 1
                )

        return profiles

    # ------------------------------------------------------------------
    # Storing and combining
    # ------------------------------------------------------------------

    def save(self, file):
        print(f"\t- Writing power state file: {IOtools.clipstr(file)}")
        with open(file, "wb") as handle:
            pickle.dump(self, handle)

    def combine_states(self, states, includeTransport=True):
        self.TransportOptions_set = [self.TransportOptions]
        self.profiles_stored_set = self.profiles_stored

        for state in states:
            for key in self.keys0D:
                self.plasma[key] = torch.cat((self.plasma[key], state[key])).to(
                    self.plasma[key]
                )
            for key in self.keys1D:
                self.plasma[key] = torch.cat((self.plasma[key], state[key])).to(
                    self.plasma[key]
                )
            for key in self.keys2D:
                self.plasma[key] = torch.cat((self.plasma[key], state[key])).to(
                    self.plasma[key]
                )

            self.TransportOptions_set.append(state.TransportOptions)
            self.profiles_stored_set += state.profiles_stored

            if includeTransport:
                for key in ["chi_e", "chi_i"]:
                    self.TransportOptions["ModelOptions"][key] = torch.cat(
                        (
                            self.TransportOptions["ModelOptions"][key],
                            state.TransportOptions["ModelOptions"][key],
                        )
                    ).to(self.TransportOptions["ModelOptions"][key])

    # ------------------------------------------------------------------
    # Flux-matching and iteration tools
    # ------------------------------------------------------------------

    def calculate(
        self, X, nameRun="test", folder="~/scratch/", storeInfo=False, extra_params={}
    ):
        """
        Provide what's needed for flux-matching
        """

        # 1. Modify gradients
        self.modify(X)

        # 2. Plasma parameters
        self.calculateProfileFunctions()

        # 3. Sources and sinks
        if self.TargetCalc == "powerstate":
            self.calculateTargets()  # Calculate targets based on powerstate functions
        else:
            pass  # Use what it is going to be calculated in the next step

        # 4. Turbulent and neoclassical transport
        self.calculateTransport(
            nameRun=nameRun,
            folder=folder,
            storeInfo=storeInfo,
            extra_params=extra_params,
        )

        # 5. Residual powers
        self.metric()

        return (
            self.plasma["P_tr"],
            self.plasma["P"],
            self.plasma["S"],
            self.plasma["residual"],
        )

    def modify(self, X):
        self.Xcurrent = X
        numeach = self.plasma["rho"].shape[1] - 1

        for c, i in enumerate(self.ProfilesPredicted):
            if X is not None:
                self.plasma[f"aL{i}"][:, 1:] = self.Xcurrent[
                    :, numeach * c : numeach * (c + 1)
                ]
            self.update_var(i)

    def findFluxMatchProfiles(
        self, algorithm="root", algorithmOptions={}, bounds=None, extra_params={}
    ):
        self.FluxMatch_plasma_orig = copy.deepcopy(self.plasma)

        print(
            f'\n- Flux matching of powerstate file ({self.plasma["rho"].shape[0]} parallel batches of {self.plasma["rho"].shape[1]-1} radii) has been requested...'
        )
        print(
            "**********************************************************************************************"
        )
        timeBeginning = datetime.datetime.now()

        if algorithm == "root":
            ITtools.fluxMatchRoot(self, extra_params=extra_params)
        if algorithm == "picard":
            ITtools.fluxMatchPicard(self, extra_params=extra_params)
        if algorithm == "simple_relax":
            self.FluxMatch_Xopt, self.FluxMatch_Yopt = ITtools.fluxMatchSimpleRelax(
                self,
                algorithmOptions=algorithmOptions,
                bounds=bounds,
                extra_params=extra_params,
            )
        if algorithm == "prfseq":
            self.FluxMatch_Xopt, self.FluxMatch_Yopt = ITtools.fluxMatchPRFseq(
                self,
                algorithmOptions=algorithmOptions,
                bounds=bounds,
                extra_params=extra_params,
            )

        print(
            "**********************************************************************************************"
        )
        print(
            f"- Flux matching of powerstate file has been found, and took {IOtools.getTimeDifference(timeBeginning)}\n"
        )

    # ------------------------------------------------------------------
    # Plotting tools
    # ------------------------------------------------------------------

    def plot(self, axs=None, axsRes=None, figs=None, c="r", label=""):
        if axs is None:
            from mitim_tools.misc_tools.GUItools import FigureNotebook

            self.fn = FigureNotebook("PowerState", geometry="1800x900")
            figMain = self.fn.add_figure(label="PowerState")

            figProf_1 = self.fn.add_figure(label="Profiles")
            figProf_2 = self.fn.add_figure(label="Powers")
            figProf_3 = self.fn.add_figure(label="Geometry")
            figProf_4 = self.fn.add_figure(label="Gradients")
            figFlows = self.fn.add_figure(label="Flows")
            figProf_6 = self.fn.add_figure(label="Other")
            fig7 = self.fn.add_figure(label="Impurities")
            figs = figProf_1, figProf_2, figProf_3, figProf_4, figFlows, figProf_6, fig7

            grid = plt.GridSpec(4, 6, hspace=0.3, wspace=0.3)

            axs = [
                figMain.add_subplot(grid[0, 1]),
                figMain.add_subplot(grid[0, 2]),
                figMain.add_subplot(grid[0, 3]),
                figMain.add_subplot(grid[0, 4]),
                figMain.add_subplot(grid[0, 5]),
                figMain.add_subplot(grid[1, 1]),
                figMain.add_subplot(grid[1, 2]),
                figMain.add_subplot(grid[1, 3]),
                figMain.add_subplot(grid[1, 4]),
                figMain.add_subplot(grid[1, 5]),
                figMain.add_subplot(grid[2, 1]),
                figMain.add_subplot(grid[2, 2]),
                figMain.add_subplot(grid[2, 3]),
                figMain.add_subplot(grid[2, 4]),
                figMain.add_subplot(grid[2, 5]),
                figMain.add_subplot(grid[3, 1]),
                figMain.add_subplot(grid[3, 2]),
                figMain.add_subplot(grid[3, 3]),
                figMain.add_subplot(grid[3, 4]),
                figMain.add_subplot(grid[3, 5]),
            ]

            axsRes = figMain.add_subplot(grid[:, 0])

        POWERplot.plot(self, axs, axsRes, figs, c=c, label=label)

    # ------------------------------------------------------------------
    # Main tools
    # ------------------------------------------------------------------

    def detach_tensors(self, includeDerived=True, DoThisPlasma=None, do_fine=True):
        if DoThisPlasma is None:
            DoThisPlasma = self.plasma

        for key in self.keys0D:
            DoThisPlasma[key] = DoThisPlasma[key].detach()
        for key in self.keys1D:
            DoThisPlasma[key] = DoThisPlasma[key].detach()
        if includeDerived:
            for key in self.keys1D_derived:
                DoThisPlasma[key] = DoThisPlasma[key].detach()
        for key in self.keys2D:
            DoThisPlasma[key] = DoThisPlasma[key].detach()

        if do_fine and (self.plasma_fine is not None):
            self.detach_tensors(
                includeDerived=includeDerived,
                DoThisPlasma=self.plasma_fine,
                do_fine=False,
            )

    def repeat(
        self, batch_size=1, pos=0, includeDerived=True, DoThisPlasma=None, do_fine=True
    ):
        """
        Repeat 1D profiles (radii) to (batch_size,radii) so that the mitim calcs are fine

        includeDerived = False will make things faster if I'm recalculating things later anyway
        """

        if DoThisPlasma is None:
            DoThisPlasma = self.plasma

        if len(DoThisPlasma["rho"].shape) == 1:
            for key in self.keys0D:
                DoThisPlasma[key] = DoThisPlasma[key].repeat(batch_size)
            for key in self.keys1D:
                DoThisPlasma[key] = DoThisPlasma[key].repeat(batch_size, 1)
            for key in self.keys2D:
                DoThisPlasma[key] = DoThisPlasma[key].repeat(batch_size, 1, 1)
            if includeDerived:
                for key in self.keys1D_derived:
                    DoThisPlasma[key] = DoThisPlasma[key].repeat(batch_size, 1)
        else:
            for key in self.keys0D:
                DoThisPlasma[key] = DoThisPlasma[key][pos].repeat(batch_size)
            for key in self.keys1D:
                DoThisPlasma[key] = DoThisPlasma[key][pos, :].repeat(batch_size, 1)
            for key in self.keys2D:
                DoThisPlasma[key] = DoThisPlasma[key][pos, :, :].repeat(
                    batch_size, 1, 1
                )
            if includeDerived:
                for key in self.keys1D_derived:
                    DoThisPlasma[key] = DoThisPlasma[key][pos, :].repeat(batch_size, 1)

        if do_fine and (self.plasma_fine is not None):
            self.repeat(
                batch_size=batch_size,
                pos=pos,
                includeDerived=includeDerived,
                DoThisPlasma=self.plasma_fine,
                do_fine=False,
            )

    def unrepeat(self, pos=0, includeDerived=True, DoThisPlasma=None, do_fine=True):
        """
        Opposite to repeat(), to extract just one profile of the batch
        """

        if DoThisPlasma is None:
            DoThisPlasma = self.plasma

        for key in self.keys0D:
            DoThisPlasma[key] = DoThisPlasma[key][pos]
        for key in self.keys1D:
            self.plasma[key] = DoThisPlasma[key][pos, :]
        for key in self.keys2D:
            DoThisPlasma[key] = DoThisPlasma[key][pos, :, :]
        if includeDerived:
            for key in self.keys1D_derived:
                DoThisPlasma[key] = DoThisPlasma[key] = DoThisPlasma[key][pos, :]

        if do_fine and (self.plasma_fine is not None):
            self.unrepeat(
                pos=pos,
                includeDerived=includeDerived,
                DoThisPlasma=self.plasma_fine,
                do_fine=False,
            )

    def update_var(
        self, name, var=None, printMessages=True, specific_deparametrizer=None
    ):
        """
        This inserts gradients and updates coarse profiles

        Notes:
                - If no var is given, assume that I have already assign it and I need to update the profile
                - Gradients must be defined on axis. (0,0) please
                - This keeps the thermal ion concentrations constant to the original case
        """

        deparametrizers_choice = (
            self.deparametrizers_coarse
            if specific_deparametrizer is None
            else specific_deparametrizer
        )

        if name in ["te"]:
            if var is not None:
                self.plasma["aLte"][: var.shape[0], :] = var[:, :]
            aLT_withZero = self.plasma["aLte"]
            _, varN = deparametrizers_choice["te"](
                self.plasma["roa"], aLT_withZero, printMessages=printMessages
            )

            # To avoid crazy values that could lead to nans or infs
            self.plasma["te"] = varN.clamp(min=0, max=200)

            # Complete full aLT
            self.plasma["aLte"] = torch.cat(
                (
                    self.plasma["aLte"][:, : -self.plasma["rho"].shape[1] + 1],
                    self.plasma["aLte"][:, 1:],
                ),
                axis=1,
            )

        elif name in ["ti"]:
            if var is not None:
                self.plasma["aLti"][: var.shape[0], :] = var[:, :]
            aLT_withZero = self.plasma["aLti"]
            _, varN = deparametrizers_choice["ti"](
                self.plasma["roa"], aLT_withZero, printMessages=printMessages
            )

            # To avoid crazy values that could lead to nans or infs
            self.plasma["ti"] = varN.clamp(min=0, max=200)

            # Complete full aLT
            self.plasma["aLti"] = torch.cat(
                (
                    self.plasma["aLti"][:, : -self.plasma["rho"].shape[1] + 1],
                    self.plasma["aLti"][:, 1:],
                ),
                axis=1,
            )

        elif name in ["ne"]:
            if var is not None:
                self.plasma["aLne"][: var.shape[0], :] = var[:, :]
            aLT_withZero = self.plasma["aLne"]
            _, varN = deparametrizers_choice["ne"](
                self.plasma["roa"], aLT_withZero, printMessages=printMessages
            )

            # If it's density, be careful about the rest of ions
            ne_0orig, ni_0orig = self.plasma["ne"].clone(), self.plasma["ni"].clone()

            # To avoid crazy values that could lead to nans or infs
            self.plasma["ne"] = varN.clamp(min=0, max=200)

            # Complete full aLT
            self.plasma["aLne"] = torch.cat(
                (
                    self.plasma["aLne"][:, : -self.plasma["rho"].shape[1] + 1],
                    self.plasma["aLne"][:, 1:],
                ),
                axis=1,
            )

            # Modify ni accordingly
            self.plasma["ni"] = ni_0orig.clone()
            for i in range(self.plasma["ni"].shape[2]):
                self.plasma["ni"][:, :, i] = self.plasma["ne"] * (
                    ni_0orig[:, :, i] / ne_0orig
                )

        elif name in ["nZ"]:
            if var is not None:
                self.plasma["aLnZ"][: var.shape[0], :] = var[:, :]
            aLT_withZero = self.plasma["aLnZ"]
            _, varN = deparametrizers_choice["nZ"](
                self.plasma["roa"], aLT_withZero, printMessages=printMessages
            )

            # To avoid crazy values that could lead to nans or infs
            self.plasma["nZ"] = varN.clamp(min=0, max=200)

            # Complete full aLT
            self.plasma["aLnZ"] = torch.cat(
                (
                    self.plasma["aLnZ"][:, : -self.plasma["rho"].shape[1] + 1],
                    self.plasma["aLnZ"][:, 1:],
                ),
                axis=1,
            )

            # Insert into ni
            self.plasma["ni"] = self.plasma["ni"].clone()
            self.plasma["ni"][:, :, self.impurityPosition - 1] = self.plasma["nZ"]

        elif name in ["w0"]:
            factor_mult = 1 / TRANSFORMtools.factorMult_w0(self)

            if var is not None:
                self.plasma["aLw0"][: var.shape[0], :] = var[:, :]
            aLT_withZero = self.plasma["aLw0"]
            _, varN = deparametrizers_choice["w0"](
                self.plasma["roa"], aLT_withZero, printMessages=printMessages
            )

            # Complete full aLT
            self.plasma["aLw0"] = torch.cat(
                (
                    self.plasma["aLw0"][:, : -self.plasma["rho"].shape[1] + 1],
                    self.plasma["aLw0"][:, 1:],
                ),
                axis=1,
            )

            self.plasma["w0"] = varN * factor_mult  # .clamp(min=0,max=200)

        return aLT_withZero

    # ------------------------------------------------------------------
    # Toolset for calculation
    # ------------------------------------------------------------------

    def calculateProfileFunctions(self, calculateRotationStuff=True, mref_u=2.01355):
        """
        Update the normalizations of the current state
        """

        # gyro-Bohm
        (
            self.plasma["Qgb"],
            self.plasma["Ggb"],
            self.plasma["Pgb"],
            self.plasma["Sgb"],
            self.plasma["Qgb_convection"],
        ) = PLASMAtools.gyrobohmUnits(
            self.plasma["te"],
            self.plasma["ne"] * 1e-1,
            mref_u,
            self.plasma["B_unit"],
            self.plasma["a"],
        )

        # Collisionality
        self.plasma["nuei"] = PLASMAtools.xnue(
            self.plasma["te"], self.plasma["ne"] * 1e-1, self.plasma["a"], mref_u
        )

        # Gyro-radius
        self.plasma["rho_s"] = PLASMAtools.rho_s(
            self.plasma["te"], mref_u, self.plasma["B_unit"]
        )
        self.plasma["c_s"] = PLASMAtools.c_s(self.plasma["te"], mref_u)

        # Other
        self.plasma["tite"] = self.plasma["ti"] / self.plasma["te"]
        self.plasma["fZ"] = self.plasma["nZ"] / self.plasma["ne"]
        self.plasma["beta_e"] = PLASMAtools.betae(
            self.plasma["te"], self.plasma["ne"] * 1e-1, self.plasma["B_unit"]
        )

        """
		Make sure that the variables are on-repeat
		------------------------------------------
		"""

        self.keys1D_derived["Qgb"] = 1
        self.keys1D_derived["Ggb"] = 1
        self.keys1D_derived["Pgb"] = 1
        self.keys1D_derived["Sgb"] = 1
        self.keys1D_derived["nuei"] = 1
        self.keys1D_derived["rho_s"] = 1
        self.keys1D_derived["c_s"] = 1
        self.keys1D_derived["tite"] = 1
        self.keys1D_derived["fZ"] = 1
        self.keys1D_derived["beta_e"] = 1

        """
		Rotation stuff
		--------------
			Reason why I not always calculate this is because here I'm combining quantities derived (w0,cs)
			which can be accessed in a finer grid, with aLw0 which is a primitive one, that works in the coarse
			grid. Therefore, aLw0_n can only be calculated on the coarse grid.
		"""
        if calculateRotationStuff:
            self.plasma["w0_n"] = self.plasma["w0"] / self.plasma["c_s"]
            self.plasma["aLw0_n"] = (
                self.plasma["aLw0"] * self.plasma["w0"] / self.plasma["c_s"]
            )  # aLw0 * w0 = -a*dw0/dr; then aLw0_n = -dw0/dr * a/c_s
            self.keys1D_derived["w0_n"] = 1
            self.keys1D_derived["aLw0_n"] = 1

    def calculateTargets(self):
        """
        Update the targets of the current state
        """

        # Fixed Targets
        if self.TargetType == 1:
            PextraE, PextraI = (
                self.plasma["PextraE_Target1"],
                self.plasma["PextraI_Target1"],
            )  # Original integrated from input.gacode
        elif self.TargetType == 2:
            PextraE, PextraI = (
                self.plasma["PextraE_Target2"],
                self.plasma["PextraI_Target2"],
            )
        elif self.TargetType == 3:
            PextraE, PextraI = self.plasma["te"] * 0.0, self.plasma["te"] * 0.0

        # **************************************************************************************************
        # Work on a finer grid for targets?
        # **************************************************************************************************

        if self.plasma_fine is not None:
            """
            Make all quantities needed on the fine resolution
            -------------------------------------------------
                    Note that the set ['te','ti','ne','nZ','w0','ni'] will automatically be substituted during the update_var() that comes next, so
                    it's ok that I lose the torch leaf here. However, I must do this copy here because if any of those variables are not updated in
                    update_var() then it would fail. But first store them for later use.
            """

            plasma_original = {}
            for variable in [
                "B_unit",
                "B_ref",
                "volp",
                "rmin",
                "roa",
                "rho",
                "te",
                "ti",
                "ne",
                "nZ",
                "w0",
                "ni",
            ]:
                plasma_original[variable] = self.plasma[variable].clone()
                self.plasma[variable] = self.plasma_fine[variable]

            # Store also the gradients that are part of the torch trees, so that the derivative is not lost
            for variable in ["aLte", "aLti", "aLne", "aLnZ", "aLw0"]:
                plasma_original[variable] = self.plasma[variable].clone()

            """
			Integrate through fine de-parameterization
			"""
            for i in self.ProfilesPredicted:
                _ = self.update_var(
                    i,
                    printMessages=False,
                    specific_deparametrizer=self.deparametrizers_coarse_middle,
                )

        """
		**************************************************************************************************
		Calculate Targets
		**************************************************************************************************
		"""
        # Start by making sub-targets equal to zero
        for i in [
            "qfuse",
            "qfusi",
            "qie",
            "qrad",
            "qrad_bremms",
            "qrad_line",
            "qrad_sync",
        ]:
            self.plasma[i] = self.plasma["te"] * 0.0

        # Compute targets

        if self.TargetType >= 2:
            TARGETStools.exchange(self)

        if self.TargetType == 3:
            TARGETStools.alpha(self)
            TARGETStools.radiation(self)

        """
		**************************************************************************************************
		Calculate integral of all targets, and then sum aux.
		Reason why I do it this convoluted way is to make it faster in mitim, not to run integrateQuadPoly all the time.
		Run once for all the batch and also for electrons and ions
		(in MW/m^2)
		**************************************************************************************************
		"""

        qe = self.plasma["qfuse"] - self.plasma["qie"] - self.plasma["qrad"]
        qi = self.plasma["qfusi"] + self.plasma["qie"]
        q = torch.cat((qe, qi)).to(qe)
        P = self.volume_integrate(q, dim=2)

        # **************************************************************************************************
        # Come back to original grid for targets?
        # **************************************************************************************************

        if self.plasma_fine is not None:
            # Interpolate results from fine to coarse (i.e. whole point is that it is better than integrate interpolated values)
            if self.TargetType >= 2:
                for i in ["qie"]:
                    self.plasma[i] = self.plasma[i][:, self.positions_targets]
            if self.TargetType == 3:
                for i in [
                    "qfuse",
                    "qfusi",
                    "qrad",
                    "qrad_bremms",
                    "qrad_line",
                    "qrad_sync",
                ]:
                    self.plasma[i] = self.plasma[i][:, self.positions_targets]
            P = P[:, self.positions_targets]

            # Recover variables calculated prior to the fine-targets method
            for i in plasma_original:
                self.plasma[i] = plasma_original[i]

        # **************************************************************************************************
        # Plug-in Targets
        # **************************************************************************************************

        self.plasma["Pe"] = (
            self.plasma["PauxE"] + P[: qe.shape[0], :] + PextraE
        )  # MW/m^2
        self.plasma["Pi"] = (
            self.plasma["PauxI"] + P[qe.shape[0] :, :] + PextraI
        )  # MW/m^2
        self.plasma["Ce"] = self.plasma["GauxE"]  # 1E20/s/m^2
        self.plasma["CZ"] = self.plasma["GauxZ"]  # 1E20/s/m^2
        self.plasma["Mt"] = self.plasma["MauxT"]  # J/m^2

        if self.useConvectiveFluxes:
            self.plasma["Ce"] = PLASMAtools.convective_flux(
                self.plasma["te"], self.plasma["Ce"]
            )  # MW/m^2
            self.plasma["CZ"] = PLASMAtools.convective_flux(
                self.plasma["te"], self.plasma["CZ"]
            )  # MW/m^2

        if (
            "forceZeroParticleFlux" in self.TransportOptions["ModelOptions"]
        ) and self.TransportOptions["ModelOptions"]["forceZeroParticleFlux"]:
            self.plasma["Ce"] = self.plasma["Ce"] * 0

        """
		**************************************************************************************************
		GB Normalized
		**************************************************************************************************
			Note: This is useful for mitim surrogate variables of targets
		"""

        self.plasma["PeGB"] = self.plasma["Pe"] / self.plasma["Qgb"]
        self.plasma["PiGB"] = self.plasma["Pi"] / self.plasma["Qgb"]
        if self.useConvectiveFluxes:
            self.plasma["CeGB"] = self.plasma["Ce"] / self.plasma["Qgb"]
            self.plasma["CZGB"] = self.plasma["CZ"] / self.plasma["Qgb"]
        else:
            self.plasma["CeGB"] = self.plasma["Ce"] / self.plasma["Ggb"]
            self.plasma["CZGB"] = self.plasma["CZ"] / self.plasma["Ggb"]
        self.plasma["MtGB"] = self.plasma["Mt"] / self.plasma["Pgb"]

        # **************************************************************************************************
        # Make sure that the variables are on-repeat
        # **************************************************************************************************
        for i in [
            "Pe",
            "Pi",
            "Ce",
            "CZ",
            "Mt",
            "qfuse",
            "qfusi",
            "qie",
            "qrad",
            "qrad_sync",
            "qrad_line",
            "qrad_bremms",
            "PeGB",
            "PiGB",
            "CeGB",
            "CZGB",
            "MtGB",
        ]:
            self.keys1D_derived[i] = 1

    def calculateTransport(
        self, nameRun="test", folder="~/scratch/", storeInfo=False, extra_params={}
    ):
        """
        Update the transport of the current state
        """

        # ******* Nothing
        if self.TransportOptions["TypeTransport"] is None:
            for i in [
                "Pe_tr",
                "Pi_tr",
                "Ce_tr",
                "CZ_tr",
                "Mt_tr",
                "Pe_tr_turb",
                "Pi_tr_turb",
                "Ce_tr_turb",
                "CZ_tr_turb",
                "Mt_tr_turb",
                "Pe_tr_neo",
                "Pi_tr_neo",
                "Ce_tr_neo",
                "CZ_tr_neo",
                "Mt_tr_neo",
            ]:
                self.plasma[i] = self.plasma["te"][:, 1:] * 0.0

            for i in ["Pe", "Pi", "Ce", "CZ", "Mt"]:
                self.plasma[i] = self.plasma[i][:, 1:]

        # ******* TGLF+NEO
        elif "tgyro" in self.TransportOptions["TypeTransport"]:
            TGYROresults = TRANSPORTtools.tgyro_model(
                self,
                self.TransportOptions["ModelOptions"],
                name=nameRun,
                folder=folder,
                provideTargets=self.TargetCalc == "tgyro",
                TypeTransport=self.TransportOptions["TypeTransport"],
                extra_params=extra_params,
            )

            if storeInfo:
                self.TGYROresults = TGYROresults

        # ******* Transport coefficients
        elif self.TransportOptions["TypeTransport"] == "chis":
            TRANSPORTtools.diffusion_model(self, self.TransportOptions["ModelOptions"])

        # ******* Surrogate Model
        elif self.TransportOptions["TypeTransport"] == "surrogate":
            TRANSPORTtools.surrogate_model(self, self.TransportOptions["ModelOptions"])

        # Make sure that the variables are on-repeat
        for i in [
            "Pe_tr",
            "Pi_tr",
            "Ce_tr",
            "CZ_tr",
            "Mt_tr",
            "Pe_tr_turb",
            "Pi_tr_turb",
            "Ce_tr_turb",
            "CZ_tr_turb",
            "Mt_tr_turb",
            "Pe_tr_neo",
            "Pi_tr_neo",
            "Ce_tr_neo",
            "CZ_tr_neo",
            "Mt_tr_neo",
        ]:
            self.keys1D_derived[i] = 1

    def metric(self):
        """
        Calculation of residual transport, dP/dt, ignoring the first (a zero)
        Define convergence metric of current state
        """

        # All fluxes in a single vector
        self.plasma["P"], self.plasma["P_tr"] = torch.Tensor().to(
            self.plasma["Pe"]
        ), torch.Tensor().to(self.plasma["Pe"])
        for c, i in enumerate(self.ProfilesPredicted):
            if i == "te":
                self.plasma["P"] = torch.cat(
                    (self.plasma["P"], self.plasma["Pe"]), dim=1
                ).to(self.plasma["P"])
                self.plasma["P_tr"] = torch.cat(
                    (self.plasma["P_tr"], self.plasma["Pe_tr"]), dim=1
                ).to(self.plasma["P"])
            if i == "ti":
                self.plasma["P"] = torch.cat(
                    (self.plasma["P"], self.plasma["Pi"]), dim=1
                ).to(self.plasma["P"])
                self.plasma["P_tr"] = torch.cat(
                    (self.plasma["P_tr"], self.plasma["Pi_tr"]), dim=1
                ).to(self.plasma["P"])
            if i == "ne":
                self.plasma["P"] = torch.cat(
                    (self.plasma["P"], self.plasma["Ce"]), dim=1
                ).to(self.plasma["P"])
                self.plasma["P_tr"] = torch.cat(
                    (self.plasma["P_tr"], self.plasma["Ce_tr"]), dim=1
                ).to(self.plasma["P"])
            if i == "nZ":
                self.plasma["P"] = torch.cat(
                    (self.plasma["P"], self.plasma["CZ"]), dim=1
                ).to(self.plasma["P"])
                self.plasma["P_tr"] = torch.cat(
                    (self.plasma["P_tr"], self.plasma["CZ_tr"]), dim=1
                ).to(self.plasma["P"])
            if i == "w0":
                self.plasma["P"] = torch.cat(
                    (self.plasma["P"], self.plasma["Mt"]), dim=1
                ).to(self.plasma["P"])
                self.plasma["P_tr"] = torch.cat(
                    (self.plasma["P_tr"], self.plasma["Mt_tr"]), dim=1
                ).to(self.plasma["P"])

        self.plasma["S"] = self.plasma["P"] - self.plasma["P_tr"]
        self.plasma["residual"] = self.plasma["S"].abs().mean(axis=1)

        self.keys1D_derived["P"] = self.keys1D_derived["P_tr"] = 1
        self.keys0D["residual"] = 1

    def volume_integrate(self, var, dim=1):
        """
        If var in MW/m^3, this gives as output the MW/m^2 profile
        """

        return CALCtools.integrateQuadPoly(
            self.plasma["rmin"].repeat(dim, 1), var * self.plasma["volp"].repeat(dim, 1)
        ) / self.plasma["volp"].repeat(dim, 1)

    # ------------------------------------------------------------------
    # Toolset for post-processsing
    # ------------------------------------------------------------------

    def determinePerformance(self, nameRun="test", folder="~/scratch/"):
        folder = IOtools.expandPath(folder)

        # ************************************
        # Calculate state
        # ************************************

        self.calculate(None, nameRun=nameRun, folder=folder, storeInfo=True)

        # ************************************
        # Postprocessing
        # ************************************

        if self.TargetCalc == "tgyro":
            """
            Full targets
            """

            tuple_rho_indeces = ()
            for rho in self.model_current.rhosToSimulate:
                tuple_rho_indeces += (np.argmin(np.abs(rho - self.TGYROresults.rho)),)

            (
                self.plasma["Pfuse"],
                self.plasma["Pfusi"],
                self.plasma["Pie"],
                self.plasma["Prad_bremms"],
                self.plasma["Prad_sync"],
                self.plasma["Prad_line"],
            ) = TRANSPORTtools.full_targets(self.TGYROresults, tuple_rho_indeces)
            for i in ["Pfuse", "Pfusi", "Pie", "Prad_bremms", "Prad_sync", "Prad_line"]:
                self.plasma[i] = (
                    torch.from_numpy(self.plasma[i]).to(self.dfT).unsqueeze(0)
                )

            self.plasma["Pfus"] = (self.plasma["Pfuse"] + self.plasma["Pfusi"])[
                :, -1
            ] * 5.0
            self.plasma["Prad"] = (
                self.plasma["Prad_bremms"]
                + self.plasma["Prad_sync"]
                + self.plasma["Prad_line"]
            )[:, -1]

        else:
            self.plasma["Pfus"] = (
                self.volume_integrate(
                    (self.plasma["qfuse"] + self.plasma["qfusi"]) * 5.0
                )
                * self.plasma["volp"]
            )[:, -1]
            self.plasma["Prad"] = (
                self.volume_integrate(self.plasma["qrad"]) * self.plasma["volp"]
            )[:, -1]

            self.insertProfiles(
                self.profiles,
                writeFile=f"{folder}/input.gacode.new.powerstate",
                PositionInBatch=0,
                applyCorrections={
                    "Tfast_ratio": False,
                    "Ti_thermals": False,
                    "ni_thermals": False,
                    "recompute_ptot": False,
                    "ensureMachNumber": None,
                },
                insertPowers=True,
                rederive_profiles=False,
                reRead=False,
            )

        self.plasma["Pin"] = (
            (self.plasma["PauxE"] + self.plasma["PauxI"]) * self.plasma["volp"]
        )[:, -1]
        self.plasma["Q"] = self.plasma["Pfus"] / self.plasma["Pin"]

        for i in ["Pfus", "Pin", "Q", "Prad"]:
            self.keys0D[i] = 1
        self.unrepeat()

        # ************************************
        # Print Info
        # ************************************

        print(
            f"Q = {self.plasma['Q'].item():.2f} (Pfus = {self.plasma['Pfus'].item():.2f}MW, Pin = {self.plasma['Pin'].item():.2f}MW)"
        )

        print(f"Prad = {self.plasma['Prad'].item():.2f}MW")
