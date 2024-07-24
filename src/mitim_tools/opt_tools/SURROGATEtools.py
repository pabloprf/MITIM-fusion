import os
import torch
import gpytorch
import botorch
import contextlib
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.opt_tools import BOTORCHtools
from mitim_tools.opt_tools.utils import BOgraphics
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools.misc_tools.CONFIGread import read_verbose_level
from IPython import embed

verbose_level = read_verbose_level()

UseCUDAifAvailable = True

# ---------------------------------------------------------------------------------
# 	Model Class
# ---------------------------------------------------------------------------------


class surrogate_model:
    """
    This is where each of the fittings take place.
    Data is given unnormalized. The GP normalizes the data if requested and fits the model for trainY
    normalized as well.

    dfT is the default torch tensor to use for .to() (device and type)

    """

    def __init__(
        self,
        Xor,
        Yor,
        Yvaror,
        surrogate_parameters,
        output=None,
        output_transformed=None,
        bounds=None,
        avoidPoints=[],
        dfT=None,
        surrogateOptions={},
        FixedValue=False,
        fileTraining=None,
    ):
        """
        Noise is variance here (square of standard deviation).
        """

        torch.manual_seed(0)

        self.avoidPoints = avoidPoints
        self.output = output
        self.output_transformed = output_transformed
        self.surrogateOptions = surrogateOptions
        self.dfT = dfT
        self.surrogate_parameters = surrogate_parameters
        self.bounds = bounds
        self.FixedValue = FixedValue
        self.fileTraining = fileTraining

        if self.dfT is None:
            self.dfT = torch.randn(
                (2, 2),
                dtype=torch.double,
                device=torch.device(
                    "cpu"
                    if ((not UseCUDAifAvailable) or (not torch.cuda.is_available()))
                    else "cuda"
                ),
            )

        self.train_X = torch.from_numpy(Xor).to(self.dfT)
        self.train_Y = torch.from_numpy(Yor).to(self.dfT)

        # Extend noise if needed
        if isinstance(Yvaror, float) or len(Yvaror.shape) == 1:
            print(
                f"\t- Noise (variance) has one value only ({Yvaror}), assuming constant for all samples and outputs in absolute terms",
                verbose=verbose_level,
            )
            Yvaror = Yor * 0.0 + Yvaror

        self.train_Yvar = torch.from_numpy(Yvaror).to(self.dfT)

        # ---------- Print ----------
        print("\t- Surrogate options:", verbose=verbose_level)
        for i in self.surrogateOptions:
            print(f"\t\t{i:20} = {self.surrogateOptions[i]}", verbose=verbose_level)

        # --------------------------------------------------------------------
        # Eliminate points if needed (not from the "added" set)
        # --------------------------------------------------------------------

        if len(self.avoidPoints) > 0:
            print(
                f"\t- Fitting without considering points: {self.avoidPoints}",
                verbose=verbose_level,
                typeMsg="w",
            )

            self.train_X = torch.Tensor(
                np.delete(self.train_X, self.avoidPoints, axis=0)
            ).to(self.dfT)
            self.train_Y = torch.Tensor(
                np.delete(self.train_Y, self.avoidPoints, axis=0)
            ).to(self.dfT)
            self.train_Yvar = torch.Tensor(
                np.delete(self.train_Yvar, self.avoidPoints, axis=0)
            ).to(self.dfT)

        # ------------------------------------------------------------------------------------
        # Input and Outcome transform (PHYSICS)
        # ------------------------------------------------------------------------------------

        dimY = self.train_Y.shape[-1]

        input_transform_physics = BOTORCHtools.Transformation_Inputs(
            self.output, self.surrogate_parameters
        )
        outcome_transform_physics = BOTORCHtools.Transformation_Outcomes(
            dimY, self.output, self.surrogate_parameters
        )

        dimTransformedDV_x = input_transform_physics(self.train_X).shape[-1]
        dimTransformedDV_y = dimY

        # -------------------------------------------------------------------------------------
        # Add points from file
        # -------------------------------------------------------------------------------------

        # Points to be added from file
        if (self.surrogateOptions["extrapointsFile"] is not None) and (self.output is not None) and (self.output in self.surrogateOptions["extrapointsModels"]):

            print(
                f"\t* Requested extension of training set by points in file {self.surrogateOptions['extrapointsFile']}"
            )

            df = pd.read_csv(self.surrogateOptions["extrapointsFile"])
            df_model = df[df['Model'] == self.output]

            # Check 1: Do the points for this output share the same x_names?
            if df_model['x_names'].nunique() > 1:
                print("Different x_names for points in the file, prone to errors", typeMsg='q')

            # Check 2: Is it consistent with the x_names of this run?
            x_names = df_model['x_names'].apply(ast.literal_eval).iloc[0]
            x_names_check = self.surrogate_parameters['physicsInformedParamsComplete'][self.output]
            if x_names != x_names_check:
                print("x_names in file do not match the ones in this run, prone to errors", typeMsg='q')            


            self.train_Y_added = torch.from_numpy(df_model['y'].to_numpy()).unsqueeze(-1).to(self.dfT)
            self.train_Yvar_added = torch.from_numpy(df_model['yvar'].to_numpy()).unsqueeze(-1).to(self.dfT)
    
            x = []
            for i in range(len(x_names)):
                x.append(df_model[f'x{i}'].to_numpy())
            self.train_X_added_full = torch.from_numpy(np.array(x).T).to(self.dfT)

            self.train_X_added = (
                self.train_X_added_full[:, :dimTransformedDV_x] if self.train_X_added_full.shape[-1] > dimTransformedDV_x else self.train_X_added_full
            )

        else:
            if self.fileTraining is not None:
                train_X_Complete, _ = self.surrogate_parameters["transformationInputs"](
                    self.train_X,
                    self.output,
                    self.surrogate_parameters,
                    self.surrogate_parameters["physicsInformedParamsComplete"],
                )
                dimTransformedDV_x_full = train_X_Complete.shape[-1]
            else:
                dimTransformedDV_x_full = self.train_X.shape[-1]

            self.train_X_added_full = torch.empty((0, dimTransformedDV_x_full))
            self.train_X_added = torch.empty((0, dimTransformedDV_x))
            self.train_Y_added = torch.empty((0, dimTransformedDV_y))
            self.train_Yvar_added = torch.empty((0, dimTransformedDV_y))
            
        # --------------------------------------------------------------------------------------
        # Make sure that very small variations are not captured
        # --------------------------------------------------------------------------------------

        if (self.train_X_added.shape[0] > 0) and (self.train_X.shape[0] > 1):
            self.ensureMinimalVariationSuppressed(input_transform_physics)

        # --------------------------------------------------------------------------------------
        # Make sure at least 2 points
        # --------------------------------------------------------------------------------------

        if self.train_X.shape[0] + self.train_X_added.shape[0] == 1:
            factor = 1.2
            print(
                f"\t- This objective had only one point, adding a point with linear interpolation (trick for mitim targets only), {factor}",
                typeMsg="w",
            )
            self.train_X = torch.cat((self.train_X, self.train_X * factor))
            self.train_Y = torch.cat((self.train_Y, self.train_Y * factor))
            self.train_Yvar = torch.cat((self.train_Yvar, self.train_Yvar * factor))

        # -------------------------------------------------------------------------------------
        # Check minimum noises
        # -------------------------------------------------------------------------------------

        self.ensureMinimumNoise()

        # -------------------------------------------------------------------------------------
        # Write file with surrogate if there are transformations
        # -------------------------------------------------------------------------------------

        if (self.fileTraining is not None) and (
            self.train_X.shape[0] + self.train_X_added.shape[0] > 0
        ):
            self.writeFileTraining(input_transform_physics, outcome_transform_physics)

        # -------------------------------------------------------------------------------------
        # Input and Outcome transform (NORMALIZATIONS)
        # -------------------------------------------------------------------------------------

        input_transform_normalization = botorch.models.transforms.input.Normalize(
            dimTransformedDV_x, bounds=None
        )
        output_transformed_standardization = (
            botorch.models.transforms.outcome.Standardize((dimTransformedDV_y))
        )

        # Obtain normalization constants now (although during training this is messed up, so needed later too)
        self.normalization_pass(
            input_transform_physics,
            input_transform_normalization,
            outcome_transform_physics,
            output_transformed_standardization,
        )

        # ------------------------------------------------------------------------------------
        # Combine transformations in chain of PHYSICS + NORMALIZATION
        # ------------------------------------------------------------------------------------

        input_transform = botorch.models.transforms.input.ChainedInputTransform(
            tf1=input_transform_physics, tf2=input_transform_normalization
        )

        outcome_transform = BOTORCHtools.ChainedOutcomeTransform(
            tf1=outcome_transform_physics, tf2=output_transformed_standardization
        )

        self.variables = (
            self.surrogate_parameters["physicsInformedParams"][self.output]
            if (
                (self.output is not None)
                and ("physicsInformedParams" in self.surrogate_parameters)
                and (self.surrogate_parameters["physicsInformedParams"] is not None)
            )
            else None
        )

        # *************************************************************************************
        # Model
        # *************************************************************************************

        print(
            f'\t- Initializing model{" for "+self.output_transformed if (self.output_transformed is not None) else ""}',
            verbose=verbose_level,
        )

        """
        self.train_X contains the untransformed of this specific run:   (batch1, dimX)
        self.train_X_added contains the transformed of the table:       (batch2, dimXtr)
        """
        self.gpmodel = BOTORCHtools.ExactGPcustom(
            self.train_X,
            self.train_Y,
            self.train_Yvar,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            surrogateOptions=self.surrogateOptions,
            variables=self.variables,
            train_X_added=self.train_X_added,
            train_Y_added=self.train_Y_added,
            train_Yvar_added=self.train_Yvar_added,
        )

    def normalization_pass(
        self,
        input_transform_physics,
        input_transform_normalization,
        outcome_transform_physics,
        outcome_transform_normalization,
    ):
        input_transform_normalization.training = True
        outcome_transform_normalization.training = True
        outcome_transform_normalization._is_trained = torch.tensor(False)

        train_X_transformed = torch.cat(
            (input_transform_physics(self.train_X), self.train_X_added), axis=0
        )
        y, yvar = outcome_transform_physics(self.train_X, self.train_Y, self.train_Yvar)
        train_Y_transformed = torch.cat((y, self.train_Y_added), axis=0)
        train_Yvar_transformed = torch.cat((yvar, self.train_Yvar_added), axis=0)

        train_X_transformed_norm = input_transform_normalization(train_X_transformed)
        (
            train_Y_transformed_norm,
            train_Yvar_transformed_norm,
        ) = outcome_transform_normalization(train_Y_transformed, train_Yvar_transformed)

        # Make sure they are not on training mode
        input_transform_normalization.training = False
        outcome_transform_normalization.training = False
        outcome_transform_normalization._is_trained = torch.tensor(True)

    def fit(self):
        print(
            f"\t- Fitting model to {self.train_X.shape[0]+self.train_X_added.shape[0]} points"
        )

        # ---------------------------------------------------------------------------------------------------
        # Define loss Function to minimize
        # ---------------------------------------------------------------------------------------------------

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.gpmodel.likelihood, self.gpmodel
        )

        # ---------------------------------------------------------------------------------------------------
        # Prepare for training
        # ---------------------------------------------------------------------------------------------------

        if self.gpmodel.train_inputs[0].dtype == torch.float64:
            self.gpmodel = self.gpmodel.double()
            self.gpmodel.likelihood = self.gpmodel.likelihood.double()

        if self.gpmodel.train_inputs[0].device.type == "cuda":
            self.gpmodel = self.gpmodel.cuda()
            self.gpmodel.likelihood = self.gpmodel.likelihood.cuda()

        """
		---------------------------------------------------------------------------------------------------
			TRAINING
		---------------------------------------------------------------------------------------------------
		"""

        # Train always in physics-transformed space, to enable mitim re-use training from file
        with fundamental_model_context(self):
            track_fval = self.perform_model_fit(mll)

        # ---------------------------------------------------------------------------------------------------
        # Asses optimization
        # ---------------------------------------------------------------------------------------------------
        self.assess_optimization(track_fval)

        # ---------------------------------------------------------------------------------------------------
        # Go back to definining the right normalizations, because the optimizer has to work on training mode...
        # ---------------------------------------------------------------------------------------------------

        self.normalization_pass(
            self.gpmodel.input_transform["tf1"],
            self.gpmodel.input_transform["tf2"],
            self.gpmodel.outcome_transform["tf1"],
            self.gpmodel.outcome_transform["tf2"],
        )

    def perform_model_fit(self, mll):
        self.gpmodel.train()
        self.gpmodel.likelihood.train()
        mll.train()

        # ---------------------------------------------------------------------------------------------------
        # Fit
        # ---------------------------------------------------------------------------------------------------

        # Approx MLL ---------------------------------------
        (train_x,) = mll.model.train_inputs
        approx_mll = len(train_x) > 2000
        if approx_mll:
            print(
                f"\t* Using approximate MLL because x has {len(train_x)} elements",
                verbose=verbose_level,
            )
        # --------------------------------------------------

        # Store first MLL value
        track_fval = [
            -mll.forward(mll.model(*mll.model.train_inputs), mll.model.train_targets)
            .detach()
            .item()
        ]

        def callback(x, y, mll=mll):
            track_fval.append(y.fval)

        mll = botorch.fit.fit_gpytorch_mll(
            mll,
            max_attempts=20,
            kwargs={"track_iterations": True, "approx_mll": approx_mll},
            optimizer_kwargs={
                "method": "L-BFGS-B",
                "bounds": None,
                "options": {"disp": verbose_level == 5},
                "callback": callback,
            },
        )

        self.gpmodel.eval()
        self.gpmodel.likelihood.eval()
        mll.eval()

        print(
            f"\n\t- Marginal log likelihood went from {track_fval[0]:.3f} to {track_fval[-1]:.3f}"
        )

        return track_fval

    def predict(self, X, produceFundamental=False, nSamples=None):
        """
        This routine can be used to make predictions with models (individual and list), outside of the definitions
        of acquisitions and objectives. If the model is invididual, you can use produceFundamental to work on transformed (x and y) space.

        Inputs (batch1...N,dimX)
                - Receives unnormalized, untransformed x (i.e. raw DVs)
                - Provides unnormalized, untransformed y (i.e. raw OFs)
        Outputs (batch1...N,dimY)
                - Upper and lower bounds are +-2*std
                - Samples if nSamples not None
        """

        # Fast
        # with gpytorch.settings.fast_computations(), gpytorch.settings.fast_pred_samples(), \
        # 	 gpytorch.settings.fast_pred_var(), gpytorch.settings.lazily_evaluate_kernels():
        # Accurate
        # with 	gpytorch.settings.fast_computations(log_prob=False, solves=False, covar_root_decomposition=False), \
        # 		gpytorch.settings.eval_cg_tolerance(1E-6), gpytorch.settings.fast_pred_samples(state=False), gpytorch.settings.num_trace_samples(0):

        with (
            fundamental_model_context(self)
            if produceFundamental
            else contextlib.nullcontext(self)
        ) as surrogate_model:
            posterior = surrogate_model.gpmodel.posterior(X)

        mean = posterior.mean
        lower, upper = posterior.mvn.confidence_region()
        samples = (
            posterior.rsample(sample_shape=torch.Size([nSamples]))
            if nSamples is not None
            else None
        )

        if lower.dim() == 1:
            lower, upper = lower.unsqueeze(1), upper.unsqueeze(1)

        return mean, upper, lower, samples

    def writeFileTraining(self, input_transform_physics, outcome_transform_physics):
        """
        --------------------------------------------------------------------
        Write file with surrogate if there are transformations
                Note: USE TRANSFORMATIONS AT COMPLETE NUMBER (AFTER TRANSITIONS) for those in this run, but
                simply use the info that was in extra_points_file
        --------------------------------------------------------------------
        """

        # ------------------------------------------------------------------------------------------------------------------------
        # Transform the points without the added from file
        # ------------------------------------------------------------------------------------------------------------------------

        # I do not use directly input_transform_physics because I need all the columns, not of this specif iteration
        train_X_Complete, _ = self.surrogate_parameters["transformationInputs"](
            self.train_X,
            self.output,
            self.surrogate_parameters,
            self.surrogate_parameters["physicsInformedParamsComplete"],
        )

        train_Y, train_Yvar = outcome_transform_physics(
            self.train_X, self.train_Y, self.train_Yvar
        )

        dv_names_Complete = (
            self.surrogate_parameters["physicsInformedParamsComplete"][self.output]
            if (
                "physicsInformedParamsComplete" in self.surrogate_parameters
                and self.surrogate_parameters["physicsInformedParamsComplete"]
                is not None
            )
            else [i for i in self.bounds]
        )

        if self.train_X_added_full.shape[-1] < train_X_Complete.shape[-1]:
            print(
                "\t\t- Points from file have less input dimensions, extending with NaNs for writing new file",
                typeMsg="w",
            )
            self.train_X_added_full = torch.cat(
                (
                    self.train_X_added_full,
                    torch.full(
                        (
                            self.train_X_added_full.shape[0],
                            train_X_Complete.shape[-1]
                            - self.train_X_added_full.shape[-1],
                        ),
                        torch.nan,
                    ),
                ),
                axis=-1,
            )
        elif self.train_X_added_full.shape[-1] > train_X_Complete.shape[-1]:
            print(
                "\t\t- Points from file have more input dimensions, removing last dimensions for writing new file",
                typeMsg="w",
            )
            self.train_X_added_full = self.train_X_added_full[
                :, : train_X_Complete.shape[-1]
            ]

        x = torch.cat((self.train_X_added_full, train_X_Complete), axis=0)
        y = torch.cat((self.train_Y_added, train_Y), axis=0)
        yvar = torch.cat((self.train_Yvar_added, train_Yvar), axis=0)


        # ------------------------------------------------------------------------------------------------------------------------
        # Merged data with existing data frame and write
        # ------------------------------------------------------------------------------------------------------------------------

        new_df = create_df_portals(x,y,yvar,dv_names_Complete,self.output)

        if os.path.exists(self.fileTraining):

            # Load the existing DataFrame from the HDF5 file
            existing_df = pd.read_csv(self.fileTraining)

            # Concatenate the existing DataFrame with the new DataFrame
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        else:

            combined_df = new_df

        # Save the combined DataFrame back to the file
        combined_df.to_csv(self.fileTraining, index=False)

    # --------------------------
    # PLOTTING AND POST-ANALYSIS
    # --------------------------
    def plot(*args, **kwargs):
        BOgraphics.plot_surrogate_model(*args, **kwargs)

    def plotSensitivities(*args, **kwargs):
        BOgraphics.plotSensitivities_surrogate_model(*args, **kwargs)

    def plotTraining(*args, **kwargs):
        BOgraphics.plotTraining_surrogate_model(*args, **kwargs)

    def localBehavior(*args, **kwargs):
        BOgraphics.localBehavior_surrogate_model(*args, **kwargs)

    def localBehavior_scan(*args, **kwargs):
        BOgraphics.localBehavior_scan_surrogate_model(*args, **kwargs)

    # --------------------------
    # Checks
    # --------------------------

    def testTraining(
        self,
        plotYN=False,
        figs=None,
        x_next=None,
        y_next=None,
        ystd_next=None,
        axs=None,
        plotsPerFigure=20,
        ylabels=None,
        stds=2,
    ):
        """
        Note that evaluating the error in relative and in absolute is by definition different, becasue
        the transformation is not a simple multiplication, I substract the mean

        """

        # --- Actually be trained on ---
        xT, y, yvar = self.train_X, self.train_Y, self.train_Yvar

        yPredicted, yU, yL, _ = self.predict(xT)

        y = y.cpu().numpy()
        yPredicted = yPredicted.detach().cpu().numpy()
        yL = yL.detach().cpu().numpy()
        yU = yU.detach().cpu().numpy()

        # --- Next points ---
        if x_next is not None:
            yPredicted_next, yU_next, yL_next, _ = self.predict(x_next)
            x_next = x_next.cpu().numpy()
            yPredicted_next = yPredicted_next.detach().cpu().numpy()
            yL_next = yL_next.detach().cpu().numpy()
            yU_next = yU_next.detach().cpu().numpy()

        # --- Print stuff ---
        maxError = np.zeros(y.shape[1])
        for j in range(y.shape[1]):
            for i in range(y.shape[0]):
                err = (
                    np.abs((y[i, j] - yPredicted[i, j]) / y[i, j]) * 100.0
                    if y[i, j] != 0.0
                    else 0.0
                )
                # if printYN and err>5.0:
                # 	print(f'\t* Trained point #{i}, y({j})={y[i,j]:.3f}, y_pred({j})={yPredicted[i,j]:.3f} ({err:.2f}% off)',typeMsg='w')
                maxError[j] = np.max([err, maxError[j]])

        # --- Plot stuff ---
        if plotYN:
            if axs is None:
                if figs is None:
                    plt.ion()
                    figs = [plt.figure()]
                    plotsPerFigure = y.shape[1]

                axs = []
                i = -1
                for i in range(len(figs) - 1):
                    axs.extend(
                        GRAPHICStools.producePlotsGrid(
                            plotsPerFigure, fig=figs[i], hspace=0.4, wspace=0.4
                        )
                    )
                axs.extend(
                    GRAPHICStools.producePlotsGrid(
                        y.shape[1] - plotsPerFigure * (i + 1),
                        fig=figs[i + 1],
                        hspace=0.4,
                        wspace=0.4,
                    )
                )

            for j in range(y.shape[1]):
                ax = axs[j]

                ax.plot(y[:, j], yPredicted[:, j], "*", c="b", markersize=5)

                # Error definitions: +/- confidence bound (2*std)
                yerr = [
                    (yPredicted[:, j] - yL[:, j]) / 2 * stds,
                    (yU[:, j] - yPredicted[:, j]) / 2 * stds,
                ]
                xerr = [
                    stds * np.sqrt(yvar[:, j].cpu().numpy()),
                    stds * np.sqrt(yvar[:, j].cpu().numpy()),
                ]

                ax.errorbar(
                    y[:, j],
                    yPredicted[:, j],
                    c="b",
                    yerr=yerr,
                    xerr=xerr,
                    capsize=5.0,
                    fmt="none",
                )
                ax.plot(
                    [np.min(y[:, j]), np.max(y[:, j])],
                    [np.min(y[:, j]), np.max(y[:, j])],
                    c="k",
                    ls="--",
                    lw=0.5,
                )

                ax.set_xlabel(
                    f'{ylabels[j] if ylabels is not None else f"y({j})"} evaluated'
                )
                ax.set_ylabel(
                    f'{ylabels[j] if ylabels is not None else f"y({j})"} predicted'
                )
                # ax.set_title(f'y({j}) Training set, {self.output_transformed}')

                colorError = (
                    "r" if maxError[j] > 2.5 else "k" if maxError[j] > 0.25 else "g"
                )

                ax.text(
                    0.45,
                    0.05,
                    f"max error = {maxError[j]:.2f}%",
                    color=colorError,
                    fontsize=6,
                    transform=ax.transAxes,
                )

                if x_next is not None:
                    # Error definitions: +/- confidence bound
                    yerr_next = [
                        (yPredicted_next[:, j] - yL_next[:, j]) / 2 * stds,
                        (yU_next[:, j] - yPredicted_next[:, j]) / 2 * stds,
                    ]

                    if (y_next is not None) and (not np.isinf(y_next[:, j]).any()):
                        yTrain = y_next[:, j]
                        xerr_next = [stds * ystd_next[:, j], stds * ystd_next[:, j]]
                        ax.errorbar(
                            yTrain,
                            yPredicted_next[:, j],
                            c="g",
                            xerr=xerr_next,
                            yerr=yerr_next,
                            capsize=5.0,
                            fmt="o",
                        )
                    else:
                        ax.plot(
                            yPredicted_next[:, j],
                            yPredicted_next[:, j],
                            "s",
                            c="g",
                            markersize=5,
                        )

            return axs

    def ensureMinimalVariationSuppressed(self, input_transform_physics, thr=1e-6):
        """
        In some cases, the added data from file might have extremely small variations in some of the fixed
        inputs, as compared to the trained data of this run. In such a case, modify this variation
        """

        # Do dimensions of the non-added points change?
        x_transform = input_transform_physics(self.train_X)
        indecesUnchanged = torch.where(
            (x_transform.max(axis=0)[0] - x_transform.min(axis=0)[0])
            / x_transform.mean(axis=0)[0]
            < thr
        )[0]

        HasThisBeenApplied = 0

        for i in indecesUnchanged:
            if (
                (self.train_X_added[:, i] - x_transform[0, i]) / x_transform[0, i]
            ).abs().max() < thr:
                HasThisBeenApplied += 1
                for j in range(self.train_X_added.shape[0]):
                    self.train_X_added[j, i] = x_transform[0, i]

        if HasThisBeenApplied > 0:
            print(
                f"\t- Supression of small variations {thr:.1e} in added data applied to {HasThisBeenApplied} dims",
                typeMsg="w",
            )

    def ensureMinimumNoise(self):
        if ("MinimumRelativeNoise" in self.surrogateOptions) and (
            self.surrogateOptions["MinimumRelativeNoise"] is not None
        ):
            maxY = (
                self.train_Y.abs().max()
                if self.train_Y.shape[0] > 0
                else torch.tensor(0.0)
            )
            maxY_added = (
                self.train_Y_added.abs().max()
                if self.train_Y_added.shape[0] > 0
                else torch.tensor(0.0)
            )
            maxVal = torch.max(maxY, maxY_added)

            minstd_constraint = maxVal * self.surrogateOptions["MinimumRelativeNoise"]

            # Actual points
            if self.train_Y.shape[0] > 0:
                std = self.train_Yvar**0.5

                if std.min().item() < minstd_constraint:
                    print(
                        f"\t* std for output {self.output} has been clipped b/c std_min = {self.surrogateOptions['MinimumRelativeNoise']*100:.2f}%, {minstd_constraint:.1e}; and had {std.min().item():.1e} ",
                        typeMsg="w",
                    )
                    std = std.clip(minstd_constraint)

                self.train_Yvar = std**2

            # Added
            if self.train_Y_added.shape[0] > 0:
                std = self.train_Yvar_added**0.5

                if std.min().item() < minstd_constraint:
                    print(
                        f"\t- std for output {self.output} has been clipped (added points) b/c std_min = {self.surrogateOptions['MinimumRelativeNoise']*100:.2f}% ({minstd_constraint:.1e}) and had {std.min().item():.1e} ",
                        typeMsg="w",
                    )
                    std = std.clip(minstd_constraint)

                self.train_Yvar_added = std**2

    def assess_optimization(self, track_fval):
        self.losses = {
            "losses": track_fval,
            "loss_ini": track_fval[0],
            "loss_final": track_fval[-1],
        }

        print("\t- Fitting summary:", verbose=verbose_level)
        if verbose_level in [4, 5]:
            print("\t\t* Model raw parameters:")
            for param_name, param in self.gpmodel.named_parameters():
                BOgraphics.printParam(param_name, param, extralab="\t\t\t")

            print("\t\t* Model constraints:")
            dictParam = {}
            for constraint_name, constraint in self.gpmodel.named_constraints():
                BOgraphics.printConstraint(constraint_name, constraint, extralab="\t\t")
                dictParam[constraint_name.replace("_constraint", "")] = constraint

            """
			This is an "inconvenient" way to calculate the actual parameters https://docs.gpytorch.ai/en/stable/examples/00_Basic_Usage/Hyperparameters.html?highlight=constraints#How-do-constraints-work?
			but I like it.
			"""
            print("\t\t* Model actual parameters:")
            for param_name, param in self.gpmodel.named_parameters():
                if param_name in dictParam:
                    param = dictParam[param_name].transform(param)
                param_name = param_name.replace("raw_", "actual_")

                BOgraphics.printParam(param_name, param, extralab="\t\t\t")


# Class to call the model posterior directly on transformed space (x and y)
class fundamental_model_context(object):
    def __init__(self, surrogate_model):
        self.surrogate_model = surrogate_model

    def __enter__(self):
        # Works for individual models, not ModelList
        self.surrogate_model.gpmodel.input_transform.tf1.flag_to_evaluate = False
        self.surrogate_model.gpmodel.outcome_transform.tf1.flag_to_evaluate = False

        return self.surrogate_model

    def __exit__(self, *args):
        self.surrogate_model.gpmodel.input_transform.tf1.flag_to_evaluate = True
        self.surrogate_model.gpmodel.outcome_transform.tf1.flag_to_evaluate = True

def create_df_portals(x, y, yvar, x_names, output, max_x = 20):

    new_data = []
    for i in range(x.shape[0]):
        data_point = {
            'Model': output,
            'y': y[i,:].item(),
            'yvar': yvar[i,:].item(),
            'x_names': x_names,
        }
        for j in range(x.shape[1]):
            data_point[f'x{j}'] = x[i,j].item()
        new_data.append(data_point)

    # Create a DataFrame for the new data
    new_df = pd.DataFrame(new_data)

    return new_df

