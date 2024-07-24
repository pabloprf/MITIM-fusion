import torch
import copy
import os
from mitim_tools.opt_tools.optimizers import optim
from mitim_modules.powertorch.physics import TRANSPORTtools
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed

def fluxMatchRoot(self, algorithmOptions={}):
    
    dimX = (self.plasma["rho"].shape[-1]-1)*len(self.ProfilesPredicted)

    Xopt, Yopt = [], []

    if algorithmOptions.get('storeValues',True):
        def evaluator(x):
            """
            Notes:
                - x comes extended, batch*dim
                - y must be returned extended as well, batch*dim
            """

            X = x.view((x.shape[0] // dimX, dimX))  # [batch*dim]->[batch,dim]

            # Evaluate source term
            _, _, y, _ = self.calculate(X)

            # Compress again  [batch,dim]->[batch*dim]
            y = y.view(x.shape)

            # Store values
            Xopt.append(X.clone().detach()[0,...])
            Yopt.append(y.abs().clone().detach())

            return y
    else:
        def evaluator(x):
            """
            Notes:
                - x comes extended, batch*dim
                - y must be returned extended as well, batch*dim
            """

            X = x.view((x.shape[0] // dimX, dimX))  # [batch*dim]->[batch,dim]

            # Evaluate source term
            _, _, y, _ = self.calculate(X)

            # Compress again  [batch,dim]->[batch*dim]
            y = y.view(x.shape)

            return y

    # ---- Initial guess

    x0 = torch.Tensor().to(self.dfT)
    for c, i in enumerate(self.ProfilesPredicted):
        x0 = torch.cat((x0, self.plasma[f"aL{i}"][:, 1:].detach()), dim=1)

    # **** Optimize ****
    _ = optim.powell(evaluator, x0, None, algorithmOptions=algorithmOptions)
    # ******************

    if algorithmOptions.get('storeValues',True):
        Xopt = torch.stack(Xopt)
        Yopt = torch.stack(Yopt)
    else:
        Xopt, Yopt = torch.Tensor(), torch.Tensor()

    return Xopt, Yopt

def fluxMatchSimpleRelax(self, algorithmOptions={}, bounds=None):
    
    # Default options
    tol = algorithmOptions.get("tol", 1e-3)
    max_it = algorithmOptions.get("max_it", 1e5)
    relax = algorithmOptions.get("relax", 0.001)
    dx_max = algorithmOptions.get("dx_max", 0.05)
    print_each = algorithmOptions.get("print_each", 1e2)
    MainFolder = algorithmOptions.get("MainFolder", "~/scratch/")
    storeValues = algorithmOptions.get("storeValues", True)
    namingConvention = algorithmOptions.get("namingConvention", "powerstate_sr_ev")

    def evaluator(X, cont=0):
        nameRun = f"{namingConvention}{cont}"
        folder = f"{MainFolder}/{nameRun}/"
        if (not os.path.exists(folder)) and (self.TransportOptions["transport_evaluator"] == TRANSPORTtools.tgyro_model):
            os.system(f"mkdir -p {folder}/model_complete/")

        # ***************************************************************************************************************
        # Calculate
        # ***************************************************************************************************************

        folderTGYRO = f"{folder}/model_complete/"

        QTransport, QTarget, _, _ = self.calculate(
            X, nameRun=nameRun, folder=folderTGYRO, evaluation_number=cont
        )

        # Save state so that I can check initializations
        if self.TransportOptions["transport_evaluator"] == TRANSPORTtools.tgyro_model:
            self.save(f"{folder}/powerstate.pkl")
            os.system(f"cp {folderTGYRO}/input.gacode {folder}/.")

        return QTransport, QTarget

    # Concatenate the input gradients
    x0 = torch.Tensor().to(self.plasma["aLte"])
    for c, i in enumerate(self.ProfilesPredicted):
        x0 = torch.cat((x0, self.plasma[f"aL{i}"][:, 1:].detach()), dim=1)

    # Make sure is properly batched
    x0 = x0.view(
        (
            self.plasma["rho"].shape[0],
            (self.plasma["rho"].shape[1] - 1) * len(self.ProfilesPredicted),
        )
    )

    # Optimize
    Xopt, Yopt = optim.relax(
        evaluator,
        x0,
        tol=tol,
        max_it=max_it,
        relax=relax,
        dx_max=dx_max,
        bounds=bounds,
        print_each=print_each,
        storeValues=storeValues,
    )

    return Xopt, Yopt


def fluxMatchPicard(self, tol=1e-6, max_it=1e3):
    """
    I should figure out what to do with the cases with too much transport that never converge
    """

    Xopt, Yopt = [], []
    def evaluator(x):
        """
        Notes:
            - x comes [batch,dim] already
            - y must be returned [batch,dim] as well
        """

        # Evaluate source term
        _, _, y, _ = self.calculate(x)

        # Store values
        Xopt.append(x.clone().detach()[0,...])
        Yopt.append(y.abs().clone().detach())

        return y

    # Concatenate the input gradients
    x0 = torch.Tensor().to(self.plasma["aLte"])
    for c, i in enumerate(self.ProfilesPredicted):
        x0 = torch.cat((x0, self.plasma[f"aL{i}"][:, 1:].detach()), dim=1)

    # **** Optimize ****
    _ = optim.picard(evaluator, x0, tol=tol, max_it=max_it)
    # ******************
    
    Xopt = torch.stack(Xopt)
    Yopt = torch.stack(Yopt)

    return Xopt, Yopt
