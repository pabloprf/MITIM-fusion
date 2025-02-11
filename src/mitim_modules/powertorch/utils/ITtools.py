import torch
import shutil
from mitim_tools.opt_tools.optimizers import optim
from mitim_modules.powertorch.physics import TRANSPORTtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools import IOtools
from IPython import embed

def fluxMatchRoot(self, algorithm_options={},jac_ad=True, bounds=None):
    
    dimX = (self.plasma["rho"].shape[-1]-1)*len(self.ProfilesPredicted)

    Xopt, Yopt = [], []

    folder = algorithm_options.get("folder", '~/scratch/')
    
    if bounds is not None:
        bound_transform = optim.logistic(l=bounds[0, :], u=bounds[1, :])


    cont = 0
    if algorithm_options.get('storeValues',True):
        def evaluator(x, bound_transform=bound_transform):
            """
            Notes:
                - x comes extended, batch*dim
                - y must be returned extended as well, batch*dim
            """
            nonlocal cont

            X = x.view((x.shape[0] // dimX, dimX))  # [batch*dim]->[batch,dim]

            # Transform from infinite bounds
            X = bound_transform.transform(X)

            # Evaluate source term
            _, _, y, _ = self.calculate(X,folder=f'{folder}/ev{cont}', nameRun=f"ev{cont}", evaluation_number=cont)
            cont += 1

            # Compress again  [batch,dim]->[batch*dim]
            y = y.view(x.shape)

            # Store values
            Xopt.append(X.clone().detach()[0,...])
            Yopt.append(y.abs().clone().detach())

            return y
    else:
        def evaluator(x, bound_transform=bound_transform):
            """
            Notes:
                - x comes extended, batch*dim
                - y must be returned extended as well, batch*dim
            """
            nonlocal cont

            X = x.view((x.shape[0] // dimX, dimX))  # [batch*dim]->[batch,dim]

            # Transform from infinite bounds
            X = bound_transform.transform(X)

            # Evaluate source term
            _, _, y, _ = self.calculate(X,folder=f'{folder}/ev{cont}', nameRun=f"ev{cont}", evaluation_number=cont)
            cont += 1

            # Compress again  [batch,dim]->[batch*dim]
            y = y.view(x.shape)

            return y

    # ---- Initial guess

    x0 = torch.Tensor().to(self.dfT)
    for c, i in enumerate(self.ProfilesPredicted):
        x0 = torch.cat((x0, self.plasma[f"aL{i}"][:, 1:].detach()), dim=1)

    x0 = bound_transform.untransform(x0)

    # **** Optimize ****
    _ = optim.powell(evaluator, x0, None, algorithm_options=algorithm_options,jac_ad=jac_ad)
    # ******************

    if algorithm_options.get('storeValues',True):
        Xopt = torch.stack(Xopt)
        Yopt = torch.stack(Yopt)
    else:
        Xopt, Yopt = torch.Tensor(), torch.Tensor()

    #Xopt = bound_transform.untransform(Xopt)

    return Xopt, Yopt

def fluxMatchSimpleRelax(self, algorithm_options={}, bounds=None):
    
    # Default options
    tol = algorithm_options.get("tol", 1e-3)
    maxiter = algorithm_options.get("maxiter", 1e5)
    relax = algorithm_options.get("relax", 0.001)
    dx_max = algorithm_options.get("dx_max", 0.05)
    dx_max_abs = algorithm_options.get("dx_max_abs", None)
    dx_min_abs = algorithm_options.get("dx_min_abs", None)
    print_each = algorithm_options.get("print_each", 1e2)
    MainFolder = algorithm_options.get("folder", "~/scratch/")
    storeValues = algorithm_options.get("storeValues", True)
    namingConvention = algorithm_options.get("namingConvention", "powerstate_sr_ev")

    MainFolder = IOtools.expandPath(MainFolder)

    def evaluator(X, cont=0):
        nameRun = f"{namingConvention}_{cont}"
        folder = MainFolder /  f"{namingConvention}_{cont}"
        if issubclass(self.TransportOptions["transport_evaluator"], TRANSPORTtools.power_transport):
            (folder / "model_complete").mkdir(parents=True, exist_ok=True)

        # ***************************************************************************************************************
        # Calculate
        # ***************************************************************************************************************

        folderTGYRO = folder / "model_complete"
        QTransport, QTarget, _, _ = self.calculate(
            X, nameRun=nameRun, folder=folderTGYRO, evaluation_number=cont
        )

        # Save state so that I can check initializations
        if issubclass(self.TransportOptions["transport_evaluator"], TRANSPORTtools.power_transport):
            self.save(folder / "powerstate.pkl")
            shutil.copy2(folderTGYRO / "input.gacode", folder)

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
        maxiter=maxiter,
        relax=relax,
        dx_max=dx_max,
        bounds=bounds,
        dx_max_abs = dx_max_abs,
        dx_min_abs = dx_min_abs,
        print_each=print_each,
        storeValues=storeValues,
    )

    return Xopt, Yopt


def fluxMatchPicard(self, tol=1e-6, maxiter=1e3):
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
    _ = optim.picard(evaluator, x0, tol=tol, maxiter=maxiter)
    # ******************
    
    Xopt = torch.stack(Xopt)
    Yopt = torch.stack(Yopt)

    return Xopt, Yopt
