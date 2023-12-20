import datetime, torch, copy, os
from IPython import embed
from mitim_tools.misc_tools import IOtools
from mitim_tools.opt_tools.optimizers import optim

from mitim_tools.misc_tools.IOtools import printMsg as print


def fluxMatchRoot(self, extra_params={}):
    # ---- Function that provides fluxes
    def evaluator(X):
        # If no batch dimension add it
        X = X.unsqueeze(0) if X.dim() == 1 else X

        # Make sure that the shape is correct. This is useful when running flux matching in batches
        X = X.view(
            (
                self.plasma["rho"].shape[0],
                (self.plasma["rho"].shape[1] - 1) * len(self.ProfilesPredicted),
            )
        )

        s = copy.deepcopy(self)

        P_tr, P, _, r = s.calculate(X, extra_params=extra_params)

        if P.shape[0] > 1:
            P, P_tr = P.view(-1).unsqueeze(0), P_tr.view(-1).unsqueeze(0)

        return P, P_tr

    # ---- Initial guess

    x0 = torch.Tensor().to(self.plasma["aLte"])
    for c, i in enumerate(self.ProfilesPredicted):
        x0 = torch.cat((x0, self.plasma[f"aL{i}"][:, 1:].detach()), dim=1)

    x0 = x0.view(
        (
            self.plasma["rho"].shape[0],
            (self.plasma["rho"].shape[1] - 1) * len(self.ProfilesPredicted),
        )
    )

    # ---- Optimize
    Xopt = optim.powell(evaluator, x0, None)

    Xopt = Xopt.view(
        (
            self.plasma["rho"].shape[0],
            (self.plasma["rho"].shape[1] - 1) * len(self.ProfilesPredicted),
        )
    )

    # ---- Evaluate state with the optimum gradients for flux matching
    _ = self.calculate(Xopt)


def fluxMatchPicard(self, tol=1e-6, max_it=1e3, extra_params={}):
    """
    I should figure out what to do with the cases with too much transport that never converge
    """

    # ---- Function that provides source term
    def evaluator(X):
        _, _, S, _ = self.calculate(X, extra_params=extra_params)
        return S

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
    Xopt = optim.picard(evaluator, x0, tol=tol, max_it=max_it)

    # Evaluate state with the optimum gradients
    _ = self.calculate(Xopt)


def fluxMatchSimpleRelax(self, algorithmOptions={}, bounds=None, extra_params={}):
    tol = algorithmOptions["tol"] if "tol" in algorithmOptions else 1e-3
    max_it = algorithmOptions["max_it"] if "max_it" in algorithmOptions else 1e5
    relax = algorithmOptions["relax"] if "relax" in algorithmOptions else 0.001
    dx_max = algorithmOptions["dx_max"] if "dx_max" in algorithmOptions else 0.05
    print_each = (
        algorithmOptions["print_each"] if "print_each" in algorithmOptions else 1e2
    )
    MainFolder = (
        algorithmOptions["MainFolder"]
        if "MainFolder" in algorithmOptions
        else "~/scratch/"
    )
    storeValues = (
        algorithmOptions["storeValues"] if "storeValues" in algorithmOptions else False
    )
    namingConvention = (
        algorithmOptions["namingConvention"]
        if "namingConvention" in algorithmOptions
        else "powerstate_sr_ev"
    )

    def evaluator(X, cont=0):
        nameRun = f"{namingConvention}{cont}"
        folder = f"{MainFolder}/{nameRun}/"
        if (not os.path.exists(folder)) and (
            "tgyro" in self.TransportOptions["TypeTransport"]
        ):
            os.system(f"mkdir {folder}")
            os.system(f"mkdir {folder}/model_complete/")

        # ***************************************************************************************************************
        # Calculate
        # ***************************************************************************************************************

        extra_params_model = copy.deepcopy(extra_params)
        extra_params_model["numPORTALS"] = f"{cont}"

        folderTGYRO = f"{folder}/model_complete/"

        QTransport, QTarget, _, _ = self.calculate(
            X, nameRun=nameRun, folder=folderTGYRO, extra_params=extra_params_model
        )

        # Save state so that I can check initializations
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


def fluxMatchPicard(self, tol=1e-6, max_it=1e3, extra_params={}):
    """
    I should figure out what to do with the cases with too much transport that never converge
    """

    # ---- Function that provides source term
    def evaluator(X):
        _, _, S, _ = self.calculate(X, extra_params=extra_params)
        return S

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
    Xopt = optim.picard(evaluator, x0, tol=tol, max_it=max_it)

    # Evaluate state with the optimum gradients
    _ = self.calculate(Xopt)


def fluxMatchPRFseq(self, algorithmOptions={}, bounds=None, extra_params={}):
    tol = algorithmOptions["tol"] if "tol" in algorithmOptions else 1e-3
    max_it = algorithmOptions["max_it"] if "max_it" in algorithmOptions else 1e5
    relax = algorithmOptions["relax"] if "relax" in algorithmOptions else 0.001
    dx_max = algorithmOptions["dx_max"] if "dx_max" in algorithmOptions else 0.05
    print_each = (
        algorithmOptions["print_each"] if "print_each" in algorithmOptions else 1e2
    )
    storeValues = (
        algorithmOptions["storeValues"] if "storeValues" in algorithmOptions else False
    )

    info_case = {"rho": self.plasma["rho"][0, 1:], "profs": self.ProfilesPredicted}

    def evaluator(X, cont=0):
        QTransport, QTarget, _, _ = self.calculate(X, extra_params=extra_params)
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
    optim.prfseq(
        evaluator,
        x0,
        tol=tol,
        max_it=max_it,
        relax=relax,
        dx_max=dx_max,
        bounds=bounds,
        print_each=print_each,
        info_case=info_case,
        storeValues=storeValues,
    )
