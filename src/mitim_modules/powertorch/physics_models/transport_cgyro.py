import copy
import shutil
import torch
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools, TGYROtools
from mitim_modules.powertorch.physics_models import transport_tgyro
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class cgyro_model(transport_tgyro.tgyro_model):
    def __init__(self, powerstate, **kwargs):
        super().__init__(powerstate, **kwargs)

    def evaluate(self):

        # Run original evaluator
        tgyro = self._evaluate_tglf_neo()

        # Run CGYRO trick
        powerstate_orig = self._trick_cgyro(tgyro)

        # Process results
        self._postprocess_results(tgyro, "cgyro_neo")

        # Some checks
        print("\t- Checking model modifications:")
        for r in ["QeMWm2_tr_turb", "QiMWm2_tr_turb", "Ce_tr_turb", "CZ_tr_turb", "MtJm2_tr_turb"]: #, "PexchTurb"]: #TODO: FIX
            print(f"\t\t{r}(tglf)  = {'  '.join([f'{k:.1e} (+-{ke:.1e})' for k,ke in zip(powerstate_orig.plasma[r][0][1:],powerstate_orig.plasma[r+'_stds'][0][1:]) ])}")
            print(f"\t\t{r}(cgyro) = {'  '.join([f'{k:.1e} (+-{ke:.1e})' for k,ke in zip(self.powerstate.plasma[r][0][1:],self.powerstate.plasma[r+'_stds'][0][1:]) ])}")

    # ************************************************************************************
    # Private functions for CGYRO evaluation
    # ************************************************************************************

    def _trick_cgyro(self, tgyro):

        FolderEvaluation_TGYRO = self.folder / "cgyro_neo"

        print("\t- Checking whether cgyro_neo folder exists and it was written correctly via cgyro_trick...")

        correctly_run = FolderEvaluation_TGYRO.exists()
        if correctly_run:
            print("\t\t- Folder exists, but was cgyro_trick run?")
            with open(FolderEvaluation_TGYRO / "mitim_flag", "r") as f:
                correctly_run = bool(float(f.readline()))

        if correctly_run:
            print("\t\t\t* Yes, it was", typeMsg="w")
        else:
            print("\t\t\t* No, it was run, repating process", typeMsg="i")

            # Remove cgyro_neo folder
            if FolderEvaluation_TGYRO.exists():
                IOtools.shutil_rmtree(FolderEvaluation_TGYRO)

            # Copy tglf_neo results
            shutil.copytree(self.folder / "tglf_neo", FolderEvaluation_TGYRO)

            # **********************************************************
            # CGYRO writter
            # **********************************************************

            # Write a flag indicating this was not performed
            with open(FolderEvaluation_TGYRO / "mitim_flag", "w") as f:
                f.write("0")

            self._cgyro_trick(FolderEvaluation_TGYRO)

            # Write a flag indicating this was performed, to avoid an issue that... the script crashes when it has copied tglf_neo, without cgyro_trick modification
            with open(FolderEvaluation_TGYRO / "mitim_flag", "w") as f:
                f.write("1")

        # Read TGYRO files and construct portals variables

        tgyro.read(label="cgyro_neo", folder=FolderEvaluation_TGYRO) 

        powerstate_orig = copy.deepcopy(self.powerstate)
        
        return powerstate_orig

    def _cgyro_trick(self,FolderEvaluation_TGYRO):

        # Print Information
        print(self._print_info())

        # Copy profiles so that later it is easy to grab all the input.gacodes that were evaluated
        self._profiles_to_store()

        # **************************************************************************************************************************
        # Evaluate CGYRO
        # **************************************************************************************************************************

        evaluateCGYRO(
            self.powerstate.TransportOptions["ModelOptions"]["extra_params"]["PORTALSparameters"],
            self.powerstate.TransportOptions["ModelOptions"]["extra_params"]["folder"],
            self.evaluation_number,
            FolderEvaluation_TGYRO,
            self.file_profs,
            self.powerstate.plasma["roa"][0,1:],
            self.powerstate.ProfilesPredicted,
        )

        # Make tensors
        for i in ["QeMWm2_tr_turb", "QiMWm2_tr_turb", "Ce_tr_turb", "CZ_tr_turb", "MtJm2_tr_turb"]:
            try:
                self.powerstate.plasma[i] = torch.from_numpy(self.powerstate.plasma[i]).to(self.powerstate.dfT).unsqueeze(0)
            except:
                pass

    def _print_info(self):

        txt = "\nFluxes to be matched by CGYRO ( TARGETS - NEO ):"

        for var, varn in zip(
            ["r/a  ", "rho  ", "a/LTe", "a/LTi", "a/Lne", "a/LnZ", "a/Lw0"],
            ["roa", "rho", "aLte", "aLti", "aLne", "aLnZ", "aLw0"],
        ):
            txt += f"\n{var}        = "
            for j in range(self.powerstate.plasma["rho"].shape[1] - 1):
                txt += f"{self.powerstate.plasma[varn][0,j+1]:.6f}   "

        for var, varn in zip(
            ["Qe (MW/m^2)", "Qi (MW/m^2)", "Ce (MW/m^2)", "CZ (MW/m^2)", "MtJm2 (J/m^2) "],
            ["QeMWm2", "QiMWm2", "Ce", "CZ", "MtJm2"],
        ):
            txt += f"\n{var}  = "
            for j in range(self.powerstate.plasma["rho"].shape[1] - 1):
                txt += f"{self.powerstate.plasma[varn][0,j+1]-self.powerstate.plasma[f'{varn}_tr_neo'][0,j+1]:.4e}   "

        return txt

"""
The CGYRO file must contain GB units, and the gb unit is MW/m^2, 1E19m^2/s
The CGYRO file must use particle flux. Convective transformation occurs later
"""

def evaluateCGYRO(PORTALSparameters, folder, numPORTALS, FolderEvaluation, unmodified_profiles, radii, ProfilesPredicted):
    print("\n ** CGYRO evaluation of fluxes has been requested before passing information to the STRATEGY module **",typeMsg="i",)

    if isinstance(numPORTALS, int):
        numPORTALS = str(numPORTALS)

    # ------------------------------------------------------------------------------------------------
    # Harcoded
    # ------------------------------------------------------------------------------------------------
    if PORTALSparameters['hardCodedCGYRO'] is not None:
        """
        train_sep is the number of initial runs in it#0 results file. Now, it's usually 1
        start_num is the number of the first iteration, usually 0
        trick_harcoded_f is the name of the file until the iteration number. E.g. 'example_run/Outputs/cgyro_results/iter_rmp_75_'

        e.g.:
            train_sep,start_num,last_one,trick_hardcoded_f = 1, 0,100, 'example_run/Outputs/cgyro_results/d3d_5chan_it_'

        """

        train_sep = PORTALSparameters["hardCodedCGYRO"]["train_sep"]
        start_num = PORTALSparameters["hardCodedCGYRO"]["start_num"]
        last_one = PORTALSparameters["hardCodedCGYRO"]["last_one"]
        trick_hardcoded_f = PORTALSparameters["hardCodedCGYRO"]["trick_hardcoded_f"]
    else:
        train_sep = None
        start_num = None
        last_one = None
        trick_hardcoded_f = None
    # ------------------------------------------------------------------------------------------------

    minErrorPercent = PORTALSparameters["percentError_stable"]
    Qi_criterion_stable = PORTALSparameters["Qi_criterion_stable"]
    percentNeo = PORTALSparameters["percentError"][1]
    useConvectiveFluxes = PORTALSparameters["useConvectiveFluxes"]

    try:
        impurityPosition = PROFILEStools.impurity_location(PROFILEStools.gacode_state(unmodified_profiles), PORTALSparameters["ImpurityOfInterest"])
    except ValueError:
        if 'nZ' in ProfilesPredicted:
            raise ValueError(f"Impurity {PORTALSparameters['ImpurityOfInterest']} not found in the profiles and needed for CGYRO evaluation")
        else:
            impurityPosition = 0
            print(f'\t- Impurity location not found. Using hardcoded value of {impurityPosition}')

    OriginalFimp = PORTALSparameters["fImp_orig"]

    cgyroing_file = (
        lambda file_cgyro, numPORTALS_this=0: cgyroing(
            FolderEvaluation,
            unmodified_profiles,
            numPORTALS,
            minErrorPercent,
            Qi_criterion_stable,
            useConvectiveFluxes,
            percentNeo,
            radii,
            OriginalFimp=OriginalFimp,
            evaluationsInFile=f"{numPORTALS_this}",
            impurityPosition=impurityPosition,
            file=file_cgyro,
        )
    )
    print(f"\t- Suggested function call for mitim evaluation {numPORTALS} (lambda for cgyroing):",typeMsg="i")
    cgyropath = IOtools.expandPath(folder, ensurePathValid=True) / 'Outputs' / 'cgyro_results' / f'cgyro_it_{numPORTALS}.txt'
    print(f"\tcgyroing_file('{cgyropath}')")

    print('\t- Then insert "exit" and RETURN', typeMsg="i")
    if (trick_hardcoded_f is None) or (int(numPORTALS) > last_one):
        embed()
    else:
        # ------------------------------------------------------------------
        # Hard-coded stuff for quick modifications
        # ------------------------------------------------------------------
        if int(numPORTALS) < train_sep:
            cgyroing_file(f"{trick_hardcoded_f}{start_num}.txt",numPORTALS_this=numPORTALS)
        else:
            cgyroing_file(f"{trick_hardcoded_f}{int(numPORTALS)-train_sep+1+start_num}.txt",numPORTALS_this=0)


def cgyroing(
    FolderEvaluation,
    unmodified_profiles,
    evaluations,
    minErrorPercent,
    Qi_criterion_stable,
    useConvectiveFluxes,
    percentNeo,
    radii,
    OriginalFimp=1.0,
    file=None,
    evaluationsInFile=0,
    impurityPosition=3,
):
    """
    Variables need to have dimensions of (evaluation,rho)
    """

    evaluations = np.array([int(i) for i in evaluations.split(",")])
    evaluationsInFile = np.array([int(i) for i in evaluationsInFile.split(",")])

    aLTe,aLTi,aLne,Q_gb,Qe,Qi,Ge,GZ,Mt,Pexch,QeE,QiE,GeE,GZE,MtE,PexchE,_,_ = readCGYROresults(file, radii)

    cont = 0
    for _ in evaluations:
        k = evaluationsInFile[cont]
        cont += 1

        print(f"\t- Modifying {IOtools.clipstr(FolderEvaluation)} with position {k} in CGYRO results file {IOtools.clipstr(file)}")

        # Get TGYRO
        tgyro = TGYROtools.TGYROoutput(FolderEvaluation,profiles=PROFILEStools.gacode_state(unmodified_profiles))

        # Quick checker of correct file
        wasThisTheCorrectRun(aLTe, aLTi, aLne, Q_gb, tgyro)

        transport_tgyro.modifyResults(
            Qe[k, :],
            Qi[k, :],
            Ge[k, :],
            GZ[k, :],
            Mt[k, :],
            Pexch[k, :],
            QeE[k, :],
            QiE[k, :],
            GeE[k, :],
            GZE[k, :],
            MtE[k, :],
            PexchE[k, :],
            tgyro,
            FolderEvaluation,
            minErrorPercent=minErrorPercent,
            useConvectiveFluxes=useConvectiveFluxes,
            Qi_criterion_stable=Qi_criterion_stable,
            percent_tr_neo=percentNeo,
            impurityPosition=impurityPosition,
            OriginalFimp=OriginalFimp,
        )


def wasThisTheCorrectRun(aLTe, aLTi, aLne, Q_gb, tgyro, ErrorRaised=0.005):
    print("\t- Checking that this was the correct run...")

    tgyro_new = copy.deepcopy(tgyro)
    tgyro_new.aLti = tgyro_new.aLti[:, 0, :]

    variables = [
        [aLTe, tgyro_new.aLte, "aLTe"],
        [aLTi, tgyro_new.aLti, "aLTi"],
        [aLne, tgyro_new.aLne, "aLne"],
        [Q_gb, tgyro_new.Q_GB, "Qgb"],
    ]

    for var in variables:
        [c, t, n] = var

        for pos in range(c.shape[0]):
            for i in range(c.shape[1]):
                error = np.max(abs((t[pos, i + 1] - c[pos, i]) / t[pos, i + 1]))
                print(
                    f"\t\t* Error in {n}[{i}] was {error*100.0:.2f}% (TGYRO {t[pos,i+1]:.3f} vs. CGYRO {c[pos,i]:.3f})",
                    typeMsg="w" if error > ErrorRaised else "",
                )


def readlineNTH(line, full_file=True, unnormalize=True):
    s = line.split()

    i = 2
    roa = float(s[i])
    i += 3
    aLne = float(s[i])
    i += 3
    aLTi = float(s[i])
    i += 3
    aLTe = float(s[i])
    i += 3

    Qi = float(s[i])
    i += 3
    Qi_std = float(s[i])
    i += 3
    Qe = float(s[i])
    i += 3
    Qe_std = float(s[i])
    i += 3
    Ge = float(s[i])
    i += 3
    Ge_std = float(s[i])
    i += 3

    if full_file:
        GZ = float(s[i])
        i += 3
        GZ_std = float(s[i])
        i += 3

        Mt = float(s[i])
        i += 3
        Mt_std = float(s[i])
        i += 3

        Pexch = float(s[i])
        i += 3
        Pexch_std = float(s[i])
        i += 3

    Q_gb = float(s[i])
    i += 3
    G_gb = float(s[i]) * 1e-1
    i += 3  # From 1E19 to 1E20

    if full_file:
        Mt_gb = float(s[i])
        i += 3
        Pexch_gb = float(s[i])
        i += 3

    tstart = float(s[i])
    i += 3
    tend = float(s[i])
    i += 3

    if unnormalize:
        QiReal = Qi * Q_gb
        QiReal_std = Qi_std * Q_gb
        QeReal = Qe * Q_gb
        QeReal_std = Qe_std * Q_gb
        GeReal = Ge * G_gb
        GeReal_std = Ge_std * G_gb
    else:
        QiReal = Qi
        QiReal_std = Qi_std
        QeReal = Qe
        QeReal_std = Qe_std
        GeReal = Ge
        GeReal_std = Ge_std

    if full_file:
        if unnormalize:
            GZReal = GZ * G_gb
            GZReal_std = GZ_std * G_gb

            MtReal = Mt * Mt_gb
            MtReal_std = Mt_std * Mt_gb

            PexchReal = Pexch * Pexch_gb
            PexchReal_std = Pexch_std * Pexch_gb
        else:
            GZReal = GZ
            GZReal_std = GZ_std

            MtReal = Mt
            MtReal_std = Mt_std

            PexchReal = Pexch
            PexchReal_std = Pexch_std

        return roa,aLTe,aLTi,aLne,Q_gb,QeReal,QiReal,GeReal,GZReal,MtReal,PexchReal,QeReal_std,QiReal_std,GeReal_std,GZReal_std,MtReal_std,PexchReal_std,tstart,tend
    else:
        return roa,aLTe,aLTi,aLne,Q_gb,QeReal,QiReal,GeReal,0.0,0.0,0.0,QeReal_std,QiReal_std,GeReal_std,0.0,0.0,0.0,tstart,tend


def readCGYROresults(file, radii, unnormalize=True):
    """
    Arrays are in (batch,radii)
    MW/m^2 and 1E20
    """

    with open(file, "r") as f:
        lines = f.readlines()

    rad = len(radii)
    num = len(lines) // rad

    roa = np.zeros((num, rad))
    aLTe = np.zeros((num, rad))
    aLTi = np.zeros((num, rad))
    aLne = np.zeros((num, rad))
    Q_gb = np.zeros((num, rad))

    Qe = np.zeros((num, rad))
    Qe_std = np.zeros((num, rad))
    Qi = np.zeros((num, rad))
    Qi_std = np.zeros((num, rad))
    Ge = np.zeros((num, rad))
    Ge_std = np.zeros((num, rad))

    GZ = np.zeros((num, rad))
    GZ_std = np.zeros((num, rad))

    Mt = np.zeros((num, rad))
    Mt_std = np.zeros((num, rad))

    Pexch = np.zeros((num, rad))
    Pexch_std = np.zeros((num, rad))

    tstart = np.zeros((num, rad))
    tend = np.zeros((num, rad))

    p = {}
    for r in range(len(radii)):
        p[r] = 0
    for i in range(len(lines)):

        # --------------------------------------------------------
        # Line not empty
        # --------------------------------------------------------
        if len(lines[i].split()) < 10:
            continue

        # --------------------------------------------------------
        # Read line
        # --------------------------------------------------------
        (
            roa_read,
            aLTe_read,
            aLTi_read,
            aLne_read,
            Q_gb_read,
            Qe_read,
            Qi_read,
            Ge_read,
            GZ_read,
            Mt_read,
            Pexch_read,
            Qe_std_read,
            Qi_std_read,
            Ge_std_read,
            GZ_std_read,
            Mt_std_read,
            Pexch_std_read,
            tstart_read,
            tend_read,
        ) = readlineNTH(lines[i], unnormalize=unnormalize)
        
        # --------------------------------------------------------
        # Radial location position
        # --------------------------------------------------------
        threshold_radii = 1E-4
        r = np.where(np.abs(radii-roa_read)<threshold_radii)[0][0]

        # --------------------------------------------------------
        # Assign to that radial location
        # --------------------------------------------------------

        roa[p[r], r],aLTe[p[r], r],aLTi[p[r], r],aLne[p[r], r],Q_gb[p[r], r],Qe[p[r], r],Qi[p[r], r],Ge[p[r], r],GZ[p[r], r],Mt[p[r], r],Pexch[p[r], r],Qe_std[p[r], r],Qi_std[p[r], r],Ge_std[p[r], r],GZ_std[p[r], r],Mt_std[p[r], r],Pexch_std[p[r], r],tstart[p[r], r],tend[p[r], r] = roa_read,aLTe_read,aLTi_read,aLne_read,Q_gb_read,Qe_read,Qi_read,Ge_read,GZ_read,Mt_read,Pexch_read,Qe_std_read,Qi_std_read,Ge_std_read,GZ_std_read,Mt_std_read,Pexch_std_read,tstart_read,tend_read

        p[r] += 1

    return aLTe,aLTi,aLne,Q_gb,Qe,Qi,Ge,GZ,Mt,Pexch,Qe_std,Qi_std,Ge_std,GZ_std,Mt_std,Pexch_std,tstart,tend
