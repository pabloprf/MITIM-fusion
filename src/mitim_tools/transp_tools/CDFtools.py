import os
import shutil
import pickle
import copy
import datetime
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict
from mitim_tools.misc_tools import (
    IOtools,
    MATHtools,
    PLASMAtools,
    GRAPHICStools,
)
from mitim_tools.transp_tools import UFILEStools
from mitim_tools.gacode_tools import TGLFtools, TGYROtools
from mitim_tools.gacode_tools.utils import GACODEplotting, GACODErun, TRANSPgacode
from mitim_tools.transp_tools.utils import (
    FBMtools,
    TORICtools,
    PRIMAtools,
    ANALYSIStools,
    TRANSPhelpers
)
from mitim_tools.gs_tools import GEQtools
from mitim_tools.gs_tools.utils import GEQplotting
from mitim_tools.misc_tools.GUItools import FigureNotebook
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

def read_cdf_transp(cdf_file):
    '''
    With the support of chatGPT 4o (08/10/2024)
    '''

    cdf_file = IOtools.expandPath(cdf_file)
    mod_file = cdf_file.parent / f'{cdf_file.name}_mod'
    src = netCDF4.Dataset(cdf_file)

    if src['TIME'].shape[0] > src['TIME3'].shape[0]:
        print(f"\t* TIME had {src['TIME'].shape[0]- src['TIME3'].shape[0]} more time slices than TIME3, possibly because of a bad time to retrieve CDF file, fixing it...",typeMsg='w')

        # Create a dataset object to store the modified data
        mod_file.unlink(missing_ok=True)
        dst = netCDF4.Dataset(mod_file, 'w', memory=None)
        
        # Copy global attributes
        dst.setncatts({attr: src.getncattr(attr) for attr in src.ncattrs()})

        # Copy dimensions, adjust TIME dimension
        for name, dimension in src.dimensions.items():
            if name == 'TIME':
                # Adjust TIME dimension size based on TIME3
                new_time_size = src['TIME3'].shape[0]
                dst.createDimension(name, new_time_size)
            else:
                dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

        # Copy variables, adjust TIME variable
        for name, variable in src.variables.items():
            # Copy variable attributes
            var_attrs = {attr: variable.getncattr(attr) for attr in variable.ncattrs()}
            
            # Adjust the TIME variable
            if name == 'TIME':
                new_var = dst.createVariable(name, variable.datatype, variable.dimensions)
                new_var.setncatts(var_attrs)
                new_var[:] = src['TIME'][:src['TIME3'].shape[0]]  # Adjust TIME variable data
            else:
                new_var = dst.createVariable(name, variable.datatype, variable.dimensions)
                new_var.setncatts(var_attrs)
                
                # Adjust variables that depend on TIME
                if 'TIME' in variable.dimensions:
                    time_index = variable.dimensions.index('TIME')
                    slicing = [slice(None)] * variable.ndim
                    slicing[time_index] = slice(0, src['TIME3'].shape[0])
                    new_var[:] = variable[tuple(slicing)]
                else:
                    new_var[:] = variable[:]

    else:
        dst = src

    return dst.variables

class transp_output:
    def __init__(
        self,
        netCDFfile,
        ZerothTime=False,
        readFBM=False,
        ask_if_fbm=False,
        readTGLF=False,
        readTORIC=False,
        readGFILE=False,
        readStructures=True,
        readGEQDSK=False,
        calculateAccurateLineAverage=None,
        calcualtePorcelli=False,
        shotNumber=None,
        machine="SPARC",
        timeExtract=1.0,
        timeExtract_av=0.1,
        EvaluateExtraAnalysis=None,
        folderScratch="~/scratch/",
    ):
        """
        calculateAccurateLineAverage = None (=_avol), False (flux surface at one time), True (accurate at each time)
        coincideTime is used when comparing to experiment: first is experiment
        """

        folderScratch = IOtools.expandPath(folderScratch)

        self.readGEQDSK = readGEQDSK

        self.mainLegendSize = 8

        np.seterr(under="ignore")

        netCDFfile = IOtools.expandPath(netCDFfile, ensurePathValid=True)

        # ~~~~~~~~ Open CDF file ~~~~~~~~

        print(f"\n>> Analyzing netCDF file {IOtools.clipstr(netCDFfile)}")

        self.LocationCDF = netCDFfile

        # Capability to provide folder and just find the CDF in there
        if self.LocationCDF.is_dir(): 
            self.LocationCDF = IOtools.findFileByExtension(self.LocationCDF, ".CDF", agnostic_to_case=True)
            if self.LocationCDF is None:
                raise ValueError(f"[MITIM] Could not find a CDF file in {self.LocationCDF}")
        # ----------------------------

        self.f = read_cdf_transp(self.LocationCDF) 

        self.info = getRunMetaInfo(self.LocationCDF)

        self.FolderCDF = self.LocationCDF.resolve().parent

        self.FolderEvaluation, _ = IOtools.reducePathLevel(
            self.FolderCDF, level=1, isItFile=False
        )

        self.folderWork, self.nameRunid = IOtools.getLocInfo(self.LocationCDF)

        print(
            f"\t- INFO - runid: {self.nameRunid}; folder: ...{IOtools.clipstr(netCDFfile)}"
        )

        self.eps00 = 1e-14

        msAfterSawtooth, msBeforeSawtooth = 10.0, 10.0

        try:
            self.LocationNML, _ = GACODErun.findNamelist(
                self.LocationCDF, folderWork=folderScratch
            )
        except:
            try:
                print("Taking first")
                self.LocationNML, _ = GACODErun.findNamelist(
                    self.LocationCDF, folderWork=folderScratch, ForceFirst=True
                )
            except:
                print("cannot retrieve namelist", typeMsg="w")
                self.LocationNML = None

        # ~~~~~~~~ Get reactor parameters ~~~~~~~~

        self.getConstants()
        self.defineCoordinates(
            msAfterSawtooth=msAfterSawtooth,
            msBeforeSawtooth=msBeforeSawtooth,
            ZerothTime=ZerothTime,
        )

        self.defineGeometricParameters()
        self.defineMagneticParameters()

        self.getProfiles(calculateAccurateLineAverage=calculateAccurateLineAverage)
        self.getCPUs()

        self.getDilutions()
        self.getStoredEnergy()
        self.getFastIons()
        self.getImpurities()

        self.getFundamentalPlasmaPhysics()
        self.getGreenwald()

        self.getPowerLCFS()
        self.calculateCurrentDiffusion()
        self.getPowers()
        self.getMagneticEnergy()
        self.getCollisionality()
        self.getConfinementTimes()
        self.getFusionPerformance(eta_ICH=1.0)
        self.calculateSawtoothCrashEnergy()

        self.getLHthresholds()
        self.getTotalBeta()
        self.getVelocities()

        self.getElectricField()
        self.getFluxes()

        self.getTransport()
        self.getPorcelli()
        self.getParticleBalance()

        self.getNeutrals()

        self.getBoundaryInfo()
        self.getStabilityLimits()

        self.getTGLFparameters()

        self.checkQNZEFF()

        self.getEquilibriumFunctions()

        self.getSpecies()

        self.inputFilesTGLF, self.TGLFstandalone = {}, {}
        for i in np.arange(self.t[0] * 1000, self.t[-1] * 1000):
            self.inputFilesTGLF[int(i)], self.TGLFstandalone[int(i)] = None, None

        # ~~~~~~~~ FBM ~~~~~~~~

        self.fbm_He4_gc, self.fbm_He4_po = None, None
        self.fbm_Dbeam_gc, self.fbm_Dbeam_po = None, None
        self.fbm_T_gc, self.fbm_T_po = None, None

        if readFBM:
            datanum = 1

            file_converted = (
                self.folderWork / f"NUBEAM_folder" / f"{self.nameRunid}_fi_{datanum}_GC.cdf"
            )
            file_notconverted = (
                self.folderWork / f"NUBEAM_folder" / f"{self.nameRunid}.DATA{datanum}"
            )

            needToConvert = file_notconverted.exists() and (
                not file_converted.exists()
            )

            if needToConvert:
                if ask_if_fbm:
                    readFBM = print("\t- Gathering FBM data ——requires conversion of FBM files",typeMsg="q",)
                else:
                    print("\t- Gathering FBM data ——requires conversion of FBM files", typeMsg="i")
            else:
                print("\t- Gathering FBM data")

        if readFBM:
            print(f"\t\t- Looking for DATA{datanum}...")

            thr = 1e-13

            # He4
            if self.nfusHe4_avol.max() > thr:
                self.fbm_He4_gc, self.fbm_He4_po = FBMtools.getFBMprocess(
                    self.folderWork,
                    self.nameRunid,
                    datanum=datanum,
                    FBMparticle="He4_FUSN",
                )
                if self.fbm_He4_gc is not None:
                    print("\t\t- Gathered He4 FBM files")

            # T (D+D)
            if self.nfusT_avol.max() > thr:
                self.fbm_T_gc, self.fbm_T_po = FBMtools.getFBMprocess(
                    self.folderWork,
                    self.nameRunid,
                    datanum=datanum,
                    FBMparticle="T_FUSN",
                )
                if self.fbm_T_gc is not None:
                    print("\t\t- Gathered T FBM files")

            # D beam
            if self.nbD_avol.max() > thr:
                self.fbm_Dbeam_gc, self.fbm_Dbeam_po = FBMtools.getFBMprocess(
                    self.folderWork,
                    self.nameRunid,
                    datanum=datanum,
                    FBMparticle="D_NBI",
                )
                if self.fbm_Dbeam_gc is not None:
                    print("\t\t- Gathered D beam FBM files")

        # ~~~~~~~~ TORIC ~~~~~~~~

        readTORIC = readTORIC and np.sum(self.PichT) > 0.0 + self.eps00 * (
            len(self.t) + 1
        )

        self.torics = []
        if readTORIC:
            self.torics, self.cdf_FI = TORICtools.getTORICfromTRANSP(
                self.folderWork, self.nameRunid
            )

        # ~~~~~~~~ PORCELLI ~~~~~~~~
        self.calcualtePorcelli = calcualtePorcelli
        if calcualtePorcelli:
            self.Porcelli = ANALYSIStools.calcualtePorcelli(self)
        else:
            self.Porcelli = None

        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------


        self.getEstimatedMachineCost()

        (
            self.R_ant,
            self.Z_ant,
            self.R_vv,
            self.Z_vv,
            self.R_lim,
            self.Z_lim,
            self.R_gyr,
            self.Z_gyr,
            self.F_gyr,
            self.beam_trajectories,
            self.ECRH_trajectories,
        ) = (None, None, None, None, None, None, None, None, None, None, None)
        if readStructures:
            self.getStructures()
        self.gfile_in = None
        if readGFILE:
            self.getGFILE()

        # ~~~~~~~~ TGLF ~~~~~~~~

        self.TGLF = None
        if readTGLF and "KYS_TGLF" in self.f:
            if self.f["KYS_TGLF"][:].max() > 0.0:
                self.TGLF = TRANSPgacode.tglfCDF(self)
            else:
                print(
                    "\t- TGLF growth rates were allocated but are 0... probably FTIME came before TGLF",
                    typeMsg="w",
                )

        self.TGLFstd = {}

        # ~~~~~~~~ Evaluate at last sawtooth to provide metrics ~~~~~~~~

        self.evaluateReactorMetrics(EvaluateExtraAnalysis=EvaluateExtraAnalysis)

        # ~~~~~~~~ EXPERIMENT ~~~~~~~~

        self.exp = None

        if shotNumber is not None:
            self.machine = machine
            self.shotNumber = shotNumber
            self.timeProfile = timeExtract
            self.timeProfile_av = timeExtract_av

            if self.machine == "CMOD" or self.machine == "SPARC":
                from mitim_tools.experiment_tools import CMODtools

                self.exp = CMODtools.CMODexperiment(
                    self.shotNumber,
                    timeProfile=self.timeProfile,
                    timeProfile_av=self.timeProfile_av,
                )
            if self.machine == "AUG":
                from mitim_tools.experiment_tools import AUGtools

                self.exp = AUGtools.AUGexperiment(self.shotNumber)

        # Isolver

        if hasattr(self, "R_lim") and self.R_lim is not None:
            RlimZlim = [self.R_lim, self.Z_lim]
        else:
            RlimZlim = None

        tok = "sprc"
        try:
            self.isolver = ANALYSIStools.transpISOLVER(
                self.f, RlimZlim=RlimZlim, tok=tok
            )
        except:
            print("\t- ISOLVER variables could not be found or interpreted")
            self.isolver = None

        # ~~~~~~~~ Special Variables  (for convergence) ~~~~~~~~
        self.SpecialVariables = {
            "VarMITIM_Q": self.Q,
            "VarMITIM_PRESS": self.p_avol,
            "VarMITIM_TE": self.Te_avol,
            "VarMITIM_TI": self.Ti_avol,
            "VarMITIM_NE": self.ne_avol,
            "VarMITIM_HE4": self.nHe4_avol,
            "VarMITIM_PRESS_0": self.p[:, 0],
            "VarMITIM_TI_0": self.Ti0,
            "VarMITIM_TE_0": self.Te0,
            "VarMITIM_NE_0": self.ne0,
            "VarMITIM_HE4_0": self.nHe4[:, 0],
        }

    def analyze_initial(self, duration=1.0, fig=None):
        ANALYSIStools.plotFullCurrentDynamics(
            self, self.t[0], self.t[0] + duration, fig=fig
        )

    def analyze_sawtooth(self, tmargin=0.01, fig=None):
        ANALYSIStools.plotFullCurrentDynamics(
            self,
            self.tlastsawU[-2],
            self.tlastsawU[-1],
            fig=fig,
            tmargin=tmargin,
            rho_plot_lim=0.52,
        )

    def evaluateReactorMetrics(self, ReactorTextFile=None, EvaluateExtraAnalysis=None):
        index = self.ind_saw_before

        # Note: These are the variables that contain all information about the TRANSP plasma that are relevant for optimiztiaon. It contains a single
        # timeslice value (typically the top of the last sawtooth) for relevant metrics. These metrics are to be used as OF or constraints

        variables = IOtools.OrderedDict(
            {
                "taue": self.taue,
                "neutrons": self.neutrons,
                "pressure": self.p_avol,
                "fG": self.fGv,
                "Qratio": self.QiQe_ratio[:, np.argmin(np.abs(self.x_lw - 0.5))],
                "Qe": self.qe_obs[:, np.argmin(np.abs(self.x_lw - 0.5))],
                "Te_avol": self.Te_avol,
                "Te0": self.Te0,
                "Te_peaking": self.Te_peaking,
                "Ti_avol": self.Ti_avol,
                "Ti0": self.Ti0,
                "Ti_peaking": self.Ti_peaking,
                "ne_avol": self.ne_avol,
                "ne0": self.ne0,
                "ne_peaking": self.ne_peaking,
                "Q_plasma": self.Q,
                "H98y2": self.H98y2_check,
                "P_input": self.utilsiliarPower,
                "Pich": self.PichT,
                "Wtherm": self.Wth,
                "Wtotal": self.Wtot,
                "Prad": self.PradT,
                "crash_W": self.Wsaw,
                "volume": self.volume,
                "surface": self.surface,
                "P_LCFS": self.P_LCFS,
                "P_out": self.Pout,
                "BetaN": self.BetaN,
                "q1_position": self.x_saw_inv,
                "tau_cr": self.tau_c,
                "q95": self.q95,
                "BetaPol": self.BetaPol,
                "Rmajor": self.Rmajor,
                "delta": self.delta,
                "epsilon": self.epsilon,
                "kappa": self.kappa,
                "Bt": self.Bt,
                "Plh_ratio": self.Plh_ratio,
                "Plh_ratio_mod": self.Plh_ratio_mod,
                "kappa_lim": self.kappa_lim,
                "penalties": self.penalties,
                "cost": self.cost,
                "LHdiff": self.LHdiff,
                "rho": self.x,
                "Te": self.Te,
                "Ti": self.Ti,
                "ne": self.ne,
            }
        )

        self.reactor = {}
        for ikey in variables:
            self.reactor[ikey] = variables[ikey][index]

        # ~~~~~~~~ Extra ones
        self.reactor["CoordinateMapping"] = {
            "rho": self.xb[index],
            "psi": self.psin[index],
        }
        self.reactor["kappaRatio"] = self.kappa_995[index] / self.kappa[index]
        self.reactor["deltaRatio"] = self.delta_995[index] / self.delta[index]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~ Some require extra analysis (do not run always)

        """
		 Extra transport analysis variables. Posibilities:

		 	- VarMITIM_CHIPERT_QE_0.60 for dQe/daLTe at rho=0.6

		 	- VarMITIM_D_W_0.60 for D particle coefficient W at rho=0.6
		 	- VarMITIM_V_W_0.60 for V particle coefficient W at rho=0.6
		 	- VarMITIM_VoD_W_0.60

			- VarMITIM_FLUCTE_LOWK_0.6 for integrated Te fluctuation levels at low-k at rho=0.6

		"""

        # ****** Settings *********
        TGLFsettings = 5
        d_perp_cm = {0.7: 0.757 / np.sqrt(2) / (np.cos(11 * (np.pi / 180)))}
        # *************************

        if EvaluateExtraAnalysis is not None:
            max_attempts = 3
            att = 0
            while att < max_attempts:
                att += 1
                print(f" ~~~ Running transport analysis (attempt #{att})")
                success = True
                for var in EvaluateExtraAnalysis:
                    if (
                        "VarMITIM_CHIPERT" in var
                        or "VarMITIM_D" in var
                        or "VarMITIM_V" in var
                        or "VarMITIM_FLUCTE" in var
                    ):
                        typeA = var.split("_")[-3]  # CHIPERT, D, V, FLUCTE, VoD
                        quantity = var.split("_")[-2]  # QE, W, LOWK
                        location = float(var.split("_")[-1])  # 0.60
                        # try:
                        self.reactor[var] = self.transportAnalysis(
                            typeAnalysis=typeA,
                            quantity=quantity,
                            rho=location,
                            time=self.t[index],
                            TGLFsettings=TGLFsettings,
                            d_perp_cm=d_perp_cm,
                        )
                        success = success and True
                        # except:	success = success and False
                if success:
                    break

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if ReactorTextFile is not None:
            with open(ReactorTextFile, "a") as outfile:
                outfile.write("-------- Output Parameters -------------\n")
                for i in self.reactor:
                    outfile.write(f"{i} = {self.reactor[i]}\n")

    def getCPUs(self):
        self.cptim = self.f["CPTIM"][:]
        try:
            self.cptim_pt = self.f["PT_CPTIME"][:]  # PT_SOLVER
        except:
            self.cptim_pt = self.t * 0.0
        self.cptim_out = self.f["CPOUT"][:]  # OUTPUT SYSTEM
        self.cptim_geom = self.f["CPGEOM"][:]  # FLUX SURFACE GEOMETRY
        self.cptim_mhd = self.f["CPMHDQ"][:]  # MHD equilibrium
        self.cptim_fsa = self.f["CPGEOCAL"][:]  # Flux Surf. Averages
        try:
            self.cptim_nubeam = self.f["CPBMAX"][:]  # MAX THREAD CPU TIME: NUBEAM
        except:
            self.cptim_nubeam = self.t * 0.0

        try:
            self.cptim_icrf = self.f["CPWAVE"][:]  # ICRF
            self.cptim_fpp = self.f["CPFPP"][:] - self.cptim_icrf  # FPP
        except:
            self.cptim_icrf = self.t * 0.0
            self.cptim_fpp = self.t * 0.0

        self.cptim_deriv = MATHtools.deriv(self.t, self.cptim)

        self.dt = np.append(np.diff(self.t), [0])

    def defineCoordinates(
        self, msAfterSawtooth=10.0, msBeforeSawtooth=10.0, ZerothTime=False
    ):
        # ~~~~~ Time coordinates

        self.t = self.f["TIME"][:]
        self.timeOri, self.timeFin = self.t[0], self.t[-1]
        if ZerothTime:
            self.t = self.t - self.timeOri

        print(
            "\t- Simulated from t={0:.3f}s to t={1:.3f}s ({2:.3f}s of simulated plasma)".format(
                self.t[0], self.t[-1], self.t[-1] - self.t[0]
            )
        )

        try:
            self.tlastsaw, self.tlastsawU = (
                self.f["TLASTSAW"][:],
                np.unique(self.f["TLASTSAW"][:])[1:],
            )
            self.numSaw = len(self.tlastsawU)
        except:
            self.tlastsaw, self.tlastsawU, self.numSaw = self.t, self.t, None
            print(">> This plasma did not sawtooth")

        if ZerothTime:
            self.tlastsaw, self.tlastsawU = (
                self.tlastsaw - self.timeOri,
                self.tlastsawU - self.timeOri,
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~ Gather indices
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        howmanybefore = 1
        self.ind_saw = findLastSawtoothIndex(
            self, howmanybefore=howmanybefore, positionInChain=-1
        )
        self.ind_sawAll = []
        for i in range(1000):
            try:
                self.ind_sawAll.append(
                    findLastSawtoothIndex(
                        self, howmanybefore=howmanybefore, positionInChain=-1 - i
                    )
                )
            except:
                break

        self.tlast = self.t[self.ind_saw]
        self.nt = len(self.t)

        # After sawtooth:
        timeAfter = self.t[self.ind_saw + howmanybefore] + msAfterSawtooth * 1e-3
        self.ind_saw_after = np.argmin(np.abs(self.t - timeAfter))

        # Before sawtooth:
        timeBefore = self.t[self.ind_saw + howmanybefore] - msBeforeSawtooth * 1e-3
        self.ind_saw_before = np.argmin(np.abs(self.t - timeBefore))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~ Spatial normalized coordinates
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~ Fluxes

        self.psi = self.f["PLFLX"][:]  # Poloidal flux (Wb/rad)
        self.phi = self.f["TRFLX"][:]  # Toroidal flux (Wb)

        self.phi_bnd = self.phi[:, -1]  # Toroidal flux at boundary (Wb)
        self.psi_bnd = self.psi[:, -1] *  2 * np.pi  # Poloidal flux at boundary (Wb)

        self.phi_check = self.f["TRFCK"][:]

        try:
            self.RBtor_bnd = self.f["GRBA_DATA"][:] * 1e-2  # T*m
        except:
            self.RBtor_bnd = self.t * 0.0

        # ~~~~~ Normalized Fluxes

        # Normalized poloidal flux
        psin = []
        for it in range(len(self.t)):
            psin.append(self.psi[it] / self.psi[it, -1])
        self.psin = np.array(psin)

        # Normalized toroidal flux
        phin = []
        for it in range(len(self.t)):
            phin.append(self.phi[it] / self.phi[it, -1])
        self.phin = np.array(phin)

        # Sqrt normalized poloidal flux
        self.xpol = np.sqrt(self.psin)

        # Sqrt normalized toroidal flux
        self.x, self.xb = self.f["X"][:], self.f["XB"][:]

        # At last sawtooth

        self.x_lw = self.x[self.ind_saw, :]
        self.xb_lw = self.xb[self.ind_saw, :]
        self.psin_lw = self.psin[self.ind_saw, :]
        self.phin_lw = self.phin[self.ind_saw, :]
        self.xpol_lw = self.xpol[self.ind_saw, :]

        self.nx = len(self.x_lw)

        # ~~~~~ Volume
        self.dvol = self.f["DVOL"][:] * 1e-6  # m^3

        self.dvol_cum = np.cumsum(self.dvol, axis=1)

        # ~~~~~ Spatial real coordinates

        # Minor radius
        self.rmin = (
            self.f["RMNMP"][:] * 1e-2
        )  # Half width of flux surface midplane intercept
        roa = []
        for it in range(len(self.t)):
            roa.append(self.rmin[it] / self.rmin[it, -1])
        self.roa = np.array(roa)

        # Major radius
        self.Rmaj = (
            self.f["RMAJM"][:] * 1e-2
        )  # Midplane radii (it goes from LF to HF, and correspond to XB + zero)
        self.Rmaj2 = (
            self.f["RMJSYM"][:] * 1e-2
        )  # midplane radii (it goes from LF to HF, but more points?)

        # ~~~~~ For surface averages
        self.ave_grad_rho = self.f["GXI"][:] * 1e2
        self.ave_grad_rhosq = self.f["GXI2"][:] * 1e4

        self.Rinv = self.f["GRI"][:] * 1e2
        self.Rinv2 = self.f["GR2I"][:] * 1e4

        self.gradXsqrt_Rsq = self.f["GX2R2I"][:] * 1e8  # in m^-2

        # ~~~~~ Quantities for comparison to GACODE
        self.dVdr = derivativeVar(self, self.dvol_cum, specialDerivative=self.rmin)
        self.dVdx = derivativeVar(self, self.dvol_cum, specialDerivative=self.x)
        self.S_x = self.f["SURF"][:] * 1e-4
        self.ave_grad_r = self.S_x / self.dVdr

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~ Transformations between entire midplane and LF/HF
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Complete, in XB
        hf = len(self.Rmaj[0, :]) // 2
        self.rmaj_HF = self.Rmaj[:, :hf]
        self.rmaj_LF = self.Rmaj[:, hf + 1 :]

        self.xb_whole = np.append(
            np.append(np.flipud(self.xb_lw), np.array([0])), self.xb_lw
        )
        self.x_whole = np.append(np.flipud(self.x_lw), self.x_lw)

        # Complete, in X anx XB
        self.rmaj_LFxb, self.rmaj_HFxb = mapRmajorToX(
            self, self.Rmaj, interpolateToX=False
        )
        self.rmaj_LFx, self.rmaj_HFx = mapRmajorToX(
            self, self.Rmaj, interpolateToX=True
        )

        # Miller-like flux surfaces
        self.RmajorS = copy.deepcopy(self.rmaj_LF)
        for it in range(len(self.t)):
            self.RmajorS[it] = (self.rmaj_LFx[it] + self.rmaj_HFx[it]) / 2
        self.epsilonS = self.rmin / self.RmajorS

    def findIndeces(self, time=-1, rho=0.5, offsets=[0, 0]):
        # If time is -1: last sawtooth, -2: second to last sawtooth, ...

        if time < 0:
            try:
                time = self.tlastsawU[time]
            except:
                time = self.t[-1]

        ind_t = np.argmin(np.abs(time - self.t))
        ind_x = np.argmin(np.abs(rho - self.x[ind_t, :]))

        ind_tb = np.argmin(np.abs(time - (self.t - offsets[0])))
        ind_ta = np.argmin(np.abs(time - (self.t - offsets[1])))

        return ind_t, ind_x, ind_tb, ind_ta

    def getProfiles(self, calculateAccurateLineAverage=False):
        self.Te = self.f["TE"][:] * 1e-3
        self.Ti = self.f["TI"][:] * 1e-3
        self.TZ = self.f["TX"][:] * 1e-3
        self.ne = self.f["NE"][:] * 1e6 * 1e-20  # in 10^20m^-3

        self.Te0 = self.Te[:, 0]
        self.Ti0 = self.Ti[:, 0]
        self.TZ0 = self.TZ[:, 0]
        self.ne0 = self.ne[:, 0]

        self.Te_avol = volumeAverage(self.f, "TE") * 1e-3
        self.Ti_avol = volumeAverage(self.f, "TI") * 1e-3
        self.TZ_avol = volumeAverage(self.f, "TX") * 1e-3
        self.ne_avol = volumeAverage(self.f, "NE") * 1e6 * 1e-20  # in 10^20m^-3

        if calculateAccurateLineAverage is not None:
            self.ne_l = self.lineAverageCentralChord(
                self.ne, accurate=calculateAccurateLineAverage
            )  # in 10^20m^-3
        else:
            print(
                "\t- Not calculating line-average (it is expensive), using volume average instead",
                typeMsg="i",
            )
            self.ne_l = self.ne_avol

        self.Te_height = self.f["TEPED"][:] * 1e-3
        self.Ti_height = self.f["TIPED"][:] * 1e-3
        self.ne_height = self.f["NEPED"][:] * 1e6 * 1e-20  # in 10^20m^-3
        self.Te_width = self.f["TEPEDW"][:]
        self.Ti_width = self.f["TIPEDW"][:]
        self.ne_width = self.f["NEPEDW"][:]

        # Aprox pedestal pressure
        self.p_height = (
            2 * (self.Te_height * 1e3 * self.e_J * self.ne_height * 1e20) * 1e-6
        )  # MPa

    def lineAverageCentralChord(self, var, accurate=False):
        Rmaj = self.Rmajor[-1]

        if accurate:
            print(
                "\t- Accurate calculation of line-av. @R={0}m vertical chord, it may take some time...".format(
                    Rmaj
                )
            )
            chord_array, _ = self.chordLengths(Rmaj=Rmaj)
        else:
            time = self.t[self.ind_saw]
            print(
                "\t- Rough calculation of line-av. @R={0:.2f}m vertical chord with surfaces @t={1:.3f}s".format(
                    Rmaj, time
                ),
                typeMsg="i",
            )
            chord_array, _ = self.chordLengths(Rmaj=Rmaj, time=time)

        z_av = np.sum(chord_array * var, axis=1) / np.sum(chord_array, axis=1)

        return z_av

    def chordLengths(self, Rmaj=1.85, time=None, thetaPoints=1e5):
        if time is None:
            times = self.t
        else:
            times = [time]

        ChordLengths, allP = [], []
        for time in times:
            ChordLength, PrevPoint, allPoints = [], 0, []
            for ix in self.xb_lw:
                RMC, YMC = getFluxSurface(self.f, time, ix, thetap=thetaPoints)
                yp = MATHtools.intersect(RMC, YMC, Rmaj)
                if yp is not None:
                    ChordLength.append(yp[0] - PrevPoint)
                    PrevPoint = yp[0]
                    allPoints.append(yp[0])
                else:
                    ChordLength.append(0)
                    allPoints.append(0)
            ChordLengths.append(ChordLength)
            allP.append(allPoints)

        return np.array(ChordLengths), np.array(allP)

    def getTGLFparameters(self, mi_u=None, correctGrid=True):
        # --------------------------------
        # correctGrid = True calculates the gyrbohm quantities on the center grid, and then
        # 	it interpolates into the boundary grid.
        # correctGrid = False does the same but it assumes that the kinetic profiles were not
        # 	defined originally in the boundary grid ("ghost" grid has an extra zero point). This
        # 	means that everything is shifted, which I don't agree with but that's the only way to
        # 	match input.profiles and tgyro gyrobohm output.
        # --------------------------------
        self.TGLF_Te = copy.deepcopy(self.Te)
        self.TGLF_ne = copy.deepcopy(self.ne)
        self.TGLF_Ti = copy.deepcopy(self.Ti)
        self.TGLF_ni = copy.deepcopy(self.ni)
        if not correctGrid:
            for i in range(len(self.t)):
                self.TGLF_Te[i] = np.interp(
                    self.x_lw, self.x_lw - self.x_lw[0], self.Te[i]
                )
                self.TGLF_ne[i] = np.interp(
                    self.x_lw, self.x_lw - self.x_lw[0], self.ne[i]
                )
                self.TGLF_Ti[i] = np.interp(
                    self.x_lw, self.x_lw - self.x_lw[0], self.Ti[i]
                )
                self.TGLF_ni[i] = np.interp(
                    self.x_lw, self.x_lw - self.x_lw[0], self.ni[i]
                )

        # --------------------------------
        # Modify reference mass
        # --------------------------------

        if mi_u is None:
            if self.mi / self.u > 1.5:
                if self.mi / self.u > 2.5:
                    self.TGLF_mref = 3.0
                else:
                    self.TGLF_mref = 2.0
            else:
                self.TGLF_mref = 1.0
        else:
            self.TGLF_mref = mi_u

        # --------------------------------
        # Definition of Bunit (in XB grid)
        # --------------------------------

        self.TGLF_Bunit, self.TGLF_Bunit_x = [], []
        for it in range(len(self.t)):
            bunit = PLASMAtools.Bunit(self.phi[it, :], self.rmin[it, :])
            self.TGLF_Bunit.append(bunit)
            self.TGLF_Bunit_x.append(np.interp(self.x_lw, self.xb_lw, bunit))
        self.TGLF_Bunit = np.array(self.TGLF_Bunit)
        self.TGLF_Bunit_x = np.array(self.TGLF_Bunit_x)

        # --------------------------------
        # Useful quantities (in X)
        # --------------------------------

        self.TGLF_cs = PLASMAtools.c_s(self.TGLF_Te, self.TGLF_mref)
        self.TGLF_rhos = PLASMAtools.rho_s(
            self.TGLF_Te, self.TGLF_mref, self.TGLF_Bunit_x
        )

        # --------------------------------
        # GyroBohm unit
        # --------------------------------

        self.TGLF_a = constant_radius(self.a, lenX=len(self.x_lw))

        self.TGLF_Qgb, self.TGLF_Ggb, _, _, _ = PLASMAtools.gyrobohmUnits(
            self.TGLF_Te, self.TGLF_ne, self.TGLF_mref, self.TGLF_Bunit_x, self.TGLF_a
        )

        # Interpolation to surface grids, where the QGB will be used
        self.TGLF_Qgb_xb = copy.deepcopy(self.TGLF_Qgb)
        self.TGLF_Ggb_xb = copy.deepcopy(self.TGLF_Ggb)
        for it in range(len(self.t)):
            self.TGLF_Qgb_xb[it] = np.interp(self.xb[it], self.x[it], self.TGLF_Qgb[it])
            self.TGLF_Ggb_xb[it] = np.interp(self.xb[it], self.x[it], self.TGLF_Ggb[it])

        # BY CONVENTION
        self.TGLF_Qgb = copy.deepcopy(self.TGLF_Qgb_xb)
        self.TGLF_Ggb = copy.deepcopy(self.TGLF_Ggb_xb)

        # --------------------------------
        # Normalizations of fluxes
        # --------------------------------

        self.qeGB_obs = self.qe_obs / self.TGLF_Qgb
        self.qiGB_obs = self.qi_obs / self.TGLF_Qgb

        self.qeGB_tr = self.qe_tr / self.TGLF_Qgb
        self.qiGB_tr = self.qi_tr / self.TGLF_Qgb

        self.qe_obs_GACODE = self.qe_obs * self.ave_grad_r
        self.qi_obs_GACODE = self.qi_obs * self.ave_grad_r
        self.Ge_obs_GACODE = self.Ge_obs * self.ave_grad_r

        self.qe_tr_GACODE = self.qe_tr * self.ave_grad_r
        self.qi_tr_GACODE = self.qi_tr * self.ave_grad_r
        self.Ge_tr_GACODE = self.Ge_tr * self.ave_grad_r

        # --------------------------------
        # Rest of things (no 100% checked)
        # --------------------------------

        self.getTGLFparameters_all()

    def getTGLFparameters_all(self):
        # GACODE geometry

        self.TGLF_R0 = 0.5 * (self.rmaj_LFx + self.rmaj_HFx)
        self.TGLF_r = 0.5 * (
            self.rmaj_LFx - self.rmaj_HFx
        )  # Equal to self.rmin actually

        # GACODE rotation

        self.TGLF_w0 = 2 * np.pi * self.VtorkHz * 1e3  # in rad/s
        self.sign_it = -1

        vpar = -self.sign_it * (self.TGLF_R0 * self.TGLF_w0) / self.cs
        w0p = derivativeVar(self, self.TGLF_w0)
        w0_norm = self.cs[:, 0] / self.TGLF_R0[:, 0]
        w0_norm = constant_radius(w0_norm, lenX=len(self.x_lw))

        f_rot = w0p / w0_norm
        gamma_p0 = -self.TGLF_R0 * f_rot * w0_norm
        gamma_eb0 = gamma_p0 * self.TGLF_r / (self.q * self.TGLF_R0)
        vexb_shear = -self.sign_it * gamma_eb0 * self.TGLF_a / self.cs
        vpar_shear = -self.sign_it * gamma_p0 * self.TGLF_a / self.cs

        # ----------------------------------------
        # ---------- TGLF Variables ----------
        # ----------------------------------------

        # ---------- COMMON
        self.TGLF_vexb_shear = vexb_shear
        self.TGLF_betae = (
            4.0
            * np.pi
            * 1e-7
            * (2.0 * self.ne * 1e20 * self.Te * self.e_J * 1e3)
            / self.TGLF_Bunit**2.0
        )
        self.TGLF_nue = self.nu_ei / (self.cs / self.TGLF_a)
        self.TGLF_zeff = self.Zeff
        self.TGLF_debye = (
            7.43e2 * np.sqrt(self.Te * 1e3 / (self.ne * 1e20)) / np.abs(self.TGLF_rhos)
        )  # From tgyro_tglf_map.f90

        # ---------- GEO
        self.TGLF_rmin = self.rmin / self.TGLF_a
        self.TGLF_rmaj = self.TGLF_R0 / self.TGLF_a
        self.TGLF_zmaj = self.Ymag
        self.TGLF_drmindx = self.TGLF_rmin / self.TGLF_rmin
        self.TGLF_drmajdx = derivativeVar(self, self.TGLF_rmaj * self.TGLF_a)
        self.TGLF_dzmajdx = copy.deepcopy(self.TGLF_zmaj) * 0.0 + self.eps00

        self.TGLF_q = self.q
        self.TGLF_q_prime = (
            (self.TGLF_q * self.TGLF_a / self.TGLF_r) ** 2
            * gradNorm(self, self.TGLF_q)
            * (-self.TGLF_r / self.TGLF_a)
        )
        self.TGLF_p_prime = (
            4.0
            * np.pi
            * 1e-7
            * (self.TGLF_q * self.TGLF_a**2 / (self.TGLF_r * self.TGLF_Bunit**2))
            * derivativeVar(self, self.p * 1e6)
        )

        self.TGLF_kappa = self.kappaS
        self.TGLF_s_kappa = gradNorm(self, self.kappaS) * (-self.rmin / self.TGLF_a)
        self.TGLF_delta = self.deltaS
        self.TGLF_s_delta = (
            gradNorm(self, self.deltaS) * (-self.rmin / self.TGLF_a) * self.deltaS
        )
        self.TGLF_zeta = self.zetaS
        self.TGLF_s_zeta = (
            gradNorm(self, self.zetaS) * (-self.rmin / self.TGLF_a) * self.zetaS
        )

        self.TGLF_zs = np.array([-1, 1])
        self.TGLF_mass = np.array([0.000272313, 1])
        self.TGLF_rlns = np.array([self.aLne, self.aLnD])
        self.TGLF_rlts = np.array([self.aLTe, self.aLTi])
        self.TGLF_taus = np.array([np.ones([len(self.t), len(self.x_lw)]), self.TiTe])
        self.TGLF_as = np.array(
            [np.ones([len(self.t), len(self.x_lw)]), self.nmain / self.ne]
        )
        self.TGLF_vpar = np.array([vpar, vpar])
        self.TGLF_vpar_shear = np.array([vpar_shear, vpar_shear])

    def getGreenwald(self):
        nG = PLASMAtools.Greenwald_density(self.Ip, self.a)

        fG = []
        for i in range(len(self.x_lw)):
            fG.append(self.ne[:, i] / nG)
        self.fG = np.transpose(fG)

        self.fGv = self.ne_avol / nG
        self.fGl = self.ne_l / nG
        self.fGh = self.ne_height / nG

        # self.fG_950 = np.zeros(len(self.t))
        # for it in range(len(self.t)):
        # 	self.fG_950[it] = np.interp([0.95],self.psin[it],self.fG[it])[0]

    def getDilutions(self):

        '''
        Note that ni in TRANSP seems to not include He3 !!!
        '''

        self.ni = self.f["NI"][:] * 1e6 * 1e-20  # in 10^20m^-3
        self.ni_avol = volumeAverage(self.f, "NI") * 1e6 * 1e-20  # in 10^20m^-3

        # ~~~~~~~~~~~~~~~~~~~~~~~~~ Main Plasma ~~~~~~~~~~~~~~~~~~~~~~~~~

        try:
            self.nD = self.f["ND"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nD_avol = volumeAverage(self.f, "ND") * 1e6 * 1e-20  # in 10^20m^-3
        except:
            print("\t- This plasma had no Deuterium")
            self.nD = self.ne * 0.0 + self.eps00
            self.nD_avol = self.ne_avol * 0.0 + self.eps00

        try:
            self.nH = self.f["NH"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nH_avol = volumeAverage(self.f, "NH") * 1e6 * 1e-20  # in 10^20m^-3
        except:
            self.nH = self.ne * 0.0 + self.eps00
            self.nH_avol = self.ne_avol * 0.0 + self.eps00

        try:
            self.nT = self.f["NT"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nT_avol = volumeAverage(self.f, "NT") * 1e6 * 1e-20  # in 10^20m^-3
        except:
            print("\t- This plasma had no Tritium")
            self.nT = self.ne * 0.0 + self.eps00
            self.nT_avol = self.ne_avol * 0.0 + self.eps00

        try:
            self.nHe4 = self.f["NHE4"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nHe4_avol = volumeAverage(self.f, "NHE4") * 1e6 * 1e-20  # in 10^20m^-3

        except:
            if self.nT_avol[-1] < self.eps00 * 1.2:
                print("\t- This plasma had no thermal alphas")
            else:
                print(
                    " \t- This plasma, although DT, had no thermal alphas being tracked"
                )
            self.nHe4 = self.ne * 0.0 + self.eps00
            self.nHe4_avol = self.ne_avol * 0.0 + self.eps00

        self.nT_particles = self.volume * self.nT_avol  # in 10^20 particles
        self.nD_particles = self.volume * self.nD_avol  # in 10^20 particles
        self.nH_particles = self.volume * self.nH_avol  # in 10^20 particles

        self.nT_frac = self.nT_particles / (
            self.nT_particles + self.nD_particles + self.nH_particles
        )
        self.nD_frac = self.nD_particles / (
            self.nT_particles + self.nD_particles + self.nH_particles
        )
        self.nH_frac = self.nH_particles / (
            self.nT_particles + self.nD_particles + self.nH_particles
        )

        self.fD = self.nD / self.ne
        self.fD_avol = self.nD_avol / self.ne_avol
        self.fT = self.nT / self.ne
        self.fT_avol = self.nT_avol / self.ne_avol
        self.fH = self.nH / self.ne
        self.fH_avol = self.nH_avol / self.ne_avol

        self.nmain = self.nD + self.nT + self.nH
        self.nmain_avol = self.nD_avol + self.nT_avol + self.nH_avol

        self.fmain = (self.nD + self.nT + self.nH) / self.ne
        self.fmain_avol = (self.nD_avol + self.nT_avol + self.nH_avol) / self.ne_avol
        self.fHe4 = self.nHe4 / self.ne
        self.fHe4_avol = self.nHe4_avol / self.ne_avol

        # ~~~~~~~~~~~~~~~~~~~~~~~~~ Average ion mass ~~~~~~~~~~~~~~~~~~~~~~~~~

        self.mD = 2.0135532 * self.u
        self.mT = 3.0160492 * self.u
        self.mHe3 = 3.0160293 * self.u
        self.mH = 1.0 * self.u
        self.mHe4 = 4.002602 * self.u

        self.mi = (
            self.nD_frac[self.ind_saw] * self.mD
            + self.nT_frac[self.ind_saw] * self.mT
            + self.nH_frac[self.ind_saw] * self.mH
        )
        self.m_mu = 1.0 / (1.0 / self.me + 1.0 / self.mi)
        self.Meff = (
            self.nD_frac[self.ind_saw] * 2.0
            + self.nT_frac[self.ind_saw] * 3.0
            + self.nH_frac[self.ind_saw] * 1.0
        )
        self.Ai = self.mi / self.u

        # ~~~~~~~~~~~~~~~~~~~~~~~~~ Total particles ~~~~~~~~~~~~~~~~~~~~~~~~~

        self.ne_tot = volumeIntegralTot(self.f, "NE") * 1e-20  # in 10^20
        try:
            self.nD_tot = volumeIntegralTot(self.f, "ND") * 1e-20  # in 10^20
        except:
            self.nD_tot = copy.deepcopy(self.ne_tot) * 0.0

        try:
            self.nT_tot = volumeIntegralTot(self.f, "NT") * 1e-20  # in 10^20
        except:
            self.nT_tot = copy.deepcopy(self.ne_tot) * 0.0

        try:
            self.nH_tot = volumeIntegralTot(self.f, "NH") * 1e-20  # in 10^20
        except:
            self.nH_tot = copy.deepcopy(self.ne_tot) * 0.0

        try:
            self.nHe4_tot = volumeIntegralTot(self.f, "NHE4") * 1e-20  # in 10^20
        except:
            self.nHe4_tot = copy.deepcopy(self.ne_tot) * 0.0

        self.nT_tot_mg = self.nT_tot * 1e20 * self.mT * 1e6
        self.nHe4_tot_mg = self.nHe4_tot * 1e20 * self.mHe4 * 1e6

    def getMagneticEnergy(self):
        # Magnetic energy

        self.Umag_pol = self.f["UBPOL"][:]  # MJ/m^3
        self.Umag_tor = self.f["UBTOR"][:]  # MJ/m^3

        phibound = constant_radius(self.phi_bnd, lenX=len(self.x_lw))
        self.Umag_pol_check = (
            1
            / (2 * self.mu0)
            * (self.iota_bar * self.xb * phibound / np.pi) ** 2
            * self.gradXsqrt_Rsq
            * 1e-6
        )

        self.UmagT_tor = volumeIntegralTot(self.f, "UBTOR") * 1e-6  # MJ
        self.UmagT_pol = volumeIntegralTot(self.f, "UBPOL") * 1e-6  # MJ

        self.Umag = self.Umag_tor + self.Umag_pol
        self.UmagT = self.UmagT_tor + self.UmagT_pol

        # Rate of change of magnetic energy

        self.Umag_pol_dt = self.f["UBPDT"][:]  # MW/m^3
        self.UmagT_pol_dt = volumeIntegralTot(self.f, "UBPDT") * 1e-6  # MW

        self.Umag_dt = self.f["UBTDT"][:]  # MW/m^3
        self.UmagT_dt = volumeIntegralTot(self.f, "UBTDT") * 1e-6  # MW

        self.Umag_tor_dt = self.Umag_dt - self.Umag_pol_dt
        self.UmagT_tor_dt = self.UmagT_dt - self.UmagT_pol_dt

        self.ExBpower = self.f["UDEXB"][:]  # MW/m^3
        self.ExBpower_T = volumeIntegralTot(self.f, "UDEXB") * 1e-6  # MW

        self.BpolComp = self.f["UBCMP"][:]  # MW/m^3
        self.BpolComp_T = volumeIntegralTot(self.f, "UBCMP") * 1e-6  # MW

        aux = []
        for i in range(len(self.x_lw)):
            aux.append(
                self.Umag_pol[:, i]
                / self.gradXsqrt_Rsq[:, i]
                / self.dVdx[:, i]
                * MATHtools.deriv(self.t, self.dVdx[:, i] * self.gradXsqrt_Rsq[:, i])
            )
        self.BpolComp_check = np.transpose(np.array(aux))

        aux = []
        for i in range(len(self.x_lw)):
            aux.append(
                self.Umag_pol[:, i]
                / self.dVdx[:, i]
                * MATHtools.deriv(self.t, self.dVdx[:, i])
            )
        self.BpolComp_check2 = np.transpose(np.array(aux))

        # --------------
        # Get energy that TRANSP does not account for that needs to come from electrons to Bpol
        # --------------

        """
		This is an attempt to resolve the energy generation by sawtooth.
		It is based on the idea that UmagT_pol does not obey a time evolution based on UmagT_pol_dt during
		a sawtooth crash. I calculate the correction.
		"""

        blend = 0.05

        # Time evolution of Umag_pol according to TRANSP
        u_transp = self.UmagT_pol_dt
        # Actual time evolution of Umag_pol
        u_mitim = np.append([0], np.diff(self.UmagT_pol) / np.diff(self.t))
        # Extra power not accounted by UmagT_pol_dt at each time
        u_diff = u_mitim - u_transp

        DeltaUpol = []
        for it in range(len(self.t)):
            DeltaUpol.append(
                profilePower(
                    self.x_lw,
                    self.dvol[it],
                    u_diff[it],
                    self.x_saw_mix[it],
                    blend=blend,
                )
            )
        self.Psaw = np.array(DeltaUpol)

        self.Poh_corr = self.Poh - self.Psaw
        self.PohT_corr = volumeIntegralTot_var(self.f, self.Poh_corr * 1e-6)

    def getMinorityIons(self):
        # ~~~~~~~~~~~~ Overall

        try:
            self.nmini = self.f["NMINI"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nmini_avol = (
                volumeAverage(self.f, "NMINI") * 1e6 * 1e-20
            )  # in 10^20m^-3

        except:
            print("\t- This plasma had no ICRF minorities")
            self.nmini = self.nD * 0.0 + self.eps00
            self.nmini_avol = self.nD_avol * 0.0 + self.eps00

        self.fmini = self.nmini / self.ne
        self.fmini_avol = self.nmini_avol / self.ne_avol

        # ~~~~~~~~~~~~ He3

        try:
            self.nminiHe3 = self.f["NMINI_3"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nminiHe3_avol = (
                volumeAverage(self.f, "NMINI_3") * 1e6 * 1e-20
            )  # in 10^20m^-3
        except:
            self.nminiHe3 = self.nD * 0.0 + self.eps00
            self.nminiHe3_avol = self.nD_avol * 0.0 + self.eps00

        self.fminiHe3 = self.nminiHe3 / self.ne
        self.fminiHe3_avol = self.nminiHe3_avol / self.ne_avol

        # ~~~~~~~~~~~~ H

        try:
            self.nminiH = self.f["NMINI_H"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nminiH_avol = (
                volumeAverage(self.f, "NMINI_H") * 1e6 * 1e-20
            )  # in 10^20m^-3
        except:
            self.nminiH = self.nD * 0.0 + self.eps00
            self.nminiH_avol = self.nD_avol * 0.0 + self.eps00

        self.fminiH = self.nminiH / self.ne
        self.fminiH_avol = self.nminiH_avol / self.ne_avol

        # ~~~~~~~~~~ Calculate average minority ions, in the same way as it was done with impurities

        if np.sum(self.fmini_avol) > 0.0 + self.eps00 * (len(self.t) + 1):
            self.nminiAVE_Z = (self.nminiHe3 * 2**2 + self.nminiH * 1) / (
                self.nminiHe3 * 2 + self.nminiH * 1
            )
            self.nminiAVE = (self.nminiHe3 * 2 + self.nminiH * 1) / self.nminiAVE_Z

            self.fminiAVE_Z = (self.fminiHe3 * 2**2 + self.fminiH * 1) / (
                self.fminiHe3 * 2 + self.fminiH * 1
            )
            self.fminiAVE = (self.fminiHe3 * 2 + self.fminiH * 1) / self.fminiAVE_Z

            self.fmini_avolAVE_Z = (
                self.fminiHe3_avol * 2**2 + self.fminiH_avol * 1
            ) / (self.fminiHe3_avol * 2 + self.fminiH_avol * 1)
            self.fmini_avolAVE = (
                self.fminiHe3_avol * 2 + self.fminiH_avol * 1
            ) / self.fmini_avolAVE_Z

        else:
            self.nminiAVE_Z = copy.deepcopy(self.ne) * 0.0 + self.eps00
            self.nminiAVE = copy.deepcopy(self.ne) * 0.0 + self.eps00
            self.fminiAVE_Z = copy.deepcopy(self.ne) * 0.0 + self.eps00
            self.fminiAVE = copy.deepcopy(self.ne) * 0.0 + self.eps00
            self.fmini_avolAVE_Z = copy.deepcopy(self.ne_avol) * 0.0 + self.eps00
            self.fmini_avolAVE = copy.deepcopy(self.ne_avol) * 0.0 + self.eps00

    def getFusionIons(self):
        try:
            self.nfus = self.f["NFI"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nfus_avol = volumeAverage(self.f, "NFI") * 1e6 * 1e-20  # in 10^20m^-3

        except:
            print("\t- This plasma had no fusion ions")
            self.nfus = self.nD * 0.0 + self.eps00
            self.nfus_avol = self.nD_avol * 0.0 + self.eps00

        try:
            self.nfusHe4 = self.f["FDENS_4"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nfusHe4_avol = (
                volumeAverage(self.f, "FDENS_4") * 1e6 * 1e-20
            )  # in 10^20m^-3
        except:
            self.nfusHe4 = self.nD * 0.0 + self.eps00
            self.nfusHe4_avol = self.nD_avol * 0.0 + self.eps00

        try:
            self.nfusHe3 = self.f["FDENS_3"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nfusHe3_avol = (
                volumeAverage(self.f, "FDENS_3") * 1e6 * 1e-20
            )  # in 10^20m^-3
        except (KeyError,IndexError):
            self.nfusHe3 = self.nD * 0.0 + self.eps00
            self.nfusHe3_avol = self.nD_avol * 0.0 + self.eps00

        try:
            self.nfusT = self.f["FDENS_T"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nfusT_avol = (
                volumeAverage(self.f, "FDENS_T") * 1e6 * 1e-20
            )  # in 10^20m^-3
        except (KeyError,IndexError):
            self.nfusT = self.nD * 0.0 + self.eps00
            self.nfusT_avol = self.nD_avol * 0.0 + self.eps00

        try:
            self.nfusH = self.f["FDENS_P"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nfusH_avol = (
                volumeAverage(self.f, "FDENS_P") * 1e6 * 1e-20
            )  # in 10^20m^-3
        except (KeyError,IndexError):
            self.nfusH = self.nD * 0.0 + self.eps00
            self.nfusH_avol = self.nD_avol * 0.0 + self.eps00

        self.ffus = self.nfus / self.ne
        self.ffus_avol = self.nfus_avol / self.ne_avol

        self.ffusHe4 = self.nfusHe4 / self.ne
        self.ffusHe4_avol = self.nfusHe4_avol / self.ne_avol
        self.ffusHe3 = self.nfusHe3 / self.ne
        self.ffusHe3_avol = self.nfusHe3_avol / self.ne_avol
        self.ffusT = self.nfusT / self.ne
        self.ffusT_avol = self.nfusT_avol / self.ne_avol
        self.ffusH = self.nfusH / self.ne
        self.ffusH_avol = self.nfusH_avol / self.ne_avol

        # ~~~~~~~~~~ Calculate average fusion ions, in the same way as it was done with impurities

        if np.sum(self.nfus_avol) > 0.0 + self.eps00 * (len(self.t) + 1):
            self.nfusAVE_Z = (
                self.nfusHe4 * 2**2
                + self.nfusHe3 * 2**2
                + self.nfusH * 1
                + self.nfusT * 1
            ) / (self.nfusHe4 * 2 + self.nfusHe3 * 2 + self.nfusH * 1 + self.nfusT * 1)
            self.nfusAVE = (
                self.nfusHe4 * 2 + self.nfusHe3 * 2 + self.nfusH * 1 + self.nfusT * 1
            ) / self.nfusAVE_Z

            self.nfusAVE = correctMasked(self.nfusAVE)
            self.nfusAVE_Z = correctMasked(self.nfusAVE_Z)

            self.ffusAVE_Z = (
                self.ffusHe4 * 2**2
                + self.ffusHe3 * 2**2
                + self.ffusH * 1
                + self.ffusT * 1
            ) / (self.ffusHe4 * 2 + self.ffusHe3 * 2 + self.ffusH * 1 + self.ffusT * 1)
            self.ffusAVE = (
                self.ffusHe4 * 2 + self.ffusHe3 * 2 + self.ffusH * 1 + self.ffusT * 1
            ) / self.ffusAVE_Z

            self.ffusAVE = correctMasked(self.ffusAVE)
            self.ffusAVE_Z = correctMasked(self.ffusAVE_Z)

            self.ffus_avolAVE_Z = (
                self.ffusHe4_avol * 2**2
                + self.ffusHe3_avol * 2**2
                + self.ffusH_avol * 1
                + self.ffusT_avol
            ) / (
                self.ffusHe4_avol * 2
                + self.ffusHe3_avol * 2
                + self.ffusH_avol * 1
                + self.ffusT_avol
            )
            self.ffus_avolAVE = (
                self.ffusHe4_avol * 2
                + self.ffusHe3_avol * 2
                + self.ffusH_avol * 1
                + self.ffusT_avol
            ) / self.ffus_avolAVE_Z

            self.ffus_avolAVE = correctMasked(self.ffus_avolAVE)
            self.ffus_avolAVE_Z = correctMasked(self.ffus_avolAVE_Z)

        else:
            self.nfusAVE_Z = copy.deepcopy(self.ne) * 0.0 + self.eps00
            self.nfusAVE = copy.deepcopy(self.ne) * 0.0 + self.eps00
            self.ffusAVE_Z = copy.deepcopy(self.ne) * 0.0 + self.eps00
            self.ffusAVE = copy.deepcopy(self.ne) * 0.0 + self.eps00
            self.ffus_avolAVE_Z = copy.deepcopy(self.ne_avol) * 0.0 + self.eps00
            self.ffus_avolAVE = copy.deepcopy(self.ne_avol) * 0.0 + self.eps00

    def getBeamIons(self):
        try:
            self.nb = self.f["BDENS"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nb_avol = volumeAverage(self.f, "BDENS") * 1e6 * 1e-20  # in 10^20m^-3
        except:
            print("\t- This plasma had no beam ions")
            self.nb = self.nD * 0.0 + self.eps00
            self.nb_avol = self.nD_avol * 0.0 + self.eps00

        try:
            self.nbD = self.f["BDENS_D"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nbD_avol = (
                volumeAverage(self.f, "BDENS_D") * 1e6 * 1e-20
            )  # in 10^20m^-3
        except:
            self.nbD = self.nD * 0.0 + self.eps00
            self.nbD_avol = self.nD_avol * 0.0 + self.eps00

        try:
            self.nbT = self.f["BDENS_T"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nbT_avol = (
                volumeAverage(self.f, "BDENS_T") * 1e6 * 1e-20
            )  # in 10^20m^-3
        except:
            self.nbT = self.nD * 0.0 + self.eps00
            self.nbT_avol = self.nD_avol * 0.0 + self.eps00

        try:
            self.nbH = self.f["BDENS_H"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nbH_avol = (
                volumeAverage(self.f, "BDENS_H") * 1e6 * 1e-20
            )  # in 10^20m^-3
        except:
            self.nbH = self.nD * 0.0 + self.eps00
            self.nbH_avol = self.nD_avol * 0.0 + self.eps00

        self.fb = self.nb / self.ne
        self.fb_avol = self.nb_avol / self.ne_avol

        self.fbD = self.nbD / self.ne
        self.fbD_avol = self.nbD_avol / self.ne_avol

        self.fbT = self.nbT / self.ne
        self.fbT_avol = self.nbT_avol / self.ne_avol

        self.fbH = self.nbH / self.ne
        self.fbH_avol = self.nbH_avol / self.ne_avol

        # ~~~~~~~~~~ Calculate average beam ions, in the same way as it was done with impurities

        if np.sum(self.nb_avol) > 0.0 + self.eps00 * (len(self.t) + 1):
            self.nbAVE_Z = (self.nbH * 1 + self.nbT * 1 + self.nbD * 1) / (
                self.nbH * 1 + self.nbT * 1 + self.nbD * 1
            )
            self.nbAVE = (self.nbH * 1 + self.nbT * 1 + self.nbD * 1) / self.nbAVE_Z

            self.fbAVE_Z = (self.fbH * 1 + self.fbT * 1 + self.fbD * 1) / (
                self.fbH * 1 + self.fbT * 1 + self.fbD * 1
            )
            self.fbAVE = (self.fbH * 1 + self.fbT * 1 + self.fbD * 1) / self.fbAVE_Z

            self.fb_avolAVE_Z = (self.fbH_avol * 1 + self.fbT_avol + self.fbD_avol) / (
                self.fbH_avol * 1 + self.fbT_avol + self.fbD_avol
            )
            self.fb_avolAVE = (
                self.fbH_avol * 1 + self.fbT_avol + self.fbD_avol
            ) / self.fb_avolAVE_Z

        else:
            self.nbAVE_Z = copy.deepcopy(self.ne) * 0.0 + self.eps00
            self.nbAVE = copy.deepcopy(self.ne) * 0.0 + self.eps00
            self.fbAVE_Z = copy.deepcopy(self.ne) * 0.0 + self.eps00
            self.fbAVE = copy.deepcopy(self.ne) * 0.0 + self.eps00
            self.fb_avolAVE_Z = copy.deepcopy(self.ne_avol) * 0.0 + self.eps00
            self.fb_avolAVE = copy.deepcopy(self.ne_avol) * 0.0 + self.eps00

    def getFastIons(self):
        # ~~~~~~~~~~~~~~~~~~~~~ Densities ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.getMinorityIons()
        self.getFusionIons()
        self.getBeamIons()

        # =========== TOTAL ===========

        self.nFast = self.nmini + self.nfus + self.nb
        self.nFast_avol = self.nmini_avol + self.nfus_avol + self.nb_avol
        self.fFast = self.fmini + self.ffus + self.fb
        self.fFast_avol = self.fmini_avol + self.ffus_avol + self.fb_avol

        self.nFastAVE_Z = (
            self.nbAVE * self.nbAVE_Z**2
            + self.nfusAVE * self.nfusAVE_Z**2
            + self.nminiAVE * self.nminiAVE_Z**2
        ) / (
            self.nbAVE * self.nbAVE_Z
            + self.nfusAVE * self.nfusAVE_Z
            + self.nminiAVE * self.nminiAVE_Z
        )
        self.nFastAVE = (
            self.nbAVE * self.nbAVE_Z
            + self.nfusAVE * self.nfusAVE_Z
            + self.nminiAVE * self.nminiAVE_Z
        ) / self.nFastAVE_Z

        self.fFastAVE_Z = (
            self.fbAVE * self.fbAVE_Z**2
            + self.ffusAVE * self.ffusAVE_Z**2
            + self.fminiAVE * self.fminiAVE_Z**2
        ) / (
            self.fbAVE * self.fbAVE_Z
            + self.ffusAVE * self.ffusAVE_Z
            + self.fminiAVE * self.fminiAVE_Z
        )
        self.fFastAVE = (
            self.fbAVE * self.fbAVE_Z
            + self.ffusAVE * self.ffusAVE_Z
            + self.fminiAVE * self.fminiAVE_Z
        ) / self.fFastAVE_Z

        self.fFast_avolAVE_Z = (
            self.fb_avolAVE * self.fb_avolAVE_Z**2
            + self.ffus_avolAVE * self.ffus_avolAVE_Z**2
            + self.fmini_avolAVE * self.fmini_avolAVE_Z**2
        ) / (
            self.fb_avolAVE * self.fb_avolAVE_Z
            + self.ffus_avolAVE * self.ffus_avolAVE_Z
            + self.fmini_avolAVE * self.fmini_avolAVE_Z
        )
        self.fFast_avolAVE = (
            self.fb_avolAVE * self.fb_avolAVE_Z
            + self.ffus_avolAVE * self.ffus_avolAVE_Z
            + self.fmini_avolAVE * self.fmini_avolAVE_Z
        ) / self.fFast_avolAVE_Z

        # ~~~~~~~~~~~~~~~~~~~~~ Energies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        try:
            self.Wperpx_fus = self.f["UFIPP"][:]  # In MJ/m^3
            self.Wparx_fus = self.f["UFIPA"][:]  # In MJ/m^3
        except:
            self.Wperpx_fus = copy.deepcopy(self.Wperp_x) * 0.0 + self.eps00
            self.Wparx_fus = copy.deepcopy(self.Wperp_x) * 0.0 + self.eps00
        self.pFast_fus = 1 / 2 * self.Wperpx_fus + self.Wparx_fus
        self.pFast_fus_avol = volumeAverage_var(self.f, self.pFast_fus)

        self.Wfast_fus = (
            volumeIntegralTot_var(self.f, self.Wperpx_fus + self.Wparx_fus) * 1e-6
        )

        try:
            self.Wperpx_fusHe4 = self.f["UFPRP_4"][:]  # In MJ/m^3
            self.Wparx_fusHe4 = self.f["UFPAR_4"][:]  # In MJ/m^3
        except:
            self.Wperpx_fusHe4 = copy.deepcopy(self.Wperp_x) * 0.0 + self.eps00
            self.Wparx_fusHe4 = copy.deepcopy(self.Wperp_x) * 0.0 + self.eps00
        self.pFast_fusHe4 = 1 / 2 * self.Wperpx_fusHe4 + self.Wparx_fusHe4

        try:
            self.Wperpx_mini = self.f["UMINPP"][:]  # In MJ/m^3
            self.Wparx_mini = self.f["UMINPA"][:]  # In MJ/m^3
        except:
            self.Wperpx_mini = copy.deepcopy(self.Wperp_x) * 0.0 + self.eps00
            self.Wparx_mini = copy.deepcopy(self.Wperp_x) * 0.0 + self.eps00

        self.pFast_mini = 1 / 2 * self.Wperpx_mini + self.Wparx_mini
        self.pFast_mini_avol = volumeAverage_var(self.f, self.pFast_mini)

        self.Wfast_mini = (
            volumeIntegralTot_var(self.f, self.Wperpx_mini + self.Wparx_mini) * 1e-6
        )

        try:
            self.Wperpx_b = self.f["UBPRP"][:]  # In MJ/m^3
            self.Wparx_b = self.f["UBPAR"][:]  # In MJ/m^3
        except:
            self.Wperpx_b = copy.deepcopy(self.Wperp_x) * 0.0 + self.eps00
            self.Wparx_b = copy.deepcopy(self.Wperp_x) * 0.0 + self.eps00

        self.pFast_b = 1 / 2 * self.Wperpx_b + self.Wparx_b
        self.pFast_b_avol = volumeAverage_var(self.f, self.pFast_b)

        self.Wfast_b = (
            volumeIntegralTot_var(self.f, self.Wperpx_b + self.Wparx_b) * 1e-6
        )

        # ~~~~~~~~~~~~~~~~~~~~~ Temperatures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.Tmini, self.Tmini_check, self.Tmini_perp, self.Tmini_par = (
            copy.deepcopy(self.Te) * 0.0 + self.eps00,
            copy.deepcopy(self.Te) * 0.0 + self.eps00,
            copy.deepcopy(self.Te) * 0.0 + self.eps00,
            copy.deepcopy(self.Te) * 0.0 + self.eps00,
        )
        for i in self.f.keys():
            if "TMINI_" in i:
                self.Tmini = self.f[i][:] * 1e-3  # keV
                self.Tmini_avol = volumeAverage(self.f, i) * 1e-3

                Emini_perp = (
                    self.Wperpx_mini * 1e6 / (self.nmini * 1e20 * self.e_J * 1e3)
                )
                Emini_par = self.Wparx_mini * 1e6 / (self.nmini * 1e20 * self.e_J * 1e3)
                self.Tmini_check = (Emini_perp + Emini_par) * 2.0 / 3.0

                self.Tmini_perp = Emini_perp
                self.Tmini_par = 2.0 * Emini_par

        if np.sum(self.nfus_avol) > 0.0 + self.eps00 * (len(self.t) + 1):
            self.Tfus = (
                2.0
                / 3.0
                * (self.Wperpx_fus + self.Wparx_fus)
                * 1e6
                / (self.nfus * 1e20)
                / self.e_J
                * 1e-3
            )
        else:
            self.Tfus = copy.deepcopy(self.ne) * 0.0 + self.eps00
        if np.sum(self.nb_avol) > 0.0 + self.eps00 * (len(self.t) + 1):
            self.Tb = (
                2.0
                / 3.0
                * (self.Wperpx_b + self.Wparx_b)
                * 1e6
                / (self.nb * 1e20)
                / self.e_J
                * 1e-3
            )
        else:
            self.Tb = copy.deepcopy(self.ne) * 0.0 + self.eps00

        if np.sum(self.nFast_avol) > 0.0 + self.eps00 * (len(self.t) + 1):
            self.Tfast = (
                2.0
                / 3.0
                * (self.Wperp_x + self.Wpar_x)
                * 1e6
                / (self.nFast * 1e20)
                / self.e_J
                * 1e-3
            )
        else:
            self.Tfast = copy.deepcopy(self.ne) * 0.0 + self.eps00

        self.Tfus_avol = volumeAverage_var(self.f, self.Tfus)
        self.Tb_avol = volumeAverage_var(self.f, self.Tb)
        self.Tfast_avol = volumeAverage_var(self.f, self.Tfast)

        # ~~~~~~ Average energy (ends up being 3/2*Tfast) ~~~~~~~~~~~

        try:
            self.Emini = (
                self.f["EMINPER_3"][:] + self.f["EMINPAR_3"][:]
            ) * 1e-3  # in keV
        except:
            self.Emini = copy.deepcopy(self.ne) * 0.0 + self.eps00

        self.Emini_check = (
            (self.Wperpx_mini + self.Wparx_mini)
            * 1e6
            / (self.nmini * 1e20)
            / self.e_J
            * 1e-3
        )

        self.Efus = (
            (self.Wperpx_fus + self.Wparx_fus)
            * 1e6
            / (self.nfus * 1e20)
            / self.e_J
            * 1e-3
        )
        # self.Efus 	= ( self.f['FEPRP2_4'][:] 	+ self.f['FEPLL2_4'][:] )*1E-3	# in keV

        self.Efast = (self.Emini * self.nmini + self.Efus * self.nfus) / (
            self.nmini + self.nfus
        )  # in keV

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fast ions TRANSPORT
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~ Total

        try:
            self.D_f = self.f["DIFB"][:] * 1e-4  # in m^2/s
            self.V_f = self.f["VELB"][:] * 1e-2  # in m/s

            self.Pf_loss_orbit = self.f["BPLIM"][:] * 1e-6  # in MW
            # self.Pf_loss_orbitPrompt 	= self.Pf_loss_orbit * self.f['BSORBPR'][:]

            self.Gf_loss_cx = self.f["SBCXX"][:] * 1e-20  # in 1E20 particles/s
        except:
            self.D_f = self.Te * 0.0
            self.V_f = self.Te * 0.0
            self.Pf_loss_orbit = self.t * 0.0
            self.Gf_loss_cx = self.t * 0.0

        try:
            self.Jr_anom = self.f["CURBRABD"][:] * 1e-6 * 1e4  # MA/m^2
        except:
            self.Jr_anom = self.j * 0.0

        # ~~~~ He4

        try:
            self.DnDt_He4f = self.f["DNFDT_4"][:] * 1e6 * 1e-20  # in 10E20/m^3/s
            self.Pf_loss_orbit_He4 = self.f["BFLIM_4"][:] * 1e-6  # in MW

            self.Pf_loss_cx_He4 = (
                self.f["BFCXI_4"][:] + self.f["BFCXX_4"][:]
            ) * 1e-6  # in MW
            self.Pf_thermalized_He4 = self.f["BFTH_4"][:] * 1e-6  # in MW
            self.Pf_stored_He4 = self.f["BFST_4"][:] * 1e-6  # in MW

            self.Gf_loss_orbit_He4 = self.f["FSORB_4"][:] * 1e-20  # in 1E20 particles/s
            self.Gf_loss_orbitPrompt_He4 = (
                self.f["FSORBPR_4"][:] * self.Gf_loss_orbit_He4
            )

        except:
            self.DnDt_He4f = self.t * 0.0
            self.Pf_loss_orbit_He4 = self.t * 0.0

            self.Pf_loss_cx_He4 = self.t * 0.0
            self.Pf_thermalized_He4 = self.t * 0.0
            self.Pf_stored_He4 = self.t * 0.0

            self.Gf_loss_orbit_He4 = self.t * 0.0
            self.Gf_loss_orbitPrompt_He4 = self.t * 0.0

    def getImpurities(self, maxNumStates=200):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~ Zeff as given as input to TRANSP
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.Zeff = self.f["ZEFFI"][:]
        self.Zeff_avol = volumeAverage(self.f, "ZEFFI")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~ Total
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.nZ = self.f["NIMP"][:] * 1e6 * 1e-20  # in 10^20m^-3
        self.nZ_avol = volumeAverage(self.f, "NIMP") * 1e6 * 1e-20  # in 10^20m^-3
        self.fZ = self.nZ / self.ne
        self.fZ_avol = self.nZ_avol / self.ne_avol
        self.aLnZ = gradNorm(self, self.nZ)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~ Individual impurities
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.nZs = {}
        self.nZs_avol = {}
        self.fZs = {}
        self.fZs_avol = {}

        for i in self.f.keys():
            if "NIMPS_" in i:
                element = i.split("_")[1]

                # ~~~~~~~~~~ Total density of impurities

                self.nZs[element] = {
                    "total": self.f[i][:] * 1e6 * 1e-20
                }  # in 10^20m^-3
                self.nZs_avol[element] = {
                    "total": volumeAverage(self.f, i) * 1e6 * 1e-20
                }  # in 10^20m^-3
                self.fZs[element] = {"total": self.nZs[element]["total"] / self.ne}
                self.fZs_avol[element] = {
                    "total": self.nZs_avol[element]["total"] / self.ne_avol
                }

                # ~~~~~~~~~~ Density of charge states

                self.nZs[element]["states"] = {}
                self.nZs_avol[element]["states"] = {}
                self.fZs[element]["states"] = {}
                self.fZs_avol[element]["states"] = {}

                Zave_imp = copy.deepcopy(self.ne) * 0.0 + self.eps00
                Zave_imp_avol = copy.deepcopy(self.ne_avol) * 0.0 + self.eps00

                Zeff_imp = copy.deepcopy(self.ne) * 0.0 + self.eps00
                Zeff_imp_avol = copy.deepcopy(self.ne_avol) * 0.0 + self.eps00

                for j in range(maxNumStates):
                    if f"NIMP_{element}_{j + 1}" not in self.f.keys():
                        self.nZs[element]["states"][j + 1] = (
                            copy.deepcopy(self.ne) * 0.0 + self.eps00
                        )
                        self.nZs_avol[element]["states"][j + 1] = (
                            copy.deepcopy(self.ne_avol) * 0.0 + self.eps00
                        )
                        self.fZs[element]["states"][j + 1] = (
                            copy.deepcopy(self.ne) * 0.0 + self.eps00
                        )
                        self.fZs_avol[element]["states"][j + 1] = (
                            copy.deepcopy(self.ne_avol) * 0.0 + self.eps00
                        )
                    else:
                        self.nZs[element]["states"][j + 1] = (
                            self.f[f"NIMP_{element}_{j + 1}"][:] * 1e6 * 1e-20
                        )  # in 10^20m^-3
                        self.nZs_avol[element]["states"][j + 1] = (
                            volumeAverage(self.f, f"NIMP_{element}_{j + 1}")
                            * 1e6
                            * 1e-20
                        )  # in 10^20m^-3
                        self.fZs[element]["states"][j + 1] = (
                            self.nZs[element]["states"][j + 1] / self.ne
                        )
                        self.fZs_avol[element]["states"][j + 1] = (
                            self.nZs_avol[element]["states"][j + 1] / self.ne_avol
                        )

                    Zave_imp += (
                        self.nZs[element]["states"][j + 1]
                        / self.nZs[element]["total"]
                        * (j + 1)
                    )
                    Zave_imp_avol += (
                        self.fZs_avol[element]["states"][j + 1]
                        / self.fZs_avol[element]["total"]
                        * (j + 1)
                    )

                    Zeff_imp += (
                        self.nZs[element]["states"][j + 1]
                        / self.nZs[element]["total"]
                        * (j + 1) ** 2
                    )
                    Zeff_imp_avol += (
                        self.fZs_avol[element]["states"][j + 1]
                        / self.fZs_avol[element]["total"]
                        * (j + 1) ** 2
                    )

                # ~~~~~~~~~~ Zave of individual impurities in radius and in volume average

                self.nZs[element]["Zave"] = Zave_imp
                self.fZs_avol[element]["Zave"] = Zave_imp_avol

                self.nZs[element]["Zeff"] = np.sqrt(Zeff_imp)
                self.fZs_avol[element]["Zeff"] = np.sqrt(Zeff_imp_avol)

        # Some times it is not stored as charge states but I can recover the impurity charge

        if (
            len(self.nZs.keys()) == 1
            and np.ma.is_masked(self.nZs[element]["Zave"])
            and element == "TOK"
        ):
            self.fZs_avol[element]["Zave"] = self.f["XZIMP"][:]
            self.nZs[element]["Zave"] = constant_radius(
                self.fZs_avol[element]["Zave"], lenX=100
            )
            self.fZs_avol[element]["Zeff"] = self.fZs_avol[element]["Zave"]
            self.nZs[element]["Zeff"] = self.nZs[element]["Zave"]

            zav = int(np.mean(self.fZs_avol[element]["Zave"]))
            self.nZs[element]["states"][zav] = self.nZ
            self.nZs_avol[element]["states"][zav] = self.nZ_avol
            self.fZs[element]["states"][zav] = (
                self.nZs[element]["states"][zav] / self.ne
            )
            self.fZs_avol[element]["states"][zav] = (
                self.nZs_avol[element]["states"][zav] / self.ne_avol
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~ Averaging
        # single impurity that fulfills quasineutrality and Zeff with only one Zave (but NIMP changes)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        summ = copy.deepcopy(self.ne) * 0.0
        summ2 = copy.deepcopy(self.ne) * 0.0
        summA = copy.deepcopy(self.ne) * 0.0
        summA2 = copy.deepcopy(self.ne) * 0.0
        summB = copy.deepcopy(self.t) * 0.0
        summB2 = copy.deepcopy(self.t) * 0.0
        for element in self.nZs:
            for state in self.nZs[element]["states"]:
                summ += state * self.nZs[element]["states"][state]
                summ2 += state**2 * self.nZs[element]["states"][state]

                summA += state * self.fZs[element]["states"][state]
                summA2 += state**2 * self.fZs[element]["states"][state]

                summB += state * self.fZs_avol[element]["states"][state]
                summB2 += state**2 * self.fZs_avol[element]["states"][state]

        self.nZAVE_Z = summ2 / summ
        self.nZAVE = summ / self.nZAVE_Z

        self.fZAVE_Z = summA2 / summA
        self.fZAVE = summA / self.fZAVE_Z

        self.fZ_avolAVE_Z = summB2 / summB
        self.fZ_avolAVE = summB / self.fZ_avolAVE_Z

    def getNeutrals(self):
        # ~~~~~~~~~~~~~~~~ Deuterium
        try:
            self.nD0_vol = self.f["DN0VD"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nD0_wall = self.f["DN0WD"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nD0 = self.nD0_vol + self.nD0_wall

            self.nD0_recy = self.f["N0SRC_D"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nD0_gasf = self.f["N0SGF_D"][:] * 1e6 * 1e-20  # in 10^20m^-3

            # Temperature
            self.TD0_vol = self.f["T0VD"][:] * 1e-3  # in keV
            self.TD0_wall = self.f["T0WD"][:] * 1e-3  # in keV

        except:
            self.nD0_vol = self.ne * 0.0 + self.eps00
            self.nD0_wall = self.ne * 0.0 + self.eps00
            self.nD0 = self.nD0_vol + self.nD0_wall

            self.nD0_recy = self.ne * 0.0 + self.eps00
            self.nD0_gasf = self.ne * 0.0 + self.eps00

            # Temperature
            self.TD0_vol = self.ne * 0.0 + self.eps00
            self.TD0_wall = self.ne * 0.0 + self.eps00

        # Pressure
        self.pD0_vol = self.TD0_vol * 1e3 * self.e_J * self.nD0_vol * 1e20 * 1e-6  # MPa
        self.pD0_wall = (
            self.TD0_wall * 1e3 * self.e_J * self.nD0_wall * 1e20 * 1e-6
        )  # MPa
        self.pD0 = self.pD0_vol + self.pD0_wall

        # ~~~~~~~~~~~~~~~~ Tritium
        try:
            self.nT0_vol = self.f["DN0VT"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nT0_wall = self.f["DN0WT"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nT0 = self.nT0_vol + self.nT0_wall

            self.nT0_recy = self.f["N0SRC_T"][:] * 1e6 * 1e-20  # in 10^20m^-3
            self.nT0_gasf = self.f["N0SGF_T"][:] * 1e6 * 1e-20  # in 10^20m^-3

            self.TT0_vol = self.f["T0VT"][:] * 1e-3  # in keV
            self.TT0_wall = self.f["T0WT"][:] * 1e-3  # in keV

            # Pressure
            self.pT0_vol = (
                self.TT0_vol * 1e3 * self.e_J * self.nT0_vol * 1e20 * 1e-6
            )  # MPa
            self.pT0_wall = (
                self.TT0_wall * 1e3 * self.e_J * self.nT0_wall * 1e20 * 1e-6
            )  # MPa
            self.pT0 = self.pT0_vol + self.pT0_wall

        except:
            self.nT0_vol = self.nD0 * 0.0 + self.eps00
            self.nT0_wall = self.nD0 * 0.0 + self.eps00
            self.nT0 = self.nD0 * 0.0 + self.eps00
            self.nT0_recy = self.nD0 * 0.0 + self.eps00
            self.nT0_gasf = self.nD0 * 0.0 + self.eps00

            self.TT0_vol = self.nD0 * 0.0 + self.eps00
            self.TT0_wall = self.nD0 * 0.0 + self.eps00

            self.pT0_vol = self.nD0 * 0.0 + self.eps00
            self.pT0_wall = self.nD0 * 0.0 + self.eps00
            self.pT0 = self.nD0 * 0.0 + self.eps00

        # ~~~~~~~~~~~~~~~~ Total main ions

        self.nmain0 = self.nD0 + self.nT0
        self.pmain0 = self.pD0 + self.pT0

        # try:
        # 	self.nmain0_avol 	= (volumeAverage(self.f,'DN0VD')+
        # 					   volumeAverage(self.f,'DN0WD')+
        # 					   volumeAverage(self.f,'DN0VT')+
        # 					   volumeAverage(self.f,'DN0WT')) * 1E6 *1E-20 # in 10^20m^-3
        # except:
        # 	self.nmain0_avol 	= (volumeAverage(self.f,'DN0VD')+
        # 					   volumeAverage(self.f,'DN0WD')) * 1E6 *1E-20 # in 10^20m^-3

    def getParticleBalance(self):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~ Electrons
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.ne_dt = self.f["DNEDT"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
        self.ne_divGamma = self.f["DIVFE"][:] * 1e6 * 1e-20  # in 10^20m^-3/s

        # ~~~~~ Sources

        self.ne_source_wall = self.f["SCEW"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
        self.ne_source_wallT = volumeIntegralTot(self.f, "SCEW") * 1e-20  # in 10^20/s

        self.ne_source_vol = self.f["SCEV"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
        self.ne_source_volT = volumeIntegralTot(self.f, "SCEV") * 1e-20  # in 10^20/s

        self.ne_source_imp = self.f["SCEZ"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
        self.ne_source_impT = volumeIntegralTot(self.f, "SCEZ") * 1e-20  # in 10^20/s

        try:
            self.ne_source_fi = self.f["SBE"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
            self.ne_source_fiT = volumeIntegralTot(self.f, "SBE") * 1e-20  # in 10^20/s
        except:
            self.ne_source_fi = copy.deepcopy(self.ne_source_wall) * 0.0 + self.eps00
            self.ne_source_fiT = copy.deepcopy(self.t) * 0.0 + self.eps00

        self.ne_source = (
            self.ne_source_wall
            + self.ne_source_vol
            + self.ne_source_imp
            + self.ne_source_fi
        )  # in 10^20m^-3/s
        self.ne_sourceT = (
            self.ne_source_wallT
            + self.ne_source_volT
            + self.ne_source_impT
            + self.ne_source_fiT
        )  # in 10^20/s

        # ~~~~~ Beams

        try:
            self.ne_source_bD = self.f["SBE_D"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
        except:
            self.ne_source_bD = copy.deepcopy(self.ne_source_wall) * 0.0 + self.eps00

        # ~~~~~ Wall

        try:
            self.ne_source_gfD = self.f["SEGF_D"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
        except:
            self.ne_source_gfD = copy.deepcopy(self.ne_source_bD) * 0.0 + self.eps00
        try:
            self.ne_source_gfT = self.f["SEGF_T"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
        except:
            self.ne_source_gfT = copy.deepcopy(self.ne_source_gfD) * 0.0 + self.eps00
        try:
            self.ne_source_gfHe4 = self.f["SEGF_4"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
        except:
            self.ne_source_gfHe4 = copy.deepcopy(self.ne_source_gfD) * 0.0 + self.eps00
        self.ne_source_gf = self.f["SESGF"][:] * 1e6 * 1e-20  # in 10^20m^-3/s

        try:
            self.ne_source_rcyD = self.f["SERC_D"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
        except:
            self.ne_source_rcyD = copy.deepcopy(self.ne_source_gfD) * 0.0 + self.eps00
        try:
            self.ne_source_rcyT = self.f["SERC_T"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
        except:
            self.ne_source_rcyT = copy.deepcopy(self.ne_source_gfD) * 0.0 + self.eps00
        try:
            self.ne_source_rcyHe4 = self.f["SERC_4"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
        except:
            self.ne_source_rcyHe4 = copy.deepcopy(self.ne_source_gfD) * 0.0 + self.eps00
        self.ne_source_rcy = self.f["SESRC"][:] * 1e6 * 1e-20  # in 10^20m^-3/s

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~` Deuterium
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        try:
            self.nD_dt = self.f["DNDDT"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
            self.nD_divGamma = self.f["DIVFD"][:] * 1e6 * 1e-20  # in 10^20m^-3/s

            self.nD_source_wall = self.f["SWD"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
            self.nD_source_wallT = (
                volumeIntegralTot(self.f, "SWD") * 1e-20
            )  # in 10^20/s

            self.nD_source_vol = self.f["SVD"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
            self.nD_source_volT = volumeIntegralTot(self.f, "SVD") * 1e-20  # in 10^20/s

            self.nD_source = self.nD_source_wall + self.nD_source_vol
            self.nD_sourceT = self.nD_source_wallT + self.nD_source_volT

        except:
            self.nD_dt = copy.deepcopy(self.ne_source_rcy) * 0.0 + self.eps00
            self.nD_divGamma = copy.deepcopy(self.ne_source_rcy) * 0.0 + self.eps00

            self.nD_source = copy.deepcopy(self.ne_source_rcy) * 0.0 + self.eps00
            self.nD_sourceT = copy.deepcopy(self.t) * 0.0 + self.eps00

            self.nD_source_wall = copy.deepcopy(self.ne_source_rcy) * 0.0 + self.eps00
            self.nD_source_wallT = copy.deepcopy(self.t) * 0.0 + self.eps00

            self.nD_source_vol = copy.deepcopy(self.ne_source_rcy) * 0.0 + self.eps00
            self.nD_source_volT = copy.deepcopy(self.t) * 0.0 + self.eps00

        try:
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~` Tritum
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            self.nT_source_wall = self.f["SWT"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
            self.nT_source_vol = self.f["SVT"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
            self.nT_dt = self.f["DNTDT"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
            self.nT_divGamma = self.f["DIVFT"][:] * 1e6 * 1e-20  # in 10^20m^-3/s

            self.nT_source_wallT = (
                volumeIntegralTot(self.f, "SWT") * 1e-20
            )  # in 10^20/s
            self.nT_source_volT = volumeIntegralTot(self.f, "SVT") * 1e-20  # in 10^20/s
            self.nT_sourceT = self.nT_source_wallT + self.nT_source_volT

        except:
            self.nT_dt = copy.deepcopy(self.ne_source_rcy) * 0.0 + self.eps00
            self.nT_divGamma = copy.deepcopy(self.ne_source_rcy) * 0.0 + self.eps00

            self.nT_source = copy.deepcopy(self.ne_source_rcy) * 0.0 + self.eps00
            self.nT_sourceT = copy.deepcopy(self.t) * 0.0 + self.eps00

            self.nT_source_wall = copy.deepcopy(self.ne_source_rcy) * 0.0 + self.eps00
            self.nT_source_wallT = copy.deepcopy(self.t) * 0.0 + self.eps00

            self.nT_source_vol = copy.deepcopy(self.ne_source_rcy) * 0.0 + self.eps00
            self.nT_source_volT = copy.deepcopy(self.t) * 0.0 + self.eps00

        self.nT_source = self.nT_source_wall + self.nT_source_vol
        self.nT_sourceT = self.nT_source_wallT + self.nT_source_volT

        #

        self.nmain_source = self.nD_source + self.nT_source
        self.nmain_sourceT = self.nD_sourceT + self.nT_sourceT

        try:
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~` He-4
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            self.nHe4_source_wall = self.f["SWHE4"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
            self.nHe4_source_vol = self.f["SVHE4"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
            self.nHe4_dt = self.f["DNHE4DT"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
            self.nHe4_divGamma = self.f["DIVHE4T"][:] * 1e6 * 1e-20  # in 10^20m^-3/s

            self.nHe4_source_wallT = (
                volumeIntegralTot(self.f, "SWHE4") * 1e-20
            )  # in 10^20/s
            self.nHe4_source_volT = (
                volumeIntegralTot(self.f, "SVHE4") * 1e-20
            )  # in 10^20/s
        except:
            self.nHe4_source_wall = (
                copy.deepcopy(self.nD_source_wall) * 0.0 + self.eps00
            )
            self.nHe4_source_vol = copy.deepcopy(self.nD_source_wall) * 0.0 + self.eps00
            self.nHe4_dt = copy.deepcopy(self.nD_source_wall) * 0.0 + self.eps00
            self.nHe4_divGamma = copy.deepcopy(self.nD_source_wall) * 0.0 + self.eps00
            self.nHe4_source_wallT = copy.deepcopy(self.t) * 0.0 + self.eps00
            self.nHe4_source_volT = copy.deepcopy(self.t) * 0.0 + self.eps00

        self.nHe4_source = self.nHe4_source_wall + self.nHe4_source_vol
        self.nHe4_sourceT = self.nHe4_source_wallT + self.nHe4_source_volT

    def calculateCurrentDiffusion(self):
        # Sawtooth period
        self.tau_saw = np.diff(self.tlastsawU) * 1e3  # in ms
        t = []
        for i in range(len(self.tlastsawU) - 1):
            t.append(np.mean(self.tlastsawU[i : i + 2]))
        tau_saw_t = np.array(t)

        if len(tau_saw_t) > 0:
            self.tau_saw = np.interp(self.t, tau_saw_t, self.tau_saw)
        else:
            self.tau_saw = np.ones(len(self.t)) * self.t[-1]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Formulas current diffusion
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # 0D formula in MARS (Bonoli?)

        self.tau_c_MARS = (
            1.4
            * self.a**2
            * self.kappa
            * self.Te_avol ** (1.5)
            / self.Zeff_avol
            * 1000.0
        )  # ms

        # 0D formula based on neoclassical resistivity (Wesson 2.16.2?)

        nu_spitz = 2.8 * 1e-8 * self.Te_avol ** (-1.5)
        nu_neo = nu_spitz * self.Zeff_avol * 0.9 / (1 - self.epsilon**0.5) ** 2
        self.tau_c_res0 = self.mu0 * self.a**2 / nu_neo * 1000.0  # ms

        # Current diffusion from volume average resistivity inside of pedestal

        self.eta = self.f["ETA_USE"][:] * 1e-2  # Ohm*m
        self.eta_avol = (
            volumeAverage(self.f, "ETA_USE") * 1e-2
        )  # ,rangex=np.argmin(np.abs(self.x_lw-0.9))) * 1E-2 # Ohm*m

        self.etas_nc = self.f["ETA_NC"][:] * 1e-2  # Ohm*m
        self.etas_sp = self.f["ETA_SP"][:] * 1e-2  # Ohm*m
        self.etas_sps = self.f["ETA_SPS"][:] * 1e-2  # Ohm*m
        self.etas_wnc = self.f["ETA_WNC"][:] * 1e-2  # Ohm*m
        self.etas_tsc = self.f["ETA_TSC"][:] * 1e-2  # Ohm*m
        self.etas_snc = self.f["ETA_SNC"][:] * 1e-2  # Ohm*m

        self.tau_c_res1 = self.mu0 * self.a**2 / self.eta_avol * 1000.0  # in ms

        # ------ Current diffusion
        self.tau_c = self.tau_c_MARS

    def getPowers(self):
        # ------------------ LOSSES ------------------------------

        # Radiation
        self.getRADinfo()

        # Charge-exchange
        self.Pcx = self.f["P0NET"][:]  # MW/m^3
        self.PcxT = volumeIntegralTot(self.f, "P0NET") * 1e-6  # MW
        self.PcxT_cum = volumeIntegral(self.f, "P0NET") * 1e-6  # MW

        self.Losses_ions = self.PcxT + self.Pi_LCFS
        self.Losses_elec = self.PradT + self.Pe_LCFS
        self.Losses = self.Losses_ions + self.Losses_elec

        # ------------------ POWER IN ---------------------------

        # Ohmic
        self.Poh = self.f["POH"][:]  # MW/m^3
        self.PohT = self.f["POHT"][:] * 1e-6

        self.Poh_checkJ = (self.eta * self.jOh * 1e6) * (self.j * 1e6) * 1e-6
        self.PohT_checkJ = volumeIntegralTot_var(self.f, self.Poh_checkJ) * 1e-6

        self.Poh_checkJ_NCLASS = (
            (self.etas_wnc * self.jOh * 1e6) * (self.j * 1e6) * 1e-6
        )
        self.PohT_checkJ_NCLASS = (
            volumeIntegralTot_var(self.f, self.Poh_checkJ_NCLASS) * 1e-6
        )

        self.Poh_checkJ_Sauter = (
            (self.etas_snc * self.jOh * 1e6) * (self.j * 1e6) * 1e-6
        )
        self.PohT_checkJ_Sauter = (
            volumeIntegralTot_var(self.f, self.Poh_checkJ_Sauter) * 1e-6
        )

        self.Poh_checkJ_Sauter2 = (self.etas_snc * self.j * 1e6) * (self.j * 1e6) * 1e-6
        self.PohT_checkJ_Sauter2 = (
            volumeIntegralTot_var(self.f, self.Poh_checkJ_Sauter2) * 1e-6
        )

        # ICRF
        self.getICRFinfo()

        # ECRF
        self.getECRFinfo()

        # LH
        self.getLHinfo()

        # NBI
        self.getNBIinfo()

        # Alpha
        try:
            self.Pfuse = self.f["PFE"][:]  # MW/m^3
            self.Pfusi = self.f["PFI"][:]  # MW/m^3
            self.PfuseT = volumeIntegralTot(self.f, "PFE") * 1e-6  # MW
            self.PfusiT = volumeIntegralTot(self.f, "PFI") * 1e-6  # MW
            if np.sum(self.PfuseT + self.PfusiT) > 0.0 + self.eps00 * (len(self.t) + 1):
                self.Pfuse_frac = np.array(
                    [
                        x / y if y else 0
                        for x, y in zip(self.PfuseT, (self.PfuseT + self.PfusiT))
                    ]
                )
                self.Pfusi_frac = 1 - self.Pfuse_frac
            else:
                self.Pfuse_frac = 0.0
                self.Pfusi_frac = 0.0

            # Cumulative power
            self.Pfuse_cum = volumeMultiplication(self.f, "PFE") * 1e-6  # in MW
            self.Pfusi_cum = volumeMultiplication(self.f, "PFI") * 1e-6  # in MW

        except:
            self.Pfuse = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Pfusi = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.PfuseT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.PfusiT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.Pfuse_frac = 0.0
            self.Pfusi_frac = 0.0
            self.Pfuse_cum = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Pfusi_cum = copy.deepcopy(self.Poh) * 0.0 + self.eps00

        self.PfuseV = self.Pfuse * (self.f["DVOL"][:] * 1e-6)
        self.PfusiV = self.Pfusi * (self.f["DVOL"][:] * 1e-6)
        self.PfusT = self.PfuseT + self.PfusiT

        # Fast ions
        try:
            self.Heat_FastIonsT = self.f["BPHTO"][:]  # MW
        except:
            self.Heat_FastIonsT = copy.deepcopy(self.f["QIE"][:]) * 0.0

        # ------------------ EXTRAS ---------------------------

        # Equilibration
        self.Pei = self.f["QIE"][
            :
        ]  # MW/m^3						# Power from electrons to ions (positive if electrons lose power)
        self.PeiT = volumeIntegralTot(self.f, "QIE") * 1e-6  # MW
        self.PeiT_cum = volumeIntegral(self.f, "QIE") * 1e-6  # MW

        # Compression
        self.Pcompe = self.f["PCMPE"][:]  # MW/m^3
        self.Pcompi = self.f["PCMPI"][:]  # MW/m^3
        self.PcompeT = volumeIntegralTot(self.f, "PCMPE") * 1e-6  # MW
        self.PcompiT = volumeIntegralTot(self.f, "PCMPI") * 1e-6  # MW

        # Gain
        self.Gaine = self.f["GAINE"][:]  # MW/m^3
        self.Gaini = self.f["GAINI"][:]  # MW/m^3
        self.GaineT = volumeIntegralTot(self.f, "GAINE") * 1e-6  # MW
        self.GainiT = volumeIntegralTot(self.f, "GAINI") * 1e-6  # MW

        self.GaineT_cum = volumeIntegral(self.f, "GAINE") * 1e-6  # MW
        self.GainiT_cum = volumeIntegral(self.f, "GAINI") * 1e-6  # MW

        self.GainT = self.GaineT + self.GainiT

        # Others
        try:
            self.Pi_others = self.f["QROT"][:] + self.f["PBTH"][:]
        except:
            self.Pi_others = self.Gaine * 0.0 + self.eps00

        # ------------------ Composition ---------------------------

        self.Piheat = self.f["IHEAT"][:]  # MW/m^3
        self.Peheat = self.f["EHEAT"][:]  # MW/m^3

        self.PiheatT = volumeIntegralTot(self.f, "IHEAT") * 1e-6  # MW
        self.PeheatT = volumeIntegralTot(self.f, "EHEAT") * 1e-6  # MW

        self.PiheatT_cum = volumeIntegral(self.f, "IHEAT") * 1e-6  # in MW
        self.PeheatT_cum = volumeIntegral(self.f, "EHEAT") * 1e-6  # in MW

        self.utilsiliarPower_elec = self.PeichT + self.PohT + self.PechT + self.PnbieT
        self.utilsiliarPower_ions = self.PiichT + self.PnbiiT
        self.utilsiliarPower_fast = self.PnbihT

        # -------
        self.Pe = (
            self.Pcompe
            + self.Peich
            + self.Pech
            + self.Pnbie
            + self.Pfuse
            + self.Poh
            + self.Plhe
        )
        self.Pi = self.Pcompi + self.Piich + self.Pnbii + self.Pfusi + self.Plhi

        self.Pe_teo = self.Pe - self.Prad - self.Pei
        self.Pi_teo = self.Pi + self.Pei - self.Pcx

        # ______________ Compound ____________________________

        # Auxiliar heating

        self.utilsiliarPower_elec = (
            self.PeichT + self.PohT + self.PechT + self.PnbieT + self.PlheT
        )
        self.utilsiliarPower_ions = (
            self.PiichT + self.PnbiiT + self.PlhiT
        )  # should be equal to IHEAT is elf.PiichT+self.PnbiiT
        self.utilsiliarPower_fast = self.PnbihT

        # self.utilsiliarPower 		= self.utilsiliarPower_elec + self.utilsiliarPower_ions
        self.utilsiliarPower = (
            self.PichT + self.PohT + self.PechT + self.PnbiT + self.PlhT
        )

        # Others
        self.PowerIn_other = self.PnbirT + self.PnbihT
        self.PowerOut_other = self.PnbicT + self.PnbilT

        # Total Heating

        self.PowerToElec = self.utilsiliarPower_elec + self.PfuseT
        self.PowerToIons = self.utilsiliarPower_ions + self.PfusiT

        # Total power expressions
        self.Ptot = self.PowerToElec + self.PowerToIons
        self.PtotExtra = self.Ptot + self.PowerIn_other
        self.Ptot_check = (
            self.f["PL2HTOT"][:] * 1e-6
        )  # should be equal to Ptot, but sometimes this variable fails?

        self.PowerToElec_frac = self.PowerToElec / self.Ptot
        self.PowerToIons_frac = 1 - self.PowerToElec_frac

    def getCollisionality(self):
        # Coulomb logarithm
        self.LambdaCoul_e, self.LambdaCoul_i = PLASMAtools.calculateCoulombLogarithm(
            self.Te, self.ne * 1e20, Z=1, ni=self.ni * 1e20, Ti=self.Ti
        )
        self.LambdaCoul_e_TRANSP = self.f["CLOGE"][:]
        self.LambdaCoul_i_TRANSP = self.f["CLOGI"][:]

        # Local collisionalities
        (
            self.nu_ei,
            self.nu_eff,
            self.nu_star,
            self.nu_norm,
            self.nu_eff_Angioni,
        ) = PLASMAtools.calculateCollisionalities(
            self.ne * 1e20,
            self.Te,
            self.Zeff,
            self.RmajorS,
            self.q,
            self.epsilonS,
            self.ne_avol,
            self.Te_avol,
            self.Zeff_avol,
            self.Rmajor,
            self.mi / self.u,
        )

        # Volume-averaged
        self.nu_ei_avol = volumeAverage_var(self.f, self.nu_ei)
        self.nu_eff_avol = volumeAverage_var(self.f, self.nu_eff)
        self.nu_star_avol = volumeAverage_var(self.f, self.nu_star)
        self.nu_norm_avol = volumeAverage_var(self.f, self.nu_norm)

        # Provided by TRANSP
        self.nusti = self.f["NUSTI"][:]
        self.nuste = self.f["NUSTE"][:]

    def getRADinfo(self):
        self.Prad = self.f["PRAD"][:]  # MW/m^3
        self.PradT = volumeIntegralTot(self.f, "PRAD") * 1e-6  # MW
        self.PradT_cum = volumeIntegral(self.f, "PRAD") * 1e-6  # MW

        # From bolo
        self.Prad_m = self.f["PRAD0"][:]  # MW/m^3
        self.PradT_m = volumeIntegralTot(self.f, "PRAD0") * 1e-6  # MW

        try:
            self.Prad_b = self.f["PRAD_BR"][:]  # MW/m^3
            self.Prad_l = self.f["PRAD_LI"][:]  # MW/m^3
            self.Prad_c = self.f["PRAD_CY"][:]  # MW/m^3

            self.PradT_b = volumeIntegralTot(self.f, "PRAD_BR") * 1e-6  # MW
            self.PradT_l = volumeIntegralTot(self.f, "PRAD_LI") * 1e-6  # MW
            self.PradT_c = volumeIntegralTot(self.f, "PRAD_CY") * 1e-6  # MW

        except:
            self.Prad_b = copy.deepcopy(self.Prad) * 0.0 + self.eps00
            self.Prad_l = copy.deepcopy(self.Prad) * 0.0 + self.eps00
            self.Prad_c = copy.deepcopy(self.Prad) * 0.0 + self.eps00

            self.PradT_b = copy.deepcopy(self.PradT) * 0.0 + self.eps00
            self.PradT_l = copy.deepcopy(self.PradT) * 0.0 + self.eps00
            self.PradT_c = copy.deepcopy(self.PradT) * 0.0 + self.eps00

        self.PradZ = {}
        self.PradZT = {}
        for i in self.nZs:
            try:
                self.PradZ[i] = self.f["PRADS_" + i][:]  # MW/m^3
                self.PradZT[i] = volumeIntegralTot(self.f, "PRADS_" + i) * 1e-6  # MW
            except:
                self.PradZ[i] = copy.deepcopy(self.ne) * 0.0
                self.PradZT[i] = copy.deepcopy(self.t) * 0.0

        self.PradZ = {}
        self.PradZT = {}
        for i in self.nZs:
            try:
                self.PradZ[i] = self.f["PRADS_" + i][:]  # MW/m^3
                self.PradZT[i] = volumeIntegralTot(self.f, "PRADS_" + i) * 1e-6  # MW
            except:
                self.PradZ[i] = copy.deepcopy(self.ne) * 0.0
                self.PradZT[i] = copy.deepcopy(self.t) * 0.0

    def getICRFinfo(self):
        try:
            self.PichT = self.f["PICHTOT"][:] * 1e-6
            self.Peich = self.f["PEICH"][:]  # MW/m^3
            self.Piich = self.f["PIICH"][:]  # MW/m^3
            self.PeichT = volumeIntegralTot(self.f, "PEICH") * 1e-6  # MW
            self.PiichT = volumeIntegralTot(self.f, "PIICH") * 1e-6  # MW
            self.PichTOrbLoss = volumeIntegralTot(self.f, "QMINORB") * 1e-6  # MW
            self.Peich_frac = np.array(
                [
                    x / y if y else 0
                    for x, y in zip(self.PeichT, (self.PeichT + self.PiichT))
                ]
            )
            self.Piich_frac = 1 - self.Peich_frac

            self.Pich = self.f["QICHA"][:]  # MW/m^3

            # Separation
            self.Pich_min = self.f["QICHMIN"][:]  # MW/m^3
            self.Pich_minRenorm = self.f["QMINICH"][:]  # MW/m^3
            self.Pich_minPTCL = self.f["QMINPSC"][:]  # MW/m^3
            self.Pich_minOrbLoss = self.f["QMINORB"][:]  # MW/m^3
            self.PichT_min = self.f["PICHMIN"][:] * 1e-6  # MW

            self.PichT_minRenorm = volumeIntegralTot(self.f, "QMINICH") * 1e-6  # MW
            self.PichT_minPTCL = volumeIntegralTot(self.f, "QMINPSC") * 1e-6  # MW

            self.PichT_MC = self.f["PICHMC"][:] * 1e-6  # MW
            self.Pich_MC = self.f["QICHMC"][:]  # MW/m^3

            self.PiichT_dir = self.f["PICHI"][:] * 1e-6  # MW
            self.Piich_dir = self.f["QICHI"][:]  # MW/m^3

            self.PeichT_dir = self.f["PICHE"][:] * 1e-6  # MW
            self.Peich_dir = self.f["QICHE"][:]  # MW/m^3

            self.PfichT_dir = self.f["PICHFAST"][:] * 1e-6  # MW
            self.Pfich_dir = self.f["QICHFAST"][:]  # MW/m^3

            # Collisional
            self.Piich_min = self.f["QMINI"][:]  # MW/m^3
            self.Peich_min = self.f["QMINE"][:]  # MW/m^3
            self.PiichT_min = volumeIntegralTot(self.f, "QMINI") * 1e-6  # MW
            self.PeichT_min = volumeIntegralTot(self.f, "QMINE") * 1e-6  # MW

            # TORIC
            self.TORIC_nphi_x = self.f["XGRID_NPHI"][:]
            self.TORIC_nphi_p = self.f["PIC_NPHI"][:] * 1e-6  # MW

            self.Gainmin = self.f["QMINDOT"][:]  # MW/m^3
            self.GainminT = volumeIntegralTot(self.f, "QMINDOT") * 1e-6  # MW

            # Different atennas
            PichT_ant = []
            for i in range(100):
                try:
                    PichT_ant.append(self.f[f"PICHA{i + 1}"][:] * 1e-6)
                except:
                    break
            self.PichT_ant = np.array(PichT_ant)

            FichT_ant = []
            for i in range(100):
                try:
                    FichT_ant.append(self.f[f"FREQA{i + 1}"][:] * 1e-6)  # in MHz
                except:
                    break
            self.FichT_ant = np.array(FichT_ant)

            # Cumulative power
            self.Peich_cum = volumeMultiplication(self.f, "PEICH") * 1e-6  # in MW
            self.Piich_cum = volumeMultiplication(self.f, "PIICH") * 1e-6  # in MW

            self.Pich_check = (
                self.Pich_min + self.Piich_dir + self.Peich_dir + self.Pfich_dir
            )
            self.PichT_check = (
                self.PichT_min + self.PiichT_dir + self.PeichT_dir + self.PfichT_dir
            )

        except:
            self.PichT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.Peich = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Piich = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.PeichT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.PiichT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.Peich_frac = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.Piich_frac = copy.deepcopy(self.PohT) * 0.0 + self.eps00

            self.Pich_minRenorm = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Pich_minPTCL = copy.deepcopy(self.Poh) * 0.0 + self.eps00

            self.PichT_min = self.t * 0.0 + self.eps00
            self.PichT_MC = self.t * 0.0 + self.eps00
            self.PiichT_dir = self.t * 0.0 + self.eps00
            self.PeichT_dir = self.t * 0.0 + self.eps00
            self.PfichT_dir = self.t * 0.0 + self.eps00

            self.Piich_min = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Peich_min = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.PiichT_min = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.PeichT_min = copy.deepcopy(self.PohT) * 0.0 + self.eps00

            self.Pich = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Pich_min = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Pich_MC = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Piich_dir = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Peich_dir = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Pfich_dir = copy.deepcopy(self.Poh) * 0.0 + self.eps00

            self.PichT_ant = np.array([copy.deepcopy(self.PohT) * 0.0 + self.eps00])
            self.FichT_ant = np.array([copy.deepcopy(self.PohT) * 0.0 + self.eps00])

            self.Pich_minOrbLoss = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.PichTOrbLoss = copy.deepcopy(self.PohT) * 0.0 + self.eps00

            self.Gainmin = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.GainminT = copy.deepcopy(self.PohT) * 0.0 + self.eps00

            self.Peich_cum = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Piich_cum = copy.deepcopy(self.Poh) * 0.0 + self.eps00

            self.Pich_check = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.PichT_check = copy.deepcopy(self.PohT) * 0.0 + self.eps00

        self.PichT_min_frac = MATHtools.divideZero(self.PichT_min, self.PichT)
        self.PiichT_dir_frac = MATHtools.divideZero(self.PiichT_dir, self.PichT)
        self.PeichT_dir_frac = MATHtools.divideZero(self.PeichT_dir, self.PichT)

        # Resonances

        B = self.Bt_ext
        Fich = self.FichT_ant[0]
        if np.sum(Fich) > 0.0 + self.eps00 * (len(self.t) + 1):
            harmonic = 1.0
            self.R_ICRF_D = PLASMAtools.findResonance(
                Fich / harmonic, B, self.Rmaj, self.qm_D
            )
            self.R_ICRF_T = PLASMAtools.findResonance(
                Fich / harmonic, B, self.Rmaj, self.qm_T
            )
            self.R_ICRF_He3 = PLASMAtools.findResonance(
                Fich / harmonic, B, self.Rmaj, self.qm_He3
            )
            self.R_ICRF_He4 = PLASMAtools.findResonance(
                Fich / harmonic, B, self.Rmaj, self.qm_He4
            )
            self.R_ICRF_H = PLASMAtools.findResonance(
                Fich / harmonic, B, self.Rmaj, self.qm_H
            )

            harmonic = 2.0
            self.R_ICRF_D_2 = PLASMAtools.findResonance(
                Fich / harmonic, B, self.Rmaj, self.qm_D
            )
            self.R_ICRF_T_2 = PLASMAtools.findResonance(
                Fich / harmonic, B, self.Rmaj, self.qm_T
            )
            self.R_ICRF_He3_2 = PLASMAtools.findResonance(
                Fich / harmonic, B, self.Rmaj, self.qm_He3
            )
            self.R_ICRF_H_2 = PLASMAtools.findResonance(
                Fich / harmonic, B, self.Rmaj, self.qm_H
            )
            self.R_ICRF_He4_2 = PLASMAtools.findResonance(
                Fich / harmonic, B, self.Rmaj, self.qm_He4
            )

    def getLCFS_time(self, time=None):
        if time is None:
            time = self.t[self.ind_saw]
        Boundary_r, Boundary_z = getFluxSurface(self.f, time, 1.0)

        return Boundary_r, Boundary_z

    def getLHinfo(self):
        try:
            self.PlheT = self.f["PLHE"][:] * 1e-6  # MW
            self.Plhe = self.f["PELH"][:]  # MW/m^3

            self.PlhiT = self.f["PLHI"][:] * 1e-6  # MW
            self.Plhi = self.f["PILH"][:]  # MW/m^3

        except:
            self.PlheT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.Plhe = copy.deepcopy(self.Poh) * 0.0 + self.eps00

            self.PlhiT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.Plhi = copy.deepcopy(self.Poh) * 0.0 + self.eps00  # MW/m^3

        self.PlhT = self.PlheT + self.PlhiT  # self.f['PLH'][:] *1E-6	# MW

    def getECRFinfo(self):
        try:
            self.PechT_delivered = self.f["PECIN"][:] * 1e-6  # ECRF (MW) input power
            self.PechT = (
                self.f["PECHT"][:] * 1e-6
            )  # ECRF (MW) electron heating (rest is lost?)
            self.PechT_lost = self.PechT_delivered - self.PechT

            self.Pech = self.f["PEECH"][:]  # MW/m^3

            self.PechT_check = volumeIntegralTot(self.f, "PEECH") * 1e-6  # MW

            PechT_ant = []
            for i in range(100):
                try:
                    PechT_ant.append(self.f[f"PECIN{i + 1}"][:] * 1e-6)
                except:
                    break
            self.PechT_ant = np.array(PechT_ant)

            Pech_ant = []
            for i in range(100):
                try:
                    Pech_ant.append(self.f[f"PEECH{i + 1}"][:])
                except:
                    break
            self.Pech_ant = np.array(Pech_ant)

            jECH_ant = []
            for i in range(100):
                try:
                    jECH_ant.append(self.f[f"ECCUR{i + 1}"][:])
                except:
                    break
            self.jECH_ant = np.array(jECH_ant) * 1e-2

            PechTor_ant = []
            for i in range(100):
                try:
                    PechTor_ant.append(self.f[f"PHAIECH{i + 1}"][:])
                except:
                    break
            self.PechTor_ant = np.array(PechTor_ant)

            PechPol_ant = []
            for i in range(100):
                try:
                    PechPol_ant.append(self.f[f"THETECH{i + 1}"][:])
                except:
                    break
            self.PechPol_ant = np.array(PechPol_ant)

        except:
            self.PechT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.PechT_delivered = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.Pech = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.PechT_ant = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.Pech_ant = copy.deepcopy(self.Poh) * 0.0 + self.eps00

            self.PechT_check = copy.deepcopy(self.PohT) * 0.0 + self.eps00

        try:
            self.Theta_gyr, self.Phi_gyr = [], []
            for i in range(8):
                self.Theta_gyr.append(self.f[f"THETECH{i + 1}"][:])
                self.Phi_gyr.append(self.f[f"PHAIECH{i + 1}"][:])
            self.Theta_gyr = np.array(self.Theta_gyr)
            self.Phi_gyr = np.array(self.Phi_gyr)
        except:
            self.Theta_gyr, self.Phi_gyr = None, None

    def getNBIinfo(self):
        try:
            # Power launched (PINJ = BPSHI + BPCAP)
            self.PnbiINJ = self.f["PINJ"][:] * 1e-6  # MW
            self.PnbiLoss_shine = self.f["BPSHI"][:] * 1e-6  # MW
            self.PnbiT = self.f["BPCAP"][:] * 1e-6  # MW

            self.PnbiStored = self.f["BPST"][:] * 1e-6  # MW

            # Total powers to species (BPCAP = Ions+Electrons+Fast+Losses+Rotation)

            self.PnbiiT = self.f["BPTI"][:] * 1e-6  # MW
            self.PnbieT = self.f["BPTE"][:] * 1e-6  # MW
            self.PnbihT = self.f["BPTH"][:] * 1e-6  # MW

            self.PnbicT = (self.f["BPCXI"][:] + self.f["BPCXX"][:]) * 1e-6
            self.PnbilT = self.f["BPLIM"][:] * 1e-6
            try:
                self.PnbirT = (
                    self.f["BPTHS"][:]
                    + self.f["BPTHR"][:]
                    + self.f["BPJXB"][:]
                    + self.f["BPCOL"][:]
                ) * 1e-6
            except:
                self.PnbirT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            # Total deposited
            self.Pnbii = self.f["PBI"][:]  # MW/m^3
            self.Pnbie = self.f["PBE"][:]  # MW/m^3

            self.Pnbih = self.f["PBTH"][:]  # MW/m^3

            # Per beam

            PnbiT_beam = []
            for i in range(8):
                try:
                    PnbiT_beam.append(self.f[f"PINJ0{i + 1}"][:] * 1e-6)
                except:
                    break
            self.PnbiT_beam = np.array(PnbiT_beam)

            Pnbie_beam = []
            for i in range(8):
                try:
                    Pnbie_beam.append(self.f[f"PBE0{i + 1}_TOT"][:])
                except:
                    break
            self.Pnbie_beam = np.array(Pnbie_beam)

            Pnbii_beam = []
            for i in range(8):
                try:
                    Pnbii_beam.append(self.f[f"PBI0{i + 1}_TOT"][:])
                except:
                    break
            self.Pnbii_beam = np.array(Pnbii_beam)

            Pnbih_beam = []
            for i in range(8):
                try:
                    Pnbih_beam.append(self.f[f"PBTH0{i + 1}_TOT"][:])
                except:
                    break
            self.Pnbih_beam = np.array(Pnbih_beam)

            # particles
            Pnbip_beam = []
            for i in range(8):
                try:
                    Pnbip_beam.append(
                        self.f[f"BDEP0{i + 1}_TOT"][:] * 1e6 * 1e-20
                    )  # 1E20/m^3/s
                except:
                    break
            self.Pnbip_beam = np.array(Pnbip_beam)

            j_beam, j_beamU = [], []
            for i in range(8):
                try:
                    j_beam.append(self.f[f"BDC0{i + 1}"][:] * 1e-6 * 1e4)
                    j_beamU.append(self.f[f"UDC0{i + 1}"][:] * 1e-6 * 1e4)
                except:
                    break
            self.jNBI_beam = np.array(j_beam)
            self.jNBI_beamU = np.array(j_beamU)

        except:
            self.PnbiINJ = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.PnbiLoss_shine = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.PnbilT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.PnbirT = copy.deepcopy(self.PohT) * 0.0 + self.eps00

            self.PnbiStored = copy.deepcopy(self.PohT) * 0.0 + self.eps00

            self.PnbiT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.PnbiiT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.PnbieT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.PnbihT = copy.deepcopy(self.PohT) * 0.0 + self.eps00

            self.PnbicT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.PnbilT = copy.deepcopy(self.PohT) * 0.0 + self.eps00
            self.PnbirT = copy.deepcopy(self.PohT) * 0.0 + self.eps00

            self.Pnbii = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Pnbie = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Pnbih = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Pnbie_beam = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Pnbii_beam = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.Pnbih_beam = copy.deepcopy(self.Poh) * 0.0 + self.eps00

            # Info about sources of particles
            self.nD_source_beams = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.nD_source_halo = copy.deepcopy(self.Poh) * 0.0 + self.eps00

        try:
            # Info about sources of particles
            self.nD_source_beams = self.f["SBTH_D"][:] * 1e6 * 1e-20  # in 10^20m^-3/s
            self.nD_source_halo = (
                (self.f["SIHALO_D"][:] - self.f["S0HALO"][:]) * 1e6 * 1e-20
            )  # in 10^20m^-3/s
        except:
            # Info about sources of particles
            self.nD_source_beams = copy.deepcopy(self.Poh) * 0.0 + self.eps00
            self.nD_source_halo = copy.deepcopy(self.Poh) * 0.0 + self.eps00

    def getFusionPower(self):
        self.FusTT = self.f["TOT2TT"][:] * (11.3) * 1e6
        self.FusDDn = self.f["TOTDDN"][:] * (0.82 + 2.45) * 1e6
        try:
            self.FusDDp = self.f["TOTDDP"][:] * (1.01 + 3.02) * 1e6
        except:
            self.FusDDp = copy.deepcopy(self.FusTT) * 0.0 + self.eps00
        self.FusDT = self.f["TOTDT"][:] * (3.5 + 14.1) * 1e6

        self.Pout = (
            (self.FusTT + self.FusDDn + self.FusDDp + self.FusDT) * self.e_J * 1e-6
        )  # MW

    def getFusionPerformance(self, eta_ICH=1.0):
        self.utilsiliarPower_real = self.utilsiliarPower / eta_ICH
        self.getFusionPower()

        self.Q = self.Pout / (self.utilsiliarPower)
        self.Q_corrected_dWdt = self.Pout / (self.utilsiliarPower - self.GainT)
        self.Qreal = self.Pout / (self.utilsiliarPower_real)

        self.neutrons = self.f["NEUTT"][:] * 1e-20  # 1E20 / s

        try:
            self.neutrons_m = self.f["MNEUT"][:] * 1e-20  # 1E20 / s
        except:
            self.neutrons_m = self.neutrons * 0.0

        self.neutrons_thr = self.f["NEUTX"][:] * 1e-20  # 1E20 / s
        self.neutrons_thr_frac = self.neutrons_thr / self.neutrons

        # Rates
        self.neutrons_x = self.f["TTNTX"][:] * 1e6 * 1e-20  # 1E20 n / s / m^3
        self.neutrons_thr_x = self.f["THNTX"][:] * 1e6 * 1e-20  # 1E20 n / s / m^3
        self.neutrons_x_cum = (
            volumeMultiplication(self.f, "TTNTX") * 1e-20
        )  # 1E20 n / s
        self.neutrons_thr_x_cum = (
            volumeMultiplication(self.f, "THNTX") * 1e-20
        )  # 1E20 n / s

        # ----
        # Beam

        try:
            self.neutrons_beambeam = self.f["BBNTS"][:] * 1e-20  # 1E20 / s
            self.neutrons_beambeam_x = (
                self.f["BBNTX"][:] * 1e6 * 1e-20
            )  # 1E20 n / s / m^3
            self.neutrons_beambeam_x_cum = (
                volumeMultiplication(self.f, "BBNTX") * 1e-20
            )  # 1E20 n / s

            self.neutrons_beamtarget = self.f["BTNTS"][:] * 1e-20  # 1E20 / s
            self.neutrons_beamtarget_x = (
                self.f["BTNTX"][:] * 1e6 * 1e-20
            )  # 1E20 n / s / m^3
            self.neutrons_beamtarget_x_cum = (
                volumeMultiplication(self.f, "BTNTX") * 1e-20
            )  # 1E20 n / s

        except:
            self.neutrons_beambeam = copy.deepcopy(self.t) * 0.0 + self.eps00
            self.neutrons_beambeam_x = copy.deepcopy(self.x) * 0.0 + self.eps00

            self.neutrons_beamtarget = copy.deepcopy(self.t) * 0.0 + self.eps00
            self.neutrons_beamtarget_x = copy.deepcopy(self.x) * 0.0 + self.eps00

            self.neutrons_beamtarget_x_cum = copy.deepcopy(self.x) * 0.0 + self.eps00
            self.neutrons_beambeam_x_cum = copy.deepcopy(self.x) * 0.0 + self.eps00

        # ---

        # DT Neutrons
        try:
            self.neutrons_thrDT = self.f["NEUTX_DT"][:] * 1e-20  # 1E20 / s
            self.neutrons_thrDT_frac = self.neutrons_thrDT / self.neutrons_thr
            self.neutrons_thrDT_x = (
                self.f["THNTX_DT"][:] * 1e6 * 1e-20
            )  # 1E20 n / s / m^3
            self.neutrons_thrDT_x_cum = (
                volumeMultiplication(self.f, "THNTX_DT") * 1e-20
            )  # 1E20 n / s
        except:
            self.neutrons_thrDT = copy.deepcopy(self.neutrons) * 0.0 + self.eps00
            self.neutrons_thrDT_frac = copy.deepcopy(self.neutrons) * 0.0 + self.eps00
            self.neutrons_thrDT_x = copy.deepcopy(self.x) * 0.0 + self.eps00
            self.neutrons_thrDT_x_cum = copy.deepcopy(self.x) * 0.0 + self.eps00

        # DD Neutrons
        try:
            self.neutrons_thrDD = self.f["NEUTX_DD"][:] * 1e-20  # 1E20 / s
            self.neutrons_thrDD_frac = self.neutrons_thrDD / self.neutrons_thr
            self.neutrons_thrDD_x = (
                self.f["THNTX_DD"][:] * 1e6 * 1e-20
            )  # 1E20 n / s / m^3
            self.neutrons_thrDD_x_cum = (
                volumeMultiplication(self.f, "THNTX_DD") * 1e-20
            )  # 1E20 n / s
        except:
            self.neutrons_thrDD = copy.deepcopy(self.neutrons) * 0.0 + self.eps00
            self.neutrons_thrDD_frac = copy.deepcopy(self.neutrons) * 0.0 + self.eps00
            self.neutrons_thrDD_x = copy.deepcopy(self.x) * 0.0 + self.eps00
            self.neutrons_thrDD_x_cum = copy.deepcopy(self.x) * 0.0 + self.eps00

        # TT Neutrons
        try:
            self.neutrons_thrTT = self.f["NEUTX_TT"][:] * 1e-20  # 1E20 / s
            self.neutrons_thrTT_frac = self.neutrons_thrTT / self.neutrons_thr
            self.neutrons_thrTT_x = (
                self.f["THNTX_TT"][:] * 1e6 * 1e-20
            )  # 1E20 n / s / m^3
            self.neutrons_thrTT_x_cum = (
                volumeMultiplication(self.f, "THNTX_TT") * 1e-20
            )  # 1E20 n / s
        except:
            self.neutrons_thrTT = copy.deepcopy(self.neutrons) * 0.0 + self.eps00
            self.neutrons_thrTT_frac = copy.deepcopy(self.neutrons) * 0.0 + self.eps00
            self.neutrons_thrTT_x = copy.deepcopy(self.x) * 0.0 + self.eps00
            self.neutrons_thrTT_x_cum = copy.deepcopy(self.x) * 0.0 + self.eps00

        # Lawson
        self.nTtau = (
            self.taue * 1e-3 * self.Ti0 * self.nmain[:, 0] * 0.1
        )  # 10^{21}keVm^{-3}s

    def getPorcelli(self):
        try:
            self.porcelli_13L = self.f["PORC13L"][:]
            self.porcelli_13R = self.f["PORC13R"][:]

            self.porcelli_14L = self.f["PORC14L"][:]
            self.porcelli_14R = self.f["PORC14R"][:]

            self.porcelli_15aL = self.f["PORC15AL"][:]
            self.porcelli_15aC = self.f["PORC15AC"][:]
            self.porcelli_15aR = self.f["PORC15AR"][:]

            self.porcelli_15bL = self.f["PORC15BL"][:]
            self.porcelli_15bR = self.f["PORC15BR"][:]

            eq13, eq14, eq15a, eq15b, eq15, q1cond = [], [], [], [], [], []
            for i in range(len(self.t)):
                if self.porcelli_13L[i] > self.porcelli_13R[i]:
                    eq13.append(1)
                else:
                    eq13.append(0)
                if self.porcelli_14L[i] > self.porcelli_14R[i]:
                    eq14.append(0.95)
                else:
                    eq14.append(0.05)
                if (
                    self.porcelli_15aL[i] < self.porcelli_15aC[i]
                    and self.porcelli_15aC[i] < self.porcelli_15aR[i]
                ):
                    eq15a.append(0.9)
                else:
                    eq15a.append(0.1)
                if self.porcelli_15bL[i] < self.porcelli_15bR[i]:
                    eq15b.append(0.85)
                else:
                    eq15b.append(0.15)
                if eq15a[i] > 0.5 and eq15b[i] > 0.5:
                    eq15.append(0.8)
                else:
                    eq15.append(0.2)
                if self.q[i, 0] <= 1.0:
                    q1cond.append(0.75)
                else:
                    q1cond.append(0.25)

            self.eq13, self.eq14, self.eq15, self.q1cond = (
                np.array(eq13),
                np.array(eq14),
                np.array(eq15),
                np.array(q1cond),
            )

            self.porcelli_rq1 = self.f["PORCDIAG1"][:]  # rminor of the q=1 surface
            self.porcelli_taumin = self.f["PORCDIAG2"][
                :
            ]  # Minimum allowed sawtooth period
            self.porcelli_ti0 = self.f["PORCDIAG3"][:]  # central ion temperature (keV)
            self.porcelli_mi = self.f["PORCDIAG4"][:]  # average ion mass (amu)
            self.porcelli_Efast = self.f["PORCDIAG5"][
                :
            ]  # energy of fast ions at q=1 surface (keV) (scalar)
            self.porcelli_s1 = self.f["PORCDIAG6"][:]  # magnetic shear at q=1 surface

        except:
            print(
                ">> This plasma did not use Porcelli model to trigger sawtooth crashes"
            )
            self.porcelli_13L = self.t * 0.0 + self.eps00
            self.porcelli_13R = self.t * 0.0 + self.eps00

            self.porcelli_14L = self.t * 0.0 + self.eps00
            self.porcelli_14R = self.t * 0.0 + self.eps00

            self.porcelli_15aL = self.t * 0.0 + self.eps00
            self.porcelli_15aC = self.t * 0.0 + self.eps00
            self.porcelli_15aR = self.t * 0.0 + self.eps00

            self.porcelli_15bL = self.t * 0.0 + self.eps00
            self.porcelli_15bR = self.t * 0.0 + self.eps00

            self.eq13 = self.t * 0.0 + self.eps00
            self.eq14 = self.t * 0.0 + self.eps00
            self.eq15 = self.t * 0.0 + self.eps00
            self.q1cond = self.t * 0.0 + self.eps00

            self.porcelli_rq1 = self.t * 0.0 + self.eps00
            self.porcelli_taumin = self.t * 0.0 + self.eps00
            self.porcelli_ti0 = self.t * 0.0 + self.eps00
            self.porcelli_mi = self.t * 0.0 + self.eps00
            self.porcelli_taumin = self.t * 0.0 + self.eps00
            self.porcelli_s1 = self.t * 0.0 + self.eps00

        # Mixing

        self.psi_heli = self.f["HLFLX"][:]  # Wb/rad

        try:
            Energy = volumeIntegral_var(self.f, self.p)
            Energy_e = volumeIntegral_var(self.f, self.p_e)
            Vols = volumeIntegral_var(self.f, self.p / self.p)
            x_saw_inv_prof, newpressure, newpressure_e = [], [], []
            for it in range(len(self.t)):
                ix = np.argmin(np.abs(self.x_saw_mix[it] - self.x_lw))
                newp = Energy[it, ix] / Vols[it, ix]
                newpe = Energy_e[it, ix] / Vols[it, ix]
                x_saw_inv_prof.append(self.x_lw[np.argmin(np.abs(self.p[it] - newp))])
                newpressure.append(newp * np.ones(len(self.x_lw)))
                newpressure_e.append(newpe * np.ones(len(self.x_lw)))
            self.x_saw_inv_prof = np.array(x_saw_inv_prof)
            self.p_expectedsaw = np.array(newpressure)
            self.p_e_expectedsaw = np.array(newpressure_e)
        except:
            self.x_saw_inv_prof = self.t * 0.0 + self.eps00
            self.p_expectedsaw = self.x_lw * 0.0 + self.eps00
            self.p_e_expectedsaw = self.x_lw * 0.0 + self.eps00

        self.Te_expectedsaw = (
            self.p_e_expectedsaw * 1e6 / (self.ne * 1e20) / (self.e_J * 1000)
        )

        self.Ue = self.f["UE"][:]  # MJ/m^3
        self.UeT = volumeIntegralTot(self.f, "UE") * 1e-6  # MJ

    def calculateSawtoothCrashEnergy(self):
        howmanybefore = 5
        howmanyafter = 5

        Delta_W = []
        for i in range(len(self.t)):
            tin = np.argmin(np.abs(self.tlastsaw[i] - self.t)) - howmanybefore
            tout = np.argmin(np.abs(self.tlastsaw[i] - self.t)) + howmanyafter

            try:
                Delta_W.append(self.Wtot[tin] - self.Wtot[tout])
            except:
                Delta_W.append(0)

        self.Wsaw = np.array(Delta_W)

    def getVelocities(self):
        # Toroidal --------------------------

        try:
            self.VtorkHz = self.f["OMEGA"][:] / (2.0 * np.pi) * 1e-3  # kHz
        except:
            self.VtorkHz = self.x * 0.0  # kHz

        try:
            self.VtorkHz_nc = self.f["OMEGA_NC"][:] / (2.0 * np.pi) * 1e-3  # kHz
        except:
            self.VtorkHz_nc = self.x * 0.0  # kHz

        try:
            self.VtorkHz_data = self.f["OMEGDATA"][:] / (2.0 * np.pi) * 1e-3  # kHz
        except:
            self.VtorkHz_data = self.x * 0.0  # kHz

        self.Vtor_LF = self.VtorkHz * (2.0 * np.pi * self.rmaj_LFx)  # km/s
        self.Mach_LF = self.Vtor_LF * 1e3 / self.vTi

        self.Vtor_HF = self.VtorkHz * (2.0 * np.pi * self.rmaj_HFx)  # km/s
        self.Mach_HF = self.Vtor_HF * 1e3 / self.vTi

        self.Vtor0kHz = self.VtorkHz[:, 0]
        self.Vtor0 = self.Vtor_LF[:, 0]
        self.Mach0 = self.Mach_LF[:, 0]

        try:
            self.VtorNC = self.f["VTORD_NC"][:] * 1e-5  # km/s
        except:
            self.VtorNC = copy.deepcopy(self.Rmaj) * 0.0 + self.eps00
        self.VtorNC_LF, self.VtorNC_HF = mapRmajorToX(self, self.VtorNC)
        self.Vtor0NC = self.VtorNC_LF[0]

        # Poloidal --------------------------

        try:
            self.VpolNC = self.f["VPOLD_NC"][:] * 1e-5  # km/s
        except:
            self.VpolNC = copy.deepcopy(self.Rmaj) * 0.0 + self.eps00
        self.VpolNC_LF, self.VpolNC_HF = mapRmajorToX(self, self.VpolNC)
        self.Vpol0NC = self.VpolNC_LF[0]

    def getElectricField(self):
        self.Er = self.f["ERTOT"][:] * 1e2  # V/m
        self.Er_LF, self.Er_HF = mapRmajorToX(self, self.Er)

        self.Er_p = self.f["ERPRESS"][:] * 1e2  # V/m
        self.Er_p_LF, self.Er_p_HF = mapRmajorToX(self, self.Er_p)

        self.Er_tor = self.f["ERVTOR"][:] * 1e2  # V/m
        self.Er_tor_LF, self.Er_tor_HF = mapRmajorToX(self, self.Er_tor)

        self.Er_pol = self.f["ERVPOL"][:] * 1e2  # V/m
        self.Er_pol_LF, self.Er_pol_HF = mapRmajorToX(self, self.Er_pol)

        self.Er_check = self.Er_p_LF + self.Er_tor_LF + self.Er_pol_LF

        #
        try:
            self.Epot = self.f["VRPOT"][:] * 1e-3  # kV
        except:
            self.Epot = copy.deepcopy(self.x) * 0.0 + self.eps00
        try:
            self.Epot_nc = self.f["EPOTNC"][:] * 1e-3  # kV
        except:
            self.Epot_nc = copy.deepcopy(self.x) * 0.0 + self.eps00
        try:
            self.Epot_rot = self.f["EPOTRO"][:] * 1e-3  # kV
        except:
            self.Epot_rot = copy.deepcopy(self.x) * 0.0 + self.eps00

        # Get rotation (in XB)

        self.w0 = -derivativeVar(self, self.Epot, specialDerivative=self.psi)

        self.VtorkHz_check = self.w0 / (2 * np.pi)
        self.VtorkHz_rot_check = -derivativeVar(
            self, self.Epot_rot, specialDerivative=self.psi
        ) / (2 * np.pi)
        self.VtorkHz_nc_check = -derivativeVar(
            self, self.Epot_nc, specialDerivative=self.psi
        ) / (2 * np.pi)

    def getFluxes(self):
        self.qe_obsCND = calculateFlux("PCNDE", self.f) * 1e-6  # MW/m^2
        self.qe_obsCNV = calculateFlux("PCNVE", self.f) * 1e-6  # MW/m^2
        self.qi_obsCND = calculateFlux("PCOND", self.f) * 1e-6  # MW/m^2
        self.qi_obsCNV = calculateFlux("PCONV", self.f) * 1e-6  # MW/m^2

        self.qe_obs = calculateFlux("EETR_OBS", self.f) * 1e-6  # MW/m^2
        self.qi_obs = calculateFlux("IETR_OBS", self.f) * 1e-6  # MW/m^2
        self.qe_tr = calculateFlux("EETR_MOD", self.f) * 1e-6  # MW/m^2
        self.qi_tr = calculateFlux("IETR_MOD", self.f) * 1e-6  # MW/m^2

        self.qe_tr_nc = calculateFlux("QFLNC_E", self.f) * 1e-6  # MW/m^2
        self.qi_tr_nc = calculateFlux("QFLNC_I", self.f) * 1e-6  # MW/m^2

        self.Ge_obs = calculateFlux("EPTR_OBS", self.f) * 1e-20  # 10^20/s/m^2
        self.Ge_tr = calculateFlux("EPTR_MOD", self.f) * 1e-20  # 10^20/s/m^2

        self.Ge_tr_nc = calculateFlux("GFLNC_E", self.f) * 1e-20  # 10^20/s/m^2

        self.Ce_obs = PLASMAtools.convective_flux(self.Te, self.Ge_obs)
        self.Ce_tr = PLASMAtools.convective_flux(self.Te, self.Ge_tr)

        #
        self.Te_divQ = self.f["EETR_OBS"][:]  # MW/m^3
        self.Ti_divQ = self.f["IETR_OBS"][:]  # MW/m^3
        self.ne_divG = self.f["EPTR_OBS"][:] * 1e-20 * 1e6  # 10^20/s/m^3

    def getPowerLCFS(self):
        # Energy
        self.Pe_obs = volumeIntegralTot(self.f, "EETR_OBS") * 1e-6  # MW
        self.Pi_obs = volumeIntegralTot(self.f, "IETR_OBS") * 1e-6  # MW
        self.Pe_tr = volumeIntegralTot(self.f, "EETR_MOD") * 1e-6  # MW
        self.Pi_tr = volumeIntegralTot(self.f, "IETR_MOD") * 1e-6  # MW

        self.Pe_obs_cum = volumeIntegral(self.f, "EETR_OBS") * 1e-6  # MW
        self.Pi_obs_cum = volumeIntegral(self.f, "IETR_OBS") * 1e-6  # MW

        self.P_obs = self.Pe_obs + self.Pi_obs
        self.P_tr = self.Pe_tr + self.Pi_tr

        self.P_LCFS = self.P_obs
        self.Pe_LCFS = self.Pe_obs
        self.Pi_LCFS = self.Pi_obs

        # Particles
        self.Le_obs = volumeIntegralTot(self.f, "EPTR_OBS") * 1e-20  # 10^20/s
        self.Le_tr = volumeIntegralTot(self.f, "EPTR_MOD") * 1e-20  # 10^20/s

        try:
            self.LD = volumeIntegralTot(self.f, "DIVFD") * 1e-20  # 10^20/s
        except:
            self.LD = copy.deepcopy(self.Le_tr) * 0.0
        try:
            self.LT = volumeIntegralTot(self.f, "DIVFT") * 1e-20  # 10^20/s
        except:
            self.LT = copy.deepcopy(self.LD) * 0.0
        try:
            self.LHe4 = volumeIntegralTot(self.f, "DIVHE4T") * 1e-20  # 10^20/s
        except:
            self.LHe4 = copy.deepcopy(self.LD) * 0.0

        # Total energy
        self.Delta_t = self.t - self.t[0]
        self.Energy_LCFS = np.cumsum(
            np.array([np.max([i, 0]) for i in self.P_LCFS]) * self.dt
        )

    def getConfinementTimes(self):
        self.taue = self.f["TAUEA"][:] * 1000.0  # in ms
        self.taueTot = self.f["TAUA1"][:] * 1000.0  # in ms

        self.taue_check = (
            self.Wth / (self.P_LCFS + self.PradT + self.PcxT) * 1000.0
        )  # in ms
        self.taueTot_check = (
            self.Wtot / (self.P_LCFS + self.PradT + self.PcxT) * 1000.0
        )  # in ms

        # for my studies
        Whatx = (
            3.0
            / 2.0
            * (self.ne * 1e20 * self.Te * 1e3 + self.ne * 1e20 * self.Ti * 1e3)
            * self.e_J
            * 1e-6
        )
        What = volumeIntegralTot_var(self.f, Whatx * 1e-6)  # to cm^3 for integration
        self.taue_hat = self.taue / self.Wth * What

        # For the calculations of taue and H, it uses the LCFS power. This means that it may not take into account the contribution
        # that electrons lose during sawtooth to the poloidal field (but that is given afterwards back). Therefore, if PohT is 1MW and
        # ExB power is 0.8MW (IN STEADY STATE), 1MW is used in the taue calculation. However,
        # self.PowerNotAcountedFor = self.ExBpower_T-self.PohT
        # self.taue_corrected = self.Wth/( self.P_LCFS + self.PradT + self.PcxT + self.PowerNotAcountedFor) * 1000.0 # in ms

        # ~~~~~~~~~~~~~~~~~
        # Particle
        # ~~~~~~~~~~~~~~~~~

        try:
            self.taup_D_x = (
                volumeIntegral(self.f, "ND")
                * 1e-20
                / (volumeIntegral(self.f, "DIVFD") * 1e-20)
                * 1000.0
            )  # in ms
        except:
            self.taup_D_x = copy.deepcopy(self.x) * 0.0
        self.taup_D = self.taup_D_x[:, -1]

        # Helium ash
        try:
            self.taup_He4_x = (
                volumeIntegral(self.f, "NHE4")
                * 1e-20
                / (volumeIntegral(self.f, "DIVHE4T") * 1e-20)
                * 1000.0
            )  # in ms
        except:
            self.taup_He4_x = copy.deepcopy(self.x) * 0.0
        self.taup_He4 = self.taup_He4_x[:, -1]

        # Tritium
        try:
            self.taup_T_x = (
                volumeIntegral(self.f, "NT")
                * 1e-20
                / (volumeIntegral(self.f, "DIVFT") * 1e-20)
                * 1000.0
            )  # in ms
        except:
            self.taup_T_x = copy.deepcopy(self.x) * 0.0
        self.taup_T = self.taup_T_x[:, -1]

        # Slowing down times
        # self.tauSD_Catto = 3*M*self.Te**(3/2)/(4*np.sqrt(2*np.pi*m)*Z**2*self.e_J**4*self.ne*1E20*self.LambdaCoul)

        self.tauSD_He4_Stix = self.getStixSD(Z=2, A=4, W=3.5e3) * 1e3  # ms
        self.tauSD_He4_Stix_avol = volumeAverage_var(self.f, self.tauSD_He4_Stix)

        try:
            self.tauSD_He4 = self.f["TFSL_4"][:] * 1e3  # ms
        except:
            self.tauSD_He4 = self.x * 0.0

        self.tauSD_He4_avol = volumeAverage_var(self.f, self.tauSD_He4)

    def getStixSD(self, Z=2, A=4, W=3.5e3):
        # as given in T H Stix 1972 Plasma Phys. 14 367

        mZ_approx = self.nZAVE_Z * 2 * self.u

        summ = 1**2 * self.nmain * 1e14 / (
            self.mi / self.u
        ) + self.nZAVE_Z**2 * self.nZAVE * 1e14 / (mZ_approx / self.u)
        Wcrit = 14.8 * self.Te * (A**1.5 / (self.ne * 1e14) * summ)

        ts = (
            6.27e8
            * (A * (self.Te * 1000) ** 1.5)
            / (Z**2 * self.ne * 1e14 * self.LambdaCoul_e)
        )

        tau = ts / 3 * np.log(1 + (W / Wcrit) ** 1.5)

        return tau

    def getLHthresholds(self):
        n = self.ne_l

        # Hughes Memo -----
        self.nLH_min = PLASMAtools.LHthreshold_nmin(
            self.Ip, self.Bt, self.a, self.Rmajor
        )

        # L-H scaling (Martin 2008)
        self.P_thr_Martin1 = LH_Martin1(n, self.Bt, self.surface, nmin=self.nLH_min)
        self.P_thr_Martin1_low = LH_Martin1_low(
            n, self.Bt, self.surface, nmin=self.nLH_min
        )
        self.P_thr_Martin1_up = LH_Martin1_up(
            n, self.Bt, self.surface, nmin=self.nLH_min
        )
        self.P_thr_Martin2 = LH_Martin2(
            n, self.Bt, self.a, self.Rmajor, nmin=self.nLH_min
        )
        self.isotopeMassEffect = (2 / self.Ai) ** 1.11

        # Qi scaling (Schmidtmayr 2018)
        self.Pi_thr_Schmi1 = LH_Schmid1(
            n, self.Bt, self.surface, nmin=self.nLH_min * 0.0 + self.eps00
        )
        self.Pi_thr_Schmi2 = LH_Schmid2(
            n, self.Bt, self.surface, nmin=self.nLH_min * 0.0 + self.eps00
        )

        # L-H as given by TRANSP
        self.P_thr_tr = self.f["PL2HREQ"][:] * 1e-6  # in MW

        # FRACTION
        self.Plh = self.P_thr_Martin1 * self.isotopeMassEffect
        self.Plh_ratio = self.P_LCFS / (self.P_thr_Martin1 * self.isotopeMassEffect)
        self.Plh_ratio_low = self.P_LCFS / (
            self.P_thr_Martin1_low * self.isotopeMassEffect
        )
        self.Plh_ratio_up = self.P_LCFS / (
            self.P_thr_Martin1_up * self.isotopeMassEffect
        )
        self.Plh_ratio_mod = (self.P_LCFS + self.PradT) / (
            self.P_thr_Martin1 * self.isotopeMassEffect
        )

        for it in range(len(self.t)):
            self.Plh_ratio[it] = np.max([0, self.Plh_ratio[it]])
            self.Plh_ratio_mod[it] = np.max([0, self.Plh_ratio_mod[it]])

        # Penalty
        expectationPlh = 1.0
        self.LHdiff = np.zeros(len(self.t))
        for i in range(len(self.t)):
            self.LHdiff[i] = MATHtools.sigmoidPenalty(
                self.Plh_ratio[i], x_unity=[expectationPlh, 1.0e3], h_transition=0.05
            )

    def defineGeometricParameters(self):
        self.a = self.rmin[:, -1]

        self.Rmag = self.f["RAXIS"][:] * 1e-2
        self.Ymag = self.f["YAXIS"][:] * 1e-2

        self.kappaS = self.f["ELONG"][:]
        self.deltaS_u = self.f["TRIANGU"][:]
        self.deltaS_l = self.f["TRIANGL"][:]
        self.deltaS = 0.5 * (self.deltaS_u + self.deltaS_l)  # self.f['TRIANG'][:]
        self.zetaS = 0.5 * (self.f["SQUARE_LO"][:] + self.f["SQUARE_UO"][:])
        self.kappa = self.kappaS[:, -1]
        self.delta = self.deltaS[:, -1]
        self.zeta = self.zetaS[:, -1]

        self.ShafShift = self.f["ASHAF"][:] * 1e-2
        self.Rmajor = self.Rmag - self.ShafShift

        self.epsilon = self.a / self.Rmajor
        self.b = self.a * self.kappa
        self.d = self.a * self.delta

        self.volume = plasmaVolume(self.f) * 1e-6  # in m^3
        self.surface = plasmaSurface(self.f) * 1e-4  # in m^2
        self.surfaceXS = plasmaXSarea(self.f) * 1e-4  # in m^2
        self.surfaceXSs = (
            self.f["DAREA"][:] * 1e-4
        )  # in m^2	# This is the XS area in between surfaces, so self.surfaceXS=sum(self.surfaceXSs)
        self.surfaceXSs_cum = np.cumsum(self.surfaceXSs, axis=1)

        # Cumulative volume ------------------------------------------------------------------------------------------
        self.volume_cum = []
        for ix in range(self.x_lw.shape[0]):
            self.volume_cum.append(plasmaVolume(self.f, rangex=ix) * 1e-6)
        self.volume_cum = np.transpose(np.array(self.volume_cum))
        # ------------------------------------------------------------------------------------------------------------

        self.dAdx = derivativeVar(self, self.surfaceXSs_cum, specialDerivative=self.x)
        self.dAdx_check = 1 / (2 * np.pi) * self.Rinv

        self.kappaITER = self.surfaceXS / (np.pi * self.a**2)

        # Params
        self.kappa_995 = copy.deepcopy(self.t)
        self.delta_995 = copy.deepcopy(self.t)
        self.kappa_950 = copy.deepcopy(self.t)
        self.delta_950 = copy.deepcopy(self.t)
        for it in range(len(self.t)):
            self.kappa_995[it] = np.interp([0.995], self.psin[it], self.kappaS[it])[0]
            self.delta_995[it] = np.interp([0.995], self.psin[it], self.deltaS[it])[0]
            self.kappa_950[it] = np.interp([0.95], self.psin[it], self.kappaS[it])[0]
            self.delta_950[it] = np.interp([0.95], self.psin[it], self.deltaS[it])[0]

        # Simplify flux-surfaces (in XB)
        self.Rmax = self.rmaj_LFxb
        self.Rmin = self.rmaj_HFxb
        Ymag = constant_radius(self.Ymag, lenX=len(self.xb_lw))
        self.Zmax = Ymag + ((self.rmaj_LFxb - self.rmaj_HFxb) / 2 * self.kappaS)
        self.Zmin = Ymag - ((self.rmaj_LFxb - self.rmaj_HFxb) / 2 * self.kappaS)

    def defineMagneticParameters(self):
        self.Bt_ext = self.f["BTX"][:]
        self.Bp_ext = self.Bt_ext * self.f["FBPBT"][:]

        # Calculate Bp ---
        self.Bp_LFx, self.Bp_HFx = mapRmajorToX(
            self, self.Bp_ext, originalCoordinate="RMAJM"
        )
        self.Bp_x = 0.5 * (self.Bp_LFx - np.flipud(self.Bp_HFx))
        self.Bp_avol = volumeAverage_var(self.f, self.Bp_x)
        # ---

        self.Bt_plasma = self.Bt_ext * self.f["FBTX"][:]

        self.Magnetism = self.f["GFUN"][:]  # I think this is F/(RBt_vacuum)
        self.Magnetism_c = self.f["GFUNC"][:]

        # Magnetic field on axes

        Bfield = []
        for indt in range(len(self.t)):
            inter = np.argmin(np.abs(self.Rmaj[indt, :] - self.Rmajor[indt]))
            Bfield.append(self.Bt_ext[indt, inter])
        self.Bt = np.array(Bfield)  # This is magnetic field on Rmajor

        Bfield = []
        for indt in range(len(self.t)):
            inter = np.argmin(np.abs(self.Rmaj[indt, :] - self.Rmag[indt]))
            Bfield.append(self.Bt_ext[indt, inter])
        self.Bt_mag = np.array(Bfield)  # This is magnetic field on Rmag

        self.Ip = self.f["PCUR"][:] * 1e-6  # in MA

        self.calculateMagneticShearQuantities()

        # Magnetic field structure
        hf = len(self.Rmaj[0, :]) // 2
        self.B_HF = self.Bt_ext[:, :hf]  # on XB grid
        self.B_LF = self.Bt_ext[:, hf + 1 :]  # on XB grid
        self.Bt_x = 0.5 * (self.Bt_ext[:, :hf] + self.Bt_ext[:, hf + 1 :])

        self.RBt_vacuum = self.f["BZXR"][:] * 1e-2  # Vacuum field T*m
        self.Bt_vacuum = self.RBt_vacuum / self.Rmajor  # Vacuum field T
        self.Bt_vacuum_avol = self.RBt_vacuum * volumeAverage_var(self.f, self.Rinv)
        self.Bt2_vacuum_avol = self.RBt_vacuum**2 * volumeAverage_var(
            self.f, self.Rinv2
        )

        # These are magnitudes of total field, no only toroidal. I checked this by comparing UBTOR+UBPOL = GB2/(2*mu0)
        self.B_av = self.f["GB1"][:]
        self.B_av_sqr = self.f["GB2"][:]
        self.B_av_sqr_inv = self.f["GB2I"][:]

        # Current profile
        self.j = self.f["CUR"][:] * 1e-6 * 1e4  # in MA/m^2
        self.jOh = self.f["CUROH"][:] * 1e-6 * 1e4  # in MA/m^2
        self.jB = self.f["CURBS"][:] * 1e-6 * 1e4  # in MA/m^2
        try:
            self.jNBI = self.f["CURB"][:] * 1e-6 * 1e4  # in MA/m^2
        except:
            self.jNBI = self.j * 0.0 + self.eps00
        try:
            self.jECH = self.f["ECCUR"][:] * 1e-6 * 1e4  # in MA/m^2
        except:
            self.jECH = self.j * 0.0 + self.eps00

        self.jprime = derivativeVar(self, self.j)

        self.Ip_j = surfaceIntegralTot(self.f, "CUR") * 1e-6  # in MA
        self.IpB = surfaceIntegralTot(self.f, "CURBS") * 1e-6  # in MA
        self.IpOH = surfaceIntegralTot(self.f, "CUROH") * 1e-6  # in MA

        self.IpB_fraction = self.IpB / self.Ip

        self.Ip_cum = surfaceIntegral(self.f, "CUR") * 1e-6  # in MA

        # Inductance

        self.Lpol = self.f["LPOL"][:] * 1e-2
        self.Lpol_a = self.Lpol[:, -1]

        phibound = constant_radius(self.phi_bnd, lenX=len(self.x_lw))

        self.Bpol = self.f["BPOL"][:]
        self.Bpol_check = 2 * phibound * self.xb / self.q * self.Lpol / self.dVdx

        self.Bpol2 = (self.f["UBPOL"][:] * 1e6) * (2 * self.mu0)
        self.Bpol2_check = (
            self.iota_bar * self.xb * phibound / np.pi
        ) ** 2 * self.gradXsqrt_Rsq

        self.Bpol2_avol = volumeAverage_var(self.f, self.Bpol2)
        self.Bpol_avol = volumeAverage_var(self.f, self.Bpol)

        self.Bpol_long = (
            self.mu0 * self.Ip_cum * 1e6 / self.Lpol
        )  # This is like a longitudinal average

        # self.Li_MITIM_1 	= self.Bpol2_avol/self.Bpol_avol**2
        # self.Li_MITIM_2x 	= self.Bpol2/self.Bpol**2
        # self.Li_MITIM_2 	= volumeAverage_var(self.f,self.Li_MITIM_2x)

        self.LiVDIFF = self.f["LI_VDIFF"][:]  # it is equal to 2*LIO2
        self.Bpol2_extrap = np.zeros(len(self.t))
        for it in range(len(self.t)):
            self.Bpol2_extrap[it] = MATHtools.extrapolate(
                1.0, self.x_lw, self.Bpol2[it]
            )
        self.LiVDIFF_check = self.Bpol2_avol / self.Bpol2_extrap

        self.Li1 = self.f["LI_1"][:]
        self.Li1_check = self.Bpol2_avol / (self.mu0 * self.Ip * 1e6 / self.Lpol_a) ** 2

        self.Li3 = self.f["LI_3"][:]
        self.Li3_check = (
            2
            * self.Bpol2_avol
            * self.volume
            / (
                (self.mu0 * self.Ip * 1e6) ** 2
                * 0.5
                * (self.Rmaj[:, 0] + self.Rmaj[:, -1])
            )
        )

        # GS errors
        self.q_check = self.f["QCHK"][:]
        try:
            self.q_MHD = self.f["QMHD"][:]
        except:
            self.q_MHD = self.q_check * self.eps00

        self.Ip_eq = self.f["PCUREQ"][:] * 1e-6  # in MA

        try:
            self.p_mhd = self.f["PMHD"][:] * 1e-6
            self.p_mhd_in = self.f["PMHD_IN"][:] * 1e-6
            self.p_mhd_sm = self.f["PMHD_SM"][:] * 1e-6
        except:
            self.p_mhd = self.j * self.eps00
            self.p_mhd_in = self.j * self.eps00
            self.p_mhd_sm = self.j * self.eps00

        self.p_mhd_check = self.f["PCHK"][:] * 1e-6

        self.GS_error = self.f["GSERROR"][:]

        try:
            self.TEQ_error = self.f["TEQRESID"][:]
        except:
            self.TEQ_error = self.t * self.eps00

        # VSURF
        self.Vsurf = self.f["VSURC"][:]
        self.Vsurf_m = self.f["VSUR"][:]

        self.calculateVoltage()

        self.Vsec = self.f["VOLTSECA"][:]  # V*s Poynting Average flux consumptio

    def calculateMagneticShearQuantities(self):
        self.q = self.f["Q"][:]
        self.q95 = self.f["Q95"][:]
        self.q95_check = []
        for it in range(len(self.t)):
            self.q95_check.append(np.interp(0.95, self.psin[it], self.q[it]))
        self.q95_check = np.array(self.q95_check)

        self.iota_bar = 1 / self.q
        self.iota = self.iota_bar / (2 * np.pi)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sawtooth stuff
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        try:
            self.x_saw_inv = self.f["XSWIN"][:]
        except:
            self.x_saw_inv = self.t * 0.0 + self.eps00

        try:
            self.x_saw_mix = self.f["XSWMX"][:]
        except:
            self.x_saw_mix = self.t * 0.0 + self.eps00

        for i in range(len(self.t)):
            self.x_saw_inv[it] = np.interp(
                self.x_saw_inv[it], self.roa[it], self.xb[it]
            )
            self.x_saw_mix[it] = np.interp(
                self.x_saw_mix[it], self.roa[it], self.xb[it]
            )

        self.q0 = self.q[:, 0]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Shat definitions
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        shat, Ls = [], []
        for i in range(len(self.t)):
            shat0, Ls0 = PLASMAtools.magneticshear(
                self.q[i], self.rmin[i], self.Rmajor[i]
            )
            shat.append(shat0)
            Ls.append(Ls0)
        self.shat_Rice, self.Ls_Rice = np.array(shat), np.array(Ls)

        # shat defined as a/q * dq/dr (typical definition of normalized scale length)
        self.shat = -gradNorm(self, self.q)
        self.shat_gR = self.shat * self.ave_grad_rho
        self.shat_rho = -gradNorm(self, self.q, specialDerivative=self.x)

    def calculateVoltage(self):
        # Other current stuff
        self.JdotB = self.f["PLJB"][:] * 1e-2  # MA T / m^2
        self.JdotB_b_used = self.f["PLJBXT"][:] * 1e-2  # MA T / m^2
        self.JdotB_b_raw = self.f["PLJBXTU"][:] * 1e-2  # MA T / m^2
        self.JdotB_oh = self.f["PLJBOH"][:] * 1e-2  # MA T / m^2

        self.EdotB = self.f["PLEB"][:] * 1e2  # V T / m, in X

        self.Jpol = self.f["PLCURPLL"][:]

        self.V = self.f["V"][:]  # V, in X

        self.Etor = self.V * self.Rinv / (2 * np.pi)

        self.RB_LFx, self.RB_HFx = mapRmajorToX(self, self.Rmaj * self.Bt_plasma)

        # Check voltage
        self.V_check = 2 * np.pi * self.EdotB / self.RB_LFx / self.Rinv2  # in X

        self.Poh_checkEJ = self.Etor * self.j * 1e6 * 1e-6  # MW/m^2
        self.Poh_checkEJ_T = (
            volumeIntegralTot_var(self.f, self.Poh_checkEJ) * 1e-6
        )  # MW

        # ~~~~~~~~~~~~~~~~~~~~
        # ----     Alternative
        # ~~~~~~~~~~~~~~~~~~~~

        # Time-derivatives at fixed square root of normalized toroidal flux
        phi_deriv = []
        for i in range(len(self.x_lw)):
            phi_deriv.append(MATHtools.deriv(self.t, self.phi[:, i]))
        self.phi_deriv = np.transpose(np.array(phi_deriv))

        psi_deriv = []
        for i in range(len(self.x_lw)):
            psi_deriv.append(MATHtools.deriv(self.t, self.psi[:, i]))
        self.psi_deriv = np.transpose(np.array(psi_deriv))

        self.ldot = 1.0 / (2 * self.phi_bnd) * self.phi_deriv[:, -1]
        self.ldot_mod = copy.deepcopy(self.phi)
        for i in range(len(self.x_lw)):
            self.ldot_mod[:, i] = self.ldot * self.x_lw[i]

        # Flux derivatives
        phi_deriv_psi = []
        for i in range(len(self.t)):
            phi_deriv_psi.append(MATHtools.deriv(self.psi[i], self.phi[i]))
        self.phi_deriv_psi = np.array(phi_deriv_psi)  # This should be equal to q*2*pi

        psi_deriv_phi = []
        for i in range(len(self.t)):
            psi_deriv_phi.append(MATHtools.deriv(self.phi[i], self.psi[i]))
        self.psi_deriv_phi = np.array(psi_deriv_phi)

        psi_deriv_x = []
        for i in range(len(self.t)):
            psi_deriv_x.append(MATHtools.deriv(self.xb[i], self.psi[i]))
        self.psi_deriv_x = np.array(psi_deriv_x)

        # Plasma toroidal voltage
        self.V_aux_psi = 2 * np.pi * (self.psi_deriv - self.ldot_mod * self.psi_deriv_x)

        self.V0 = self.Etor[:, 0] * 2 * np.pi * self.Rmag
        self.V0 = constant_radius(self.V0, lenX=len(self.x_lw))
        self.V_aux = self.V_aux_psi + self.V0

    def getTotalBeta(self):
        # Electron Beta

        B_pol_e = self.f["BETAE"][:]
        B_tor_e = volumeAverage(self.f, "BTE")
        self.B_e = sumBetas(B_pol_e, B_tor_e)

        # Diamagnetic beta

        B_pol_dia = self.f["BPDIA"][:]
        B_tor_dia = self.f["BTDIA"][:]
        self.B_dia = sumBetas(B_pol_dia, B_tor_dia)

        # -------------------------
        # BETA POLOIDAL
        # -------------------------

        self.BetaPol = self.f["BPEQ"][
            :
        ]  # This is equal to BETAE+BETAI+0.5*(fast betas). BETAT is equal to BETAE+BETAI

        Bpol2 = self.Bpol2_extrap
        self.BetaPol_check = self.p_kin_avol * 1e6 / (Bpol2 / (2 * self.mu0))

        self.BetaPol_1D = self.f["BPEQ1"][
            :
        ]  # This should be using a 1D definition of the Bpol, but I haven't been able to reproduce it yet
        Bpol2 = (self.mu0 * self.Ip * 1e6 / (2 * np.pi * self.a)) ** 2
        self.BetaPol_1D_check = self.p_kin_avol * 1e6 / (Bpol2 / (2 * self.mu0))

        # -------------------------
        # BETA TOROIDAL
        # -------------------------

        self.BetaTor_x = self.f["BTTOT"][:]
        self.BetaTor = volumeAverage_var(self.f, self.BetaTor_x)

        Bphi2 = constant_radius(self.Bt2_vacuum_avol, lenX=len(self.x_lw))  # TO IMPROVE
        self.BetaTor_x_check = self.p_kin * 1e6 / (Bphi2 / (2 * self.mu0))
        self.BetaTor_check = volumeAverage_var(self.f, self.BetaTor_x_check)

        self.BetaTor_thr_x = self.f["BTPL"][:]

        # -------------------------
        # Normalized beta
        # -------------------------

        self.Beta = sumBetas(self.BetaPol, self.BetaTor)

        BtParam = self.Ip / (self.a * self.Bt) / 100.0

        # Beta Normalized

        self.BetaN = self.Beta / BtParam
        self.BetaN_tor = self.BetaTor / BtParam

    def getStoredEnergy(self):
        # ----- Total, thermal and fast ions stored energy (MJ/m^3)

        self.Wth_x = self.f["UTHRM"][:]  # MJ / m^3
        self.Wtot_x = self.f["UTOTL"][:]  # MJ / m^3
        self.Wperp_x = self.f["UFASTPP"][:]  # MPa
        self.Wpar_x = self.f["UFASTPA"][:]  # MPa
        self.Wfast_x = self.Wtot_x - self.Wth_x  # This is equal also to Wperp_x+Wpar_x

        self.Wth = volumeIntegralTot(self.f, "UTHRM") * 1e-6  # MJ
        self.Wtot = volumeIntegralTot(self.f, "UTOTL") * 1e-6  # MJ
        self.Wperp = volumeIntegralTot(self.f, "UFASTPP") * 1e-6  # MJ
        self.Wpar = volumeIntegralTot(self.f, "UFASTPA") * 1e-6  # MJ
        self.Wfast = self.Wtot - self.Wth

        self.Wth_frac = self.Wth / self.Wtot
        self.Wfrac = self.Wfast / self.Wtot

        # ----- Pressures

        self.p = (
            self.f["PPLAS"][:] * 1e-6
        )  # MPa = MJ/m^3	    # This is equal to 2/3 * Wth
        self.p_kin = self.f["PTOWB"][:] * 1e-6
        self.pFast = self.p_kin - self.p  # This is equal to 1/2*Wperp_x + Wpar_x

        self.p_avol = volumeAverage_var(self.f, self.p)
        self.p_kin_avol = volumeAverage_var(self.f, self.p_kin)
        self.pFast_avol = volumeAverage_var(self.f, self.pFast)

        # ----- Checks

        # self.Wth_x_check = 3/2* self.ne*1E20 * self.Te*1E3*self.e_J + 3/2* self.ni*1E20 * self.Ti*1E3*self.e_J

        self.p_e = self.Te * 1e3 * self.e_J * self.ne * 1e20 * 1e-6
        self.p_i = self.Ti * 1e3 * self.e_J * self.ni * 1e20 * 1e-6

        self.p_check = self.p_e + self.p_i

        self.p_e_avol = volumeAverage_var(self.f, self.p_e)
        self.p_i_avol = volumeAverage_var(self.f, self.p_i)

    def getTransport(self):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Peakings
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.Te_peakingX = self.Te / np.transpose(np.atleast_2d(self.Te_avol))
        self.Ti_peakingX = self.Ti / np.transpose(np.atleast_2d(self.Ti_avol))
        self.ne_peakingX = self.ne / np.transpose(np.atleast_2d(self.ne_avol))
        self.nmain_peakingX = self.nmain / np.transpose(np.atleast_2d(self.nmain_avol))

        ix_pol = np.argmin(np.abs(self.xpol_lw - 0.2))
        self.Te_peaking = self.Te_peakingX[:, ix_pol]
        self.Ti_peaking = self.Ti_peakingX[:, ix_pol]
        self.ne_peaking = self.ne_peakingX[:, ix_pol]
        self.nmain_peaking = self.nmain_peakingX[:, ix_pol]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Ratios
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.TiTe = self.Ti / self.Te
        self.TiTe_avol = self.Ti_avol / self.Te_avol
        self.TiTe0 = self.Ti0 / self.Te0

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Gradients
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.aLTe = gradNorm(self, self.Te)
        self.aLTi = gradNorm(self, self.Ti)
        self.aLne = gradNorm(self, self.ne)
        self.aLnD = gradNorm(self, self.nD)
        self.aLnHe4 = gradNorm(self, self.nHe4)
        self.aLTmini = gradNorm(self, self.Tmini)

        self.aLTe_rho = gradNorm(self, self.Te, specialDerivative=self.x)
        self.aLTi_rho = gradNorm(self, self.Ti, specialDerivative=self.x)
        self.aLne_rho = gradNorm(self, self.ne, specialDerivative=self.x)
        self.aLnD_rho = gradNorm(self, self.nD, specialDerivative=self.x)
        self.aLnHe4_rho = gradNorm(self, self.nHe4, specialDerivative=self.x)
        self.aLTmini_rho = gradNorm(self, self.Tmini, specialDerivative=self.x)

        self.aLTe_gR = self.aLTe_rho * self.ave_grad_rho
        self.aLTi_gR = self.aLTi_rho * self.ave_grad_rho
        self.aLne_gR = self.aLne_rho * self.ave_grad_rho
        self.aLnD_gR = self.aLnD_rho * self.ave_grad_rho
        self.aLnHe4_gR = self.aLnHe4_rho * self.ave_grad_rho
        self.aLTmini_gR = self.aLTmini_rho * self.ave_grad_rho

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Confinement scalings
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Provided by TRANSP

        self.H98y2 = self.f["H98Y2"][:]
        self.tau98y2 = self.f["TAUE98Y2"][:] * 1e3
        self.H89p = self.f["H89P"][:]
        self.tau89p = self.f["TAUE89P"][:] * 1e3

        # Check H98
        self.tau98y2_check, _ = PLASMAtools.tau98y2(
            self.Ip,
            self.Rmajor,
            self.kappaITER,
            self.ne_l,
            self.epsilon,
            self.Bt,
            self.Meff,
            self.Ptot,
        )
        self.tau98y2_check = self.tau98y2_check * 1e3
        self.H98y2_check = self.taue_check / self.tau98y2_check
        self.H98y2_tot_check = self.taueTot_check / self.tau98y2_check

        # Check H98
        self.tau89p_check, _ = PLASMAtools.tau89p(
            self.Ip,
            self.Rmajor,
            self.kappaITER,
            self.ne_l,
            self.epsilon,
            self.Bt,
            self.Meff,
            self.Ptot,
        )
        self.tau89p_check = self.tau89p_check * 1e3
        self.H89p_check = self.taue_check / self.tau89p_check
        self.H89p_tot_check = self.taueTot_check / self.tau89p_check

        # Neo-Alcator (LOC) scaling
        self.tauNA, _ = PLASMAtools.tauNA(
            self.Rmajor,
            self.kappa,
            self.ne_l,
            self.a,
            self.Meff,
            self.Bt,
            self.Ip,
            delta=self.delta,
        )
        self.tauNA = self.tauNA * 1e3
        self.HNA = self.taue / self.tauNA

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Transport coefficients
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Energy

        self.Chi_e = self.f["CONDE"][:] * 1e-4  # in m^2/s
        self.Chi_i = self.f["CONDI"][:] * 1e-4  # in m^2/s

        self.Chi_eff = self.f["CONDEF"][:] * 1e-4  # in m^2/s

        # Particle (effective)

        self.Deff_e = self.f["DIFFE"][:] * 1e-4  # in m^2/s
        self.Deff_i = self.f["DIFFI"][:] * 1e-4  # in m^2/s
        try:
            self.Deff_D = self.f["DIFFD"][:] * 1e-4  # in m^2/s
        except:
            self.Deff_D = copy.deepcopy(self.Deff_e) * 0.0 + self.eps00
        try:
            self.Deff_T = self.f["DIFFT"][:] * 1e-4  # in m^2/s
        except:
            self.Deff_T = copy.deepcopy(self.Deff_D) * 0.0 + self.eps00
        try:
            self.Deff_He4 = self.f["DIFFHE4"][:] * 1e-4  # in m^2/s
        except:
            self.Deff_He4 = copy.deepcopy(self.Deff_D) * 0.0 + self.eps00

        # Particle (input profiles)

        try:
            self.D_He4 = self.f["DFI_HE4"][:] * 1e-4  # in m^2/s
            self.V_He4_R = self.f["VC4_USE"][:] * 1e-2  # in m/s
            self.V_He4, self.V_He4_HF = mapRmajorToX(
                self, self.V_He4_R, originalCoordinate="RMJSYM"
            )
        except:
            self.D_He4 = copy.deepcopy(self.Deff_D) * 0.0 + self.eps00
            self.V_He4 = copy.deepcopy(self.Deff_D) * 0.0 + self.eps00

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Parallel Transport
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Electron collisional time (from NRL for cgs or Fitzpatrick notes for SI)
        factor = (
            (
                6
                * np.sqrt(2)
                * np.pi**1.5
                * self.Eps0**2
                * self.e_J ** (1.5 - 4)
                * np.sqrt(self.me)
            )
            * 1e-20
            * 1000**1.5
        )
        self.tau_coll_e = (
            factor * self.Te**1.5 / self.ne / self.LambdaCoul_e
        )  # in seconds

        # Electron parallel diffusivity
        self.cond_e = (
            3.2
            * self.ne
            * 1e20
            * (self.Te * 1e3 * self.e_J)
            / self.me
            * self.tau_coll_e
        )  # Units: m^-1*s-1
        self.Chi_par_e = self.cond_e / (self.ne * 1e20)  # Units: m^2/s

        # Ratio
        self.Chi_ratio_e = self.Chi_par_e / self.Chi_e

        # From LaHaye 1998
        vTi = np.sqrt(2 * self.Ti_J / self.mi)
        nuTi = (
            4.4e11
            * np.sqrt(self.me / self.mi)
            * self.ne
            * 1e20
            * (self.Ti) ** (-3.0 / 2)
        )
        self.Chi_par_i = vTi**2 / nuTi
        self.Chi_ratio_i = self.Chi_par_i / self.Chi_i

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Turbulence features
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Shear rate
        self.GammaExB = self.f["SREXBV2"][:] * 2 * np.pi * 1e-3  # kHz
        self.GammaExB_tor = self.f["SREXBPHI"][:] * 2 * np.pi * 1e-3  # kHz
        self.GammaExB_pol = self.f["SREXBTHT"][:] * 2 * np.pi * 1e-3  # kHz
        self.GammaExB_p = self.f["SREXBGRP"][:] * 2 * np.pi * 1e-3  # kHz

        self.vExB = copy.deepcopy(self.GammaExB) * 0.0 + self.eps00
        self.vExB_shear = copy.deepcopy(self.GammaExB) * 0.0 + self.eps00

        # self.GammaExB_norm = self.GammaExB / (self.cs/self.a)

        # Ratios
        self.QiQe_ratio = self.qi_obs / self.qe_obs

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Checks
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        negradTe = (
            (self.ne * 1e20)
            * derivativeVar(self, self.Te_J, specialDerivative=self.x)
            * self.ave_grad_rho
        )
        nigradTi = (
            (self.ni * 1e20)
            * derivativeVar(self, self.Ti_J, specialDerivative=self.x)
            * self.ave_grad_rho
        )

        # Chi's and fluxes are defined on the boundary grid, so interpolating the gradient to there
        for it in range(len(self.t)):
            negradTe[it] = np.interp(self.xb[it], self.x[it], negradTe[it])
            nigradTi[it] = np.interp(self.xb[it], self.x[it], nigradTi[it])

        self.qe_MITIM = -self.Chi_e * negradTe * 1e-6  # MW/m^2
        self.qi_MITIM = -self.Chi_i * nigradTi * 1e-6  # MW/m^2
        self.Ge_MITIM = (
            -self.Deff_e
            * derivativeVar(self, self.ne * 1e20, specialDerivative=self.x)
            * self.ave_grad_rho
            * 1e-20
        )

        self.Chi_e_MITIM = self.qe_obs / (-negradTe) * 1e6
        self.Chi_i_MITIM = self.qi_obs / (-nigradTi) * 1e6

        self.Chi_eff_check = -(self.qe_obs * 1e6 + self.qi_obs * 1e6) / (
            negradTe + nigradTi
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Neoclassical
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # This is assuming Chang-Hinton (https://w3.pppl.gov/ntcc/Kapisn/)

        self.Chi_i_neo = self.f["XKINC"][:] * 1e-4  # in m^2/s
        self.qi_neo = (
            -(self.ni * 1e20)
            * self.Chi_i_neo
            * derivativeVar(self, self.Ti_J, specialDerivative=self.x)
            * self.ave_grad_rho
            * 1e-6
        )  # MW/m^2

        self.Chi_e_neo = self.Chi_i_neo * np.sqrt(self.me / self.mi)
        self.qe_neo = (
            -(self.ne * 1e20)
            * self.Chi_e_neo
            * derivativeVar(self, self.Te_J, specialDerivative=self.x)
            * self.ave_grad_rho
            * 1e-6
        )  # MW/m^2

        #
        self.D_e_NCLASS = self.f["DFENC"][:] * 1e-4  # in m^2/s
        self.V_e_NCLASS = self.f["VNDNC_E"][:] * 1e-2  # in m/s
        self.Ge_NCLASS = (
            -self.D_e_NCLASS
            * derivativeVar(self, self.ne * 1e20, specialDerivative=self.x)
            * self.ave_grad_rho
            + self.V_e_NCLASS * (self.ne * 1e20)
        ) * 1e-20
        self.Deff_e_NCLASS = -self.Ge_NCLASS / (
            derivativeVar(self, self.ne * 1e20, specialDerivative=self.x)
            * self.ave_grad_rho
            * 1e-20
        )

    def getConstants(self):
        self.u = 1.66053886e-27  # kg
        self.e_J = 1.60218e-19
        self.me = 9.10938356e-31
        self.Eps0 = 8.854187817e-12
        self.mu0 = 4 * np.pi * 1e-7
        self.c = 2.99792458e8
        self.press_atm = 101325.0

    def getFundamentalPlasmaPhysics(self):
        self.Te_J = self.Te * 1e3 * self.e_J
        self.Ti_J = self.Ti * 1e3 * self.e_J

        # Fundamental Frequencies

        self.wpe = PLASMAtools.calculatePlasmaFrequency(self.ne * 1e20)
        self.wpe = PLASMAtools.calculatePlasmaFrequency(self.ne * 1e20)
        self.cs = PLASMAtools.c_s(self.Te, self.mi / self.u)
        self.vTe = PLASMAtools.vThermal(self.Te, self.me)
        self.vTi = PLASMAtools.vThermal(self.Ti, self.mi)

        self.vA = np.sqrt(self.B_LF**2 / (self.mu0 * self.mi * self.ne * 1e20))
        self.Oce = self.e_J * self.B_LF / self.me
        self.Oci = self.e_J * self.B_LF / self.mi
        self.OceHF = self.e_J * self.B_HF / self.me
        self.OciHF = self.e_J * self.B_HF / self.mi

        # Fundamental Spatial Scales

        self.rhoi = np.sqrt(self.mi * self.Ti_J) / (self.e_J * self.B_LF)
        self.rhos = np.sqrt(self.mi * self.Te_J) / (self.e_J * self.B_LF)
        self.rhoe = np.sqrt(self.me * self.Te * 1e3) / (np.sqrt(self.e_J) * self.B_LF)
        self.lD = PLASMAtools.calculateDebyeLength(self.Te, self.ne * 1e20)
        self.ls = self.c / self.wpe

        rhos_a = []
        for i in range(len(self.x_lw)):
            rhos_a.append(self.rhos[:, i] / self.a)
        self.rhos_a = np.transpose(np.array(rhos_a))

        self.rhos_a_avol = volumeAverage_var(self.f, self.rhos_a)

        # Charge mass ratio

        self.qm_e = (self.e_J * 1) / (self.me)

        self.qm_D = (self.e_J * 1) / (self.mD)
        self.qm_T = (self.e_J * 1) / (self.mT)
        self.qm_He3 = (self.e_J * 2) / (self.mHe3)
        self.qm_He4 = (self.e_J * 2) / (self.mHe4)
        self.qm_H = (self.e_J * 1) / (self.mH)

    def getBoundaryInfo(self):
        R_OMP, self.Bp_OMP, Bt_OMP, Psol, ne = (
            self.Rmaj[:, -1],
            self.Bp_ext[:, -1],
            self.Bt_ext[:, -1],
            np.abs(self.P_LCFS),
            self.ne_avol,
        )

        # ~~~~~~~~ Lambda_q (mm) ~~~~~~~~

        self.Lambda_q_Brunner = PLASMAtools.calculateHeatFluxWidth_Brunner(
            self.p_avol * 1e6 / self.press_atm
        )
        (
            self.Lambda_q_Eich14,
            self.Lambda_q_Eich15,
        ) = PLASMAtools.calculateHeatFluxWidth_Eich(
            self.Bp_OMP, Psol, self.Rmajor, self.epsilon
        )

        Av = 2.5
        Zv = 1.0
        self.Lambda_q_Goldston = (
            5671e3
            * (Psol * 1e6) ** (1.0 / 8.0)
            * (
                (1 + self.kappa**2) ** (5.0 / 8.0)
                * self.a ** (17.0 / 8.0)
                * self.Bt ** (1.0 / 4.0)
                / ((self.Ip * 1e6) ** (9.0 / 8) * self.Rmajor)
            )
            * (2 * Av / (Zv**2 * (1 + Zv))) ** (7.0 / 16.0)
            * ((self.Zeff_avol + 4) / 5) ** (1.0 / 8.0)
            * self.Rmajor
            * self.Bp_avol
            / (R_OMP * self.Bp_OMP)
        )

        # ~~~~~~~~ Upstream temperature ~~~~~~~~

        self.Te_u_Eich14, self.Te_u_Eich15, self.Te_u_Brunner, self.Te_u_Goldston = (
            np.zeros(len(self.t)),
            np.zeros(len(self.t)),
            np.zeros(len(self.t)),
            np.zeros(len(self.t)),
        )
        self.Ti_u_Eich14, self.Ti_u_Eich15, self.Ti_u_Brunner, self.Ti_u_Goldston = (
            np.zeros(len(self.t)),
            np.zeros(len(self.t)),
            np.zeros(len(self.t)),
            np.zeros(len(self.t)),
        )
        for i in range(len(self.t)):
            (
                self.Te_u_Eich14[i],
                self.Ti_u_Eich14[i],
            ) = PLASMAtools.calculateUpstreamTemperature(
                self.Lambda_q_Eich14[i],
                self.q95[i],
                ne[i],
                Psol[i],
                R_OMP[i],
                self.Bp_OMP[i],
                Bt_OMP[i],
            )
            (
                self.Te_u_Eich15[i],
                self.Ti_u_Eich15[i],
            ) = PLASMAtools.calculateUpstreamTemperature(
                self.Lambda_q_Eich15[i],
                self.q95[i],
                ne[i],
                Psol[i],
                R_OMP[i],
                self.Bp_OMP[i],
                Bt_OMP[i],
            )
            (
                self.Te_u_Brunner[i],
                self.Ti_u_Brunner[i],
            ) = PLASMAtools.calculateUpstreamTemperature(
                self.Lambda_q_Brunner[i],
                self.q95[i],
                ne[i],
                Psol[i],
                R_OMP[i],
                self.Bp_OMP[i],
                Bt_OMP[i],
            )
            (
                self.Te_u_Goldston[i],
                self.Ti_u_Goldston[i],
            ) = PLASMAtools.calculateUpstreamTemperature(
                self.Lambda_q_Goldston[i],
                self.q95[i],
                ne[i],
                Psol[i],
                R_OMP[i],
                self.Bp_OMP[i],
                Bt_OMP[i],
            )

    def getStabilityLimits(self):
        # ~~~~ kink
        self.qstarsep = PLASMAtools.evaluate_qstar(
            self.Ip,
            self.Rmajor,
            self.kappa,
            self.Bt,
            self.epsilon,
            self.delta,
            isInputIp=True,
        )
        self.qstarsepITER = PLASMAtools.evaluate_qstar(
            self.Ip,
            self.Rmajor,
            self.kappa,
            self.Bt,
            self.epsilon,
            self.delta,
            isInputIp=True,
            ITERcorrection=True,
        )

        self.qstar95 = PLASMAtools.evaluate_qstar(
            self.Ip,
            self.Rmajor,
            self.kappa_950,
            self.Bt,
            self.epsilon,
            self.delta_950,
            isInputIp=True,
        )
        self.qstar95ITER = PLASMAtools.evaluate_qstar(
            self.Ip,
            self.Rmajor,
            self.kappa_950,
            self.Bt,
            self.epsilon,
            self.delta_950,
            isInputIp=True,
            ITERcorrection=True,
        )

        # ~~~~ kappa limit
        self.kappa_lim = np.zeros(len(self.t))
        for i in range(len(self.t)):
            self.kappa_lim[i] = PLASMAtools.calculateKappaLimit(
                self.epsilon[i], self.delta[i], self.Li3[i], self.Bp_OMP[i]
            )

        # ~~~~ Disruptivity
        self.penalties = np.zeros(len(self.t))
        for i in range(len(self.t)):
            self.penalties[i] = definePenalties(
                self.q95[i],
                self.fGv[i],
                self.kappa[i],
                self.BetaN[i],
                maxKappa=self.kappa_lim[i],
            )

    def getEquilibriumFunctions(self):
        """
        F is a flux function, so F_LF=F
        F here works in XB
        """

        self.F_LF, self.F = mapRmajorToX(
            self,
            self.Bt_plasma * self.Rmaj,
            originalCoordinate="RMAJM",
            interpolateToX=False,
        )

        self.FFprime = self.F * derivativeVar(self, self.F, specialDerivative=self.psi)
        self.pprime = derivativeVar(self, self.p_kin * 1e6, specialDerivative=self.psi)

    def checkQNZEFF(self):
        # Checks
        self.Zeff_check = (
            self.fmain_avol
            + self.fHe4_avol * 2**2
            + self.fZ_avolAVE * self.fZ_avolAVE_Z**2
            + self.fFast_avolAVE * self.fFast_avolAVE_Z**2
        )

        self.QN_check = (
            self.fmain_avol
            + self.fHe4_avol * 2
            + self.fZ_avolAVE * self.fZ_avolAVE_Z
            + self.fFast_avolAVE * self.fFast_avolAVE_Z
        )

        self.Zeff_error = (self.Zeff_avol - self.Zeff_check) / self.Zeff_avol
        self.QN_error = 1 - self.QN_check

        # ~~~~~~~~~~~~~~~~~~~~~~~~~ ne_check ~~~~~~~~~~~~~~~~~~~~~~~~~

        self.ne_check = self.nmain + (self.nmini + self.nHe4) * 2
        self.nZ_quas = copy.deepcopy(self.nZ) * 0.0 + self.eps00
        for i in self.fZs_avol:
            for j in range(len(self.fZs_avol[i]["states"])):
                self.nZ_quas += self.nZs[i]["states"][j + 1] * int(j + 1)
        self.ne_check += self.nZ_quas

        # Average plasma
        self.Zbar = self.QN_check / (
            self.fmain_avol + self.fHe4_avol + self.fZ_avolAVE + self.fFast_avolAVE
        )

    def checkRun(self, fig=None, time=None, printYN=True):
        if time is None:
            time = self.ind_saw
        else:
            time = np.argmin(np.abs(self.t - time))
        timeReq = self.t[time]

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)

        # _________________________________

        rads = [0.4, 0.8]
        # Total Powers
        ax0 = fig.add_subplot(grid[:, 0])
        ax1 = fig.add_subplot(grid[0, 1])
        ax2 = fig.add_subplot(grid[1, 1])

        indTime = self.plotConvSolver_x(
            timeReq, avt=0.0, ax=ax0, alsoParticle=False, alsoTR=True, colorStart=0
        )
        GRAPHICStools.addDenseAxis(ax0)
        indX = self.plotConvSolver_t(
            rads[0], ax=ax1, meanLim=True, leg=False, alsoParticle=False
        )
        GRAPHICStools.addDenseAxis(ax1)
        indX2 = self.plotConvSolver_t(
            rads[1], ax=ax2, meanLim=True, leg=False, alsoParticle=False
        )
        GRAPHICStools.addDenseAxis(ax2)
        ax0.axvline(x=self.x_lw[indX], c="k", ls="--")
        ax0.axvline(x=self.x_lw[indX2], c="k", ls="--")
        ax1.axvline(x=self.t[indTime], c="k", ls="--")

    def plotAroundSawtooth(
        self,
        quantity2D,
        x=None,
        ax=None,
        sawtooth=-2,
        fractionBefore=0.5,
        fractionAfter=0.5,
        boldPlusMinus=2,
        lw=0.2,
        alpha=1.0,
        legend=True,
        plotLinesSawVerbose=1,
        ):
        sawtooth += 1

        if len(self.ind_sawAll) > 2:
            nextP = 1  # boldPlusMinus

            if ax is None:
                fig, ax = plt.subplots()

            it = self.ind_sawAll[-sawtooth]
            t = self.t[it]
            it_prev = self.ind_sawAll[-sawtooth + nextP]
            tprev = self.t[it_prev]
            t1 = t - fractionBefore * (t - tprev)
            it1 = np.argmin(np.abs(self.t - t1))

            it_next = self.ind_sawAll[-sawtooth - nextP]
            tnext = self.t[it_next]
            t2 = t + fractionAfter * (tnext - t)
            it2 = np.argmin(np.abs(self.t - t2))

            it2 = np.min([len(self.t) - 1, it2])

            itBig1 = it - boldPlusMinus
            itBig2 = np.min([len(self.t) - 1, it + boldPlusMinus])

            if x is None:
                x = self.x

            # plot before sawtooth
            _ = GRAPHICStools.plotRange(
                self.t,
                x,
                quantity2D,
                ax=ax,
                it1=it1,
                it2=it,
                itBig=[itBig1],
                colors=["m", "r"],
                colorsBig=["r"],
                lw=lw,
                legend=legend,
                alpha=alpha,
            )

            # plot after sawtooth
            _ = GRAPHICStools.plotRange(
                self.t,
                x,
                quantity2D,
                ax=ax,
                it1=it,
                it2=it2,
                itBig=[itBig2],
                colors=["b", "g"],
                colorsBig=["b"],
                lw=lw,
                legend=legend,
                alpha=alpha,
            )

            ax.set_xlim([0, 1])
            if legend:
                ax.legend()
            ax.set_xlabel("$\\rho_N$")

            # my definitions
            exx = 1
            w_sheet = 0.05

            thr = 1e-2
            q = self.q[it + nextP] - self.q[it + nextP, 15]
            x_saw_r1l = self.xb_lw[np.where(q > thr)[0][0] - exx]
            x_saw_r1 = x_saw_r1l + w_sheet
            x_saw_r1r = x_saw_r1 + w_sheet

            # try:
            # 	x_saw_r2l= self.xb_lw[self.q[it+nextP]==1.0][-1]
            # except:
            # 	thr = 1E-5
            # 	x_saw_r2l = self.xb_lw[np.where(self.q[it+nextP]-1>thr)[0][0]-exx-exx]
            x_saw_r2l = self.x_saw_mix[it] - w_sheet

            x_saw_r2 = x_saw_r2l + w_sheet
            x_saw_r2r = x_saw_r2 + w_sheet

            if plotLinesSawVerbose > 0:
                pos = np.interp(x_saw_r1, self.xb[it], x[it])
                ax.axvline(x=pos, alpha=0.5, lw=2, ls="--", c="c")
                pos = np.interp(self.x_saw_inv[it], self.xb[it], x[it])
                ax.axvline(x=pos, alpha=0.5, lw=2, ls="--", c="k")
                pos = np.interp(self.x_saw_mix[it], self.xb[it], x[it])
                ax.axvline(x=pos, alpha=0.5, lw=2, ls="--", c="g")
                pos = np.interp(self.x_saw_inv_prof[it], self.xb[it], x[it])
                ax.axvline(x=pos, alpha=0.5, lw=2, ls="--", c="m")

                if plotLinesSawVerbose > 1:
                    pos = np.interp(x_saw_r1l, self.xb[it], x[it])
                    ax.axvline(x=pos, alpha=0.5, lw=1, ls="-.", c="c")
                    pos = np.interp(x_saw_r1r, self.xb[it], x[it])
                    ax.axvline(x=pos, alpha=0.5, lw=1, ls="-.", c="c")
                    pos = np.interp(x_saw_r2l, self.xb[it], x[it])
                    ax.axvline(x=pos, alpha=0.5, lw=1, ls="-.", c="y")
                    pos = np.interp(x_saw_r2, self.xb[it], x[it])
                    ax.axvline(x=pos, alpha=0.5, lw=2, ls="--", c="y")
                    pos = np.interp(x_saw_r2r, self.xb[it], x[it])
                    ax.axvline(x=pos, alpha=0.5, lw=1, ls="-.", c="y")

            compoundX = [x_saw_r1l, x_saw_r1, x_saw_r1r, x_saw_r2l, x_saw_r2, x_saw_r2r]

        else:
            it, compoundX = 0, [0, 0, 0, 0, 0, 0]

        return it, compoundX

    def plotComparison(self, fig=None, Xrange=[0, 1.0]):
        if fig is None:
            fig = plt.figure(figsize=(22, 9))

        time_s = self.t[self.ind_saw]
        # if self.coincidentTime[1] is None: 	time_s = self.t[self.ind_saw]
        # else:								time_s = self.coincidentTime[1]

        if self.machine == "CMOD" or self.machine == "SPARC":
            t_exp = self.exp.time  # - self.coincidentTime[0]
            t_exp2 = self.exp.Te0_t  # - self.coincidentTime[0]
            Trange = [t_exp[0] - 0.5, t_exp[-1] + 0.5]

            # print(' --> Transforming experiment time from t = {0:.3f}s to t = 0s (t0 = {1:.3f}s, {2:.3f}s)'.format(self.coincidentTime[0],t_exp[0],t_exp2[0]))

            t_sim = self.t  # - time_s

            # print(' --> Transforming simulation time from t = {0:.3f}s to t = 0s (t0 = {1:.3f}s)'.format(time_s,t_sim[0]))

            grid = plt.GridSpec(2, 5, hspace=0.6, wspace=0.6)

            # --------------------------------------------------------------------------------

            ax = fig.add_subplot(grid[0, 0])
            GRAPHICStools.compareExpSim(
                t_exp,
                [self.exp.Wexp],
                t_sim,
                [self.Wtot, self.Wth],
                ax=ax,
                lab_exp=["($W_{efit}$)"],
                lab_sim=["($W_{tot}$)", "($W_{thr}$)"],
                title="Stored Energy",
                ylabel="MJ",
                plotError=False,
                xrrange=Trange,
                yrrange=True,
            )

            ax = fig.add_subplot(grid[0, 1])
            GRAPHICStools.compareExpSim(
                t_exp,
                [self.exp.neut],
                t_sim,
                [self.neutrons],
                ax=ax,
                lab_exp=[""],
                lab_sim=[""],
                title="Neutron rate",
                ylabel="$10^{20} n/s$",
                plotError=False,
                xrrange=Trange,
                yrrange=True,
            )

            ax = fig.add_subplot(grid[0, 2])
            GRAPHICStools.compareExpSim(
                t_exp,
                [np.abs(self.exp.Vsurf)],
                t_sim,
                [np.abs(self.Vsurf)],
                ax=ax,
                lab_exp=[""],
                lab_sim=[""],
                title="Loop voltage",
                ylabel="|V|",
                plotError=False,
                xrrange=Trange,
            )

            ax = fig.add_subplot(grid[0, 3])
            GRAPHICStools.compareExpSim(
                t_exp,
                [self.exp.Li],
                t_sim,
                [self.Li1, self.Li3],
                ax=ax,
                lab_exp=[""],
                lab_sim=["Li1", "Li3"],
                title="Internal Inductance",
                ylabel="",
                plotError=False,
                xrrange=Trange,
                yrrange=True,
            )

            ax = fig.add_subplot(grid[0, 4])
            GRAPHICStools.compareExpSim(
                t_exp,
                [self.exp.q95],
                t_sim,
                [self.q95],
                ax=ax,
                lab_exp=["(EFIT)"],
                lab_sim=[""],
                title="q95",
                ylabel="",
                plotError=False,
                xrrange=Trange,
                yrrange=True,
            )

            ax = fig.add_subplot(grid[1, 0])
            GRAPHICStools.compareExpSim(
                t_exp,
                [self.exp.neL],
                t_sim,
                [self.ne_l, self.ne_avol],
                ax=ax,
                lab_exp=["(nebar)"],
                lab_sim=["(line)", "(vol)"],
                title="Line Average Density",
                ylabel="",
                plotError=False,
                xrrange=Trange,
                yrrange=True,
            )

            ax = fig.add_subplot(grid[1, 1:3])
            ix = np.argmin(np.abs(self.x_lw - self.exp.Te0_x))
            GRAPHICStools.compareExpSim(
                t_exp2,
                [self.exp.Te0],
                t_sim,
                [self.Te[:, ix]],
                ax=ax,
                lab_exp=["(ECE ch1)"],
                lab_sim=[""],
                title=f"Temperature @ rho={self.exp.Te0_x:.2f}",
                ylabel="",
                plotError=False,
                lw=1,
                xrrange=Trange,
                yrrange=True,
            )

            # 2D -----------------------------------

            it1 = np.argmin(np.abs(self.t - (self.timeProfile - self.timeProfile_av)))
            it2 = np.argmin(np.abs(self.t - (self.timeProfile + self.timeProfile_av)))

            Te = timeAverage(self.t[it1 : it2 + 1], self.Te[it1 : it2 + 1])
            Ti = timeAverage(self.t[it1 : it2 + 1], self.Ti[it1 : it2 + 1])
            ne = timeAverage(self.t[it1 : it2 + 1], self.ne[it1 : it2 + 1])

            ax = fig.add_subplot(grid[1, 3])
            GRAPHICStools.compareExpSim(
                None,
                [self.exp.Te_TS, self.exp.Te_ECE],
                self.t,
                [Te, Ti],
                x_exp=[self.exp.x_TS, self.exp.x_ECE],
                x_sim=self.x_lw,
                ax=ax,
                lab_exp=["(TS)", "(ECE)"],
                lab_sim=["Te (sim)", "Ti (sim)"],
                title="Electron Temperature",
                ylabel="$T$ (keV)",
                z_exp_err=[self.exp.TeError_TS, self.exp.TeError_ECE],
                xrrange=Xrange,
                yrrange=True,
            )

            ax = fig.add_subplot(grid[1, 4])
            GRAPHICStools.compareExpSim(
                None,
                [self.exp.ne_TS],
                self.t,
                [ne],
                x_exp=[self.exp.x_TS],
                x_sim=self.x_lw,
                ax=ax,
                lab_exp=["(TS)"],
                lab_sim=[""],
                title="Electron Density",
                ylabel="$n_e$ ($10^{20} n/s$)",
                z_exp_err=[self.exp.neError_TS],
                xrrange=Xrange,
                yrrange=True,
            )

        if self.machine == "AUG":
            if fig is None:
                fig = plt.figure(figsize=(10, 5))

            grid = plt.GridSpec(1, 3, hspace=0.6, wspace=0.6)

            ax = fig.add_subplot(grid[0, 0])
            Te = sawtoothAverage(self.t, self.Te, self.tlastsawU, time=time_s)
            GRAPHICStools.compareExpSim(
                None,
                [self.exp.Te],
                None,
                [Te],
                x_exp=[self.exp.rho_Te],
                x_sim=self.x_lw,
                alphaE=0.05,
                inRhoPol={"rho_pol": self.xpol_lw, "rho_tor": self.x_lw},
                ax=ax,
                lab_exp=["(AUGPED)"],
                lab_sim=[""],
                title="Electron Temperature",
                ylabel="$T_e$ (keV)",
                z_exp_err=[self.exp.Te_err],
                xrrange=Xrange,
                yrrange=True,
            )

            ax = fig.add_subplot(grid[0, 1])
            Ti = sawtoothAverage(self.t, self.Ti, self.tlastsawU, time=time_s)
            GRAPHICStools.compareExpSim(
                None,
                [self.exp.Ti],
                None,
                [Ti],
                x_exp=[self.exp.rho_Ti],
                x_sim=self.x_lw,
                alphaE=0.1,
                inRhoPol={"rho_pol": self.xpol_lw, "rho_tor": self.x_lw},
                ax=ax,
                lab_exp=["(AUGPED)"],
                lab_sim=[""],
                title="Ion Temperature",
                ylabel="$T_i$ (keV)",
                z_exp_err=[self.exp.Ti_err],
                xrrange=Xrange,
                yrrange=True,
            )

            ax = fig.add_subplot(grid[0, 2])
            ne = sawtoothAverage(self.t, self.ne, self.tlastsawU, time=time_s)
            GRAPHICStools.compareExpSim(
                None,
                [self.exp.ne],
                None,
                [ne],
                x_exp=[self.exp.rho_ne],
                x_sim=self.x_lw,
                alphaE=0.5,
                inRhoPol={"rho_pol": self.xpol_lw, "rho_tor": self.x_lw},
                ax=ax,
                lab_exp=["(AUGPED)"],
                lab_sim=[""],
                title="Electron Density",
                ylabel="$n_e$ ($10^{20} n/s$)",
                z_exp_err=[self.exp.ne_err],
                xrrange=Xrange,
                yrrange=True,
            )

    def plotMachine(self, fig=None, color="b", plotComplete=True, time=None):
        if time is None:
            time = self.t[self.ind_saw]

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(6, 3, hspace=0.6, wspace=0.2)

        axField = fig.add_subplot(grid[:2, 0])
        axGeo = fig.add_subplot(grid[2:, 0])  # ,sharex=axField)

        self.plotField(
            ax=axField, color=color, plotComplete=plotComplete, time=time, notAxis=False
        )
        indt = self.plotGeometry(
            ax=axGeo,
            color=color,
            plotComplete=plotComplete,
            time=time,
            plotVV=True,
            Aspect=True,
        )
        axGeo.legend(loc="upper right", prop={"size": self.mainLegendSize})

        axParams = fig.add_subplot(grid[0:3, 1])  # fig.add_subplot(6,3,())
        axGeoParams = fig.add_subplot(
            grid[3:, 1]
        )  # ,sharex=axParams) #fig.add_subplot(6,3,())

        self.plotOperation(ax=axParams, ax1=axGeoParams)
        axParams.axvline(x=self.t[indt], c="k", lw=0.5, ls="--")
        axGeoParams.axvline(x=self.t[indt], c="k", lw=0.5, ls="--")

        ax1 = fig.add_subplot(
            grid[0:3, 2]
        )  # ,sharex=axParams) #fig.add_subplot(6,3,())
        ax1.axvline(x=self.t[indt], c="k", lw=0.5, ls="--")
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.tick_params(labelbottom=True)
        ax1.set_xlabel("Time (s)")

        GRAPHICStools.addDenseAxis(ax1)

        # ----
        ax1.plot(self.t, self.Vsurf, lw=2, label="$V_{surf}$ (V)")
        ax1.plot(self.t, self.Vsurf_m, lw=1, ls="--", label="$V_{surf,meas}$ (V)")
        ax1.plot(self.t, self.IpB_fraction, lw=2, label="$I_{p,B}/I_{p}$")

        ax1.plot(self.t, self.fGv, lw=2, label="$f_{G,vol}$")
        ax1.plot(self.t, self.fGl, lw=2, label="$f_{G,lin}$")

        ax1.legend(loc="best", prop={"size": self.mainLegendSize})
        ax1.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax1)

        # ----
        ax2 = fig.add_subplot(grid[3:, 2])
        it = np.argmin(np.abs(self.t - time))
        ax2.plot(self.x_lw, self.Bt_x[it], lw=2, label="$B^{av LF-HF}_{\\phi,midpl}$")
        ax2.plot(self.x_lw, self.Bp_x[it], lw=2, label="$B^{av LF-HF}_{\\theta,midpl}$")
        ax2.plot(self.x_lw, self.B_av[it], lw=3, label="$\\langle B\\rangle$")
        ax2.plot(self.x_lw, self.TGLF_Bunit[it], lw=3, label="$B_{unit,TGLF}$")
        ax2.plot(self.x_lw, self.Bpol[it], lw=3, label="$\\langle B_{\\theta}\\rangle$")
        ax2.plot(
            self.x_lw,
            self.Bpol_check[it],
            lw=2,
            c="y",
            ls="--",
            label="$\\langle B_{\\theta}\\rangle$ check",
        )
        ax2.plot(
            self.x_lw,
            self.Bpol_long[it],
            lw=3,
            label="$\\langle B_{\\theta}\\rangle_L=\\mu_0 I_{p,encl}/L_{\\theta}$ ",
        )

        ax2.legend(loc="best", prop={"size": self.mainLegendSize})
        ax2.set_ylim(bottom=0)

        ax2.set_ylabel("Magnetic fields (T)")
        ax2.set_xlabel("$\\rho_N$")
        ax2.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax2)

    def plotOperation(self, ax=None, ax1=None):
        if ax is None:
            fig, ax = plt.subplots()
        if ax1 is None:
            fig, ax1 = plt.subplots()

        ax.plot(self.t, self.Ip, lw=2, label="$I_p$ (MA)")
        ax.plot(self.t, self.Bt, lw=2, label="$B_T$ (T)")
        ax.plot(self.t, self.utilsiliarPower, lw=2, label="Power In(MW)")
        ax.plot(self.t, self.ne_avol, lw=2, label="$n_e$ vol ($10^{20}m^{-3}$)")
        ax.plot(self.t, self.ne_l, lw=2, label="$n_e$ lin ($10^{20}m^{-3}$)")
        ax.plot(self.t, self.Zeff_avol, lw=2, label="$Z_{eff}$")

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

        ax1.plot(self.t, self.Rmajor, lw=2, label="$R_{major}$ (m)")
        ax1.plot(self.t, self.a, lw=2, label="$a$ (m)")
        ax1.plot(self.t, self.kappa, lw=2, label="$\\kappa$")
        ax1.plot(self.t, self.delta, lw=2, label="$\\delta$")
        ax1.plot(self.t, self.q95, lw=2, label="$q_{95}$")
        ax1.plot(self.t, self.q95_check, lw=1, c="y", label="$q_{95,check}$")

        ax1.legend(loc="best", prop={"size": self.mainLegendSize})
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel("Time (s)")

        GRAPHICStools.addDenseAxis(ax1)

    def plotPressures(self, fig=None, time=None):
        if fig is None:
            fig = plt.figure()
        grid = plt.GridSpec(nrows=2, ncols=2, hspace=0.2, wspace=0.4)

        if time is None:
            time = self.t[self.ind_saw]

        it = np.argmin(np.abs(self.t - time))

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(grid[0, 1])
        ax4 = fig.add_subplot(grid[1, 1], sharex=ax3)

        ax1.plot(self.t, self.Wtot, lw=2, label="$W_{tot}$")
        ax1.plot(self.t, self.Wth, lw=2, label="$W_{th}$")
        ax1.plot(self.t, self.Wfast, lw=2, label="$W_{fast}$")
        ax1.plot(self.t, self.Wfast_fus, lw=1, label="$W_{fast,fus}$")
        ax1.plot(self.t, self.Wfast_mini, lw=1, label="$W_{fast,mini}$")
        ax1.plot(self.t, self.Wfast_b, lw=1, label="$W_{fast,b}$")
        ax1.plot(self.t, self.Wperp, lw=1, label="$W_{fast,\\perp}$")
        ax1.plot(self.t, self.Wpar, lw=1, label="$W_{fast,\\parallel}$")
        ax1.plot(
            self.t,
            self.Wth + self.Wfast,
            lw=1,
            c="y",
            ls="--",
            label="$W_{th}+W_{fast}$",
        )
        ax1.plot(
            self.t,
            self.Wperp + self.Wpar,
            lw=1,
            c="g",
            ls="--",
            label="$W_{fast,\\perp}+W_{fast,\\parallel}$",
        )

        ax1.axvline(x=self.t[it], c="m", lw=1.0, ls="--")

        ax1.legend(loc="best", prop={"size": self.mainLegendSize})
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("W (MJ)")
        GRAPHICStools.addDenseAxis(ax1)

        ax2.plot(self.t, self.BetaN, lw=2, label="$\\beta_N$ (%)")
        ax2.plot(self.t, self.BetaTor * 100, lw=2, label="$\\beta_{\\phi}$ (%)")
        ax2.plot(
            self.t,
            self.BetaTor_check * 100,
            lw=1,
            ls="--",
            label="$\\beta_{\\phi}$ (%) check",
        )
        ax2.plot(self.t, self.BetaPol, lw=2, label="$\\beta_{\\theta}$")
        ax2.plot(
            self.t, self.BetaPol_check, lw=1, ls="--", label="$\\beta_{\\theta}$ check"
        )
        ax2.plot(self.t, self.BetaPol_1D, lw=2, label="$\\beta_{\\theta,1D}$")
        ax2.plot(
            self.t,
            self.BetaPol_1D_check,
            lw=1,
            ls="--",
            label="$\\beta_{\\theta,1D}$ check",
        )
        # ax2.plot(self.t,self.Beta*100,lw=2,c='c',ls='--',label='$\\beta$ (%)')

        ax2.legend(loc="best", prop={"size": self.mainLegendSize})
        ax2.set_ylabel("$\\beta_N$ (%), $\\beta_{\\phi}$ (%), $\\beta_{\\theta}$")
        ax2.set_xlabel("Time (s)")
        # ax2.set_ylim([0,3.0])
        ax2.axvline(x=self.t[it], c="m", lw=1.0, ls="--")
        GRAPHICStools.addDenseAxis(ax2)

        ax3.plot(self.x_lw, self.p_kin[it], lw=2, label="$p_{kin}$")
        ax3.plot(self.x_lw, self.p[it], lw=2, label="$p$")
        ax3.plot(
            self.x_lw,
            self.p_check[it],
            c="orange",
            ls="--",
            lw=1,
            label="$n_eT_e+n_iT_i$",
        )
        ax3.plot(self.x_lw, self.pFast[it], lw=2, label="$p_{fast}$")
        ax3.plot(self.x_lw, self.pFast_fus[it], lw=1, label="$p_{fast,fus}$")
        ax3.plot(self.x_lw, self.pFast_mini[it], lw=1, label="$p_{fast,mini}$")
        ax3.plot(self.x_lw, self.pFast_b[it], lw=1, label="$p_{fast,b}$")
        ax3.plot(
            self.x_lw,
            self.p[it] + self.pFast[it],
            c="y",
            ls="--",
            lw=2,
            label="$p+p_{fast}$ check",
        )

        ax3.legend(loc="best", prop={"size": self.mainLegendSize})
        ax3.set_ylim(bottom=0)
        ax3.set_xlim([0, 1])
        ax3.set_xlabel("$\\rho_N$")
        ax3.set_ylabel("p (MPa)")
        GRAPHICStools.addDenseAxis(ax3)

        ax4.plot(self.x_lw, self.BetaTor_x[it], lw=2, label="$\\beta_{\\phi}$")
        ax4.plot(
            self.x_lw,
            self.BetaTor_x_check[it],
            lw=1,
            ls="--",
            label="$\\beta_{\\phi}$ check",
        )

        ax4.legend(loc="best", prop={"size": self.mainLegendSize})
        ax4.set_ylim(bottom=0)
        ax4.set_xlim([0, 1])
        ax4.set_xlabel("$\\rho_N$")
        ax4.set_ylabel("$\\beta$")
        GRAPHICStools.addDenseAxis(ax4)

    def plotGeometry(
        self,
        ax=None,
        time=None,
        rhoS=np.linspace(0.1, 1, 10),
        rhoPol=False,
        ls="-",
        color="b",
        colorsurfs=None,
        alphasurfs=1.0,
        plotComplete=True,
        plotStructures=True,
        plotVV=False,
        Aspect=True,
        plotSurfs=False,
        inverted=False,
        sqrt=True,
        lwB=3,
        lw = 1,
        label="",
        labelS="",
        plotBound=True,
    ):
        if colorsurfs is None:
            colorsurfs = color

        if inverted:
            mult = -1.0
        else:
            mult = 1.0

        if ax is None:
            fig, ax = plt.subplots()
        if time is None:
            time = self.t[self.ind_saw]

        indt = np.argmin(np.abs(self.t - time))

        for rho in rhoS:
            RMC, YMC = getFluxSurface(self.f, time, rho, rhoPol=rhoPol, sqrt=sqrt)
            if plotComplete or plotSurfs:
                co = colorsurfs
                ax.plot(
                    mult * RMC, YMC, lw=lw, c=co, ls=ls, alpha=alphasurfs, label=labelS
                )

        RMC, YMC = getFluxSurface(self.f, time, 1.0)
        ax.plot(mult * RMC, YMC, lw=lwB, c=color, ls=ls, label=label)

        sizep = 30
        if plotComplete:
            ax.scatter([mult * self.Rmag[indt]], [self.Ymag[indt]], sizep, colorsurfs)

        if plotStructures:
            ax.scatter([mult * self.Rmajor[indt]], [self.Ymag[indt]], sizep, "k")
            ax.axhline(y=self.Ymag[indt], color="k", lw=1, ls="--")
            ax.axvline(x=mult * self.Rmajor[indt], color="k", lw=1, ls="--")
            ax.axvline(x=mult * self.Rmag[indt], color="g", lw=1, ls="--")
            if hasattr(self, "R_lim") and self.R_lim is not None:
                ax.plot(mult * self.R_lim, self.Z_lim, c="k", lw=3, label="Limiters")
            if hasattr(self, "R_ant") and self.R_ant is not None:
                for icha in range(self.R_ant.shape[0]):
                    ax.plot(
                        mult * self.R_ant[icha, :],
                        self.Z_ant[icha, :],
                        c="g",
                        lw=3,
                        label=f"Antenna #{icha+1}",
                    )

        if plotVV and hasattr(self, "R_vv") and self.R_vv is not None:
            ax.plot(mult * self.R_vv, self.Z_vv, c="m", lw=1, label="VV")

        if plotSurfs:
            ax.scatter([mult * self.Rmag[indt]], [self.Ymag[indt]], 50, colorsurfs)

        if plotBound and hasattr(self, "bound_R"):
            ax.plot(self.bound_R, self.bound_Z, ls="-", lw=1, c="k")

        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        if Aspect:
            ax.set_aspect("equal")
            # ax.set_ylim([-self.b[indt]*1.3,self.b[indt]*1.3])
            ax.set_xlim(
                [mult * self.Rmaj[indt, 0] * 0.7, mult * self.Rmaj[indt, -1] * 1.3]
            )

        return indt

    def plotGeometry_Above(
        self,
        ax=None,
        time=None,
        rhoS=np.linspace(0.1, 1, 10),
        ls="-",
        color="b",
        colorsurfs="k",
        plotComplete=True,
        plotStructures=True,
        Aspect=True,
        plotSurfs=False,
    ):
        if ax is None:
            fig, ax = plt.subplots()

        if time is None:
            time = self.t[self.ind_saw]

        indt = np.argmin(np.abs(self.t - time))

        for ix in rhoS:
            RMC, YMC = getFluxSurface(self.f, time, ix)
            label = "$\\rho_N=$" + str(ix)
            if plotComplete or plotSurfs:
                lw = 1
                co = colorsurfs
                R, Z = MATHtools.circle(RMC.min())
                ax.plot(R, Z, lw=lw, c=co, ls=ls)
                R, Z = MATHtools.circle(RMC.max())
                ax.plot(R, Z, lw=lw, c=co, ls=ls)

        RMC, YMC = getFluxSurface(self.f, time, ix)
        R, Z = MATHtools.circle(RMC.min())
        ax.plot(R, Z, lw=3, c=color, ls=ls)
        R, Z = MATHtools.circle(RMC.max())
        ax.plot(R, Z, lw=3, c=color, ls=ls)

        if plotComplete:
            R, Z = MATHtools.circle(self.Rmag[indt])
            ax.plot(R, Z, lw=3, c="g", ls="--")
            ax.axvline(x=(self.Rmajor[indt] + self.a[indt]), color="k", lw=1, ls="--")
            ax.axvline(x=(self.Rmajor[indt] - self.a[indt]), color="k", lw=1, ls="--")
            ax.axvline(x=-(self.Rmajor[indt] + self.a[indt]), color="k", lw=1, ls="--")
            ax.axvline(x=-(self.Rmajor[indt] - self.a[indt]), color="k", lw=1, ls="--")
            ax.axvline(x=self.Rmag[indt], color="g", lw=1, ls="--")
            ax.axvline(x=-self.Rmag[indt], color="g", lw=1, ls="--")

        if plotStructures:
            if hasattr(self, "R_lim") and self.R_lim is not None:
                R, Z = MATHtools.circle(self.R_lim.min())
                ax.plot(R, Z, c="k", lw=3, label="Limiters")
                R, Z = MATHtools.circle(self.R_lim.max())
                ax.plot(R, Z, c="k", lw=3, label="Limiters")
            if hasattr(self, "R_vv") and self.R_vv is not None:
                R, Z = MATHtools.circle(self.R_vv.min())
                ax.plot(R, Z, c="m", lw=1, label="VV")
                R, Z = MATHtools.circle(self.R_vv.max())
                ax.plot(R, Z, c="m", lw=1, label="VV")

        if plotSurfs:
            ax.scatter([self.Rmag[indt]], [self.Ymag[indt]], 50, colorsurfs)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        if Aspect:
            ax.set_aspect("equal")
            # ax.set_ylim([-self.b[indt]*1.3,self.b[indt]*1.3])
            ax.set_xlim([-self.Rmaj[indt, -1] * 1.3, self.Rmaj[indt, -1] * 1.3])

        return indt

    def plotField(
        self, ax=None, time=None, color="b", plotComplete=True, notAxis=False
    ):
        if ax is None:
            fig, ax = plt.subplots()

        if time is None:
            time = self.t[self.ind_saw]

        indt = np.argmin(np.abs(self.t - time))

        ax.plot(
            self.Rmaj[indt, :],
            self.Bt_ext[indt, :],
            lw=2,
            c=color,
            label="$B_{\\phi,ext}$",
        )
        ax.plot(
            self.Rmaj[indt, :],
            self.Bt_plasma[indt, :],
            lw=1,
            c="r",
            label="$B_{\\phi}$",
        )

        ax.plot(
            self.Rmaj[indt, :], self.Bp_ext[indt, :], lw=2, c="m", label="$B_{\\theta}$"
        )

        ax.axhline(y=0.0, c="k", ls="--", lw=1)

        inter = np.argmin(np.abs(self.Rmaj[indt, :] - self.Rmajor[indt]))
        ax.scatter([self.Rmajor[indt]], [self.Bt[indt]], 50, color)
        ix = np.argmin(np.abs(self.Rmaj[indt, :] - self.Rmag[indt]))
        ax.scatter([self.Rmag[indt]], [self.Bt_plasma[indt, ix]], 50, "r")

        if plotComplete:
            ax.axvline(x=self.Rmajor[indt], color="k", lw=1, ls="--")
            ax.axvline(x=self.Rmag[indt], color="g", lw=1, ls="--")
            ax.axvline(x=self.Rmajor[indt] + self.a[indt], color="k", lw=1, ls="--")
            ax.axvline(x=self.Rmajor[indt] - self.a[indt], color="k", lw=1, ls="--")
        if not notAxis:
            ax.set_xlabel("R (m)")
        # else:			ax.get_xaxis().set_ticks([])

        ax.set_ylabel("B (T)")

        ax.legend(loc="best", prop={"size": self.mainLegendSize})

    def plotLH(self, fig=None, time=None, plotTrajectory=True):
        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 2, hspace=0.2, wspace=0.2)

        # _________________________________

        # Total Powers
        ax = fig.add_subplot(grid[0, 0])
        ax.plot(self.t, self.P_thr_Martin1, lw=1, c="g", ls="--")
        ax.plot(self.t, self.P_thr_Martin2, lw=1, c="g", ls="--")
        ax.fill_between(
            self.t,
            self.P_thr_Martin1,
            self.P_thr_Martin2,
            facecolor="g",
            alpha=0.3,
            label="Martin08",
        )

        # ax.plot(self.t,self.P_thr_Martin1_low,lw=0.5,ls='-.',c='r',label='Martin08 - $1\\sigma$')
        # ax.plot(self.t,self.P_thr_Martin1_up,lw=0.5,ls='-.',c='r')

        # Isotope-corrected

        ax.plot(self.t, self.P_thr_Martin1 * self.isotopeMassEffect, lw=1, c="r")
        ax.plot(self.t, self.P_thr_Martin2 * self.isotopeMassEffect, lw=1, c="r")
        ax.fill_between(
            self.t,
            self.P_thr_Martin1 * self.isotopeMassEffect,
            self.P_thr_Martin2 * self.isotopeMassEffect,
            facecolor="r",
            alpha=0.2,
            label="Martin08 (w/ isotope)",
        )

        # ax.plot(self.t,self.P_thr_Martin1_low*self.isotopeMassEffect,lw=0.5,ls='-.',c='orange')
        # ax.plot(self.t,self.P_thr_Martin1_up*self.isotopeMassEffect,lw=0.5,ls='-.',c='orange')

        # ax.plot(self.t,self.P_thr_SchmiTot1,lw=1,c='b')
        # ax.plot(self.t,self.P_thr_SchmiTot2,lw=1,c='b')
        # ax.fill_between(self.t,self.P_thr_SchmiTot1,self.P_thr_SchmiTot2, facecolor='b', alpha=.3,label='2x $Q_i$ Schm18')

        # ax.plot(self.t,self.P_thr_tr,lw=1,label='$P$, TRANSP',c='y',ls='-')

        ax.plot(self.t, self.Ptot, lw=2, label="$P$ (MW)", c="g")
        ax.plot(self.t, self.Ptot - self.PradT, lw=2, label="$P-P_{rad}$ (MW)", c="m")

        ax.set_ylim([0, 1.2 * np.max([np.max(self.P_thr_Martin1), np.max(self.Ptot)])])
        ax.legend(loc="lower right")
        ax.set_ylabel("Power (MW)")

        ax.set_xlabel("Time (s)")

        GRAPHICStools.addDenseAxis(ax)

        # Only Qi
        ax = fig.add_subplot(grid[1, 0], sharex=ax)
        ax.plot(self.t, self.Pi_thr_Schmi1, lw=1, c="b")
        ax.plot(self.t, self.Pi_thr_Schmi2, lw=1, c="b")
        ax.fill_between(
            self.t,
            self.Pi_thr_Schmi1,
            self.Pi_thr_Schmi2,
            facecolor="b",
            alpha=0.3,
            label="Schm18",
        )
        ax.plot(self.t, self.Pi_LCFS, lw=2, label="$Q_i$ (MW)", c="c")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (MW)")

        ax.set_ylim(
            [0, 1.2 * np.max([np.max(self.Pi_thr_Schmi1), np.max(self.Pi_LCFS)])]
        )
        ax.legend(loc="lower right")

        GRAPHICStools.addDenseAxis(ax)

        # Only Qi
        ax1 = fig.add_subplot(grid[0, 1])
        ax2 = fig.add_subplot(grid[1, 1], sharex=ax1)
        axs = [ax1, ax2]

        # Plot u-shape
        self.plotDensityLH(axs=axs, time=time, plotTrajectory=plotTrajectory)

        GRAPHICStools.addDenseAxis(ax1)
        GRAPHICStools.addDenseAxis(ax2)

    def plotDensityLH(self, axs=None, time=None, plotTrajectory=True):
        if axs is None:
            fig, axs = plt.subplots(ncols=2)
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        isotopeMeaningful = np.abs(self.isotopeMassEffect - 1.0) > 0.02

        nmin = self.nLH_min[it]

        ax = axs[0]

        n = np.linspace(
            np.min([nmin * 0.5, self.ne_avol[it]]),
            np.max([nmin * 3, self.ne_avol[it]]),
            100,
        )
        PlhS1 = []
        PlhS2 = []
        PlhM1 = []
        PlhM2 = []
        PlhM1_low = []
        PlhM1_up = []

        for i in n:
            PlhS1.append(
                LH_Schmid1(
                    i, self.Bt[it], self.surface[it], nmin=[nmin * 0.0 + self.eps00]
                )
            )
            PlhS2.append(
                LH_Schmid2(
                    i, self.Bt[it], self.surface[it], nmin=[nmin * 0.0 + self.eps00]
                )
            )
            PlhM1.append(LH_Martin1(i, self.Bt[it], self.surface[it], nmin=[nmin]))
            PlhM2.append(
                LH_Martin2(i, self.Bt[it], self.a[it], self.Rmajor[it], nmin=[nmin])
            )
            PlhM1_low.append(
                LH_Martin1_low(i, self.Bt[it], self.surface[it], nmin=[nmin])
            )
            PlhM1_up.append(
                LH_Martin1_up(i, self.Bt[it], self.surface[it], nmin=[nmin])
            )

        PlhS1 = np.array(PlhS1)
        PlhS2 = np.array(PlhS2)
        PlhM1 = np.array(PlhM1)
        PlhM2 = np.array(PlhM2)
        PlhM1_low = np.array(PlhM1_low)
        PlhM1_up = np.array(PlhM1_up)

        if isotopeMeaningful:
            c = "g"
            ls = "--"
        else:
            c = "r"
            ls = "-"

        ax.plot(n, PlhM1, lw=1, c=c, ls=ls)
        ax.plot(n, PlhM2, lw=1, c=c, ls=ls)
        ax.fill_between(n, PlhM1, PlhM2, facecolor=c, alpha=0.3, label="Martin08")
        # ax.plot(n,PlhM1_low,lw=1,c='orange',ls='-.',label='Martin08 +- $1\\sigma$')
        # ax.plot(n,PlhM1_up,lw=1,c='orange',ls='-.')

        if isotopeMeaningful:
            ax.plot(n, PlhM1 * self.isotopeMassEffect, lw=1, c="r")
            ax.plot(n, PlhM2 * self.isotopeMassEffect, lw=1, c="r")
            ax.fill_between(
                n,
                PlhM1 * self.isotopeMassEffect,
                PlhM2 * self.isotopeMassEffect,
                facecolor="r",
                alpha=0.2,
                label="Martin08 (w/ isotope)",
            )

            # ax.plot(n,PlhM1_low*self.isotopeMassEffect,lw=1,c='r',ls='-.')
            # ax.plot(n,PlhM1_up*self.isotopeMassEffect,lw=1,c='r',ls='-.')

        ax.scatter([self.ne_avol[it]], [self.Ptot[it]], s=150, c="g", label="$P_{tot}$")
        if plotTrajectory:
            ax.plot(self.ne_avol, self.Ptot, lw=1, c="g")

        ax.scatter(
            [self.ne_avol[it]],
            [self.Ptot[it] - self.PradT[it]],
            marker="*",
            s=200,
            c="m",
            label="$P_{tot}-P_{rad}$",
        )
        if plotTrajectory:
            ax.plot(self.ne_avol, self.Ptot - self.PradT, lw=1, c="m")

        ax.set_ylabel("Power (MW)")

        ax.set_ylim(bottom=0)
        ax.legend(loc="lower right")
        # ax.set_title('Low and High Density Branches',size=20)
        ax.set_xlabel("$<n_e>$ ($10^{20}m^{-3}$)")

        ax = axs[1]

        ax.plot(n, PlhS1, lw=1, c="b")
        ax.plot(n, PlhS2, lw=1, c="b")
        ax.fill_between(n, PlhS1, PlhS2, facecolor="b", alpha=0.3, label="Schm18")

        ax.scatter(
            [self.ne_avol[it]], [self.Pi_LCFS[it]], s=200, c="c", label="$Q_{i}^{LCFS}$"
        )
        if plotTrajectory:
            ax.plot(self.ne_avol, self.Pi_LCFS, lw=1, c="c")

        ax.set_xlabel("$<n_e>$ ($10^{20}m^{-3}$)")
        ax.set_ylabel("$Q_i$ (MW)")

        ax.set_ylim([0, np.max([self.Pi_LCFS[it], PlhS2[-1]])])
        ax.legend(loc="lower right")

    def plotFBM(self, particleFile=None, fig=None, finalProfile=None):
        if particleFile is None:
            particleFile = self.fbm_He4_gc

        # I think TIME in the FBM corresponds to the namelist OUTIM-AVGTIM
        avtime = particleFile.avtime
        time = particleFile.time

        it1 = np.argmin(np.abs(self.t - (particleFile.time + avtime)))
        it2 = np.argmin(np.abs(self.t - (particleFile.time + 0.0)))

        R = [self.Rmag[it1], self.Rmag[it1] + self.a[it1] / 2.0]
        Z = [0, 0]

        rhos_plotted, ax1, ax2, ax3, ax4 = particleFile.plotComplete(R=R, Z=Z, fig=fig)

        self.plotGeometry(
            ax=ax1,
            Aspect=False,
            rhoS=particleFile.rhoUsurf,
            colorsurfs="w",
            color="r",
            alphasurfs=0.25,
        )

        if finalProfile is not None:
            ax3.fill_between(
                self.x_lw,
                finalProfile[it1],
                finalProfile[it2],
                color="b",
                alpha=0.5,
                label=f"CDF, t={self.t[it2]:.3f}-{self.t[it1]:.3f}s",
            )
        ax3.legend()

        for rho, c in zip(rhos_plotted, ["r", "m", "b", "c"]):
            ix = np.argmin(np.abs(self.x_lw - rho))
            ax4.plot(
                self.t, finalProfile[:, ix], color=c, label=f"CDF, $\\rho_N$={rho:.2f}"
            )
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Density ($10^{20} m^{-3}$)")
        GRAPHICStools.addDenseAxis(ax4)
        ax4.set_title("Extraction of FBM")
        GRAPHICStools.addLegendApart(ax4, ratio=0.8)
        ax4.axvspan(
            self.t[it1],
            self.t[it2],
            facecolor="k",
            alpha=0.2,
            edgecolor="none",
        )

    def plotBirth(self, particleFile=None, fig=None):
        if particleFile is None:
            particleFile = self.fbm_D_gc

        if fig is None:
            fig = plt.figure(figsize=(12, 8))

        grid = plt.GridSpec(1, 2, hspace=0.2, wspace=0.4)

        ax1 = fig.add_subplot(grid[0])
        ax2 = fig.add_subplot(grid[1])

        self.plotGeometry_Above(ax=ax1, color="b")
        self.plotNBItrajectories(ax=ax1, topDown=True)
        particleFile.birth.plotTopDown(ax=ax1, color="r")

        self.plotGeometry(ax=ax2, color="b")
        self.plotNBItrajectories(ax=ax2, topDown=False, leg=True)
        particleFile.birth.plotPol(ax=ax2, color="r")

    def plotAllGeos(self, ax, time):
        levels = np.arange(0, 1.0, 0.1)
        levels = np.append(levels, 1 - 1e-6)
        levels = np.append(levels, np.arange(1.0, 1.3, 0.1))
        self.plotGeometry(
            ax=ax,
            time=time,
            colorsurfs="b",
            color="b",
            rhoS=levels,
            rhoPol=True,
            sqrt=False,
            label="TRANSP",
            Aspect=False,
        )
        self.isolver.plotSurfaces(
            ax,
            time,
            levels=levels,
            colorBoundary="r",
            colorSurfs="r",
            lw=[0.5, 0.5],
            label="ISOLVER",
        )

        # # Poloidal
        if self.gfile_in is not None:
            self.gfile_in.plotFluxSurfaces(
                ax=ax, fluxes=levels, color="g", alpha=0.5, rhoPol=True, sqrt=False
            )
            gfile_txt = ", GFILE (green)"
        else:
            gfile_txt = ""

        if hasattr(self, "isolver") and self.isolver.inputs is not None:
            self.isolver.inputs.plotCoils(ax, names=False, lw=0.5, c="k")

        if hasattr(self, "bound_R") and self.bound_R is not None:
            bound_txt = ", RZFS (black)"
        else:
            bound_txt = ""

        suggested_title = "TRANSP (blue), ISOLVER (red)" + gfile_txt + bound_txt

        return suggested_title

    def plotISOLVER(self, fn=None, time=None):
        if fn is None:
            self.fnIsolver = FigureNotebook(f"ISOLVER Notebook, run #{self.nameRunid}")
        else:
            self.fnIsolver = fn

        fig1 = self.fnIsolver.add_figure(label="ISOLVER_eq")
        fig2 = self.fnIsolver.add_figure(label="ISOLVER_coils 1")
        fig2_e = self.fnIsolver.add_figure(label="ISOLVER_coils 2")
        fig3 = self.fnIsolver.add_figure(label="ISOLVER_sum")
        fig4 = self.fnIsolver.add_figure(label="ISOLVER_lcfs")

        if time is None:
            time = self.t[self.ind_saw]

        it = np.argmin(np.abs(self.t - time))

        # ----
        grid = plt.GridSpec(nrows=2, ncols=3, hspace=0.3, wspace=0.6)
        ax1 = fig4.add_subplot(grid[:, 0])
        ax2 = fig4.add_subplot(grid[0, 1])
        ax3 = fig4.add_subplot(grid[1, 1])
        ax4 = fig4.add_subplot(grid[0, 2])
        ax5 = fig4.add_subplot(grid[1, 2])

        ax = ax1
        suggested_title = self.plotAllGeos(ax, time)
        ax.set_aspect("equal")
        ax.set_title(suggested_title)

        diff = 0.1
        mm0 = 0

        ax = ax2
        _ = self.plotAllGeos(ax, time)
        mm = self.Rmajor[it] - self.a[it]
        ax.set_xlim([mm - diff, mm + diff])
        ax.set_ylim([mm0 - diff, mm0 + diff])
        ax.set_title("Zoom inner midplane")
        # xpl = 0.005
        # GRAPHICStools.gradientSPAN(ax,mm,x2,color='b',startingalpha=1.0,endingalpha=0.0,orientation='vertical')

        ax = ax4
        _ = self.plotAllGeos(ax, time)
        mm = self.Rmajor[it] + self.a[it]
        ax.set_xlim([mm - diff, mm + diff])
        ax.set_ylim([mm0 - diff, mm0 + diff])
        ax.set_title("Zoom outer midplane")

        diff = 0.15

        ax = ax3
        _ = self.plotAllGeos(ax, time)
        mm = self.isolver.xpoints[0][0, it]
        mm0 = self.isolver.xpoints[0][1, it]
        ax.set_xlim([mm - diff, mm + diff])
        ax.set_ylim([mm0 - diff, mm0 + diff])
        ax.set_title("Zoom x-point 1")

        ax = ax5
        _ = self.plotAllGeos(ax, time)
        mm = self.isolver.xpoints[1][0, it]
        mm0 = self.isolver.xpoints[1][1, it]
        ax.set_xlim([mm - diff, mm + diff])
        ax.set_ylim([mm0 - diff, mm0 + diff])
        ax.set_title("Zoom x-point 2")

        # ------
        self.isolver.plotCoils(fig=fig2, MAturns=True)
        self.isolver.plotCoils(fig=fig2_e, MAturns=False)
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = self.isolver.plotSummary(
            time, fig=fig3, V=self.V
        )

        ax = ax5
        ax.plot(self.t, self.psi_bnd, c="r", lw=1, ls="-", label="TRANSP")
        ax.legend()

        ax = ax8
        ax.plot(
            self.t, self.V0[:, 0], c="r", lw=1, ls="-", label="$2\\pi R_{m}E_{\\phi,0}$"
        )
        ax.legend()

        self.isolver.plotEquilibria(time, fig=fig1)

    def plotTGLF(self, figGR=None, figFL=None):
        ax1, ax2, ax3 = self.TGLF.plotComplete_GR(fig=figGR)
        ax1, ax2, ax3 = self.TGLF.plotComplete_FL(fig=figFL)

    def plotTORIC(self, fig=None, position=0):
        ax1, ax2, ax2e, ax3, ax4 = self.torics[position].plotComplete(fig=fig)

        self.plotGeometry(ax=ax1, Aspect=False, colorsurfs="k", color="k")
        self.plotGeometry(ax=ax2, Aspect=False, colorsurfs="w", color="w")
        self.plotGeometry(ax=ax2e, Aspect=False, colorsurfs="w", color="w")

        ax3.axvline(x=np.min(self.Rmaj[self.ind_saw]), c="b", lw=2)
        ax3.axvline(x=np.max(self.Rmaj[self.ind_saw]), c="b", lw=2)
        ax4.axvline(x=np.min(self.Rmaj[self.ind_saw]), c="b", lw=2)
        ax4.axvline(x=np.max(self.Rmaj[self.ind_saw]), c="b", lw=2)
        ax3.set_xlim(
            [
                np.min(self.Rmaj[self.ind_saw]) * 0.9,
                np.max(self.Rmaj[self.ind_saw]) * 1.1,
            ]
        )
        ax4.set_xlim(
            [
                np.min(self.Rmaj[self.ind_saw]) * 0.9,
                np.max(self.Rmaj[self.ind_saw]) * 1.1,
            ]
        )
        # ax4.axvline(x=self.Rmag[self.ind_saw],c='g',lw=2,ls='--')
        if self.R_ant is not None:
            ax4.axvline(x=self.R_ant.max(), c="g", lw=2)

        # for i in range(len(self.FichT_ant)):	self.plotRelevantResonances(ax1,self.FichT_ant[i],legendYN=True,lw=2)

    def plotTimeScales(self, fig=None, time=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 3, hspace=0.2, wspace=0.4)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[1, 0])
        ax4 = fig.add_subplot(grid[1, 1])

        ax5 = fig.add_subplot(grid[0, 2])
        ax6 = fig.add_subplot(grid[1,2])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Variation of electromagnetic quantities
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.ddt_t, self.dVdt = MATHtools.characteristicTime(
            self.t, self.xb_lw, self.V, self.tlastsaw
        )
        _, self.dqdt = MATHtools.characteristicTime(
            self.t, self.xb_lw, self.q, self.tlastsaw
        )
        _, self.djdt = MATHtools.characteristicTime(
            self.t, self.xb_lw, self.j, self.tlastsaw
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Variation of kinetic quantities
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        _, self.dTedt = MATHtools.characteristicTime(
            self.t, self.xb_lw, self.Te, self.tlastsaw
        )
        _, self.dTidt = MATHtools.characteristicTime(
            self.t, self.xb_lw, self.Ti, self.tlastsaw
        )
        _, self.dnedt = MATHtools.characteristicTime(
            self.t, self.xb_lw, self.ne, self.tlastsaw
        )
        _, self.dpdt = MATHtools.characteristicTime(
            self.t, self.xb_lw, self.p, self.tlastsaw
        )

        # Electromagnetic quantities
        ax = ax1  #
        ax.plot(self.ddt_t, self.dqdt, lw=2, label="max($\\Delta q$)")
        ax.plot(self.ddt_t, self.djdt, lw=2, label="max($\\Delta j$)")
        ax.plot(self.ddt_t, self.dVdt, lw=2, label="max($\\Delta V$)")
        ax.plot(self.ddt_t, self.dTedt, lw=2, label="max($\\Delta T_e$)")
        ax.plot(self.ddt_t, self.dTidt, lw=2, label="max($\\Delta T_i$)")
        ax.plot(self.ddt_t, self.dnedt, lw=2, label="max($\\Delta n_e$)")
        ax.plot(self.ddt_t, self.dpdt, lw=2, label="max($\\Delta p$)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Percent Variation (%)")
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})
        ax.set_title("Sawtooth-smoothed variation")
        ax.set_ylim([0, 20])

        ax.axhline(y=5.0, ls="--", lw=2, c="k")

        GRAPHICStools.addDenseAxis(ax)

        ax = ax2
        rhos = [0.2, 0.4, 0.6, 0.8, 1.0]
        ax.plot(self.t, self.V[:, 0] / self.V[0, 0], lw=1, label="$V$", c="r")
        for i in rhos:
            ax.plot(
                self.t,
                self.V[:, np.argmin(np.abs(self.xb_lw - i))]
                / self.V[0, np.argmin(np.abs(self.xb_lw - i))],
                lw=1,
            )  # ,label='$V$@$\\rho_N={}$'.format(i))
        ax.plot(self.t, self.q[:, 0], lw=1, label="$q$", c="r", ls="--")
        for i in rhos:
            ax.plot(
                self.t,
                self.q[:, np.argmin(np.abs(self.xb_lw - i))]
                / self.q[0, np.argmin(np.abs(self.xb_lw - i))],
                lw=1,
                ls="--",
            )  # ,label='$V$@$\\rho_N={}$'.format(i))
        ax.plot(self.t, self.j[:, 0], lw=1, label="$J$", c="r", ls="-.")
        for i in rhos:
            ax.plot(
                self.t,
                self.j[:, np.argmin(np.abs(self.xb_lw - i))]
                / self.j[0, np.argmin(np.abs(self.xb_lw - i))],
                lw=1,
                ls="-.",
            )  # ,label='$V$@$\\rho_N={}$'.format(i))

        ax.set_xlabel("Time(s)")
        ax.set_ylabel("Normalized to original values")
        ax.set_title("Quantities at 0,0.2,0.4,0.6,0.8,1.0")
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)

        # times
        ax = ax3
        ax.plot(self.t, self.taue * 1e-3, lw=2, label="$\\tau_{e}$")
        ax.plot(self.t, self.taup_He4 * 1e-3, lw=2, label="$\\tau_{p,He4}$ (/4)")
        ax.plot(self.t, self.tau_c * 1e-3 * 1e-1, lw=2, label="$\\tau_{curr}$ (/10)")
        ax.plot(self.t, self.tau_saw * 1e-3, lw=2, label="$\\tau_{saw}$")
        ax.plot(
            self.t,
            self.tauSD_He4_avol * 1e-3,
            lw=2,
            label="$\\langle\\tau_{SD,He4}\\rangle$",
        )
        ax.plot(
            self.t,
            self.tauSD_He4_Stix_avol * 1e-3,
            lw=2,
            label="$\\langle\\tau_{SD,He4}\\rangle$ Stix",
        )
        # ax7.plot(self.tau_saw_t,self.tau_q95*1E-3*1E-3,lw=3,label='$\\tau_{q95}$ (/1000)',marker='o')

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Time (s)")
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

        # tq profiles
        ax = ax4
        _ = GRAPHICStools.plotRange(
            self.t, self.xb, self.q, ax=ax, it1=0, it2=self.ind_saw, howmany=100
        )

        ax.axvline(
            x=self.x_saw_inv[it], alpha=0.5, lw=1, ls="-.", c="g", label="$r_{q=1}$"
        )
        ax.axvline(
            x=self.x_saw_mix[it], alpha=0.5, lw=1, ls="-.", c="y", label="$r_{mix}$"
        )
        ax.axvline(
            x=self.x_saw_inv_prof[it],
            alpha=0.5,
            lw=1,
            ls="-.",
            c="m",
            label="$r_{prof,inv}$",
        )

        ax.axhline(y=1.0, c="k", ls="--", lw=1)
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylim(bottom=0)
        ax.set_xlim([0, 1])
        ax.set_ylabel("q-profile")

        GRAPHICStools.addDenseAxis(ax)

        # _
        ax = ax5
        ax.plot(self.x_lw, self.tauSD_He4[it] * 1e-3, lw=2, label="$\\tau_{SD,He4}$")
        ax.plot(
            self.x_lw,
            self.tauSD_He4_Stix[it] * 1e-3,
            lw=2,
            label="$\\tau_{SD,He4}$ Stix",
        )

        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylim(bottom=0)
        ax.set_xlim([0, 1])
        ax.set_ylabel("Time (s)")

        GRAPHICStools.addDenseAxis(ax)

        ax = ax6
        rhos = [0.2, 0.4, 0.6, 0.8, 0.95]
        for i in rhos:
            ix = np.argmin(np.abs(self.x_lw - i))
            ax.plot(self.t, self.q[:, ix], lw=1, label=f"$\\rho_N={i:.2f}$")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("q")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.axhline(y=1.0, c="k", ls="--", lw=1)
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)


    def plotElectricField(self, fig=None, time=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.5)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(grid[0, 1], sharex=ax1)
        ax4 = fig.add_subplot(grid[1, 1])

        # _________________________________

        # Pot
        ax = ax1
        ax.plot(self.xb_lw, self.Epot[it, :], lw=3, c="r", label="V_r")
        ax.plot(self.xb_lw, self.Epot_rot[it, :], lw=2, c="b", label="V_{r,rot}")
        ax.plot(self.xb_lw, self.Epot_nc[it, :], lw=2, c="g", label="V_{r,nc}")
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("Electric potential $\\Phi$ (kV)")
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)

        ax.set_xlim([0, 1])

        # Er
        ax = ax4
        ax.plot(self.Rmaj[it], self.Er[it, :] * 1e-3, lw=2, c="r", label="$E_r$")
        ax.plot(
            self.Rmaj[it], self.Er_tor[it, :] * 1e-3, lw=1, c="b", label="$E_r$ (tor)"
        )
        ax.plot(
            self.Rmaj[it], self.Er_pol[it, :] * 1e-3, lw=1, c="g", label="$E_r$ (pol)"
        )
        ax.plot(
            self.Rmaj[it], self.Er_p[it, :] * 1e-3, lw=1, c="m", label="$E_r$ (diam)"
        )
        ax.plot(
            self.Rmaj[it],
            (self.Er_p[it, :] + self.Er_pol[it, :] + self.Er_tor[it, :]) * 1e-3,
            lw=1,
            ls="--",
            c="y",
            label="check",
        )
        ax.set_xlabel("R (m)")
        ax.set_ylabel("$E_r$ (kV/m)")
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})
        ax.set_title("Neoclassical Er")

        ax.axvline(x=self.Rmajor[it], ls="--", lw=2, c="k")
        ax.axvline(x=self.Rmag[it], ls="--", lw=2, c="g")

        GRAPHICStools.addDenseAxis(ax)

        # Er
        ax = ax2  #
        ax.plot(self.x_lw, self.Er_LF[it, :] * 1e-3, lw=2, c="r", label="$E_r$, LF")
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$E_r$ (kV/m)")
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)

        ax = ax3  # .twinx()
        ax.plot(self.x_lw, self.GammaExB[it, :], lw=2, c="m", label="$\\gamma_{ExB}$")
        ax.set_ylabel("$\\gamma$ (kHz)")
        ax.legend(loc="upper center")
        ax.set_xlabel("$\\rho_N$")

        GRAPHICStools.addDenseAxis(ax)

    def plotEM(self, fig=None, time=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 4, hspace=0.3, wspace=0.5)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0])
        ax3 = fig.add_subplot(grid[0, 1])  # ,sharex=ax2)
        ax4 = fig.add_subplot(grid[1, 1])  # ,sharex=ax2)
        ax5 = fig.add_subplot(grid[0, 2])  # ,sharex=ax2)
        ax6 = fig.add_subplot(grid[1, 2])  # ,sharex=ax2)
        ax7 = fig.add_subplot(grid[0, 3])  # ,sharex=ax2)
        ax8 = fig.add_subplot(grid[1, 3])  # ,sharex=ax2)

        # _________________________________

        # q profile radial
        ax = ax1
        i1, i2 = prepareTimingsSaw(time, self)
        ax.plot(
            self.xb_lw,
            self.q[i1, :],
            ls="-",
            c="b",
            lw=2,
            label=f"q, @ {self.t[i1]:.3f}s",
        )
        ax.plot(
            self.xb_lw,
            self.q[i2, :],
            ls="--",
            c="b",
            lw=2,
            label=f"q, @ {self.t[i2]:.3f}s",
        )

        ax.axhline(y=1.0, ls="--", c="k", lw=1)

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("q")
        ax.set_ylim(bottom=0)
        ax.set_xlim([0, 1])
        # ax.set_title('SOLID {0:.3f}s, DASHED {1:.3f}s'.format(self.t[i1],self.t[i2]))

        GRAPHICStools.addDenseAxis(ax)

        # J profile radial
        ax = ax3
        i1, i2 = prepareTimingsSaw(time, self)
        ax.plot(
            self.x_lw,
            self.j[i1, :],
            ls="-",
            lw=2,
            label="J $\\equiv \\langle J_\\phi\\rangle_A$",
            c="b",
        )
        ax.plot(self.x_lw, self.j[i2, :], ls="--", lw=2, c="b")
        ax.plot(self.x_lw, self.jB[i1, :], ls="-", lw=2, label="$J_B$", c="r")
        if self.jNBI[i1, :].max() > 1e-10:
            ax.plot(
                self.x_lw, self.jNBI[i1, :], ls="-", lw=2, label="$J_{{NBI}}$", c="m"
            )
        if self.jECH[i1, :].max() > 1e-10:
            ax.plot(
                self.x_lw, self.jECH[i1, :], ls="-", lw=2, label="$J_{{ECH}}$", c="c"
            )
        ax.plot(self.x_lw, self.jOh[i1, :], ls="-", lw=2, label="$J_{{OH}}$", c="k")
        if self.isolver is not None and self.isolver.jAnom is not None:
            ax.plot(
                self.x_lw,
                self.isolver.jAnom[i1, :],
                ls="-",
                lw=2,
                label="$J_{{IS,Anom}}$",
                c="g",
            )
            z = self.isolver.jAnom[i1, :]
        else:
            z = self.j[i1, :] * 0.0
        ax.plot(
            self.x_lw,
            self.jOh[i1, :] + self.jB[i1, :] + self.jNBI[i1, :] + self.jECH[i1, :] + z,
            ls="--",
            c="y",
            lw=1,
            label="sum",
        )

        ax.legend(loc="best", fontsize=7)
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("J ($MA/m^2$)")
        ax.set_title("XS-Average Currents")

        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        # Loop voltage
        ax = ax2  #
        ax.plot(self.xb_lw, self.V[it - 1], lw=3, c="r", label="$V$")
        ax.plot(
            self.xb_lw,
            self.V_check[it - 1],
            "--",
            lw=2,
            c="y",
            label="$V = 2\\pi \\frac{\\langle\\vec{E}\\cdot\\vec{B}\\rangle}{(RB_{\\phi})\\langle\\frac{1}{R^2}\\rangle}$",
        )
        ax.plot(
            self.xb_lw,
            self.V_aux[it - 1],
            "-.",
            lw=2,
            c="c",
            label="$V = \\frac{\\partial\\psi}{\\partial t}=\\frac{\\partial\\psi_{encl.}}{\\partial t} + 2\\pi R_{m}E_{\\phi,0}$",
        )
        ax.plot(
            self.xb_lw,
            self.V_aux_psi[it - 1],
            lw=0.5,
            ls="-.",
            c="b",
            label="$V = \\frac{\\partial\\psi_{encl.}}{\\partial t}$",
        )
        ax.plot(
            [1.0], [self.Vsurf[it - 1]], "s", c="m", label="$V_{surf}$", markersize=10
        )
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$V$ (V)")
        ax.legend(loc="best", fontsize=7)
        ax.set_ylim([0, np.max(self.V[it - 1]) * 1.5])
        ax.set_title("Loop Voltage Profile")

        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlim([0, 1])

        ax = ax4
        ax.plot(
            self.xb_lw,
            self.Etor[it],
            lw=3,
            c="b",
            label="$\\langle E_{\\phi} \\rangle = V \\frac{1}{2\\pi}\\langle \\frac{1}{R} \\rangle $",
        )
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$E$ (V/m)")
        # ax.legend(loc='lower right')
        ax.legend(loc="best", fontsize=7)
        ax.set_title("Toroidal electric field")
        ax.set_ylim([self.Etor[it].min() - 0.1, self.Etor[it].max() + 0.1])

        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        # Ohmic Power
        ax = ax6  #
        ax.plot(self.x_lw, self.Poh[it, :], lw=3, c="r", label="$P_{OH}$")
        ax.plot(
            self.x_lw,
            self.Poh_checkJ[it, :],
            lw=1,
            c="b",
            label="$\\eta \\langle J_{\\phi}\\rangle_{OH}\\cdot \\langle J_{\\phi}\\rangle$",
        )
        ax.plot(
            self.x_lw,
            self.Poh_checkJ_NCLASS[it, :],
            lw=0.5,
            c="m",
            label="$\\eta_{NCLASS} \\langle J_{\\phi}\\rangle_{OH}\\cdot \\langle J_{\\phi}\\rangle$",
        )
        ax.plot(
            self.x_lw,
            self.Poh_checkJ_Sauter[it, :],
            lw=0.5,
            c="c",
            label="$\\eta_{Sau} \\langle J_{\\phi}\\rangle_{OH}\\cdot \\langle J_{\\phi}\\rangle$",
        )
        ax.plot(
            self.x_lw,
            self.Poh_checkJ_Sauter2[it, :],
            lw=0.5,
            c="y",
            label="$\\eta_{Sau} \\langle J_{\\phi}\\rangle\\cdot \\langle J_{\\phi}\\rangle$",
        )

        ax.plot(
            self.x_lw,
            self.Poh_checkEJ[it, :],
            lw=2,
            ls="--",
            c="g",
            label="$\\langle E_{\\phi}\\rangle\\cdot \\langle J_{\\phi}\\rangle$",
        )
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("Power ($MW/m^3$)")
        ax.legend(loc="best", fontsize=7)
        ax.set_title("Ohmic power")
        ax.set_ylim([0, np.max(self.Poh[it]) * 1.2])

        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlim([0, 1])

        # -------------------------------------------

        # Resistivity
        ax = ax7  #
        ax.plot(self.x_lw, self.eta[it, :], lw=3, c="r", label="$\\eta$")
        ax.axhline(
            y=self.eta_avol[it],
            ls="-",
            alpha=0.3,
            lw=3,
            c="r",
            label="$\\langle\\eta\\rangle_v$",
        )
        ax.plot(
            self.x_lw, self.etas_sp[it, :], lw=1, c="b", label="$\\eta_{Spitzer,cl}$"
        )
        ax.plot(self.x_lw, self.etas_wnc[it, :], lw=1, c="m", label="$\\eta_{NCLASS}$")
        ax.plot(self.x_lw, self.etas_tsc[it, :], lw=1, c="g", label="$\\eta_{TSC}$")
        ax.plot(self.x_lw, self.etas_snc[it, :], lw=1, c="y", label="$\\eta_{Sauter}$")

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$\\eta$ (Ohm*m)")
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})
        ax.set_title("Resistivity")

        # ax.set_ylim([0,self.eta_avol[it]*2.0])
        ax.set_yscale("log")

        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlim([0, 1])

        #
        ax = ax5
        ax.plot(
            self.x_lw,
            self.Umag_pol[i1, :],
            ls="-",
            lw=2,
            c="r",
            label="$U_{B_{\\theta}}$",
        )
        ax.plot(
            self.x_lw,
            self.Umag_pol_check[i1, :],
            ls="--",
            c="y",
            lw=1,
            label="$U_{B_{\\theta}}$ check",
        )

        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("U ($MJ/m^3$)")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

        #
        ax = ax8
        ax.plot(
            self.x_lw,
            self.JdotB[i1, :],
            ls="-",
            c="b",
            lw=2,
            label="$\\langle J\\cdot B\\rangle$",
        )
        ax.plot(
            self.x_lw,
            self.JdotB_b_used[i1, :],
            ls="-",
            c="r",
            lw=2,
            label="$\\langle J_B\\cdot B\\rangle$",
        )
        ax.plot(
            self.x_lw,
            self.JdotB_b_raw[i1, :],
            ls="--",
            c="r",
            lw=1,
            label="$\\langle J_B\\cdot B\\rangle$ (raw)",
        )
        ax.plot(
            self.x_lw,
            self.JdotB_oh[i1, :],
            c="k",
            ls="-",
            lw=2,
            label="$\\langle J_{OH}\\cdot B\\rangle$",
        )
        if self.isolver is not None and self.isolver.JdotB_Anom is not None:
            ax.plot(
                self.x_lw,
                self.isolver.JdotB_Anom[i1, :],
                c="g",
                ls="-",
                lw=2,
                label="$\\langle J\\cdot B\\rangle_{IS, Anom}$",
            )

        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$\\langle J\\cdot B\\rangle$ ($MA\\cdot T/m^2$)")
        ax.set_xlim([0, 1])

        ax.axhline(y=0.0, ls="--", c="k", lw=0.5)

        GRAPHICStools.addDenseAxis(ax)
        ax.set_xlim([0, 1])

    def plotTimeAverages(self, fig=None, times=None):
        if times is None:
            it1 = np.argmin(np.abs(self.t - (self.t[self.ind_saw] - 2.5)))
            it2 = np.argmin(np.abs(self.t - (self.t[self.ind_saw] - 0.1)))
        else:
            it1 = np.argmin(np.abs(self.t - times[0]))
            it2 = np.argmin(np.abs(self.t - times[1]))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0])
        ax3 = fig.add_subplot(grid[0, 1])
        # ax4 = fig.add_subplot(grid[1,1])

        # --------------------------------------------------------

        ax = ax1

        z = timeAverage(self.t[it1 : it2 + 1], self.Poh[it1 : it2 + 1])
        ax.plot(self.x_lw, z, lw=2, c="b", label="$P_{OH}$")
        l1 = f"{timeAverage(self.t[it1:it2 + 1], self.PohT[it1:it2 + 1]):.2f}MW"
        z = timeAverage(self.t[it1 : it2 + 1], self.Poh_corr[it1 : it2 + 1])
        ax.plot(self.x_lw, z, lw=2, c="r", label="$P_{OH,corr}$")
        l2 = f"{timeAverage(self.t[it1:it2 + 1], self.PohT_corr[it1:it2 + 1]):.2f}MW"

        l3 = "(V*I={0:.2f}MW)".format(
            timeAverage(self.t[it1 : it2 + 1], (self.Vsurf * self.Ip)[it1 : it2 + 1])
        )

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$MW/m^3$")
        ax.set_title(f"Av. Ohmic power ({self.t[it1]:.3f}-{self.t[it2]:.3f}s)")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})

        ax.text(
            0.85,
            0.9,
            l1,
            fontweight="bold",
            color="b",
            fontsize=12,
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.85,
            0.8,
            l2,
            fontweight="bold",
            color="r",
            fontsize=12,
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.85,
            0.7,
            l3,
            fontweight="bold",
            color="g",
            fontsize=10,
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        GRAPHICStools.addDenseAxis(ax)

        # --------------------------------------------------------

        ax = ax2

        z = timeAverage(self.t[it1 : it2 + 1], self.Te[it1 : it2 + 1])
        ax.plot(self.x_lw, z, lw=2, c="r", label="$T_e$")
        z = timeAverage(self.t[it1 : it2 + 1], self.Ti[it1 : it2 + 1])
        ax.plot(self.x_lw, z, lw=2, c="b", label="$T_i$")

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$keV$")
        ax.set_title(f"Av. Temperatures ({self.t[it1]:.3f}-{self.t[it2]:.3f}s)")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)

        ax = ax3

        z = timeAverage(self.t[it1 : it2 + 1], self.qe_obs[it1 : it2 + 1])
        ax.plot(self.x_lw, z, lw=2, c="r", label="$q_e$")
        z = timeAverage(self.t[it1 : it2 + 1], self.qi_obs[it1 : it2 + 1])
        ax.plot(self.x_lw, z, lw=2, c="b", label="$q_i$")

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$MW/m^2$")
        ax.set_title(f"Av. Fluxes ({self.t[it1]:.3f}-{self.t[it2]:.3f}s)")
        ax.set_xlim([0, 1])
        ax.axhline(y=0, ls="--", c="k")
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)

    def plotUmag(self, fig=None, time=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.5)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0])
        ax3 = fig.add_subplot(grid[0, 1])  # ,sharex=ax1)
        ax4 = fig.add_subplot(grid[1, 1])  # ,sharex=ax1)

        ax5 = fig.add_subplot(grid[0, 2])
        ax6 = fig.add_subplot(grid[1, 2])

        # _________________________________

        # Balance

        self.plotPoyntingFluxBalance_total(ax1, ax2=ax2, onlyBeforeSaw=True, time=time)

        GRAPHICStools.addDenseAxis(ax1)
        GRAPHICStools.addDenseAxis(ax2)

        ##
        ax = ax4
        ax.plot(self.t, self.psi_bnd, c="r", ls="-", lw=2, label="$\\psi_{bound}$")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Poloidal Flux (Wb)")
        ax.legend(loc="lower left")
        ax.set_ylim([0, np.max(self.psi_bnd) * 2])

        GRAPHICStools.addDenseAxis(ax)

        ax = ax4.twinx()
        ax.plot(self.t, self.phi_bnd, c="b", ls="-", lw=2, label="$\\phi_{bound}$")
        ax.legend(loc="lower right")
        ax.set_ylabel("Toroidal Flux (Wb)")
        ax.set_ylim([0, np.max(self.phi_bnd) * 2])

        # Magn
        ax = ax3
        ax.plot(
            self.t, self.Bp_ext[:, -1], c="r", ls="-", lw=2, label="$B_{\\theta,R_0+a}$"
        )
        ax.set_ylabel("Poloidal Magnetic field (T)")
        ax.legend(loc="lower left")
        ax.set_xlabel("Time (s)")
        ax.set_ylim([0, np.max(self.Bp_ext[:, -1]) * 2])

        GRAPHICStools.addDenseAxis(ax)

        ax = ax3.twinx()
        ax.plot(
            self.t, self.Bt_ext[:, -1], c="b", ls="-", lw=2, label="$B_{\\phi,R_0+a}$"
        )
        ax.set_ylabel("Toroidal Magnetic field (T)")
        ax.legend(loc="lower right")
        ax.set_ylim([0, np.max(self.Bt_ext[:, -1]) * 2])

        # Magn
        ax = ax5
        # ax.plot(self.t,self.Li_MITIM_1,lw=4,label='$l_{i}=\\langle \\langle B_{\\theta}^2\\rangle_{FSA}\\rangle_v / \\langle \\langle B_{\\theta}\\rangle_{FSA}^2\\rangle_v$')
        # ax.plot(self.t,self.Li_MITIM_2,lw=4,label='$l_{i}=\\langle \\langle B_{\\theta}^2\\rangle_{FSA} / \\langle B_{\\theta}\\rangle_{FSA}^2\\rangle_v$')
        ax.plot(self.t, self.Li1, lw=3, ls="-", label="$l_{i,1}$")
        ax.plot(
            self.t,
            self.Li1_check,
            lw=2,
            ls="--",
            label="$l_{i}=\\frac{\\langle B_{\\theta}^2\\rangle}{(\\mu_0I_p/L_\\theta)^2}$",
        )
        ax.plot(self.t, self.Li3, lw=3, ls="-", label="$l_{i,3}$")
        ax.plot(
            self.t,
            self.Li3_check,
            lw=2,
            ls="--",
            label="$l_{i}=\\frac{\\langle B_{\\theta}^2\\rangle \\cdot V}{\\frac{1}{2}(\\mu_0 I_p)^2\\frac{1}{2} (R_{min}+R_{max})}$",
        )
        ax.plot(self.t, self.LiVDIFF, lw=3, ls="-", label="$l_{i,Vdiff}$")
        ax.plot(
            self.t,
            self.LiVDIFF_check,
            lw=2,
            ls="--",
            label="$l_{i}=\\frac{\\langle B_{\\theta}^2\\rangle}{\\langle B_{\\theta}^2\\rangle_{FSA,\\xi=1}}$",
        )
        ax.set_ylabel("Internal Inductance, $l_i$")
        ax.set_xlabel("Time (s)")
        # ax.set_ylim(bottom=0)
        # ax.legend(loc='best',prop={'size':self.mainLegendSize})
        GRAPHICStools.addLegendApart(ax, ratio=0.8)

        GRAPHICStools.addDenseAxis(ax)

        # Magn
        ax = ax6
        ax.plot(self.t, self.UmagT_pol, c="r", ls="-", lw=2, label="$U_{B_{\\theta}}$")
        ax.set_ylabel("Poloidal magnetic energy (MJ)")
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})
        ax.set_xlabel("Time (s)")

        GRAPHICStools.addDenseAxis(ax)

        ax = ax6.twinx()
        ax.plot(
            self.t, self.UmagT_tor * 1e-3, c="b", ls="-", lw=2, label="$U_{B_{\\phi}}$"
        )
        ax.set_ylabel("Toroidal magnetic energy (GJ)")
        ax.legend(loc="lower right")

    def plotPoyntingFluxBalance_total(
        self, ax, ax2=None, legend=True, onlyBeforeSaw=False, time=None
    ):
        if len(self.tlastsawU) < 2 or not onlyBeforeSaw:
            alpha = 1.0
        else:
            alpha = 0.2

        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        z, t = MATHtools.smoothThroughSawtooth(
            self.t, self.ExBpower_T, self.tlastsaw, 1
        )
        if onlyBeforeSaw:
            ax.plot(self.t, self.ExBpower_T, lw=3, c="b", alpha=alpha)
            ax.plot(
                t,
                z,
                lw=4,
                c="b",
                label="$\\frac{1}{\\mu_0}\\int_S |\\vec{E}x\\vec{B}|d\\vec{S}$",
            )
        else:
            ax.plot(
                self.t,
                self.ExBpower_T,
                lw=3,
                c="b",
                alpha=alpha,
                label="$\\frac{1}{\\mu_0}\\int_S |\\vec{E}x\\vec{B}|d\\vec{S}$",
            )

        if onlyBeforeSaw:
            z, t = MATHtools.smoothThroughSawtooth(
                self.t, self.Vsurf * self.Ip, self.tlastsaw, 1
            )
            ax.plot(self.t, self.Vsurf * self.Ip, lw=2, c="c", ls="--", alpha=alpha)
            ax.plot(t, z, lw=2, c="c", ls="--", label="check ($V_{\\phi}\\cdot I_p$)")

            z, t = MATHtools.smoothThroughSawtooth(
                self.t, self.Vsurf_m * self.Ip, self.tlastsaw, 1
            )
            ax.plot(
                self.t, self.Vsurf_m * self.Ip, lw=2, c="orange", ls="-.", alpha=alpha
            )
            ax.plot(
                t,
                z,
                lw=2,
                c="orange",
                ls="-.",
                label="check ($V_{\\phi,meas.}\\cdot I_p$)",
            )
        else:
            ax.plot(
                self.t,
                self.Vsurf * self.Ip,
                lw=2,
                c="c",
                ls="-",
                alpha=alpha,
                label="check ($V_{\\phi}\\cdot I_p$)",
            )
            ax.plot(
                self.t,
                self.Vsurf_m * self.Ip,
                lw=2,
                c="orange",
                ls="-",
                alpha=alpha,
                label="check ($V_{\\phi,meas.}\\cdot I_p$)",
            )

        z, t = MATHtools.smoothThroughSawtooth(self.t, self.PohT, self.tlastsaw, 1)
        if onlyBeforeSaw:
            ax.plot(self.t, self.PohT, lw=3, c="r", alpha=alpha)
            ax.plot(
                t, z, lw=2, c="r", label="$P_{OH}=\\int_V \\langle P_{OH}\\rangle dV$"
            )
        else:
            ax.plot(
                self.t,
                self.PohT,
                lw=3,
                c="r",
                alpha=alpha,
                label="$P_{OH}=\\int_V \\langle P_{OH}\\rangle dV$",
            )

        z, t = MATHtools.smoothThroughSawtooth(
            self.t, self.UmagT_pol_dt, self.tlastsaw, 1
        )
        if onlyBeforeSaw:
            ax.plot(self.t, self.UmagT_pol_dt, lw=3, c="g", alpha=alpha)
            ax.plot(
                t,
                z,
                lw=2,
                c="g",
                label="$\\frac{dU_{B_{\\theta}}}{dt}=\\frac{1}{2\\mu_0}\\int_V \\frac{d \\langle B_{\\theta}^2\\rangle}{dt}dV$",
            )
        else:
            ax.plot(
                self.t,
                self.UmagT_pol_dt,
                lw=3,
                c="g",
                alpha=alpha,
                label="$\\frac{dU_{B_{\\theta}}}{dt}=\\frac{1}{2\\mu_0}\\int_V \\frac{d \\langle B_{\\theta}^2\\rangle}{dt}dV$",
            )

        z, t = MATHtools.smoothThroughSawtooth(
            self.t, self.BpolComp_T, self.tlastsaw, 1
        )
        if onlyBeforeSaw:
            ax.plot(self.t, self.BpolComp_T, lw=3, c="m", alpha=alpha)
            ax.plot(t, z, lw=2, c="m", label="$P_{B_{\\theta},compr}$")
        else:
            ax.plot(
                self.t,
                self.BpolComp_T,
                lw=3,
                c="m",
                alpha=alpha,
                label="$P_{B_{\\theta},compr}$",
            )

        check = self.PohT + self.UmagT_pol_dt - self.BpolComp_T
        ll = "check ($P_{OH}+\\frac{dU_{B_{\\theta}}}{dt}-P_{B_{\\theta},compr}$)"

        z, t = MATHtools.smoothThroughSawtooth(self.t, check, self.tlastsaw, 1)
        if onlyBeforeSaw:
            ax.plot(self.t, check, lw=2, c="y", ls="--", alpha=alpha)
            ax.plot(t, z, lw=2, c="y", ls="--", label=ll)
        else:
            ax.plot(self.t, check, lw=2, c="y", ls="--", alpha=alpha, label=ll)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MW")
        ax.set_title("Poynting Flux Balance")
        pp = self.PohT[self.ind_saw]
        ax.set_ylim([-pp * 1.0, pp * 1.5])
        ax.axhline(y=0.0, ls="--", lw=1, c="k")

        if legend:
            GRAPHICStools.addLegendApart(ax, ratio=0.8)

        if ax2 is not None:
            # Balance

            ax = ax2
            ax.plot(
                self.x_lw,
                self.ExBpower[it],
                lw=3,
                c="b",
                label="$\\frac{1}{\\mu_0}\\nabla\\cdot|\\vec{E}x\\vec{B}|$",
            )
            ax.plot(
                self.x_lw, self.Poh[it], lw=2, c="r", label="$\\langle P_{OH}\\rangle$"
            )
            ax.plot(
                self.x_lw,
                self.Umag_pol_dt[it],
                lw=2,
                c="g",
                label="$\\frac{dU_{B\\theta}}{dt}=\\frac{1}{2\\mu_0}\\frac{d \\langle B_{\\theta}^2\\rangle}{dt}$",
            )
            ax.plot(
                self.x_lw,
                self.BpolComp[it],
                lw=2,
                c="m",
                label="$P_{B_{\\theta},compr}$",
            )

            ll = "check ($P_{OH}+\\frac{dU_{B\\theta}}{dt}-P_{B_{\\theta},compr}$)"
            ax.plot(
                self.x_lw,
                self.Poh[it] + self.Umag_pol_dt[it] - self.BpolComp[it],
                lw=2,
                c="y",
                ls="--",
                label=ll,
            )

            ax.axvline(x=self.x_saw_inv[it], ls="--", c="k", lw=2, alpha=0.5)

            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("$MW/m^{3}$")
            ax.set_xlim([0, 1.0])
            # pp = self.PohT[self.ind_saw]*1.5
            # ax.set_ylim([-pp,pp])
            ax.axhline(y=0.0, ls="--", lw=1, c="k")
            ax.legend(loc="upper right", prop={"size": 8})
            GRAPHICStools.addLegendApart(ax, ratio=0.8)
            # GRAPHICStools.addLegendApart(ax,ratio=0.8)

    def plotSawtooth(self, fig=None, time=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(4, 2, hspace=0.6, wspace=0.4)
        ax1 = fig.add_subplot(grid[:2, 0])
        ax3 = fig.add_subplot(grid[2, 0], sharex=ax1)
        ax2 = fig.add_subplot(grid[3, 0], sharex=ax1)

        ax1s = fig.add_subplot(grid[:2, 1], sharex=ax1)
        ax3e = fig.add_subplot(grid[2, 1], sharex=ax1)
        ax2s = fig.add_subplot(grid[3, 1], sharex=ax1)

        # _________________________________

        ax1.plot(
            self.t,
            self.porcelli_13L,
            "r",
            ls="-",
            lw=3,
            label="$-\\delta\\hat{W}_{core}$ (Eq. 13L)",
        )
        ax1.plot(
            self.t,
            self.porcelli_13R,
            "r",
            ls="--",
            lw=3,
            label="$c_h\\omega_{Dh}\\tau_A$ (Eq. 13R)",
        )

        ax1.plot(
            self.t,
            self.porcelli_14L,
            "b",
            ls="-",
            lw=2,
            label="$-\\delta\\hat{W}$ (Eq. 14L, Eq. 15aC)",
        )

        ax1.plot(
            self.t,
            self.porcelli_15aL,
            "g",
            ls="-",
            lw=2,
            label="$-c_\\rho\\hat{\\rho}$ (Eq. 15aL)",
        )
        ax1.plot(
            self.t,
            self.porcelli_15aR,
            "g",
            ls="-.",
            lw=2,
            label="$0.5\\omega_{*i}\\tau_A$ (Eq. 14R, Eq. 15aR)",
        )

        ax1.plot(
            self.t,
            self.porcelli_15bL * 1e-6,
            "m",
            ls="--",
            lw=2,
            label="$\\omega_{*i}$ *1E-6 (Eq. 15bL)",
        )
        ax1.plot(
            self.t,
            self.porcelli_15bR * 1e-6,
            "m",
            ls="-",
            lw=2,
            label="$c_*\\gamma_\\rho$ *1E-6 (Eq. 15bR)",
        )

        GRAPHICStools.addLegendApart(ax1, ratio=0.7)

        for i in self.tlastsawU:
            ax1.axvline(x=i, c="k", ls="--", linewidth=0.2)

        ax1.set_xlabel("Time (s)")

        GRAPHICStools.addDenseAxis(ax1)

        # Detail
        if self.calcualtePorcelli:
            self.plotPorcelliInternals(ax=ax1s)
            GRAPHICStools.addLegendApart(ax1s, ratio=0.7)
        ax1s.set_title("Porcelli Parameters")
        ax1s.set_xlabel("Time (s)")
        GRAPHICStools.addDenseAxis(ax1s)

        ax2s.plot(self.t, self.porcelli_s1, c="b", label="$s_1$")
        ax2s.legend(loc="upper left", prop={"size": self.mainLegendSize})
        ax2s.set_ylabel("Magnetic shear")

        ax = ax2s.twinx()
        ax.plot(self.t, self.porcelli_rq1, c="r", label="$r_1$")
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})
        ax.set_ylabel("Minor radius q=1")

        GRAPHICStools.addDenseAxis(ax2s)

        ax2s.set_xlabel("Time (s)")

        GRAPHICStools.addLegendApart(ax2s, ratio=0.7)

        # Conditions
        ax2.scatter(
            self.t,
            self.eq13,
            c="r",
            edgecolors="none",
            s=10,
            label="Eq. 13? ($-\\delta\\hat{W}_{core}>c_h\\omega_{Dh}\\tau_A$)",
        )
        ax2.scatter(
            self.t,
            self.eq14,
            c="b",
            edgecolors="none",
            s=10,
            label="Eq. 14? ($-\\delta\\hat{W}>0.5\\omega_{*i}\\tau_A$)",
        )
        ax2.scatter(
            self.t,
            self.eq15,
            c="orange",
            edgecolors="none",
            s=10,
            label="Eq. 15? ($-c_\\rho\\hat{\\rho}<-\\delta\\hat{W}<0.5\\omega_{*i}\\tau_A$   AND   $\\omega_{*i}<c_*\\gamma_\\rho$)",
        )
        ax2.scatter(
            self.t, self.q1cond, c="y", edgecolors="none", s=10, label="$q_0<1.0$?"
        )

        for i in [0, 0.5, 1.0]:
            ax2.axhline(y=i, color="k", linewidth=1, ls="--", alpha=1.0)

        for i in self.tlastsawU:
            ax2.axvline(x=i, c="k", ls="--", linewidth=0.2)

        ax2.set_xlim([self.t[0] - 0.1, self.t[-1] + 0.1])

        GRAPHICStools.addLegendApart(ax2, ratio=0.7)

        GRAPHICStools.addDenseAxis(ax2)

        ax2.set_xlabel("Time (s)")

        # q profile trace
        
        for irho in range(10):
            ax3e.plot(
                self.t,
                self.q[:, irho],
                ls="-",
                lw=2,
                label=f"q @ $\\rho_N$={self.xb_lw[irho]:.3f}",
            )

        ax3e.legend(loc="best", prop={"size": self.mainLegendSize})
        ax3e.set_ylabel("q")
        ax3e.set_title("First 10 radii of q-profile")
        ax3e.axhline(y=1.0, ls="--", c="k", lw=0.5)

        GRAPHICStools.addLegendApart(ax3e, ratio=0.7, size=6)

        GRAPHICStools.addDenseAxis(ax3e)

        ax3e.set_xlabel("Time (s)")

        # q profile trace
        rhos = [0, 0.1, 0.2, 0.3, 0.4]
        minn, maxx = 10, 0
        for i in rhos:
            irho = np.argmin(np.abs(self.x_lw - i))
            ax3.plot(
                self.t,
                self.q[:, irho],
                ls="-",
                lw=2,
                label=f"q @ $\\rho_N$={i}",
            )
            minn = np.min([minn, self.q[:, irho].min()])
            maxx = np.max([maxx, self.q[:, irho].max()])

        ax3.axhline(y=1.0, ls="--", c="k", lw=1)

        ax3.legend(loc="best", prop={"size": self.mainLegendSize})
        ax3.set_ylabel("q")
        ax3.set_ylim([np.max([0.80, minn]), np.min([1.3, maxx])])

        GRAPHICStools.addLegendApart(ax3, ratio=0.7)

        ax3.set_xlim([self.t[0] - 0.01, self.t[-1] + 0.01])

        GRAPHICStools.addDenseAxis(ax3)

        ax3.set_xlabel("Time (s)")

    def plotPorcelliInternals(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            self.t, self.Porcelli.dWBussac, c="g", label="$\\delta\\hat{W}_{Bussac}$"
        )
        ax.plot(self.t, self.Porcelli.dWel, c="k", label="$\\delta\\hat{W}_{el.}$")
        ax.plot(self.t, self.Porcelli.dWko, c="m", label="$\\delta\\hat{W}_{K.O.}$")
        ax.plot(
            self.t,
            self.Porcelli.dWmhd,
            c="c",
            label="$\\delta\\hat{W}_{MHD}=\\delta\\hat{W}_{Bussac}+\\delta\\hat{W}_{el.}$",
        )
        ax.plot(
            self.t,
            self.Porcelli.dWcore,
            c="b",
            lw=3,
            label="$\\delta\\hat{W}_{core}=\\delta\\hat{W}_{MHD}+\\delta\\hat{W}_{K.O.}$",
        )

        ax.plot(
            self.t,
            -self.porcelli_13L,
            c="r",
            lw=3,
            label="$\\delta\\hat{W}_{core}$ TRANSP",
        )

        ax.axhline(y=0.0, lw=0.5, ls="--", c="k")

        ax.set_ylim(
            [np.min(-self.porcelli_13L) - 0.02, np.max(-self.porcelli_13L) + 0.02]
        )

        # ax.legend(loc='best',prop={'size':self.mainLegendSize})

        ax.set_xlabel("Time (s)")

    def plotSawtoothMixing(self, fig=None):
        sawIndexAfter = 2

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)

        ax = fig.add_subplot(grid[0, 0])
        it, _ = self.plotAroundSawtooth(
            self.psi_heli,
            x=self.roa,
            ax=ax,
            legend=True,
            plotLinesSawVerbose=2,
            fractionBefore=0.0,
            fractionAfter=0.0,
            boldPlusMinus=sawIndexAfter,
        )
        ax.set_title("Helical flux")
        ax.set_ylabel("$\\psi^*$ (Wb/rad)")
        ax.set_ylim([-0.01, self.psi_heli[it].max() * 1.1])
        ax.set_xlabel("r/a")
        ax.axhline(y=0.0, ls="--", c="k", lw=1)
        GRAPHICStools.addDenseAxis(ax)

        ax1 = fig.add_subplot(grid[1, 0], sharex=ax)
        it, comp = self.plotAroundSawtooth(
            self.q,
            x=self.roa,
            ax=ax1,
            legend=False,
            plotLinesSawVerbose=2,
            fractionBefore=0.0,
            fractionAfter=0.0,
            boldPlusMinus=sawIndexAfter,
        )
        ax1.set_title("q profile")
        ax1.set_ylabel("q")
        ax1.set_xlabel("r/a")
        ax1.set_ylim([self.q[it].min(), 1.3])
        ax1.axhline(y=1.0, ls="--", c="k", lw=1)
        GRAPHICStools.addDenseAxis(ax1)

        [x_saw_r1l, x_saw_r1, x_saw_r1r, x_saw_r2l, x_saw_r2, x_saw_r2r] = comp

        if it + sawIndexAfter > len(self.t) - 1:
            sawIndexAfter -= 1

        flx_i = np.interp(self.x_saw_mix[it], self.xb[it], self.psi_heli[it])
        flx_f = np.interp(
            self.x_saw_mix[it] - 0.1,
            self.xb[it + sawIndexAfter],
            self.psi_heli[it + sawIndexAfter],
        )
        flx = flx_f - flx_i

        newflux = self.psi_heli[it + sawIndexAfter] - flx
        # ax.plot(self.roa[it+sawIndexAfter],newflux,ls='--',c='b',lw=2,label='$\\psi^*_f$ mod')

        ax.axhline(y=newflux.max(), ls="--", c="k")

        y01 = np.interp(
            0.1, self.roa[it + sawIndexAfter], self.psi_heli[it + sawIndexAfter]
        )
        a_coeff = y01 / 0.1**2
        xx = np.linspace(0, 0.5, 100)
        yy = a_coeff * xx**2
        ax.plot(xx, yy, ls="-.", lw=1.0, c="k", label="$\\sim r^2$")

        ax.legend()
        GRAPHICStools.addDenseAxis(ax)

        # -------

        ax = fig.add_subplot(grid[0, 1])
        it, _ = self.plotAroundSawtooth(
            self.p,
            x=self.roa,
            ax=ax,
            legend=False,
            plotLinesSawVerbose=2,
            fractionBefore=0.0,
            fractionAfter=0.0,
        )
        ax.set_title("Thermal pressure")
        ax.set_ylabel("MPA")
        ax.set_ylim([0, self.p[it].max() * 1.1])
        ax.set_xlabel("r/a")
        pos = np.interp(self.x_saw_inv_prof[it], self.x[it], self.roa[it])
        # ax.axvline(x=pos,ls='--',c='m',lw=1)
        ix = np.argmin(
            np.abs(
                self.roa[it] - np.interp(self.x_saw_mix[it], self.x[it], self.roa[it])
            )
        )
        ax.plot(
            self.roa[it, :ix],
            self.p_expectedsaw[it, :ix],
            c="b",
            ls="-.",
            label="$p_{expected}$",
        )
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)

        GRAPHICStools.fillGraph(
            ax,
            self.roa[it],
            self.p[it + sawIndexAfter],
            y_down=None,
            y_up=self.p[it],
            alpha=0.1,
            color="orange",
        )

        ax = ax.twinx()
        var = volumeIntegral_var(self.f, self.p)
        it, _ = self.plotAroundSawtooth(
            var,
            x=self.roa,
            ax=ax,
            legend=False,
            plotLinesSawVerbose=2,
            fractionBefore=0.0,
            fractionAfter=0.0,
        )
        # ax.set_title('Enclosed thermal pressure');
        ax.set_ylabel("MPA$\\cdot V$")
        ax.set_ylim([0, var[it].max() * 1.1])
        ax.set_xlabel("r/a")
        pos = np.interp(self.x_saw_inv_prof[it], self.x[it], self.roa[it])
        ax.axvline(x=pos, ls="--", c="m", lw=1)

        GRAPHICStools.fillGraph(
            ax,
            self.roa[it],
            var[it + sawIndexAfter],
            y_down=None,
            y_up=var[it],
            alpha=0.1,
            color="orange",
        )

        ax = fig.add_subplot(grid[1, 1])
        it, _ = self.plotAroundSawtooth(
            self.Te,
            x=self.roa,
            ax=ax,
            legend=False,
            plotLinesSawVerbose=2,
            fractionBefore=0.0,
            fractionAfter=0.0,
        )
        ax.set_title("Electron temperature")
        ax.set_ylabel("keV")
        ax.set_ylim([0, self.Te[it].max() * 1.1])
        ax.set_xlabel("r/a")
        pos = np.interp(self.x_saw_inv_prof[it], self.x[it], self.roa[it])
        # ax.axvline(x=pos,ls='--',c='m',lw=1)
        ax.plot(
            self.roa[it, :ix],
            self.Te_expectedsaw[it, :ix],
            c="b",
            ls="-.",
            label="$T_{expected}$",
        )
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})
        GRAPHICStools.addDenseAxis(ax)

        ax = fig.add_subplot(grid[0, 2])
        ax.plot(self.t, self.UmagT_pol, lw=2, c="r", label="$U_{B_{\\theta}}$")
        ax.plot(self.t, self.UeT, lw=2, c="b", label="$U_{e}$")
        ax.plot(
            self.t,
            self.UeT + self.UmagT_pol,
            lw=2,
            c="g",
            label="$U_{B_{\\theta}}+U_{e}$",
        )
        for i in self.tlastsawU:
            ax.axvline(x=i, ls="--", lw=1, c="k")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MJ")
        ax.set_title("Electrons-Bpol Balance")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        GRAPHICStools.addDenseAxis(ax)

        ax = fig.add_subplot(grid[1, 2])
        Delta1 = self.Umag_pol[self.ind_saw_after] - self.Umag_pol[self.ind_saw_before]
        ax.plot(self.x_lw, Delta1, lw=1, c="r", label="$\\Delta_{saw} U_{B_{\\theta}}$")
        Delta2 = self.Ue[self.ind_saw_after] - self.Ue[self.ind_saw_before]
        ax.plot(self.x_lw, Delta2, lw=1, c="b", label="$\\Delta_{saw} U_{e}$")
        ax.plot(
            self.x_lw,
            Delta1 + Delta2,
            lw=2,
            c="g",
            label="$\\Delta_{saw} (U_{B_{\\theta}}+U_{e})$",
        )
        ax.axvline(
            x=self.x_saw_inv[self.ind_saw_after], ls="--", lw=2, c="k", alpha=0.5
        )
        ax.axhline(y=0, ls="--", lw=1, c="k")

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$MJ/m^3$")
        ax.legend(loc="lower right")

        GRAPHICStools.addDenseAxis(ax)

    def plotAroundSawtoothQuantities(self, fig=None, fractionExtend=0.5, alpha=0.3):
        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(3, 3, hspace=0.4, wspace=0.3)

        ax1 = fig.add_subplot(grid[0, 0])
        _, _ = self.plotAroundSawtooth(
            self.q,
            x=self.xb,
            fractionBefore=fractionExtend,
            fractionAfter=fractionExtend,
            ax=ax1,
            legend=False,
            alpha=alpha,
        )
        ax1.set_title("q profile")
        ax1.set_ylabel("q")
        ax1.set_ylim([0.8, 1.2])
        ax1.axhline(y=1.0, ls="--", c="k", lw=1)
        ax1.grid()

        ax = fig.add_subplot(grid[0, 1], sharex=ax1)
        _, _ = self.plotAroundSawtooth(
            self.j,
            x=self.xb,
            fractionBefore=fractionExtend,
            fractionAfter=fractionExtend,
            ax=ax,
            legend=False,
            alpha=alpha,
        )
        ax.set_title("Current profile")
        ax.set_ylabel("j ($MA/m^2$)")
        ax.set_ylim(bottom=0)
        ax.grid()

        ax = fig.add_subplot(grid[1, 0], sharex=ax1)
        _, _ = self.plotAroundSawtooth(
            self.Te,
            x=self.x,
            fractionBefore=fractionExtend,
            fractionAfter=fractionExtend,
            ax=ax,
            legend=False,
            alpha=alpha,
        )
        ax.set_title("Electron temperature")
        ax.set_ylabel("Te (keV)")
        ax.set_ylim(bottom=0)
        ax.grid()

        ax = fig.add_subplot(grid[1, 1], sharex=ax1)
        _, _ = self.plotAroundSawtooth(
            self.Ti,
            x=self.x,
            fractionBefore=fractionExtend,
            fractionAfter=fractionExtend,
            ax=ax,
            legend=False,
            alpha=alpha,
        )
        ax.set_title("Ion temperature")
        ax.set_ylabel("Ti (keV)")
        ax.set_ylim(bottom=0)
        ax.grid()

        ax = fig.add_subplot(grid[0, 2], sharex=ax1)
        _, _ = self.plotAroundSawtooth(
            self.Piich + self.Peich,
            x=self.x,
            fractionBefore=fractionExtend,
            fractionAfter=fractionExtend,
            ax=ax,
            legend=True,
            alpha=alpha,
        )
        ax.set_title("ICRF power to Bulk")
        ax.set_ylabel("P ($MW/m^3$)")
        ax.set_ylim(bottom=0)
        ax.grid()

        ax = fig.add_subplot(grid[1, 2], sharex=ax1)
        _, _ = self.plotAroundSawtooth(
            self.Pfusi + self.Pfuse,
            x=self.x,
            fractionBefore=fractionExtend,
            fractionAfter=fractionExtend,
            ax=ax,
            legend=False,
            alpha=alpha,
        )
        ax.set_title("Alpha power to Bulk")
        ax.set_ylabel("P ($MW/m^3$)")
        ax.set_ylim(bottom=0)
        ax.grid()

        ax = fig.add_subplot(grid[2, 0], sharex=ax1)
        _, _ = self.plotAroundSawtooth(
            self.ne,
            x=self.x,
            fractionBefore=fractionExtend,
            fractionAfter=fractionExtend,
            ax=ax,
            legend=False,
            alpha=alpha,
        )
        ax.set_title("Electron density")
        ax.set_ylabel("ne ($10^{20}m^{-3}$)")
        ax.set_ylim(bottom=0)
        ax.grid()

        ax = fig.add_subplot(grid[2, 1], sharex=ax1)
        _, _ = self.plotAroundSawtooth(
            self.pFast,
            x=self.x,
            fractionBefore=fractionExtend,
            fractionAfter=fractionExtend,
            ax=ax,
            legend=False,
            alpha=alpha,
        )
        ax.set_title("Fast ion pressure")
        ax.set_ylabel("p (MPa)")
        ax.set_ylim(bottom=0)
        ax.grid()

        ax = fig.add_subplot(grid[2, 2], sharex=ax1)
        _, _ = self.plotAroundSawtooth(
            self.qe_obs + self.qi_obs,
            x=self.xb,
            fractionBefore=fractionExtend,
            fractionAfter=fractionExtend,
            ax=ax,
            legend=False,
            alpha=alpha,
        )
        ax.set_title("Bulk Heat flux")
        ax.set_ylabel("$q_e+q_i$ ($MWm^{-2}$)")
        ax.set_ylim(
            [
                -0.2,
                3 * (self.qe_obs[self.ind_saw].max() + self.qi_obs[self.ind_saw].max()),
            ]
        )
        ax.grid()

    def plotICRF(self, fig=None, time=None):
        if time is None:
            i1 = self.ind_saw
        else:
            i1 = np.argmin(np.abs(self.t - time))
        timeReq = self.t[i1]

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.4)

        ax1 = fig.add_subplot(grid[0, 1])
        ax2 = fig.add_subplot(grid[0, 2], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(grid[1, 1], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(grid[1, 2], sharex=ax1, sharey=ax1)

        ax6 = fig.add_subplot(grid[0, 0])
        ax7 = fig.add_subplot(grid[1, 0])

        # ELECTRONS
        ax = ax1
        ax.plot(self.x_lw, self.Peich[i1], c="r", lw=3, label="$P$")
        ax.plot(self.x_lw, self.Peich_min[i1], lw=2, label="$P_{min->e}$")
        ax.plot(self.x_lw, self.Peich_dir[i1], lw=2, label="$P_{ICH->e}$")
        ax.plot(
            self.x_lw,
            self.Peich_dir[i1] + self.Peich_min[i1],
            lw=2,
            ls="--",
            c="y",
            label="check (sum)",
        )

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Electrons")
        ax.set_ylabel("Power ($MWm^{-3}$)")

        ax.axhline(y=0, ls="--", c="k", lw=1)
        ax.set_xlabel("$\\rho_N$")

        GRAPHICStools.addDenseAxis(ax)

        ax.set_xlim([0, 1])
        # IONS
        ax = ax2
        ax.plot(self.x_lw, self.Piich[i1], c="r", lw=3, label="$P$")
        ax.plot(self.x_lw, self.Piich_min[i1], lw=2, label="$P_{min->i}$")
        ax.plot(self.x_lw, self.Piich_dir[i1], lw=2, label="$P_{ICH->i}$")
        ax.plot(
            self.x_lw,
            self.Piich_dir[i1] + self.Piich_min[i1],
            lw=2,
            ls="--",
            c="y",
            label="check (sum)",
        )

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Ions")
        ax.set_ylabel("Power ($MWm^{-3}$)")

        ax.axhline(y=0, ls="--", c="k", lw=1)
        ax.set_xlabel("$\\rho_N$")

        GRAPHICStools.addDenseAxis(ax)

        # MINORITY
        ax = ax3
        ax.plot(
            self.x_lw, self.Pich_min[i1], c="r", ls="--", lw=2, label="$P_{ICH->min}$"
        )
        ax.plot(
            self.x_lw,
            self.Pich_minRenorm[i1],
            c="r",
            ls="-",
            lw=3,
            label="$P_{ICH->min,renorm}$",
        )
        ax.plot(self.x_lw, self.Pich_minPTCL[i1], c="c", lw=2, label="$P_{PTCL->min}$")
        ax.plot(self.x_lw, self.Piich_min[i1], c="b", lw=2, label="$P_{min->i}$")
        ax.plot(self.x_lw, self.Peich_min[i1], c="g", lw=2, label="$P_{min->e}$")
        ax.plot(self.x_lw, self.Gainmin[i1], c="m", lw=2, label="$dW_{min}/dt$")
        ax.plot(
            self.x_lw, self.Pich_minOrbLoss[i1], c="k", lw=2, label="$P_{loss,orb}$"
        )
        ax.plot(
            self.x_lw,
            self.Peich_min[i1]
            + self.Piich_min[i1]
            + self.Gainmin[i1]
            - self.Pich_minPTCL[i1]
            + self.Pich_minOrbLoss[i1],
            lw=2,
            ls="--",
            c="y",
            label="check (bal)",
        )

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Minorities Balance")
        ax.set_ylabel("Power ($MWm^{-3}$)")
        ax.set_xlabel("$\\rho_N$")

        ax.axhline(y=0, ls="--", c="k", lw=1)

        GRAPHICStools.addDenseAxis(ax)

        # TOTAL
        ax = ax4

        # ax.plot(self.x_lw,tot,lw=3,label='$P_{ICH}$')
        ax.plot(self.x_lw, self.Pich[i1], c="r", lw=3, label="$P_{ICH}$")
        ax.plot(self.x_lw, self.Pich_min[i1], lw=2, label="$P_{ICH->min}$")
        ax.plot(self.x_lw, self.Piich_dir[i1], lw=2, label="$P_{ICH->i}$")
        ax.plot(self.x_lw, self.Peich_dir[i1], lw=2, label="$P_{ICH->e}$")
        ax.plot(self.x_lw, self.Pfich_dir[i1], lw=2, label="$P_{ICH->fast}$")
        ax.plot(
            self.x_lw, self.Pich_check[i1], lw=2, ls="--", c="y", label="check (sum)"
        )

        l2 = f"$P_{{ICH}}$ = {np.sum(self.Pich[i1] * self.dvol[i1]):.2f} MW"
        ax.text(
            0.6,
            0.9,
            l2,
            fontweight="bold",
            color="b",
            fontsize=12,
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        l2 = f"$P_{{ICH,real}}$ = {self.PichT[i1]:.2f} MW"
        ax.text(
            0.6,
            0.8,
            l2,
            fontweight="bold",
            color="b",
            fontsize=12,
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        ax.axhline(y=0, ls="--", c="k", lw=1)

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Total Balance")
        ax.set_ylabel("Power ($MWm^{-3}$)")
        ax.set_xlabel("$\\rho_N$")

        ax.set_ylim(bottom=-2.0)

        GRAPHICStools.addDenseAxis(ax)

        # Machine
        ax = ax6
        self.plotGeometry(ax=ax, color="k")
        self.plotRelevantResonances(
            ax, self.FichT_ant[0], time=self.t[i1], legendYN=True
        )
        ax.set_title("Resonances (external B)")

        # Per volume

        # -----------
        ax = ax7
        ax.bar(
            self.x_lw,
            self.Piich_cum[i1],
            width=(self.x_lw[1] - self.x_lw[0]) * 0.5,
            label="$i^+$",
            color="b",
            alpha=0.5,
        )
        ax.bar(
            self.x_lw,
            self.Peich_cum[i1],
            width=(self.x_lw[1] - self.x_lw[0]) * 0.5,
            label="$e^-$",
            color="r",
            alpha=0.5,
        )

        ax.set_title("ICRF power per zone-volume")
        ax.set_ylabel("Power ($MW$)")
        ax.set_xlabel("$\\rho_N$")
        ax.set_xlim([0, 1])
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)

    def plotICRF_t(self, fig=None):
        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.4)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1], sharex=ax1)
        ax3 = fig.add_subplot(grid[1, 0], sharex=ax1)
        ax4 = fig.add_subplot(grid[1, 1], sharex=ax1)
        ax5 = fig.add_subplot(grid[0, 2], sharex=ax1)
        ax6 = fig.add_subplot(grid[1, 2], sharex=ax1)

        # ELECTRONS
        ax = ax1
        ax.plot(self.t, self.PeichT, lw=3, c="r", label="$P$")
        ax.plot(self.t, self.PeichT_min, lw=2, label="$P_{min->e}$")
        ax.plot(self.t, self.PeichT_dir, lw=2, label="$P_{ICH->e}$")
        ax.plot(
            self.t,
            self.PeichT_dir + self.PeichT_min,
            lw=2,
            ls="--",
            c="y",
            label="check (sum)",
        )

        GRAPHICStools.addLegendApart(ax, ratio=0.85, size=self.mainLegendSize)
        GRAPHICStools.addDenseAxis(ax)

        ax.set_title("Electrons")
        ax.set_ylabel("Power ($MW$)")
        ax.set_xlabel("Time (s)")

        ax.axhline(y=0, ls="--", c="k", lw=1)

        # IONS
        ax = ax2
        ax.plot(self.t, self.PiichT, lw=3, c="r", label="$P$")
        ax.plot(self.t, self.PiichT_min, lw=2, label="$P_{min->i}$")
        ax.plot(self.t, self.PiichT_dir, lw=2, label="$P_{ICH->i}$")
        ax.plot(
            self.t,
            self.PiichT_dir + self.PiichT_min,
            lw=2,
            ls="--",
            c="y",
            label="check (sum)",
        )

        ax.set_title("Ions")
        ax.set_ylabel("Power ($MW$)")
        ax.set_xlabel("Time (s)")

        GRAPHICStools.addLegendApart(ax, ratio=0.85, size=self.mainLegendSize)
        GRAPHICStools.addDenseAxis(ax)

        # MINORITY
        ax = ax3
        ax.plot(self.t, self.PichT_min, c="r", ls="--", lw=3, label="$P_{ICH->min}$")
        ax.plot(
            self.t,
            self.PichT_minRenorm,
            c="r",
            ls="-",
            lw=2,
            label="$P_{ICH->min,renorm}$",
        )
        ax.plot(self.t, self.PichT_minPTCL, c="c", lw=2, label="$P_{PTCL->min}$")
        ax.plot(self.t, self.PiichT_min, c="b", lw=2, label="$P_{min->i}$")
        ax.plot(self.t, self.PeichT_min, c="g", lw=2, label="$P_{min->e}$")
        ax.plot(self.t, self.PichTOrbLoss, c="k", lw=2, label="$P_{loss,orb}$")
        ax.plot(self.t, self.GainminT, c="m", lw=2, label="$dW_{min}/dt$")
        ax.plot(
            self.t,
            self.PeichT_min
            + self.PiichT_min
            + self.GainminT
            - self.PichT_minPTCL
            + self.PichTOrbLoss,
            lw=2,
            ls="--",
            c="y",
            label="check (bal)",
        )

        ax.set_title("Minorities Balance")
        ax.set_ylabel("Power ($MW$)")
        ax.set_xlabel("Time (s)")

        GRAPHICStools.addLegendApart(ax, ratio=0.85, size=self.mainLegendSize)
        GRAPHICStools.addDenseAxis(ax)

        # TOTAL
        ax = ax4

        # ax.plot(self.x_lw,tot,lw=3,label='$P_{ICH}$')
        ax.plot(self.t, self.PichT, c="r", lw=3, label="$P_{ICH}$")
        ax.plot(self.t, self.PiichT_dir, lw=2, label="$P_{ICH->i}$")
        ax.plot(self.t, self.PeichT_dir, lw=2, label="$P_{ICH->e}$")
        ax.plot(self.t, self.PfichT_dir, lw=2, label="$P_{ICH->fast}$")
        ax.plot(self.t, self.PichT_min, lw=2, label="$P_{ICH->min}$")
        ax.plot(self.t, self.PichT_check, lw=2, ls="--", c="y", label="check (sum)")

        ax.set_title("Total Balance")
        ax.set_ylabel("Power (MW)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)

        GRAPHICStools.addLegendApart(ax, ratio=0.85, size=self.mainLegendSize)
        GRAPHICStools.addDenseAxis(ax)

        # TOTAL
        ax = ax6

        ax.plot(self.t, self.PichT, c="r", lw=3, label="$P_{ICH}$")
        ax.plot(self.t, self.PiichT, lw=2, label="$P_{ICH,i}$")
        ax.plot(self.t, self.PeichT, lw=2, label="$P_{ICH,e}$")
        ax.plot(self.t, self.PfichT_dir, lw=2, label="$P_{ICH,fast}$")
        ax.plot(self.t, self.GainminT, lw=2, label="$dW_{min}/dt$")
        P = self.PeichT + self.PiichT + self.PfichT_dir + self.GainminT
        ax.plot(self.t, P, lw=2, ls="--", c="y", label="check (sum)")

        ax.set_title("Total Balance (after thermalization)")
        ax.set_ylabel("Power (MW)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)

        GRAPHICStools.addLegendApart(ax, ratio=0.85, size=self.mainLegendSize)
        GRAPHICStools.addDenseAxis(ax)


    def plotRelevantResonances(self, ax, Fich, time=None, legendYN=False, lw=3):
        if time is None:
            i1 = self.ind_saw
        else:
            i1 = np.argmin(np.abs(self.t - time))

        if np.sum(Fich) > 0.0 + self.eps00 * (len(self.t) + 1):
            if self.fD_avol[i1] > 0.0 + self.eps00:
                if not np.isnan(self.R_ICRF_D[i1]):
                    ax.axvline(x=self.R_ICRF_D[i1], lw=lw, c="b", label="D")
                if not np.isnan(self.R_ICRF_D_2[i1]):
                    ax.axvline(
                        x=self.R_ICRF_D_2[i1], lw=lw, ls="--", c="b", label="D x2"
                    )
            if self.fT_avol[i1] > 0.0 + self.eps00:
                if not np.isnan(self.R_ICRF_T[i1]):
                    ax.axvline(x=self.R_ICRF_T[i1], lw=lw, c="g", label="T")
                if not np.isnan(self.R_ICRF_T_2[i1]):
                    ax.axvline(
                        x=self.R_ICRF_T_2[i1], lw=lw, ls="--", c="g", label="T x2"
                    )
            if self.fHe4_avol[i1] > 0.0 + self.eps00:
                if not np.isnan(self.R_ICRF_He4[i1]):
                    ax.axvline(x=self.R_ICRF_He4[i1], lw=lw, c="m", label="He4")
                if not np.isnan(self.R_ICRF_He4_2[i1]):
                    ax.axvline(
                        x=self.R_ICRF_He4_2[i1], lw=lw, ls="--", c="m", label="He4 x2"
                    )
            if self.fminiHe3_avol[i1] > 0.0 + self.eps00:
                if not np.isnan(self.R_ICRF_He3[i1]):
                    ax.axvline(x=self.R_ICRF_He3[i1], lw=lw, c="r", label="He3")
                if not np.isnan(self.R_ICRF_He3_2[i1]):
                    ax.axvline(
                        x=self.R_ICRF_He3_2[i1], lw=lw, ls="--", c="r", label="He3 x2"
                    )
            if self.fminiH_avol[i1] > 0.0 + self.eps00:
                if not np.isnan(self.R_ICRF_H[i1]):
                    ax.axvline(x=self.R_ICRF_H[i1], lw=lw, c="c", label="H")
                if not np.isnan(self.R_ICRF_H_2[i1]):
                    ax.axvline(
                        x=self.R_ICRF_H_2[i1], lw=lw, ls="--", c="c", label="H x2"
                    )

        if legendYN:
            ax.legend(loc="upper right", prop={"size": self.mainLegendSize})

    def plotLowerHybrid(self, fig=None, time=None):
        if time is None:
            i1 = self.ind_saw
        else:
            i1 = np.argmin(np.abs(self.t - time))
        timeReq = self.t[i1]

        if fig is None:
            fig = plt.figure()

        rhos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        grid = plt.GridSpec(2, 2, hspace=0.4, wspace=0.4)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0])
        ax3 = fig.add_subplot(grid[0, 1])
        ax4 = fig.add_subplot(grid[1, 1])

        # Profiles
        ax = ax1
        ax.plot(self.x_lw, self.Plhe[i1], lw=4, c="r", label="Electrons")
        ax.plot(self.x_lw, self.Plhi[i1], lw=4, c="b", label="Ions")

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Deposited Power")
        ax.set_ylabel("Power ($MWm^{-3}$)")
        ax.set_xlabel("$\\rho_N$")
        ax.set_xlim([0, 1.0])
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})
        for i in rhos:
            ax.axvline(x=i, ls="--", c="k", lw=0.5)

        # Time
        ax = ax2
        ax.plot(self.t, self.PlheT, lw=4, c="r", label="Electrons")
        ax.plot(self.t, self.PlhiT, lw=4, c="b", label="Ions")

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Deposited Power")
        ax.set_ylabel("Power ($MW$)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})

        # Time
        ax = ax4
        for i in rhos:
            ix = np.argmin(np.abs(self.x_lw - i))
            ax.plot(self.t, self.Plhe[:, ix], lw=2, c="r")
            ax.plot(self.t, self.Plhi[:, ix], lw=2, c="b")

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Deposited Power")
        ax.set_ylabel("Power ($MWm^{-3}$)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})

    def plotECRF(self, fig=None, time=None):
        if time is None:
            i1 = self.ind_saw
        else:
            i1 = np.argmin(np.abs(self.t - time))
        timeReq = self.t[i1]

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 3, hspace=0.4, wspace=0.4)

        ax1 = fig.add_subplot(grid[0, 2])
        ax2 = fig.add_subplot(grid[1, 2])

        ax0 = fig.add_subplot(grid[:, 1])
        ax0_extra = fig.add_subplot(grid[:, 0])

        # ELECTRONS
        col = ["b", "r", "g", "m", "c", "y", "orange", "sienna"]
        ax = ax1
        ax.plot(self.x_lw, self.Pech[i1], lw=4, c="k", label="$P$")
        for i in range(len(self.Pech_ant)):
            if np.sum(self.Pech_ant[i][i1]) > 0.0 + self.eps00 * (len(self.t) + 1):
                ax.plot(
                    self.x_lw,
                    self.Pech_ant[i][i1],
                    lw=2,
                    c=col[i],
                    label=f"ant #{i + 1}",
                )
        ptot = np.sum(self.Pech_ant[:, i1, :], axis=0)
        ax.plot(self.x_lw, ptot, lw=2, c="y", ls="--")  # ,label='check')

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Deposited Power")
        ax.set_ylabel("Power ($MWm^{-3}$)")
        ax.set_xlabel("$\\rho_N$")
        ax.set_xlim([0, 1.0])

        # CD
        ax = ax2
        ax.plot(self.x_lw, self.jECH[i1], lw=4, c="k", label="$j_{ECH}$")
        for i in range(len(self.jECH_ant)):
            if np.sum(np.abs(self.jECH_ant[i][i1])) > 0.0 + self.eps00 * (
                len(self.t) + 1
            ):
                ax.plot(
                    self.x_lw,
                    self.jECH_ant[i][i1],
                    lw=2,
                    c=col[i],
                    label=f"ant #{i + 1}",
                )
        ptot = np.sum(self.jECH_ant[:, i1, :], axis=0)
        ax.plot(self.x_lw, ptot, lw=2, c="y", ls="--")  # ,label='check')

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Deposited Current Drive")
        ax.set_ylabel("Current density($MAm^{-2}$)")
        ax.set_xlabel("$\\rho_N$")
        ax.set_xlim([0, 1.0])

        # Machine
        self.plotGeometry(ax=ax0, color="b")
        self.plotGeometry_Above(ax=ax0_extra, color="b")

        if hasattr(self, "ECRH_trajectories") and self.ECRH_trajectories is not None:
            for i in range(len(self.Pech_ant)):
                if np.sum(self.Pech_ant[i][i1]) > 0.0 + self.eps00 * (len(self.t) + 1):
                    f = i / 8
                    ax0.plot(
                        self.ECRH_trajectories["rlin"][i],
                        self.ECRH_trajectories["zlin"][i],
                        lw=3 - f,
                        c=col[i],
                        label=f"ant #{i + 1}",
                    )
                    ax0_extra.plot(
                        self.ECRH_trajectories["xlin"][i],
                        self.ECRH_trajectories["ylin"][i],
                        lw=3 - f,
                        c=col[i],
                        label=f"ant #{i + 1}",
                    )

        if hasattr(self, "R_gyr") and self.R_gyr is not None:
            for i in range(len(self.Pech_ant)):
                if np.sum(self.Pech_ant[i][i1]) > 0.0 + self.eps00 * (len(self.t) + 1):
                    f = i * 10
                    ax0.scatter(
                        [self.R_gyr[i]], [self.Z_gyr[i]], 100 - f, marker="s", c=col[i]
                    )

        if hasattr(self, "F_gyr") and self.F_gyr is not None:
            B = self.Bt_ext
            for i in range(len(self.Pech_ant)):
                if np.sum(self.Pech_ant[i][i1]) > 0.0 + self.eps00 * (len(self.t) + 1):
                    Fech = self.F_gyr[i] * np.ones(len(self.t))
                    Rres = PLASMAtools.findResonance(
                        Fech * 1e3 / 2, B, self.Rmaj, self.qm_e
                    )

                    ax0.axvline(x=Rres[i1], lw=2, c=col[i], ls="-.")

    def plotNBI(self, fig=None, time=None):
        if time is None:
            i1 = self.ind_saw
        else:
            i1 = np.argmin(np.abs(self.t - time))
        timeReq = self.t[i1]

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 3, hspace=0.2, wspace=0.4)

        ax1 = fig.add_subplot(grid[0, 1])
        ax2 = fig.add_subplot(grid[1, 1], sharex=ax1, sharey=ax1)

        ax3 = fig.add_subplot(grid[1, 2], sharex=ax1)
        ax4 = fig.add_subplot(grid[0, 2], sharex=ax1)

        ax0 = fig.add_subplot(grid[0, 0])
        ax0e = fig.add_subplot(grid[1, 0])

        # Ions
        col = ["b", "r", "g", "m", "c", "y", "orange", "sienna"]
        ax = ax1
        ax.plot(self.x_lw, self.Pnbii[i1], lw=4, c="k", label="$P$")
        for i in range(len(self.Pnbii_beam)):
            if np.sum(self.Pnbii_beam[i][i1]) > 0.0 + self.eps00 * (len(self.t) + 1):
                ax.plot(
                    self.x_lw,
                    self.Pnbii_beam[i][i1],
                    lw=2,
                    c=col[i],
                    label=f"beam #{i + 1}",
                )
        ptot = np.sum(self.Pnbii_beam[:, i1, :], axis=0)
        ax.plot(self.x_lw, ptot, lw=3, c="y", ls="--", label="check")

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Deposited Power to Ions")
        ax.set_ylabel("Power ($MWm^{-3}$)")
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylim(bottom=0)

        # Electrons
        ax = ax2
        ax.plot(self.x_lw, self.Pnbie[i1], lw=4, c="k", label="$P$")
        for i in range(len(self.Pnbie_beam)):
            if np.sum(self.Pnbie_beam[i][i1]) > 0.0 + self.eps00 * (len(self.t) + 1):
                ax.plot(
                    self.x_lw,
                    self.Pnbie_beam[i][i1],
                    lw=2,
                    c=col[i],
                    label=f"beam #{i + 1}",
                )
        ptot = np.sum(self.Pnbie_beam[:, i1, :], axis=0)
        ax.plot(self.x_lw, ptot, lw=3, c="y", ls="--", label="check")

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Deposited Power to Electrons")
        ax.set_ylabel("Power ($MWm^{-3}$)")
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylim(bottom=0)

        # Particle
        ax = ax3
        parttot = copy.deepcopy(self.x_lw) * 0.0
        for i in range(len(self.Pnbie_beam)):
            if np.sum(self.Pnbip_beam[i][i1]) > 0.0 + self.eps00 * (len(self.t) + 1):
                ax.plot(
                    self.x_lw,
                    self.Pnbip_beam[i][i1],
                    lw=2,
                    c=col[i],
                    label=f"beam #{i + 1}",
                )
                parttot += self.Pnbip_beam[i][i1]
        ax.plot(self.x_lw, parttot, lw=3, c="k", label="S")

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Ion deposition")
        ax.set_ylabel("Source ($10^{20}/m^{-3}/s$)")
        ax.set_xlabel("$\\rho_N$")

        ax.set_xlim([0, 1.0])
        ax.set_ylim(bottom=0)

        # Current
        ax = ax4
        ax.plot(self.x_lw, self.jNBI[i1], lw=4, c="k", label="$J_{NBI}$")
        for i in range(len(self.Pnbii_beam)):
            if np.sum(self.Pnbii_beam[i][i1]) > 0.0 + self.eps00 * (len(self.t) + 1):
                ax.plot(
                    self.x_lw,
                    self.jNBI_beam[i][i1],
                    lw=2,
                    c=col[i],
                    label=f"beam #{i + 1}",
                )
        ptot = np.sum(self.jNBI_beam[:, i1, :], axis=0)
        ax.plot(self.x_lw, ptot, lw=3, c="y", ls="--", label="check")

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Beam current drive")
        ax.set_ylabel("Current density ($MAm^{-2}$)")
        ax.set_xlabel("$\\rho_N$")

        # Machine
        self.plotGeometry(ax=ax0, color="b")
        self.plotNBItrajectories(time=timeReq, ax=ax0, topDown=False, col=col)

        self.plotGeometry_Above(ax=ax0e, color="b")
        self.plotNBItrajectories(time=timeReq, ax=ax0e, topDown=True, col=col)

    def plotNBItrajectories(
        self,
        time=None,
        ax=None,
        topDown=False,
        col=["b", "r", "g", "m", "c", "y", "orange", "sienna"],
        leg=False,
    ):
        if time is None:
            i1 = self.ind_saw
        else:
            i1 = np.argmin(np.abs(self.t - time))

        if ax is None:
            fig, ax = plt.subplots()

        cont = 0
        if hasattr(self, "beam_trajectories") and self.beam_trajectories is not None:
            for i in range(len(self.Pnbii_beam)):
                if np.sum(self.Pnbii_beam[i][i1]) > 0.0 + self.eps00 * (
                    len(self.t) + 1
                ):
                    f = cont**2 / 8
                    cont += 1
                    if topDown:
                        ax.plot(
                            self.beam_trajectories["xlin"][i],
                            self.beam_trajectories["ylin"][i],
                            lw=3 - f,
                            c=col[i],
                            label=f"beam #{i + 1}",
                        )
                    else:
                        ax.plot(
                            self.beam_trajectories["rlin"][i],
                            self.beam_trajectories["zlin"][i],
                            lw=3 - f,
                            c=col[i],
                            label=f"beam #{i + 1}",
                        )

        if leg:
            ax.legend(loc="best", prop={"size": self.mainLegendSize})

    def plotSeparateSystems(self, fig=None):
        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 4, hspace=0.3, wspace=0.4)

        ax1 = fig.add_subplot(grid[0, 1])
        ax2 = fig.add_subplot(grid[0, 2])
        ax3 = fig.add_subplot(grid[0, 3])

        ax0 = fig.add_subplot(grid[0, 0])

        ax0e = fig.add_subplot(grid[1, 0])
        ax1e = fig.add_subplot(grid[1, 1])
        ax2e = fig.add_subplot(grid[1, 2])
        # ax3e = fig.add_subplot(grid[1,3])

        # Ohmic
        ax = ax0
        ax.plot(self.t, self.PohT, lw=2, label="$P_{OH}$")
        # ax.legend(loc='best',prop={'size':self.mainLegendSize})
        ax.set_title("Ohmic")
        ax.set_ylabel("Power ($MW$)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)

        ax = ax0e
        ax.plot(self.t, self.IpOH, lw=2, label="$I_{p,OH}$")
        ax.plot(self.t, self.Ip, lw=2, label="$I_{p,tot}$")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_ylabel("$I_{p}$ ($MA$)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)

        # ICRF
        ax = ax1
        if np.sum(self.PichT) > 1.0e-5:
            for i in range(len(self.PichT_ant)):
                ax.plot(self.t, self.PichT_ant[i], lw=2, label=f"{i + 1}")
            ax.plot(self.t, self.PeichT + self.PiichT + self.PfichT_dir + self.GainminT, c="y", ls="--", label="to species (e+i+f+dWmin/dt)")

            timeb = 0.25
            it1 = np.argmin(np.abs(self.t - (self.t[-1] - timeb)))
            mean = np.mean(self.PeichT[it1:] + self.PiichT[it1:] + self.PfichT_dir[it1:] + self.GainminT[it1:])
            ax.axhline(
                y=mean,
                alpha=0.5,
                c="g",
                lw=2,
                ls="-.",
                label=f"average t={self.t[it1]:.3f}-{self.t[-1]:.3f}s",
            )

            ax.plot(
                self.t,
                self.PeichT + self.PiichT + self.PfichT_dir + self.GainminT + self.PichTOrbLoss,
                c="c",
                ls="--",
                label="+ orb losses",
            )

            # ax1 = ax.twinx()
            # ax1.plot(self.t,(self.PichT-mean)/self.PichT*100.0,ls='--',c='r')
            # ax1.set_ylim([0,20])
            # ax1.set_ylabel('Error (%)')

            ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("ICRF")
        ax.set_ylabel("Power Antenna ($MW$)")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Time (s)")
        GRAPHICStools.addDenseAxis(ax)

        ax = ax1e
        if np.sum(self.PichT) > 1.0e-5:
            for i in range(len(self.FichT_ant)):
                ax.plot(self.t, self.FichT_ant[i], lw=2, label=f"{i + 1}")
            ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_ylabel("Frequency Antenna ($MHz$)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)

        # ECRF
        ax = ax2
        if np.sum(self.PechT) > 1.0e-5:
            for i in range(len(self.PechT_ant)):
                ax.plot(self.t, self.PechT_ant[i], lw=2, label=f"{i + 1}")
            ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("ECRF")
        ax.set_ylabel("Power Antenna ($MW$)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)

        ax = ax2e
        if hasattr(self, "F_gyr") and self.F_gyr is not None:
            for i in range(len(self.F_gyr)):
                ax.plot(
                    self.t,
                    self.F_gyr[i] * np.ones(len(self.t)),
                    lw=2,
                    label=f"{i + 1}",
                )
            ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_ylabel("Frequency Antenna ($GHz$)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)

        # NBI
        ax = ax3
        if np.sum(self.PnbiINJ) > 1.0e-5:
            for i in range(len(self.PnbiT_beam)):
                ax.plot(
                    self.t,
                    self.PnbiT_beam[i] * np.ones(len(self.t)),
                    lw=2,
                    label=f"{i + 1}",
                )
            ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("NBI")
        ax.set_ylabel("Power Beam ($MW$)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)

        GRAPHICStools.adjust_figure_layout(fig)

    def plotEquilParams(self, fig=None, time=None):
        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(nrows=2, ncols=4, hspace=0.2, wspace=0.2)

        # gfiles
        ax0 = fig.add_subplot(grid[:, 0])
        ax0e = fig.add_subplot(grid[:, 1], sharex=ax0, sharey=ax0)

        # psi
        ax1 = fig.add_subplot(grid[:, 2])

        # params
        ax2 = fig.add_subplot(grid[0, 3])
        ax3 = fig.add_subplot(grid[1, 3])

        if time is None:
            i1 = self.ind_saw
        else:
            i1 = np.argmin(np.abs(self.t - time))
        time = self.t[i1]

        # GFILE
        fluxes = np.linspace(0.1, 1.5, 20)
        if self.gfile_in is not None:
            self.gfile_in.plotXpointEnvelope(ax=ax0, color="r", alpha=0.5, rhoPol=False)
            self.gfile_in.plotFluxSurfaces(
                ax=ax0, fluxes=fluxes, color="r", alpha=0.5, rhoPol=False
            )
        self.plotGeometry(
            ax=ax0,
            time=time,
            rhoS=fluxes,
            plotComplete=False,
            plotStructures=True,
            color="b",
            colorsurfs="b",
            plotSurfs=True,
            rhoPol=False,
            sqrt=True,
        )

        ax0.set_title("SQRT normalized TOROIDAL flux")

        # Poloidal
        fluxes = np.linspace(0.1, 1.0, 20)
        if self.gfile_in is not None:
            self.gfile_in.plotXpointEnvelope(ax=ax0e, color="r", alpha=0.5, rhoPol=True)
            self.gfile_in.plotFluxSurfaces(
                ax=ax0e, fluxes=fluxes, color="r", alpha=0.5, rhoPol=True
            )
        self.plotGeometry(
            ax=ax0e,
            time=time,
            rhoS=fluxes,
            plotComplete=False,
            plotStructures=True,
            color="b",
            colorsurfs="b",
            plotSurfs=True,
            rhoPol=True,
            sqrt=False,
        )

        ax0e.set_title("Normalized POLOIDAL flux")

        # Coordinates
        ax1.plot(
            self.xb[i1],
            self.xb[i1],
            ls="--",
            lw=2,
            label="$\\rho_{tor}=\\sqrt{\\phi_n}$",
        )
        ax1.plot(
            self.xb[i1],
            self.xpol[i1],
            ls="-",
            lw=3,
            label="$\\rho_{pol}=\\sqrt{\\psi_n}$",
        )
        ax1.plot(self.xb[i1], self.phin[i1], ls="-", lw=3, label="$\\phi_n$ (toroidal)")
        ax1.plot(self.xb[i1], self.psin[i1], ls="-", lw=3, label="$\\psi_n$ (poloidal)")
        ax1.plot(self.xb[i1], self.roa[i1], ls="-", lw=3, label="$r/a$")

        ax1.set_xlabel("$\\rho_n=\\rho_{tor}$")
        ax1.legend(loc="upper left", prop={"size": self.mainLegendSize})

        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        GRAPHICStools.addDenseAxis(ax1)

        # Elong
        ax2.plot(self.psin[i1], self.kappaS[i1], "r", ls="-", lw=3, label="$\\kappa$")
        ax2.scatter(self.psin[i1], self.kappaS[i1])

        ax2.set_xlabel("$\\psi_n$")
        ax2.set_ylabel("$\\kappa$")
        ax2.set_ylim([1.0, np.max(self.kappaS[i1])])
        ax2.legend(loc="upper left", prop={"size": self.mainLegendSize})

        ax2.axvline(x=0.95, c="k", ls="--")
        ax2.axvline(x=0.995, c="k", ls="--")
        ax2.axvline(x=1.0, c="k", ls="--")

        ax2.set_xlim([0, 1])

        l2 = [
            f"$\\kappa_{{sep}}$ = {self.kappa[i1]:.3f}",
            f"$\\kappa_{{99.5}}$ = {self.kappa_995[i1]:.3f}",
            f"$\\kappa_{{95}}$ = {self.kappa_950[i1]:.3f}",
            f"$\\kappa_{{a}}$ = {self.kappaITER[i1]:.3f}",
        ]
        ax2.text(
            0.2,
            0.7,
            "\n".join(l2),
            color="k",
            fontsize=15,
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax2.transAxes,
        )

        GRAPHICStools.addDenseAxis(ax2)

        # Triang
        ax22 = ax3
        ax22.plot(self.psin[i1], self.deltaS[i1], "b", ls="-", lw=3, label="$\\delta$")
        ax22.scatter(self.psin[i1], self.deltaS[i1])

        ax22.plot(
            self.psin[i1], self.deltaS_u[i1], "c", ls="--", lw=1, label="$\\delta_u$"
        )
        ax22.plot(
            self.psin[i1], self.deltaS_l[i1], "g", ls="--", lw=1, label="$\\delta_l$"
        )

        ax22.axvline(x=0.95, c="k", ls="--")
        ax22.axvline(x=0.995, c="k", ls="--")
        ax22.axvline(x=1.0, c="k", ls="--")

        ax22.axhline(y=0.0, c="k", ls="--")

        ax22.set_xlabel("$\\psi_n$")
        ax22.set_ylabel("$\\delta$")
        # ax22.set_ylim([np.min(self.deltaS[i1]),np.max(self.deltaS[i1])])
        ax22.legend(loc="upper left", prop={"size": self.mainLegendSize})

        ax22.set_xlim([0, 1])

        l2 = [
            f"$\\delta_{{sep}}$ = {self.delta[i1]:.3f}",
            f"$\\delta_{{99.5}}$ = {self.delta_995[i1]:.3f}",
            f"$\\delta_{{95}}$ = {self.delta_950[i1]:.3f}",
        ]
        ax22.text(
            0.2,
            0.7,
            "\n".join(l2),
            color="k",
            fontsize=15,
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax22.transAxes,
        )

        GRAPHICStools.addDenseAxis(ax22)

    def plotHeating(self, fig=None):
        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(nrows=2, ncols=4, hspace=0.3, wspace=0.4)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[0, 2])
        ax4 = fig.add_subplot(grid[1, 0])
        ax5 = fig.add_subplot(grid[1, 1])
        ax6 = fig.add_subplot(grid[1, 2])

        ax7 = fig.add_subplot(grid[0, 3])
        ax8 = fig.add_subplot(grid[1, 3])

        # ICRF
        ax1.plot(self.t, self.PichT, "r", ls="-", lw=2, label="$P_{ICH}$")
        ax1.plot(self.t, self.PeichT, "b", ls="-", lw=1, label="$P_{ICH,e}$")
        ax1.plot(self.t, self.PiichT, "g", ls="-", lw=1, label="$P_{ICH,i}$")
        ax1.plot(self.t, self.PeichT + self.PiichT, "y", ls="--", lw=1, label="$P_{ICH,e+i}$")

        ax1.plot(self.t, self.PichT_min, "r", ls="--", lw=1, label="$P_{min}$")
        ax1.plot(self.t, self.PeichT_dir, "r", ls="-.", lw=1, label="$P_{dir,e}$")
        ax1.plot(self.t, self.PiichT_dir, "r", ls=":", lw=1, label="$P_{dir,i}$")

        ax1.plot(self.t, self.PichTOrbLoss, "c", ls="-.", lw=1, label="$P_{loss,orb}$")
        ax1.plot(self.t, self.PichT_MC, "k", ls="-.", lw=1, label="$P_{MC}$")
        ax1.plot(self.t, self.PfichT_dir, "k", ls=":", lw=1, label="$P_{dir,f}$")

        GRAPHICStools.addDenseAxis(ax1)

        ax1.set_ylabel("Power (MW)")
        ax1.set_title("ICRF")
        # ax1.legend(loc='best')
        GRAPHICStools.addLegendApart(ax1, ratio=0.8, size=self.mainLegendSize)

        ax1.set_xlabel("Time (s)")

        # ECRF
        ax2.plot(self.t, self.PechT_delivered, "r", ls="-", lw=2, label="$P_{ECH}$ in")
        ax2.plot(self.t, self.PechT, "r", ls="-", lw=1, label="$P_{ECH}$")
        ax2.plot(self.t, self.PechT_check, "y", ls="--", lw=1, label="check (int)")

        ax2.set_title("ECRF")
        GRAPHICStools.addLegendApart(ax2, ratio=0.8, size=self.mainLegendSize)
        GRAPHICStools.addDenseAxis(ax2)

        ax2.set_xlabel("Time (s)")

        # NBI
        ax3.plot(self.t, self.PnbiINJ, "r", ls="-", lw=2, label="$P_{NBI}$ in")
        ax3.plot(self.t, self.PnbiT, "r", ls="-", lw=1, label="$P_{NBI}$")
        ax3.plot(self.t, self.PnbieT, "b", ls="-", lw=1, label="$P_{NBI,e}$")
        ax3.plot(self.t, self.PnbiiT, "g", ls="-", lw=1, label="$P_{NBI,i}$")
        ax3.plot(self.t, self.PnbihT, "k", ls="-", lw=1, label="$P_{NBI,th.f}$")
        ax3.plot(
            self.t, self.PowerOut_other, "c", ls="-", lw=1, label="$P_{NBI,lost}$"
        )  # cx+lim
        ax3.plot(self.t, self.PnbirT, "m", ls="-", lw=1, label="$P_{NBI,rot}$")
        ax3.plot(
            self.t, self.PnbiStored, "chocolate", ls="-", lw=1, label="$P_{NBI,stored}$"
        )

        ax3.plot(
            self.t,
            self.PnbiStored
            + self.PnbieT
            + self.PnbiiT
            + self.PnbihT
            + self.PnbirT
            + self.PnbilT
            + self.PnbicT,
            "y",
            ls="--",
            lw=1,
            label="check (sum)",
        )

        ax3.set_title("NBI")
        GRAPHICStools.addDenseAxis(ax3)
        GRAPHICStools.addLegendApart(ax3, ratio=0.8, size=self.mainLegendSize)

        ax3.set_xlabel("Time (s)")

        # Alphas
        ax7.plot(self.t, self.PfusT, "r", ls="-", lw=2, label="$P_{fus}$")
        ax7.plot(self.t, self.PfuseT, "b", ls="-", lw=1, label="$P_{fus,e}$")
        ax7.plot(self.t, self.PfusiT, "g", ls="-", lw=1, label="$P_{fus,i}$")

        ax7.set_title("ALPHAS")
        GRAPHICStools.addDenseAxis(ax7)
        GRAPHICStools.addLegendApart(ax7, ratio=0.8, size=self.mainLegendSize)

        ax7.set_xlabel("Time (s)")

        # Ion heating
        ax4.plot(self.t, self.PiheatT, "r", ls="-", lw=2, label="IHEAT")
        ax4.plot(
            self.t, self.utilsiliarPower_ions, "g", ls="-", lw=1, label="$P_{aux,i}$"
        )
        ax4.plot(self.t, self.PcompiT, "k", ls="-", lw=1, label="$P_{compr,i}$")
        ax4.plot(self.t, self.PfusiT, "m", ls="-", lw=1, label="$P_{fus,i}$")

        ax4.plot(
            self.t,
            self.utilsiliarPower_ions + self.PcompiT + self.PfusiT,
            "y",
            ls="--",
            lw=1,
            label="check (sum)",
        )

        ax4.set_title("Ion Heating")
        GRAPHICStools.addDenseAxis(ax4)
        GRAPHICStools.addLegendApart(ax4, ratio=0.8, size=self.mainLegendSize)

        ax4.set_ylabel("Power (MW)")
        ax4.set_xlabel("Time (s)")

        # Electron heating
        ax5.plot(self.t, self.PeheatT, "r", ls="-", lw=2, label="EHEAT")
        ax5.plot(
            self.t, self.utilsiliarPower_elec, "g", ls="-", lw=1, label="$P_{aux,e}$"
        )
        ax5.plot(self.t, self.PohT, "g", ls="--", lw=1, label="$P_{aux,e,OH}$")
        ax5.plot(self.t, self.PcompeT, "k", ls="-", lw=1, label="$P_{compr,e}$")
        ax5.plot(self.t, self.PfuseT, "m", ls="-", lw=1, label="$P_{fus,e}$")
        ax5.plot(
            self.t,
            self.utilsiliarPower_elec + self.PcompeT + self.PfuseT,
            "c",
            ls="--",
            lw=1,
            label="check (sum)",
        )

        ax5.set_title("Electron Heating")
        GRAPHICStools.addDenseAxis(ax5)
        GRAPHICStools.addLegendApart(ax5, ratio=0.8, size=self.mainLegendSize)
        ax5.set_xlabel("Time (s)")

        # Total heating
        ax6.plot(self.t, self.Ptot, "r", ls="-", lw=2, label="PL2HTOT")
        ax6.plot(
            self.t,
            self.utilsiliarPower_ions + self.utilsiliarPower_elec,
            "g",
            ls="-",
            lw=1,
            label="$P_{aux}$",
        )
        ax6.plot(
            self.t, self.PcompeT + self.PcompiT, "k", ls="-", lw=1, label="$P_{compr}$"
        )
        ax6.plot(self.t, self.PfusT, "m", ls="-", lw=1, label="$P_{fus}$")
        ax6.plot(
            self.t,
            self.utilsiliarPower_ions
            + self.utilsiliarPower_elec
            + self.PcompeT
            + self.PcompiT
            + self.PfusT,
            "y",
            ls="--",
            lw=1,
            label="check (sum)",
        )

        ax6.plot(
            self.t, self.PowerIn_other, "k", ls="-", lw=1, label="$P_{nbi,rot+th.f}$"
        )

        ax6.set_title("Ions+Electrons")
        GRAPHICStools.addDenseAxis(ax6)
        GRAPHICStools.addLegendApart(ax6, ratio=0.8, size=self.mainLegendSize)
        ax6.set_xlabel("Time (s)")

        # Total loss
        TOT = (
            self.P_LCFS
            + self.PradT
            + self.PcxT
            + self.PowerOut_other
            + self.PnbiLoss_shine
            + self.PichTOrbLoss
        )
        ax8.plot(self.t, TOT, "r", ls="-", lw=2, label="Total")

        ax8.plot(self.t, self.P_LCFS, "c", ls="-", lw=1, label="$P_{LCFS}$")
        ax8.plot(
            self.t, self.PradT + self.PcxT, "g", ls="-", lw=1, label="$P_{rad}+P_{cx}$"
        )
        ax8.plot(
            self.t,
            self.PowerOut_other + self.PnbiLoss_shine + self.PichTOrbLoss,
            "m",
            ls="-",
            lw=1,
            label="$P_{fast,lost+sh.th.}$",
        )

        Gains = self.GainT + self.GainminT
        ax8.plot(
            self.t,
            self.PnbiINJ + self.PechT + self.PichT + self.PfusT + self.PohT - Gains,
            "y",
            ls="--",
            lw=1,
            label="check (all in -Gain)",
        )

        ax8.set_title("Losses")
        GRAPHICStools.addDenseAxis(ax8)
        GRAPHICStools.addLegendApart(ax8, ratio=0.8, size=self.mainLegendSize)
        ax8.set_xlabel("Time (s)")

    def plotRadialPower(self, ax=None, time=None, fig=None, figCum=None):
        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.2)
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1], sharex=ax1)
        ax3 = fig.add_subplot(grid[1, 0], sharex=ax1)
        ax4 = fig.add_subplot(grid[1, 1], sharex=ax1)

        if figCum is None:
            figCum = plt.figure()
        grid = plt.GridSpec(1, 3, hspace=0.3, wspace=0.2)
        ax1C = figCum.add_subplot(grid[0, 0])
        ax2C = figCum.add_subplot(grid[0, 1], sharex=ax1C, sharey=ax1C)
        ax3C = figCum.add_subplot(grid[0, 2], sharex=ax1C)

        if time is None:
            i1 = self.ind_saw
        else:
            i1 = np.argmin(np.abs(self.t - time))
        timeReq = self.t[i1]

        # --------- Electrons Sources

        ax1.plot(
            self.x_lw, self.Peich[i1, :], "orange", ls="-", lw=1, label="$P_{ICH,e}$"
        )
        ax1.plot(self.x_lw, self.Pech[i1, :], "b", ls="-", lw=1, label="$P_{ECH,e}$")
        ax1.plot(self.x_lw, self.Pnbie[i1, :], "g", ls="-", lw=1, label="$P_{NBI,e}$")
        ax1.plot(self.x_lw, self.Pfuse[i1, :], "m", ls="-", lw=1, label="$P_{fus,e}$")
        ax1.plot(
            self.x_lw, self.Pcompe[i1, :], "k", ls="-", lw=1, label="$P_{compr,e}$"
        )
        ax1.plot(self.x_lw, self.Poh[i1, :], "c", ls="-", lw=1, label="$P_{OH}$")

        ax1.plot(self.x_lw, self.Pe[i1, :], "r", ls="-", lw=2, label="$P_{e,in}$ total")

        ax1.plot(
            self.x_lw, self.Peheat[i1, :], "y", ls="--", lw=1, label="check (EHEAT)"
        )

        ax1.axhline(y=0, ls="--", c="k", lw=1)

        ax1.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax1)

        ax1.legend(loc="best", prop={"size": self.mainLegendSize})
        ax1.set_title("Electrons Power Sources")
        ax1.set_ylabel("Power ($MWm^{-3}$)")
        ax1.set_xlabel("$\\rho_N$")
        ax1.tick_params(labelbottom=True)

        # --------- Ion Sources

        ax2.plot(
            self.x_lw, self.Piich[i1, :], "orange", ls="-", lw=1, label="$P_{ICH,i}$"
        )
        ax2.plot(self.x_lw, self.Pnbii[i1, :], "g", ls="-", lw=1, label="$P_{NBI,i}$")
        ax2.plot(self.x_lw, self.Pfusi[i1, :], "m", ls="-", lw=1, label="$P_{fus,i}$")
        ax2.plot(
            self.x_lw, self.Pcompi[i1, :], "k", ls="-", lw=1, label="$P_{compr,i}$"
        )

        ax2.plot(self.x_lw, self.Pi[i1, :], "r", ls="-", lw=2, label="$P_{i,in}$ total")

        ax2.plot(
            self.x_lw, self.Piheat[i1, :], "y", ls="--", lw=1, label="check (IHEAT)"
        )

        ax2.axhline(y=0, ls="--", c="k", lw=1)

        GRAPHICStools.addDenseAxis(ax2)

        ax2.legend(loc="best", prop={"size": self.mainLegendSize})
        ax2.set_title("Ions Power Sources")
        ax2.set_ylabel("Power ($MWm^{-3}$)")
        ax2.set_xlabel("$\\rho_N$")
        ax2.tick_params(labelbottom=True)

        # --------- Electrons Sinks

        ax3.plot(self.x_lw, self.Prad[i1, :], "orange", ls="-", lw=1, label="$P_{rad}$")
        ax3.plot(
            self.x_lw, self.Te_divQ[i1, :], "c", ls="-", lw=1, label="$\\nabla Q_e$"
        )

        # ax3.plot(self.x_lw,self.Prad[i1,:]+self.Te_divQ[i1,:],'red',ls='-',lw=3,label='$P_{losses,e}$')

        ax3.plot(self.x_lw, self.Pei[i1, :], "b", ls="-", lw=1, label="$Q_{e->i}$")

        # ax3.plot(self.x_lw,self.Peheat[i1,:]-self.Gaine[i1,:]-self.Pei[i1,:],'y',ls='--',lw=2,label='check (EHEAT-gain-$Q_{e->i}$)')

        ax3.plot(
            self.x_lw,
            self.Prad[i1, :] + self.Te_divQ[i1, :] + self.Pei[i1, :],
            "r",
            lw=2,
            label="$P_{e,out}$ total",
        )
        ax3.plot(
            self.x_lw,
            self.Pe[i1, :] - self.Gaine[i1, :],
            "y",
            ls="--",
            lw=1,
            label="$P_{e,in}-dW/dt$",
        )

        ax3.axhline(y=0, ls="--", c="k", lw=1)

        GRAPHICStools.addDenseAxis(ax3)

        ax3.legend(loc="best", prop={"size": self.mainLegendSize})
        ax3.set_title("Electrons Power Sinks")
        ax3.set_ylabel("Power ($MWm^{-3}$)")
        ax3.set_xlabel("$\\rho_N$")

        # --------- Ions Sinks

        ax4.plot(self.x_lw, self.Pcx[i1, :], "orange", ls="-", lw=1, label="$P_{cx}$")
        ax4.plot(
            self.x_lw, self.Ti_divQ[i1, :], "c", ls="-", lw=1, label="$\\nabla Q_i$"
        )

        # ax4.plot(self.x_lw,self.Pcx[i1,:]+self.Ti_divQ[i1,:],'red',ls='-',lw=3,label='$P_{losses,i}$')

        ax4.plot(self.x_lw, -self.Pei[i1, :], "b", ls="-", lw=1, label="$Q_{i->e}$")

        # ax4.plot(self.x_lw,self.Piheat[i1,:]+self.Pi_others[i1,:]-self.Gaini[i1,:]+self.Pei[i1,:],'y',ls='--',lw=2,label='check (IHEAT+QROT+PBTH- gain-$Q_{i->e}$)')

        ax4.plot(
            self.x_lw,
            self.Pcx[i1, :] + self.Ti_divQ[i1, :] - self.Pei[i1, :],
            "r",
            lw=2,
            label="$P_{i,out}$ total",
        )
        ax4.plot(
            self.x_lw,
            self.Pi[i1, :] - self.Gaini[i1, :],
            "y",
            ls="--",
            lw=1,
            label="$P_{i,in}-dW/dt$",
        )

        ax4.axhline(y=0, ls="--", c="k", lw=1)

        GRAPHICStools.addDenseAxis(ax4)

        ax4.legend(loc="best", prop={"size": self.mainLegendSize})
        ax4.set_title("Ions Power Sinks")
        ax4.set_ylabel("Power ($MWm^{-3}$)")
        ax4.set_xlabel("$\\rho_N$")

        # ----------------------------------

        # Cumulative
        ax = ax1C
        ax.plot(self.x_lw, self.PeheatT_cum[i1, :], "r", ls="-", lw=3, label="$P_{IN}$")
        ax.plot(self.x_lw, self.PradT_cum[i1, :], "b", ls="-", lw=3, label="$P_{rad}$")
        ax.plot(self.x_lw, self.PeiT_cum[i1, :], "g", ls="-", lw=3, label="$P_{ei}$")
        ax.plot(self.x_lw, self.GaineT_cum[i1, :], "c", ls="-", lw=3, label="$dW_e/dt$")
        ax.plot(self.x_lw, self.Pe_obs_cum[i1, :], "m", ls="-", lw=3, label="$Q_e$")
        ax.plot(
            self.x_lw,
            self.PeheatT_cum[i1, :]
            - self.PradT_cum[i1, :]
            - self.PeiT_cum[i1, :]
            - self.GaineT_cum[i1, :],
            "y",
            ls="--",
            lw=2,
            label="$Q_e$ check",
        )
        ax.set_ylabel("Cumulative Power ($MW$)")
        ax.set_title("Electrons")
        ax.set_xlabel("$\\rho_N$")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax.axhline(y=0, ls="--", lw=1, c="k")

        GRAPHICStools.addDenseAxis(ax)

        ax = ax2C
        ax.plot(self.x_lw, self.PiheatT_cum[i1, :], "r", ls="-", lw=3, label="$P_{IN}$")
        ax.plot(self.x_lw, self.PcxT_cum[i1, :], "b", ls="-", lw=3, label="$P_{cx}$")
        ax.plot(self.x_lw, self.PeiT_cum[i1, :], "g", ls="-", lw=3, label="$P_{ei}$")
        ax.plot(self.x_lw, self.GainiT_cum[i1, :], "c", ls="-", lw=3, label="$dW_i/dt$")
        ax.plot(self.x_lw, self.Pi_obs_cum[i1, :], "m", ls="-", lw=3, label="$Q_i$")
        ax.plot(
            self.x_lw,
            self.PiheatT_cum[i1, :]
            - self.PcxT_cum[i1, :]
            + self.PeiT_cum[i1, :]
            - self.GainiT_cum[i1, :],
            "y",
            ls="--",
            lw=2,
            label="$Q_i$ check",
        )
        ax.set_ylabel("Cumulative Power ($MW$)")
        ax.set_title("Ions")
        ax.set_xlabel("$\\rho_N$")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax.axhline(y=0, ls="--", lw=1, c="k")

        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        ax = ax3C
        ax.plot(
            self.x_lw,
            self.Pi_obs_cum[i1, :] / self.Pe_obs_cum[i1, :],
            "m",
            ls="-",
            lw=3,
            label="Transport",
        )
        ax.plot(
            self.x_lw,
            self.PiheatT_cum[i1, :] / self.PeheatT_cum[i1, :],
            "r",
            ls="-",
            lw=3,
            label="Input Power",
        )
        ax.set_ylabel("$Q_i/Q_e$")
        ax.set_title("Power ratios")
        ax.set_xlabel("$\\rho_N$")

        ax.axhline(y=1, ls="--", lw=1, c="k")

        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

    def plotConvSolver_x(
        self,
        time,
        avt=0.0,
        ax=None,
        alsoParticle=True,
        alsoTR=True,
        colorStart=0,
        xlab=True,
    ):
        colorsC = ["r", "b", "g", "orange", "m", "k"]

        if ax is None:
            fig, ax = plt.subplots()

        if avt == 0.0:
            indTime = np.argmin(np.abs(time - self.t))
            qe_obs = self.qe_obs[indTime, :]
            qe_tr = self.qe_tr[indTime, :]
            qi_obs = self.qi_obs[indTime, :]
            qi_tr = self.qi_tr[indTime, :]
            Ge_obs = self.Ge_obs[indTime, :]
            Ge_tr = self.Ge_tr[indTime, :]

            Ce_obs = self.Ce_obs[indTime, :]
            Ce_tr = self.Ce_tr[indTime, :]

        else:
            indTime1 = np.argmin(np.abs(time - avt - self.t))
            indTime2 = np.argmin(np.abs(time + avt - self.t))
            indTime = np.argmin(np.abs(time - self.t))
            qe_obs = np.mean(self.qe_obs[indTime1:indTime2, :], 0)
            qe_tr = np.mean(self.qe_tr[indTime1:indTime2, :], 0)
            qi_obs = np.mean(self.qi_obs[indTime1:indTime2, :], 0)
            qi_tr = np.mean(self.qi_tr[indTime1:indTime2, :], 0)
            Ge_obs = np.mean(self.Ge_obs[indTime1:indTime2, :], 0)
            Ge_tr = np.mean(self.Ge_tr[indTime1:indTime2, :], 0)
            Ce_obs = np.mean(self.Ce_obs[indTime1:indTime2, :], 0)
            Ce_tr = np.mean(self.Ce_tr[indTime1:indTime2, :], 0)

        ax.plot(self.x_lw, qe_obs, lw=2, c=colorsC[colorStart], label="$q_e$ PB")
        if alsoTR and np.abs(qe_tr).sum() > self.eps00:
            ax.plot(
                self.x_lw,
                qe_tr,
                lw=2,
                c=colorsC[colorStart],
                ls="--",
                label="$q_e$ MODEL",
            )
        ax.plot(self.x_lw, qi_obs, lw=2, c=colorsC[colorStart + 1], label="$q_i$ PB")
        if alsoTR and np.abs(qi_tr).sum() > self.eps00:
            ax.plot(
                self.x_lw,
                qi_tr,
                lw=2,
                c=colorsC[colorStart + 1],
                ls="--",
                label="$q_i$ MODEL",
            )
        ax.plot(
            self.x_lw, Ce_obs, lw=2, c=colorsC[colorStart + 2], label="$q_{conv}$ PB"
        )
        if alsoTR and np.abs(qi_tr).sum() > self.eps00:
            ax.plot(
                self.x_lw,
                Ce_tr,
                lw=2,
                c=colorsC[colorStart + 3],
                ls="--",
                label="$q_{conv}$ MODEL",
            )

        if xlab:
            ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$Q$ ($MW/m^2$)")
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})
        # ax.set_ylim(bottom=0)

        if alsoParticle:
            ax1 = ax.twinx()
            ax1.plot(
                self.x_lw,
                Ge_obs,
                lw=2,
                c=colorsC[colorStart + 3],
                label="$\\Gamma_e$ PB",
            )
            if alsoTR and np.abs(Ge_tr).sum() > self.eps00:
                ax1.plot(
                    self.x_lw,
                    Ge_tr,
                    lw=2,
                    c=colorsC[colorStart + 2],
                    ls="--",
                    label="$\\Gamma_e$ MODEL",
                )
            ax1.set_ylabel("$\\Gamma_e$ ($10^{20}/s/m^2$)")
            ax1.legend(loc="lower right")
            # ax1.set_ylim(bottom=0)

        return indTime

    def plotConvSolver_t(
        self, rho, ax=None, meanLim=False, alsoParticle=True, leg=True
    ):
        if ax is None:
            fig, ax = plt.subplots()

        indx = np.argmin(np.abs(rho - self.x_lw))

        ax.plot(self.t, self.qe_obs[:, indx], lw=1, c="r", label="qe PB")
        if np.abs(self.qe_tr[:, indx]).sum() > self.eps00:
            ax.plot(self.t, self.qe_tr[:, indx], lw=1, c="r", ls="--", label="qe MODEL")
        ax.plot(self.t, self.qi_obs[:, indx], lw=1, c="b", label="qi PB")
        if np.abs(self.qi_tr[:, indx]).sum() > self.eps00:
            ax.plot(self.t, self.qi_tr[:, indx], lw=1, c="b", ls="--", label="qi MODEL")
        ax.plot(self.t, self.Ce_obs[:, indx], lw=1, c="g", label="Ce PB")
        if np.abs(self.Ce_tr[:, indx]).sum() > self.eps00:
            ax.plot(
                self.t, self.Ce_tr[:, indx], lw=1, c="orange", ls="--", label="Ce MODEL"
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$Q$ ($MW/m^2$)")
        if leg:
            ax.legend()

        if alsoParticle:
            ax1 = ax.twinx()
            ax1.plot(self.t, self.Ge_obs[:, indx], lw=1, c="g", label="$\\Gamma_e$ PB")
            if np.abs(self.Ge_tr[:, indx]).sum() > self.eps00:
                ax1.plot(
                    self.t,
                    self.Ge_tr[:, indx],
                    lw=1,
                    c="orange",
                    ls="--",
                    label="$\\Gamma_e$ MODEL",
                )
            ax1.set_ylabel("$\\Gamma_e$ ($10^{20}/s/m^2$)")
            if leg:
                ax1.legend(loc="lower right")

        ax.set_title(f"Time traces at rho={rho}")

        return indx

    def plotCorrelation(self, ax, ypos=0, time=1.0, mult=1.0, rhos=False):
        ind = np.argmin(np.abs(time - self.t))

        if rhos:
            var = self.rhos
        else:
            var = self.rhoi

        leftextremes = self.rmin[ind, :] - var[ind, :] * mult

        i = 0

        while i < len(self.rmin[ind, :]):
            cc = plt.Circle(
                (self.rmin[ind, i] * 100.0, ypos),
                var[ind, i] * 100.0 * mult,
                alpha=0.5,
                facecolor="none",
                edgecolor="b",
            )
            ax.add_artist(cc)

            rightextreme = self.rmin[ind, i] + var[ind, i] * mult
            if np.argmin(np.abs(leftextremes - rightextreme)) > i:
                i = np.argmin(np.abs(leftextremes - rightextreme))
            else:
                break

    def plotT(self, axs=None, time=None, rhos=[0.0, 0.35, 0.7]):
        if axs is None:
            fig, axs = plt.subplots(ncols=2, sharex=False, sharey=True, figsize=(14, 6))

        # Temperature profiles
        ax = axs[0]
        i1, i2 = prepareTimingsSaw(time, self)
        ax.plot(
            self.x_lw,
            self.Te[i1, :],
            lw=3,
            c="r",
            label=f"$T_e$, @ {self.t[i1]:.3f}s",
        )
        ax.plot(
            self.x_lw,
            self.Te[i2, :],
            lw=3,
            ls="--",
            c="r",
            label=f"$T_e$, @ {self.t[i2]:.3f}s",
        )
        # ax.axhline(y = self.Te_avol[i1],c= 'r',lw=3,alpha=0.7,label='$T_e$ vol.av.')
        ax.plot(self.x_lw, self.Ti[i1, :], lw=3, c="b", label="$T_i$")
        ax.plot(self.x_lw, self.Ti[i2, :], lw=3, ls="--", c="b")
        ax.plot(self.x_lw, self.TZ[i1, :], lw=1, c="g", label="$T_Z$")
        ax.plot(self.x_lw, self.TZ[i2, :], lw=1, ls="--", c="g")
        # ax.axhline(y = self.Ti_avol[i1],c= 'b',lw=3,alpha=0.7)
        # ax.axhline(y = self.TZ_avol[i1],c= 'g',lw=1,alpha=0.7)
        addDetailsRho(ax, label="$T_e$, $T_i$ (keV)")  # ,title='Temperature Profiles')
        GRAPHICStools.addDenseAxis(ax)

        # Temperature trace
        ax = axs[1]
        ax, i = preparePlotTime(rhos, self, ax)
        for it in i:
            ax.plot(
                self.t, self.Te[:, it], lw=1, c="r"
            )  # ,label='$T_e$, @ $\\rho_N=${0:.1f}'.format(self.x_lw[it]))
        for it in i:
            ax.plot(self.t, self.Ti[:, it], lw=1, c="b")
        for it in i:
            ax.plot(self.t, self.TZ[:, it], lw=1, c="g")
        ax.plot(self.t, self.Te_avol, c="r", lw=4, alpha=1.0, label="$T_e$ vol.av.")
        ax.plot(self.t, self.Ti_avol, c="b", lw=4, alpha=1.0, label="$T_i$ vol.av.")
        ax.plot(self.t, self.TZ_avol, c="g", lw=1, alpha=1.0, label="$T_Z$ vol.av.")
        addDetailsTime(ax, label="$T_e$, $T_i$ (keV)")  # ,title='Temperature Traces')
        GRAPHICStools.addDenseAxis(ax)

        # Combinations
        for it in i:
            axs[0].axvline(x=self.x_lw[it], c="m", lw=1, ls="--")
        axs[1].axvline(x=self.t[i1], c="m", lw=1, ls="--")
        axs[1].axvline(x=self.t[i2], c="m", lw=1, ls="--")

    def plotN(self, axs=None, time=None, rhos=[0.0], complete=True):
        if axs is None:
            fig, axs = plt.subplots(ncols=2, sharex=False, sharey=True, figsize=(14, 6))

        # profiles
        ax = axs[0]
        i1, i2 = prepareTimingsSaw(time, self)
        # Electrons
        ax.plot(
            self.x_lw,
            self.ne[i1, :],
            lw=3,
            c="r",
            label=f"$n_e$, @ {self.t[i1]:.3f}s",
        )
        ax.plot(
            self.x_lw,
            self.ne[i2, :],
            lw=3,
            ls="--",
            c="r",
            label=f"$n_e$, @ {self.t[i2]:.3f}s",
        )
        # ax.axhline(y = self.ne_avol[i1],c= 'r',lw=3,alpha=0.7,label='$n_e$ vol.av.')
        # ax.axhline(y = self.ne_l[i1],c= 'r',ls=':',lw=3,alpha=0.7,label='$n_e$ lin.av.')
        # Main ions
        ax.plot(self.x_lw, self.nmain[i1, :], lw=3, c="b", label="$n_D+n_T$")
        ax.plot(self.x_lw, self.nmain[i2, :], lw=3, ls="--", c="b")
        # ax.axhline(y = self.nmain_avol[i1],c= 'b',lw=3,alpha=0.7)

        if complete:
            # Impurities
            fact = 20.0
            ax.plot(
                self.x_lw,
                self.nZ[i1, :] * fact,
                lw=3,
                c="g",
                label=f"$n_Z$x{fact}",
            )
            ax.plot(self.x_lw, self.nZ[i2, :] * fact, lw=3, ls="--", c="g")
            # ax.axhline(y = self.nZ_avol[i1]*fact,c= 'g',lw=3,alpha=0.7)
            #  Minorities
            fact = 10.0
            ax.plot(
                self.x_lw,
                self.nmini[i1, :] * fact,
                lw=3,
                c="c",
                label=f"$n_{{min}}$x{fact}",
            )
            ax.plot(self.x_lw, self.nmini[i2, :] * fact, lw=3, ls="--", c="c")
            # ax.axhline(y = self.nmini_avol[i1]*fact,c= 'c',lw=3,alpha=0.7)
            #  Alphas
            fact = 10.0
            ax.plot(
                self.x_lw,
                self.nHe4[i1, :] * fact,
                lw=3,
                c="m",
                label=f"$n_{{He4}}$x{fact}",
            )
            ax.plot(self.x_lw, self.nHe4[i2, :] * fact, lw=3, ls="--", c="m")
            # ax.axhline(y = self.nHe4_avol[i1]*fact,c= 'm',lw=3,alpha=0.7)

        addDetailsRho(ax, label="$n$ ($10^{20}m^{-3}$)")  # ,title='Density Profiles')
        GRAPHICStools.addDenseAxis(ax)

        # traces
        ax = axs[1]
        ax, i = preparePlotTime(rhos, self, ax)
        # Electrons
        for it in i:
            ax.plot(
                self.t, self.ne[:, it], lw=1, c="r"
            )  # ,label='$T_e$, @ $\\rho_N=${0:.1f}'.format(self.x_lw[it]))
        ax.plot(self.t, self.ne_avol, c="r", lw=4, alpha=1.0, label="$n_e$ vol.av.")
        ax.plot(
            self.t, self.ne_l, c="r", ls=":", lw=4, alpha=1.0, label="$n_e$ lin.av."
        )
        # Main ions
        for it in i:
            ax.plot(self.t, self.nmain[:, it], lw=1, c="b")
        ax.plot(self.t, self.nmain_avol, c="b", lw=4, alpha=1.0)

        if complete:
            # Impurities
            fact = 20.0
            for it in i:
                ax.plot(
                    self.t, self.nZ[:, it] * fact, lw=1, c="g"
                )  # ,label='$T_e$, @ $\\rho_N=${0:.1f}'.format(self.x_lw[it]))
            ax.plot(self.t, self.nZ_avol * fact, c="g", lw=4, alpha=1.0)
            #  Minorities
            fact = 10.0
            for it in i:
                ax.plot(
                    self.t, self.nmini[:, it] * fact, lw=1, c="c"
                )  # ,label='$T_e$, @ $\\rho_N=${0:.1f}'.format(self.x_lw[it]))
            ax.plot(self.t, self.nmini_avol * fact, c="c", lw=4, alpha=1.0)
            #  Alphas
            fact = 10.0
            for it in i:
                ax.plot(
                    self.t, self.nHe4[:, it] * fact, lw=1, c="m"
                )  # ,label='$T_e$, @ $\\rho_N=${0:.1f}'.format(self.x_lw[it]))
            ax.plot(self.t, self.nHe4_avol * fact, c="m", lw=4, alpha=1.0)

        addDetailsTime(ax, label="$n$ ($10^{20}m^{-3}$)")  # ,title='Density Traces')
        GRAPHICStools.addDenseAxis(ax)

        # Combinations
        for it in i:
            axs[0].axvline(x=self.x_lw[it], c="m", lw=1, ls="--")
        axs[1].axvline(x=self.t[i1], c="m", lw=1.0, ls="--")
        axs[1].axvline(x=self.t[i2], c="m", lw=1.0, ls="--")

    def plotProfiles(self, fig=None, time=None):
        # fig,axs = plt.subplots(nrows=2,ncols=2, sharex=False, sharey=True, figsize=(14,14))

        if fig is None:
            fig = plt.figure(figsize=(14, 8))
        axTr = fig.add_subplot(2, 2, 1)
        axTt = fig.add_subplot(2, 2, 2)
        axnr = fig.add_subplot(2, 2, 3)
        axnt = fig.add_subplot(2, 2, 4)

        self.plotT(axs=[axTr, axTt], time=time)
        self.plotN(axs=[axnr, axnt], time=time)

        return axTr, axTt, axnr, axnt

    def plotPulse(self, fig=None, time=None, avt=0.05, rhoRange=[0.3, 0.7]):
        if time is None:
            try:
                time = self.t[self.ind_sawAll[1]]
            except:
                time = self.t[self.ind_saw]

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(3, 3, hspace=0.4, wspace=0.3)

        # Posi
        positions = np.arange(rhoRange[0], rhoRange[1] + 0.1, 0.1)
        cols, _ = GRAPHICStools.colorTableFade(
            len(positions), startcolor="r", endcolor="b"
        )

        ax1 = fig.add_subplot(grid[0, 0])
        GRAPHICStools.addDenseAxis(ax1)
        ax2 = fig.add_subplot(grid[0, 1], sharex=ax1)
        GRAPHICStools.addDenseAxis(ax2)
        ax3 = fig.add_subplot(grid[1, 0], sharex=ax1)
        GRAPHICStools.addDenseAxis(ax3)
        ax4 = fig.add_subplot(grid[1, 1], sharex=ax1)
        GRAPHICStools.addDenseAxis(ax4)
        ax4b = fig.add_subplot(grid[0, 2])
        GRAPHICStools.addDenseAxis(ax4b)
        ax5 = fig.add_subplot(grid[1, 2])
        GRAPHICStools.addDenseAxis(ax5)

        ax6 = fig.add_subplot(grid[2, 0])
        GRAPHICStools.addDenseAxis(ax6)
        ax7 = fig.add_subplot(grid[2, 1])
        GRAPHICStools.addDenseAxis(ax7)
        ax8 = fig.add_subplot(grid[2, 2])
        GRAPHICStools.addDenseAxis(ax8)

        title = "rho = "
        for cont, i in enumerate(positions):
            ix, it = np.argmin(np.abs(self.x_lw - i)), np.argmin(np.abs(self.t - time))
            ax = ax1
            ax.plot(self.t, self.Te[:, ix], c=cols[cont])
            ax.axhline(y=self.Te[it, ix], ls="--", lw=0.1, c=cols[cont])
            ax = ax2
            ax.plot(self.t, self.Ti[:, ix], c=cols[cont])
            ax.axhline(y=self.Ti[it, ix], ls="--", lw=0.1, c=cols[cont])
            ax = ax3
            ax.plot(self.t, self.ne[:, ix], c=cols[cont])
            ax.axhline(y=self.ne[it, ix], ls="--", lw=0.1, c=cols[cont])
            ax = ax4
            ax.plot(self.t, self.qe_obs[:, ix], c=cols[cont])
            ax.axhline(y=self.qe_obs[it, ix], ls="--", lw=0.1, c=cols[cont])
            ax = ax7
            ax.plot(self.t, self.aLTe[:, ix], c=cols[cont])
            ax.axhline(y=self.aLTe[it, ix], ls="--", lw=0.1, c=cols[cont])
            ax = ax6
            ax.plot(self.t, self.aLTi[:, ix], c=cols[cont])
            ax.axhline(y=self.aLTi[it, ix], ls="--", lw=0.1, c=cols[cont])

            title += f"{i:.1f}, "

        ax = ax1.twinx()
        ix = np.argmin(np.abs(self.x_lw - rhoRange[0]))
        ax.plot(self.t, self.PlhT, c="g")
        ax.set_ylim([0, self.PlhT.max() * 5])

        ax1.set_ylabel("Te (keV)")
        ax2.set_ylabel("Ti (keV)")
        ax3.set_ylabel("ne ($10^{20}m^{-3}$)")
        ax4.set_ylabel("qe ($MW/m^2$)")
        ax7.set_ylabel("$a/L_{Te}$")
        ax6.set_ylabel("$a/L_{Ti}$")

        for ax in [ax1, ax2, ax3, ax4, ax7, ax6]:
            ax.set_xlabel("Time (s)")
            ax.set_xlim([time - avt, time + avt])
            ax.axvline(x=time, ls="--", lw=1, c="k")

            # GRAPHICStools.autoscale_y(ax)

        ax1.set_title(title[:-2], fontsize=7)

        # Chi Pert

        if "Creely_ChiPert" not in self.__dict__ or self.Creely_ChiPert is None:
            self.getChiPertPulse(pulseTimes=[time], rhoRange=rhoRange, timeRange=avt)
        self.chiCalc.prepareData(axs=[ax4b, ax5])
        ax4b.axhline(y=0.0, ls="--", lw=1, c="k")
        ax5.axhline(y=0.0, ls="--", lw=1, c="k")

        ax4b.text(
            0.4,
            0.8,
            f"chi = {self.Creely_ChiPert:.1f} m^2/s",
            fontweight="bold",
            color="b",
            fontsize=10,
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax4b.transAxes,
        )

        # Hysteresis
        ax = ax8
        it1 = np.argmin(np.abs(self.t - (time)))
        it2 = np.argmin(np.abs(self.t - (time + avt)))
        for cont, i in enumerate(positions):
            ix = np.argmin(np.abs(self.x_lw - i))
            ax.plot(self.aLTe[it1:it2, ix], self.qe_obs[it1:it2, ix], c=cols[cont])

        ax.set_xlabel("$a/L_{Te}$")
        ax.set_ylabel("qe ($MW/m^2$)")

    def plotCPUperformance(self, fig=None, time=None):
        maxx = 0.89

        if fig is None:
            fig = plt.figure()

        if time is None:
            i1 = self.ind_saw
        else:
            i1 = np.argmin(np.abs(self.t - time))
        timeReq = self.t[i1]

        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        # ax3 = fig.add_subplot(grid[1,0])
        # ax4 = fig.add_subplot(grid[1,1])

        # -------------

        ax = ax1

        if self.cptim[-1] < 1.0:
            factor = 60.0
            lab = "min"
        else:
            factor = 1.0
            lab = "h"

        ax.plot(self.t, self.cptim*factor, lw=2, label="Total")
        ax.plot(self.t, self.cptim_pt*factor, lw=2, label="PT_SOLVER")
        ax.plot(self.t, self.cptim_out*factor, lw=2, label="OUT")
        ax.plot(
            self.t,
            self.cptim_geom + self.cptim_mhd + self.cptim_fsa,
            lw=2,
            label="EQ+MHD+FSA",
        )
        ax.plot(self.t, self.cptim_nubeam*factor, lw=2, label="NUBEAM")
        ax.plot(self.t, self.cptim_icrf*factor, lw=2, label="ICRF")
        ax.plot(self.t, self.cptim_fpp*factor, lw=2, label="FPP")

        ax.plot(
            self.t,
            (self.cptim + self.cptim_out)*factor,
            lw=2,
            ls="--",
            c="c",
            label="Total + OUT",
        )
        ax.plot(
            self.t,
            (self.cptim_icrf
            + self.cptim_fpp
            + self.cptim_nubeam
            + self.cptim_pt
            + self.cptim_out
            + self.cptim_geom
            + self.cptim_mhd
            + self.cptim_fsa)*factor,
            lw=2,
            ls="--",
            c="y",
            label="check",
        )

        ax.set_ylabel(f"Wall-time $t_{{CPU}}$ ({lab})")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)

        ax.legend(loc="upper left", prop={"size": 10})

        GRAPHICStools.addDenseAxis(ax)

        ax = ax2
        ax.axhline(y=(self.dt * 1000.0).min(), ls="--", c="k", lw=0.3)
        ax.axhline(y=(self.dt * 1000.0).max(), ls="--", c="k", lw=0.3)

        ax.scatter(self.t, self.dt * 1000.0, s=1, label="$\\Delta t$")
        ax.set_ylabel("Time-step $\\Delta t$ (ms)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)

        # ax.legend(loc='upper left',prop={'size':self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)

    def plotTransport(self, fig=None, time=None, rhos=[0.0], complete=True):
        maxx = 0.89

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 4, hspace=0.25, wspace=0.4)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[1, 0])
        ax4 = fig.add_subplot(grid[1, 1])
        ax5 = fig.add_subplot(grid[0, 2])
        ax6 = fig.add_subplot(grid[1, 2])

        ax7 = fig.add_subplot(grid[0, 3])
        ax8 = fig.add_subplot(grid[1, 3])

        if time is None:
            i1 = self.ind_saw
        else:
            i1 = np.argmin(np.abs(self.t - time))
        timeReq = self.t[i1]

        # -------------

        ax = ax1
        ax.plot(self.x_lw, self.qi_obs[i1, :] / self.qe_obs[i1, :], lw=2)
        ax.set_ylabel("$Q_i/Q_e$")
        ax.set_xlabel("$\\rho_N$")
        ax.set_xlim([0, maxx])
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        ax.axhline(y=1.0, ls="--", c="k", lw=1)

        GRAPHICStools.addDenseAxis(ax)

        # -------------

        ax = ax2
        ax.plot(self.x_lw, self.Chi_e[i1, :], lw=2, c="r", label="$\\chi_e$")
        ax.plot(self.x_lw, self.Chi_i[i1, :], lw=2, c="b", label="$\\chi_i$")
        ax.plot(self.x_lw, self.Chi_eff[i1, :], lw=2, c="g", label="$\\chi_{eff}$")
        ax.plot(
            self.x_lw,
            self.Chi_eff_check[i1, :],
            lw=1,
            ls="--",
            c="y",
            label="$\\chi_{eff}$ check",
        )
        ax.plot(self.x_lw, self.Chi_e_MITIM[i1, :], lw=1, c="c", label="$\\chi_e$ check")
        ax.set_ylabel("$\\chi$ ($m^2/s$)")
        ax.set_xlabel("$\\rho_N$")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax.set_xlim([0, maxx])
        GRAPHICStools.autoscale_y(ax)

        GRAPHICStools.addDenseAxis(ax)

        # -------------

        ax = ax5
        ax.plot(self.x_lw, self.Deff_e[i1, :], lw=2, c="orange", label="$D_{eff,e}$")

        ax.set_ylabel("$D_{eff}$ ($m^2/s$)")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax.set_xlim([0, maxx])
        GRAPHICStools.autoscale_y(ax)

        ax.axhline(y=0.0, lw=2, c="k", ls="--")

        GRAPHICStools.addDenseAxis(ax)

        # -------------

        ax = ax4
        ax.plot(self.x_lw, self.qe_obs[i1, :], lw=2, c="r", ls="-", label="$q_e$ PB")
        ax.plot(
            self.x_lw,
            self.qe_MITIM[i1, :],
            lw=1,
            c="r",
            ls="--",
            label="$-n_e \\chi_e \\nabla_\\rho T_e |\\nabla\\rho|$",
        )

        ax.plot(self.x_lw, self.qi_obs[i1, :], lw=2, c="b", ls="-", label="$q_i$ PB")
        ax.plot(
            self.x_lw,
            self.qi_MITIM[i1, :],
            lw=1,
            c="b",
            ls="--",
            label="$-n_i \\chi_i \\nabla_\\rho T_i |\\nabla\\rho|$",
        )

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$q_{e,i}$ ($MW/m^2$)")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Check heat transport")

        ax.set_xlim([0, maxx])
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        GRAPHICStools.addDenseAxis(ax)

        # -------------

        ax = ax6
        ax.plot(
            self.x_lw,
            self.Ge_obs[i1, :],
            lw=2,
            c="orange",
            ls="-",
            label="$\\Gamma_e$ PB",
        )
        ax.plot(
            self.x_lw,
            self.Ge_MITIM[i1, :],
            lw=1,
            c="orange",
            ls="--",
            label="$-D_{eff,e} \\nabla_\\rho n_e |\\nabla\\rho|$",
        )

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$\\Gamma_e$ ($10^{20}/s/m^2$)")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Check particle transport")

        ax.set_xlim([0, maxx])
        GRAPHICStools.autoscale_y(ax)

        ax.axhline(y=0.0, lw=2, c="k", ls="--")

        GRAPHICStools.addDenseAxis(ax)

        # --------
        ax = ax3
        ax.plot(self.x_lw, self.qeGB_obs[i1, :], lw=2, c="r", label="$q_e$ PB (GB)")
        ax.plot(self.x_lw, self.qiGB_obs[i1, :], lw=2, c="b", label="$q_i$ PB (GB)")

        ax.legend(loc="best")
        ax.set_xlabel("$\\rho_N$")
        ax.set_yscale("log")
        ax.set_ylabel("Fluxes (GB units)")

        ax.set_xlim([0, maxx])
        GRAPHICStools.autoscale_y(ax)

        GRAPHICStools.addDenseAxis(ax)

        # GyroBohm unit
        ax = ax3.twinx()
        ax.plot(self.xb_lw, self.TGLF_Qgb[i1, :], lw=2)

        # ax4.set_xlabel('$\\rho_N$')
        ax.set_ylabel("GyroBohm unit ($MW/m^2$)")
        ax.set_ylim(bottom=0)

        ax.set_xlim([0, maxx])
        GRAPHICStools.autoscale_y(ax)

        # Neoclassical
        ax = ax7

        ax.plot(self.x_lw, self.Chi_e[i1, :], lw=2, c="r", label="$\\chi_e$")
        ax.plot(self.x_lw, self.Chi_e_neo[i1, :], lw=1, c="m", label="$\\chi_{e,nc}$")
        ax.plot(self.x_lw, self.Chi_i[i1, :], lw=2, c="b", label="$\\chi_i$")
        ax.plot(self.x_lw, self.Chi_i_neo[i1, :], lw=1, c="c", label="$\\chi_{i,nc}$")
        ax.set_ylabel("$\\chi$ ($m^2/s$)")
        ax.set_xlabel("$\\rho_N$")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax.set_yscale("log")

        ax.set_xlim([0, maxx])
        GRAPHICStools.autoscale_y(ax)

        GRAPHICStools.addDenseAxis(ax)

        # -----------

        ax = ax8

        ax.plot(self.x_lw, self.qe_MITIM[i1, :], lw=2, c="r", label="$q_e$")
        ax.plot(self.x_lw, self.qe_neo[i1, :], lw=1, c="m", label="$q_{e,nc}$")
        ax.plot(self.x_lw, self.qi_MITIM[i1, :], lw=2, c="b", label="$q_i$")
        ax.plot(self.x_lw, self.qi_neo[i1, :], lw=1, c="c", label="$q_{i,nc}$")
        ax.set_ylabel("$q_{e,i}$ ($MW/m^2$)")
        ax.set_xlabel("$\\rho_N$")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax.set_xlim([0, maxx])
        GRAPHICStools.autoscale_y(ax, bottomy=0)

        GRAPHICStools.addDenseAxis(ax)

    def plotDerivatives(self, fig=None, time=None, rhos=[0.0], complete=True):
        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 3, hspace=0.25, wspace=0.3)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[1, 0])
        ax4 = fig.add_subplot(grid[1, 1])
        ax5 = fig.add_subplot(grid[:, 2])
        # ax6 = fig.add_subplot(grid[1,2])

        if time is None:
            i1 = self.ind_saw
        else:
            i1 = np.argmin(np.abs(self.t - time))
        timeReq = self.t[i1]

        # ~~~~~~~~~~~
        # Scale lengths
        # ~~~~~~~~~~~

        # ~~~
        g = self.aLTe
        ax1.plot(
            self.x_lw,
            g[i1, :],
            lw=2,
            c="r",
            label="$-a/T_e$  $\\cdot$ $\\nabla_{r_{min}} T_e$",
        )
        g = self.aLTi
        ax1.plot(
            self.x_lw,
            g[i1, :],
            lw=2,
            c="b",
            label="$-a/T_i$  $\\cdot$ $\\nabla_{r_{min}} T_i$",
        )
        g = self.aLTe_gR
        ax1.plot(
            self.x_lw,
            g[i1, :],
            lw=2,
            c="r",
            ls="--",
            label="$-a/T_e$  $\\cdot$ $\\nabla_\\rho T_e$  $\\cdot$ $|\\nabla\\rho|$",
        )
        g = self.aLTi_gR
        ax1.plot(
            self.x_lw, g[i1, :], lw=2, c="b", ls="--"
        )  # ,label='$-a/T_i$  $\\cdot$ $\\nabla_\\rho T_i$  $\\cdot$ $|\\nabla\\rho|$')
        g = self.aLTe_rho
        ax1.plot(
            self.x_lw,
            g[i1, :],
            lw=2,
            c="r",
            ls=":",
            label="$-a/T_e$  $\\cdot$ $\\nabla_{\\rho} T_e$",
        )
        g = self.aLTi_rho
        ax1.plot(
            self.x_lw, g[i1, :], lw=2, c="b", ls=":"
        )  # ,label='$-a/T_e$  $\\cdot$ $\\nabla_{\\rho} T_i$')

        ax1.set_xlabel("$\\rho_N$")
        ax1.set_ylabel("$a/L_{T}$")
        ax1.legend(loc="upper left", prop={"size": 10})
        ax1.set_xlim([0, 0.9])
        GRAPHICStools.autoscale_y(ax1)
        ax1.set_title("Temperature gradients")

        GRAPHICStools.addDenseAxis(ax1)

        # ~~~
        g = self.aLne
        ax2.plot(
            self.x_lw,
            g[i1, :],
            lw=2,
            c="g",
            label="$-a/n_e$  $\\cdot$ $\\nabla_{r_{min}} n_e$",
        )
        g = self.aLne_gR
        ax2.plot(
            self.x_lw,
            g[i1, :],
            lw=2,
            c="g",
            ls="--",
            label="$-a/n_e$  $\\cdot$ $\\nabla_\\rho n_e$  $\\cdot$ $|\\nabla\\rho|$",
        )
        g = self.aLne_rho
        ax2.plot(
            self.x_lw,
            g[i1, :],
            lw=2,
            c="g",
            ls=":",
            label="$-a/n_e$  $\\cdot$ $\\nabla_{\\rho} n_e$",
        )

        ax2.set_xlabel("$\\rho_N$")
        ax2.set_ylabel("$a/L_{n}$")
        ax2.legend(loc="upper left", prop={"size": 10})
        ax2.set_xlim([0, 0.9])
        GRAPHICStools.autoscale_y(ax2)
        ax2.set_title("Density gradients")

        GRAPHICStools.addDenseAxis(ax2)

        # ~~~
        g = self.shat
        ax3.plot(
            self.x_lw,
            g[i1, :],
            lw=2,
            c="m",
            label="$a/q$  $\\cdot$ $\\nabla_{r_{min}} q$",
        )
        g = self.shat_gR
        ax3.plot(
            self.x_lw,
            g[i1, :],
            lw=2,
            c="m",
            ls="--",
            label="$a/q$  $\\cdot$ $\\nabla_\\rho q$  $\\cdot$ $|\\nabla\\rho|$",
        )
        g = self.shat_rho
        ax3.plot(
            self.x_lw,
            g[i1, :],
            lw=2,
            c="m",
            ls=":",
            label="$a/q$  $\\cdot$ $\\nabla_{\\rho} q$",
        )

        ax3.set_xlabel("$\\rho_N$")
        ax3.set_ylabel("$a/L_{q}$")
        ax3.legend(loc="upper left", prop={"size": 10})
        ax3.set_xlim([0, 0.9])
        GRAPHICStools.autoscale_y(ax3)
        ax3.set_title("Safety Factor gradients")

        GRAPHICStools.addDenseAxis(ax3)

        # ~~~
        g = self.shat_Rice
        ax4.plot(
            self.x_lw,
            g[i1, :],
            lw=2,
            c="c",
            label="$\\hat{s} = $ $r/q$  $\\cdot$ $\\nabla_{r_{min}} q$",
        )
        ax4.set_xlabel("$\\rho_N$")
        ax4.set_ylabel("$\\hat{s}$")
        ax4.legend(loc="upper center", prop={"size": 10})
        ax4.set_xlim([0, 0.9])
        GRAPHICStools.autoscale_y(ax4)
        ax4.set_title("$\\hat{s}$ metrics")

        GRAPHICStools.addDenseAxis(ax4)

        ax = ax4.twinx()
        g = self.Ls_Rice
        ax.plot(
            self.x_lw, g[i1, :], lw=2, c="k", label="$L_s = $ $R_0\\cdot q/\\hat{s}$"
        )

        ax.set_ylabel("$L_s$ (m)")
        ax.legend(loc="center", prop={"size": 10})
        GRAPHICStools.autoscale_y(ax)

        # ~~~
        ax5.plot(self.x_lw, self.qe_obs[i1], lw=2, c="r", label="$q_e$")
        ax5.plot(
            self.x_lw,
            self.qe_obs_GACODE[i1],
            lw=2,
            c="r",
            ls="--",
            label="$\\langle q_e |\\nabla r_{min}|\\rangle$",
        )
        ax5.plot(self.x_lw, self.qi_obs[i1], lw=2, c="b", label="$q_i$")
        ax5.plot(
            self.x_lw,
            self.qi_obs_GACODE[i1],
            lw=2,
            c="b",
            ls="--",
            label="$\\langle q_i |\\nabla r_{min}|\\rangle$",
        )

        ax5.set_xlabel("$\\rho_N$")
        ax5.set_ylabel("$MW/m^2$")
        ax5.legend(loc="upper right", prop={"size": 10})
        ax5.set_xlim([0, 0.9])
        GRAPHICStools.autoscale_y(ax5)
        ax5.set_title("Flux definitions")

        GRAPHICStools.addDenseAxis(ax5)

    def plotImp(
        self,
        impLab,
        lw=1,
        ax=None,
        time=None,
        howmany=None,
        varPlot=None,
        lab="$n_Z$ ($10^{20}m^{-3}$)",
        colorsC=None,
        multY=1.0,
    ):
        if varPlot is None:
            varPlot = self.nZs

        threshold_nZs_overtotal = 1e-2

        if ax is None:
            fig, ax = plt.subplots()

        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        ax.plot(
            self.x_lw,
            multY * varPlot[impLab]["total"][it, :],
            lw=2,
            c="k",
            label="Total",
        )

        ActiveStates = []
        for i in range(len(varPlot[impLab]["states"])):
            stateX = (
                varPlot[impLab]["states"][i + 1][it, :]
                / varPlot[impLab]["total"][it, :]
            )
            if np.sum(stateX) > threshold_nZs_overtotal:
                ActiveStates.append(i + 1)

        if howmany is None or howmany > len(ActiveStates):
            howmany = len(ActiveStates)

        if colorsC is None:
            colorsC, _ = GRAPHICStools.colorTableFade(howmany)

        for i in range(howmany):
            val = ActiveStates[-1] - i * len(ActiveStates) // howmany
            stateX = varPlot[impLab]["states"][val][it, :]
            ax.plot(self.x_lw, multY * stateX, lw=lw, label=f"Z={val}", c=colorsC[i])

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel(lab)

        ax.set_ylim(bottom=0)

    def plotImpurities(self, fig=None, time=None, maxStates=10):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        num = len(self.nZs) + 1

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(num, 3, hspace=0.4, wspace=0.2)

        axs = []
        for j in range(3):
            for i in range(num):
                axs.append(fig.add_subplot(grid[i, j]))

        minPlot = 1e-10

        cont = -1
        for i in self.nZs:
            cont += 1

            howmanyStates = 0
            for j in range(len(self.nZs[i]["states"])):
                if np.sum(self.nZs[i]["states"][j + 1][it, :]) > 0.0 + self.eps00 * (
                    len(self.t) + 1
                ):
                    howmanyStates += 1

            howmany = None
            if howmanyStates > maxStates:
                howmany = maxStates

            if self.nZs[i]["total"].max() > minPlot:
                # Density
                ax = axs[3 * cont]
                self.plotImp(
                    i, ax=ax, time=self.t[it], howmany=howmany, varPlot=self.nZs
                )
                ax.set_title(f"Impurity {cont+1}: {i}")
                ax.set_position(
                    [
                        ax.get_position().x0,
                        ax.get_position().y0,
                        ax.get_position().width * 0.9,
                        ax.get_position().height,
                    ]
                )
                ax.set_ylim(bottom=0)

                GRAPHICStools.addDenseAxis(ax)
                GRAPHICStools.addLegendApart(ax, ratio=0.9, withleg=True)

                # Fraction
                ax = axs[3 * cont + 1]
                self.plotImp(
                    i,
                    ax=ax,
                    time=self.t[it],
                    howmany=howmany,
                    varPlot=self.fZs,
                    lab="$n_Z/n_e$",
                )
                # ax.set_title(i)
                ax.set_position(
                    [
                        ax.get_position().x0,
                        ax.get_position().y0,
                        ax.get_position().width * 0.9,
                        ax.get_position().height,
                    ]
                )

                l2 = f"$f_Z$ = {self.fZs_avol[i]['total'][it]:.2e}"
                ax.text(
                    0.1,
                    0.9,
                    l2,
                    fontweight="bold",
                    color="r",
                    fontsize=8,
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.set_ylim(bottom=0)

                GRAPHICStools.addDenseAxis(ax)
                GRAPHICStools.addLegendApart(ax, ratio=0.9, withleg=False)

                # Ave
                ax = axs[3 * cont + 2]
                ax.plot(
                    self.x_lw, self.nZs[i]["Zave"][it, :], lw=3, c="r", label="Total"
                )
                ax.set_position(
                    [
                        ax.get_position().x0,
                        ax.get_position().y0,
                        ax.get_position().width * 0.9,
                        ax.get_position().height,
                    ]
                )
                ax.set_xlim([0, 1])
                ax.set_ylim([0, self.nZs[i]["Zave"][it, 0] + 5])
                ax.set_xlabel("$\\rho_N$")
                ax.set_ylabel("$Z_{ave}$")

                l2 = f"$Zave$ = {self.fZs_avol[i]['Zave'][it]:2.1f}"
                ax.text(
                    0.1,
                    0.1,
                    l2,
                    fontweight="bold",
                    color="r",
                    fontsize=8,
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

                GRAPHICStools.addDenseAxis(ax)
                GRAPHICStools.addLegendApart(ax, ratio=0.9, withleg=False)

        cont += 1

        # TOTAL
        ax = axs[3 * cont]
        ax.plot(self.x_lw, self.nZ[it, :], lw=2, c="r", label="Total")
        if self.nZAVE[it, :].max() > minPlot:
            ax.plot(self.x_lw, self.nZAVE[it, :], lw=2, c="g", label="Average")

        ax.set_position(
            [
                ax.get_position().x0,
                ax.get_position().y0,
                ax.get_position().width * 0.9,
                ax.get_position().height,
            ]
        )
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$n_Z$ ($10^{20}m^{-3}$)")
        ax.set_title("Total impurities")
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})

        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

        ax_z = ax.twinx()
        ax_z.plot(self.x_lw, self.Zeff[it, :], lw=2, c="b", label="$Z_{eff}$")
        ax_z.set_ylim(bottom=0)
        ax_z.set_ylabel("$Z_{eff}$")
        ax_z.legend(loc="lower left")
        ax_z.set_position(
            [
                ax_z.get_position().x0,
                ax_z.get_position().y0,
                ax_z.get_position().width * 0.9,
                ax_z.get_position().height,
            ]
        )

        ax = axs[3 * cont + 1]
        ax.plot(self.x_lw, self.fZ[it, :], lw=3, c="r", label="Total")

        l2 = f"$f_{{Z_tot}}$ = {self.fZ_avol[it]:.2e}"
        ax.text(
            0.05,
            0.5,
            l2,
            fontweight="bold",
            color="r",
            fontsize=8,
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        if self.fZ_avolAVE[it] > minPlot:
            ax.plot(self.x_lw, self.fZAVE[it, :], lw=3, c="g", label="Average")
            l3 = f"$f_{{Z,av}}$ = {self.fZ_avolAVE[it]:.2e}"
            ax.text(
                0.65,
                0.5,
                l3,
                fontweight="bold",
                color="g",
                fontsize=8,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        ax.set_position(
            [
                ax.get_position().x0,
                ax.get_position().y0,
                ax.get_position().width * 0.9,
                ax.get_position().height,
            ]
        )
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$n_Z/n_e$")

        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})

        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

        ax = axs[3 * cont + 2]
        GRAPHICStools.addDenseAxis(ax)

        if self.fZ_avolAVE[it] > minPlot:
            ax.plot(self.x_lw, self.fZAVE_Z[it, :], lw=2, c="g", label="Average")
            ax.set_position(
                [
                    ax.get_position().x0,
                    ax.get_position().y0,
                    ax.get_position().width * 0.9,
                    ax.get_position().height,
                ]
            )
            ax.set_xlim([0, 1])
            ax.set_xlabel("$\\rho_N$")
            ax.set_ylabel("$n_Z$ ($10^{20}m^{-3}$)")
            ax.set_ylabel("$Zave$")
            l2 = f"$Zave$ = {self.fZ_avolAVE_Z[it]:2.1f}"
            ax.text(
                0.1,
                0.1,
                l2,
                fontweight="bold",
                color="g",
                fontsize=8,
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.legend(loc="best", prop={"size": self.mainLegendSize})

            ax.set_ylim([0, self.fZAVE_Z[it, 0] + 5])

    def diagramSpecies(self, time=None, fn=None, label="", fn_color=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fn is None:
            plt.ion()
            _, axs = plt.subplots(ncols=1, sharex=True, sharey=False, figsize=(14, 9))
        else:
            fig = fn.add_figure(label=label, tab_color=fn_color)
            axs = fig.subplots(ncols=1, sharex=True, sharey=False)

        ax = axs
        ax.axis("off")

        startingSize = 15
        startingPosx = 0.01
        startingPosy = 0.95
        jumpLayer = 0.05
        sizeLayer = 2
        sizeLevel = 0.05

        minPlot = 1e-10  # 0.0+self.eps00

        layer = 0
        level = 0
        l2 = f"FUEL ({self.fmain_avol[it] * 100.0:.1f}%)"
        ax.text(
            startingPosx + layer * jumpLayer,
            startingPosy - level * sizeLevel,
            l2,
            fontweight="bold",
            color="k",
            fontsize=startingSize - layer * sizeLayer,
            bbox=dict(facecolor="red", alpha=0.1),
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        layer = 1
        level += 1
        l2 = "D ({0:.1f}%; f = {1:1.2e} of total)".format(
            self.fD_avol[it] / self.fmain_avol[it] * 100.0, self.fD_avol[it]
        )
        ax.text(
            startingPosx + layer * jumpLayer,
            startingPosy - level * sizeLevel,
            l2,
            fontweight="bold",
            color="k",
            fontsize=startingSize - layer * sizeLayer,
            bbox=dict(facecolor="red", alpha=0.1),
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        if self.fT_avol[it] > minPlot:
            layer = 1
            level += 1
            l2 = "T ({0:.1f}%; f = {1:1.2e} of total)".format(
                self.fT_avol[it] / self.fmain_avol[it] * 100.0, self.fT_avol[it]
            )
            ax.text(
                startingPosx + layer * jumpLayer,
                startingPosy - level * sizeLevel,
                l2,
                fontweight="bold",
                color="k",
                fontsize=startingSize - layer * sizeLayer,
                bbox=dict(facecolor="red", alpha=0.1),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        if self.fH_avol[it] > minPlot:
            layer = 1
            level += 1
            l2 = "H ({0:.1f}%; f = {1:1.2e} of total)".format(
                self.fH_avol[it] / self.fmain_avol[it] * 100.0, self.fH_avol[it]
            )
            ax.text(
                startingPosx + layer * jumpLayer,
                startingPosy - level * sizeLevel,
                l2,
                fontweight="bold",
                color="k",
                fontsize=startingSize - layer * sizeLayer,
                bbox=dict(facecolor="red", alpha=0.1),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        if self.fHe4_avol[it] > minPlot:
            layer = 0
            level += 1
            l2 = f"FUSION ASH ({self.fHe4_avol[it] * 100.0:.1f}%)"
            ax.text(
                startingPosx + layer * jumpLayer,
                startingPosy - level * sizeLevel,
                l2,
                fontweight="bold",
                color="k",
                fontsize=startingSize - layer * sizeLayer,
                bbox=dict(facecolor="b", alpha=0.1),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

            layer = 1
            level += 1
            l2 = "He4 ({0:.1f}%; f = {1:1.2e} of total)".format(
                self.fHe4_avol[it] / self.fHe4_avol[it] * 100.0, self.fHe4_avol[it]
            )
            ax.text(
                startingPosx + layer * jumpLayer,
                startingPosy - level * sizeLevel,
                l2,
                fontweight="bold",
                color="k",
                fontsize=startingSize - layer * sizeLayer,
                bbox=dict(facecolor="b", alpha=0.1),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        layer = 0
        level += 1
        l2 = f"IMPURITIES ({self.fZ_avol[it] * 100.0:.1f}%)"
        ax.text(
            startingPosx + layer * jumpLayer,
            startingPosy - level * sizeLevel,
            l2,
            fontweight="bold",
            color="k",
            fontsize=startingSize - layer * sizeLayer,
            bbox=dict(facecolor="g", alpha=0.1),
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        for i in self.nZs:
            if self.fZs_avol[i]["total"][it] > minPlot:
                layer = 1
                level += 1
                l2 = "{2} ({0:.3f}%; f = {1:1.2e} of total)".format(
                    self.fZs_avol[i]["total"][it] / self.fZ_avol[it] * 100.0,
                    self.fZs_avol[i]["total"][it],
                    i,
                )
                ax.text(
                    startingPosx + layer * jumpLayer,
                    startingPosy - level * sizeLevel,
                    l2,
                    fontweight="bold",
                    color="k",
                    fontsize=startingSize - layer * sizeLayer,
                    bbox=dict(facecolor="g", alpha=0.1),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

        layer = 0
        level += 1
        l2 = f"FAST ({self.fFast_avol[it] * 100.0:.1f}%)"
        ax.text(
            startingPosx + layer * jumpLayer,
            startingPosy - level * sizeLevel,
            l2,
            fontweight="bold",
            color="k",
            fontsize=startingSize - layer * sizeLayer,
            bbox=dict(facecolor="m", alpha=0.1),
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        if self.ffus_avol[it] > minPlot:
            layer = 1
            level += 1
            l2 = "FUSION ({0:.1f}%; f = {1:1.2e} of total)".format(
                self.ffus_avol[it] / self.fFast_avol[it] * 100.0, self.ffus_avol[it]
            )
            ax.text(
                startingPosx + layer * jumpLayer,
                startingPosy - level * sizeLevel,
                l2,
                fontweight="bold",
                color="k",
                fontsize=startingSize - layer * sizeLayer,
                bbox=dict(facecolor="m", alpha=0.1),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

            if self.ffusHe4_avol[it] > minPlot:
                layer = 2
                level += 1
                l2 = "He4 ({0:.2f}%; f = {1:1.2e} of total)".format(
                    self.ffusHe4_avol[it] / self.ffus_avol[it] * 100.0,
                    self.ffusHe4_avol[it],
                )
                ax.text(
                    startingPosx + layer * jumpLayer,
                    startingPosy - level * sizeLevel,
                    l2,
                    fontweight="bold",
                    color="k",
                    fontsize=startingSize - layer * sizeLayer,
                    bbox=dict(facecolor="m", alpha=0.1),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
            if self.ffusHe3_avol[it] > minPlot:
                layer = 2
                level += 1
                l2 = "He3 ({0:.2f}%; f = {1:1.2e} of total)".format(
                    self.ffusHe3_avol[it] / self.ffus_avol[it] * 100.0,
                    self.ffusHe3_avol[it],
                )
                ax.text(
                    startingPosx + layer * jumpLayer,
                    startingPosy - level * sizeLevel,
                    l2,
                    fontweight="bold",
                    color="k",
                    fontsize=startingSize - layer * sizeLayer,
                    bbox=dict(facecolor="m", alpha=0.1),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
            if self.ffusH_avol[it] > minPlot:
                layer = 2
                level += 1
                l2 = "H ({0:.2f}%; f = {1:1.2e} of total)".format(
                    self.ffusH_avol[it] / self.ffus_avol[it] * 100.0, self.ffusH_avol[it]
                )
                ax.text(
                    startingPosx + layer * jumpLayer,
                    startingPosy - level * sizeLevel,
                    l2,
                    fontweight="bold",
                    color="k",
                    fontsize=startingSize - layer * sizeLayer,
                    bbox=dict(facecolor="m", alpha=0.1),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
            if self.ffusT_avol[it] > minPlot:
                layer = 2
                level += 1
                l2 = "T ({0:.2f}%; f = {1:1.2e} of total)".format(
                    self.ffusT_avol[it] / self.ffus_avol[it] * 100.0, self.ffusT_avol[it]
                )
                ax.text(
                    startingPosx + layer * jumpLayer,
                    startingPosy - level * sizeLevel,
                    l2,
                    fontweight="bold",
                    color="k",
                    fontsize=startingSize - layer * sizeLayer,
                    bbox=dict(facecolor="m", alpha=0.1),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

        if self.fmini_avol[it] > minPlot:
            layer = 1
            level += 1
            l2 = "RF ({0:.1f}%; f = {1:1.2e} of total)".format(
                self.fmini_avol[it] / self.fFast_avol[it] * 100.0, self.fmini_avol[it]
            )
            ax.text(
                startingPosx + layer * jumpLayer,
                startingPosy - level * sizeLevel,
                l2,
                fontweight="bold",
                color="k",
                fontsize=startingSize - layer * sizeLayer,
                bbox=dict(facecolor="m", alpha=0.1),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            if self.fminiHe3_avol[it] > 0.0 + self.eps00:
                layer = 2
                level += 1
                l2 = "He3 ({0:.2f}%; f = {1:1.2e} of total)".format(
                    self.fminiHe3_avol[it] / self.fmini_avol[it] * 100.0,
                    self.fminiHe3_avol[it],
                )
                ax.text(
                    startingPosx + layer * jumpLayer,
                    startingPosy - level * sizeLevel,
                    l2,
                    fontweight="bold",
                    color="k",
                    fontsize=startingSize - layer * sizeLayer,
                    bbox=dict(facecolor="m", alpha=0.1),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
            if self.fminiH_avol[it] > 0.0 + self.eps00:
                layer = 2
                level += 1
                l2 = "H ({0:.2f}%; f = {1:1.2e} of total)".format(
                    self.fminiH_avol[it] / self.fmini_avol[it] * 100.0,
                    self.fminiH_avol[it],
                )
                ax.text(
                    startingPosx + layer * jumpLayer,
                    startingPosy - level * sizeLevel,
                    l2,
                    fontweight="bold",
                    color="k",
                    fontsize=startingSize - layer * sizeLayer,
                    bbox=dict(facecolor="m", alpha=0.1),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

        if self.fb_avol[it] > minPlot:
            layer = 1
            level += 1
            l2 = "BEAM ({0:.1f}%; f = {1:1.2e} of total)".format(
                self.fb_avol[it] / self.fFast_avol[it] * 100.0, self.fb_avol[it]
            )
            ax.text(
                startingPosx + layer * jumpLayer,
                startingPosy - level * sizeLevel,
                l2,
                fontweight="bold",
                color="k",
                fontsize=startingSize - layer * sizeLayer,
                bbox=dict(facecolor="m", alpha=0.1),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
            )

            if self.fbD_avol[it] > 0.0 + self.eps00:
                layer = 2
                level += 1
                l2 = "D ({0:.2f}%; f = {1:1.2e} of total)".format(
                    self.fbD_avol[it] / self.fb_avol[it] * 100.0, self.fbD_avol[it]
                )
                ax.text(
                    startingPosx + layer * jumpLayer,
                    startingPosy - level * sizeLevel,
                    l2,
                    fontweight="bold",
                    color="k",
                    fontsize=startingSize - layer * sizeLayer,
                    bbox=dict(facecolor="m", alpha=0.1),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

            if self.fbT_avol[it] > 0.0 + self.eps00:
                layer = 2
                level += 1
                l2 = "T ({0:.2f}%; f = {1:1.2e} of total)".format(
                    self.fbT_avol[it] / self.fb_avol[it] * 100.0, self.fbT_avol[it]
                )
                ax.text(
                    startingPosx + layer * jumpLayer,
                    startingPosy - level * sizeLevel,
                    l2,
                    fontweight="bold",
                    color="k",
                    fontsize=startingSize - layer * sizeLayer,
                    bbox=dict(facecolor="m", alpha=0.1),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

            if self.fbH_avol[it] > 0.0 + self.eps00:
                layer = 2
                level += 1
                l2 = "H ({0:.2f}%; f = {1:1.2e} of total)".format(
                    self.fbH_avol[it] / self.fb_avol[it] * 100.0, self.fbH_avol[it]
                )
                ax.text(
                    startingPosx + layer * jumpLayer,
                    startingPosy - level * sizeLevel,
                    l2,
                    fontweight="bold",
                    color="k",
                    fontsize=startingSize - layer * sizeLayer,
                    bbox=dict(facecolor="m", alpha=0.1),
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

    def diagramFlows(self, axs=None, time=None, fn=None, label="", fn_color=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if axs is None:
            if fn is None:
                fig, axs = plt.subplots(
                    ncols=1, sharex=True, sharey=False, figsize=(14, 9)
                )
            else:
                fig = fn.add_figure(label=label, tab_color=fn_color)
                axs = fig.subplots(ncols=1, sharex=True, sharey=False)

        ax = axs
        ax.axis("off")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # IONS & ELECTRONS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        arrowsOutIons = OrderedDict()
        arrowsOutElec = OrderedDict()

        arrowsOutIons[f"$Q_i$ = {self.Pi_LCFS[it]:.1f}MW"] = [1.0, 0.7]
        if np.abs(self.PcxT[it]) > 0.05:
            arrowsOutIons[f"$P_{{cx}}$ = {self.PcxT[it]:.1f}MW"] = [0.5, 1.0]

        arrowsOutElec[f"$Q_e$ = {self.Pe_LCFS[it]:.1f}MW"] = [1.0, 0.3]
        arrowsOutElec[f"$P_{{rad}}$ = {self.PradT[it]:.1f}MW"] = [0.5, 0.0]

        if self.PeiT[it] > 0:
            arrowsOutElec[f"$P_{{ei}}$ = {np.abs(self.PeiT[it]):.1f}MW"] = [
                0.5,
                0.65,
            ]
        else:
            arrowsOutIons[f"$P_{{ei}}$ = {np.abs(self.PeiT[it]):.1f}MW"] = [
                0.5,
                0.35,
            ]

        GRAPHICStools.diagram_plotModule(
            ax,
            f"IONS, $dW/dt$={self.GainiT[it]:.1f}MW",
            [0.5, 0.7],
            noLab=False,
            c="b",
            typeBox="round",
            arrowsOut=arrowsOutIons,
            carrowsOut=["r", "m", "orange"],
        )

        GRAPHICStools.diagram_plotModule(
            ax,
            f"ELECTRONS, $dW/dt$={self.GaineT[it]:.1f}MW",
            [0.5, 0.3],
            noLab=False,
            c="b",
            typeBox="round",
            arrowsOut=arrowsOutElec,
            carrowsOut=["r", "m", "orange"],
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fusion
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if self.PfuseT[it] + self.PfusiT[it] > 1.0e-5:
            GRAPHICStools.diagram_plotModule(
                ax,
                "FUSION",
                [0.75, 0.5],
                noLab=False,
                c="c",
                typeBox="roundtooth",
                arrowsOut={
                    f"$P_{{fus,i}}$ = {self.PfusiT[it]:.1f}MW": [0.55, 0.66],
                    f"$P_{{fus,e}}$ = {self.PfuseT[it]:.1f}MW": [0.55, 0.34],
                    f"$P_{{neutrons}}$ = {self.Pout[it] * 4.0 / 5.0:.1f}MW": [
                        1.0,
                        0.5,
                    ],
                },
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ICRF
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if self.PiichT[it] + self.PeichT[it] > 1.0e-5:

            arrowsOut={
                    f"$P_{{ICH,i}}$ = {self.PiichT[it]:.1f}MW": [0.41, 0.7],
                    f"$P_{{ICH,e}}$ = {self.PeichT[it]:.1f}MW": [0.39, 0.3],
                }

            percent_plot = 0.05
            # If minorities are not in steady-state, some portion goes to their dW/dt
            if self.GainminT[it] > percent_plot*(self.PiichT[it] + self.PeichT[it]):
                arrowsOut[f"$dW/dt$ = {self.GainminT[it]:.1f}MW"] = [0.15,0.85]
                print(f'\t- ICRF Minorities were not in steady state (dWdt>{percent_plot*100:.1f}% of power to bulk, {self.GainminT[it]/(self.PiichT[it] + self.PeichT[it])*100.0:.1f} %)', typeMsg='w')
            # Fast heating
            if self.PfichT_dir[it] > percent_plot*(self.PiichT[it] + self.PeichT[it]):
                arrowsOut[f"$P_{{ICH,fast}}$ = {self.PfichT_dir[it]:.1f}MW"] = [0.35,0.85]
                print(f'\t- ICRF heated fast particles non-negligibly (>{percent_plot*100:.1f}% of power to bulk, {self.PfichT_dir[it]/(self.PiichT[it] + self.PeichT[it])*100.0:.1f} %))', typeMsg='w')
            
            GRAPHICStools.diagram_plotModule(
                ax,
                "ICRF",
                [0.2, 0.7],
                noLab=False,
                c="g",
                typeBox="roundtooth",
                arrowsOut=arrowsOut,
                arrowsIn={f"$P_{{ICH}}$ = {self.PichT[it]:.1f}MW": [0.0, 0.7]},
            )


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # NBI
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if self.PnbiiT[it] + self.PnbieT[it] > 1.0e-5:
            GRAPHICStools.diagram_plotModule(
                ax,
                "NBI",
                [0.2, 0.55],
                noLab=False,
                c="b",
                typeBox="roundtooth",
                arrowsOut={
                    f"$P_{{NBI,i}}$ = {self.PnbiiT[it]:.1f}MW": [0.41, 0.7],
                    f"$P_{{NBI,e}}$ = {self.PnbieT[it]:.1f}MW": [0.39, 0.3],
                },
                arrowsIn={f"$P_{{NBI}}$ = {self.PnbiT[it]:.1f}MW": [0.0, 0.55]},
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ECRH
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if self.PechT[it] > 1.0e-5:
            GRAPHICStools.diagram_plotModule(
                ax,
                "ECH",
                [0.2, 0.45],
                noLab=False,
                c="m",
                typeBox="roundtooth",
                arrowsOut={f"$P_{{ECH}}$ = {self.PechT[it]:.1f}MW": [0.39, 0.3]},
                arrowsIn={f"$P_{{ECH}}$ = {self.PechT[it]:.1f}MW": [0.0, 0.45]},
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # LH
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if self.PlheT[it] + self.PlhiT[it] > 1.0e-5:
            GRAPHICStools.diagram_plotModule(
                ax,
                "LH",
                [0.2, 0.3],
                noLab=False,
                c="y",
                typeBox="roundtooth",
                arrowsOut={
                    f"$P_{{LH,i}}$ = {self.PlhiT[it]:.1f}MW": [0.41, 0.7],
                    f"$P_{{LH,e}}$ = {self.PlheT[it]:.1f}MW": [0.39, 0.3],
                },
                arrowsIn={f"$P_{{LH}}$ = {self.PlhT[it]:.1f}MW": [0.0, 0.3]},
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # OHMIC
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        GRAPHICStools.diagram_plotModule(
            ax,
            "OHMIC",
            [0.2, 0.15],
            noLab=False,
            c="k",
            typeBox="roundtooth",
            arrowsOut={f"$P_{{OH}}$ = {self.PohT[it]:.1f}MW": [0.39, 0.3]},
            arrowsIn={f"$P_{{OH}}$ = {self.PohT[it]:.1f}MW": [0.0, 0.15]},
        )

        ax.add_patch(
            patches.Rectangle(
                (0.1, 0.1),
                0.8,
                0.8,
                linewidth=1,
                ls="--",
                edgecolor="b",
                facecolor="none",
            )
        )

    def plotRadiation(self, fig=None, time=None, maxStates=10):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.2)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0])
        ax3 = fig.add_subplot(grid[0, 1])
        ax4 = fig.add_subplot(grid[1, 1])

        ax = ax1
        ax.plot(self.x_lw, self.Prad[it, :], lw=3, label="$P_{rad}$")
        tot = np.zeros(len(self.Prad[it, :]))
        for i in self.PradZ:
            ax.plot(self.x_lw, self.PradZ[i][it, :], lw=2, label="$P_{rad,%s}$" % i)
            tot += self.PradZ[i][it, :]
        ax.plot(self.x_lw, tot, c="y", ls="--", lw=1, label="sum")
        ax.plot(
            self.x_lw,
            self.Prad[it, :] - tot,
            c="c",
            ls="--",
            lw=1,
            label="$P_{rad}$-sum",
        )
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("Prad ($MW/m^3$)")
        ax.set_title("Radiation per impurity")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_ylim(bottom=0)

        l2 = f"$P$ = {self.PradT[it]:.1f}MW"
        ax.text(
            0.5,
            0.9,
            l2,
            fontweight="bold",
            color="k",
            fontsize=15,
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )

        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

        ax = ax2
        ax.plot(self.x_lw, self.Prad[it, :], lw=3, label="$P_{rad}$")
        ax.plot(self.x_lw, self.Prad_b[it, :], lw=2, label="$P_{rad,bremss}$")
        ax.plot(self.x_lw, self.Prad_l[it, :], lw=2, label="$P_{rad,line}$")
        ax.plot(self.x_lw, self.Prad_c[it, :], lw=2, label="$P_{rad,cycl}$")
        tot = self.Prad_b[it, :] + self.Prad_l[it, :] + self.Prad_c[it, :]
        ax.plot(self.x_lw, tot, c="y", ls="--", lw=1, label="sum")
        # ax.plot(self.x_lw,self.Prad[it,:]-tot,c='c',ls='--',lw=1,label='$P_{rad}$-sum')
        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("Prad ($MW/m^3$)")
        ax.set_title("Radiation per type")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

        ax = ax3
        ax.plot(self.t, self.PradT, lw=3, label="$P_{rad}$")

        tot = np.zeros(len(self.PradT))
        for i in self.PradZT:
            ax.plot(self.t, self.PradZT[i], lw=2, label="$P_{rad,%s}$" % i)
            tot += self.PradZT[i]
        ax.plot(self.t, tot, c="y", ls="--", lw=1, label="sum")
        ax.plot(self.t, self.PradT - tot, c="c", ls="--", lw=1, label="$P_{rad}$-sum")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Prad (MW)")
        ax.set_title("Radiation per impurity")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.axvline(x=self.t[it], c="k", ls="--")

        GRAPHICStools.addDenseAxis(ax)

        ax = ax4
        ax.plot(self.t, self.PradT, lw=3, label="$P_{rad}$")
        ax.plot(self.t, self.PradT_b, lw=2, label="$P_{rad,bremss}$")
        ax.plot(self.t, self.PradT_l, lw=2, label="$P_{rad,line}$")
        ax.plot(self.t, self.PradT_c, lw=2, label="$P_{rad,cycl}$")
        tot = self.PradT_b + self.PradT_l + self.PradT_c
        ax.plot(self.t, tot, c="y", ls="--", lw=1, label="sum")
        # ax.plot(self.t,self.PradT-tot,c='c',ls='--',lw=1,label='$P_{rad}$-sum')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Prad (MW)")
        ax.set_title("Radiation per type")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

    def plotStability(self, fig=None, time=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0])
        ax3 = fig.add_subplot(grid[0, 1])
        ax4 = fig.add_subplot(grid[1, 1])

        ax = ax1
        ax.plot(self.t, self.qstar95, lw=2, label="$q^*_{95}$")
        ax.plot(self.t, self.qstar95ITER, lw=2, label="$q^*_{95,ITER}$")
        ax.plot(self.t, self.qstarsep, lw=2, label="$q^*_{sep}$")
        ax.plot(self.t, self.qstarsepITER, lw=2, label="$q^*_{sep,ITER}$")
        ax.plot(self.t, self.q95, lw=2, label="$q_{95}$")

        ax.set_ylim([0, 5])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("[]")
        ax.set_title("Current limits")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        for i in np.arange(1, 5, 1):
            ax.axhline(y=i, ls="--", c="k", lw=1)

        ax.fill_between(
            self.t,
            np.zeros(len(self.t)),
            np.ones(len(self.t)) * 3.0,
            color="r",
            alpha=0.05,
        )

        ax = ax2
        ax.plot(self.t, self.fGv, lw=2, label="$f_{G,vol}$")
        ax.plot(self.t, self.fGl, lw=2, label="$f_{G,lin}$")
        # ax.plot(self.t,self.fG_950,lw=3,label='$f_{G,95.0\%}$')

        for i in np.arange(0.1, 1.0, 0.1):
            ax.axhline(y=i, ls="--", c="k", lw=1)
        ax.axhline(y=1.0, ls="--", c="k", lw=3)

        ax.fill_between(
            self.t,
            np.ones(len(self.t)),
            np.ones(len(self.t)) * 2.0,
            color="r",
            alpha=0.05,
        )
        ax.fill_between(
            self.t,
            np.ones(len(self.t)) * 0.6,
            np.ones(len(self.t)) * 1.0,
            color="magenta",
            alpha=0.05,
        )

        ax.set_ylim([0, 1.1])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("[]")
        ax.set_title("Density limits")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax = ax3
        ax.plot(self.x_lw, self.j[it], c="b", lw=2, label="$J$")
        ax.set_xlim([0, 1.0])
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("J ($MA/m^2$)")
        ax.legend(loc="lower left")
        ax.set_ylim(bottom=0)
        ax.set_title("Current profile features")
        for i in [1, 2, 3]:
            GRAPHICStools.drawLineWithTxt(
                ax,
                self.xb[it, np.argmin(np.abs(self.q[it] - i))],
                label="q=" + str(i),
                orientation="vertical",
                color="m",
                lw=5,
                ls="-",
                alpha=0.2,
                fontsize=10,
            )

        GRAPHICStools.addDenseAxis(ax)

        ax = ax4

        ax.plot(
            self.x_lw, -self.jprime[it], lw=2, c="r", label="$-\\nabla_{r_{min}} J$"
        )
        ax.set_ylabel("dJ/dr ($MA/m^3$)")
        ax.legend(loc="lower center")
        ax.axhline(y=0, ls="--", c="k", lw=1)

        for i in [1, 2, 3]:
            GRAPHICStools.drawLineWithTxt(
                ax,
                self.xb[it, np.argmin(np.abs(self.q[it] - i))],
                label="q=" + str(i),
                orientation="vertical",
                color="m",
                lw=5,
                ls="-",
                alpha=0.2,
                fontsize=10,
            )

        GRAPHICStools.addDenseAxis(ax)

    def plotRotation(self, fig=None, time=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0])
        ax3 = fig.add_subplot(grid[0, 1])
        ax4 = fig.add_subplot(grid[1, 1])

        ax = ax1
        ax.plot(
            np.append(-np.flipud(self.x_lw), self.x_lw),
            np.append(np.flipud(self.Vtor_HF[it]), self.Vtor_LF[it]),
            lw=2,
            label="Plasma toroidal rotation",
        )
        ax.plot(
            np.append(-np.flipud(self.x_lw), self.x_lw),
            np.append(np.flipud(self.VtorNC_HF[it]), self.VtorNC_LF[it]),
            lw=1,
            label="NC toroidal rotation",
        )
        ax.plot(
            np.append(-np.flipud(self.x_lw), self.x_lw),
            np.append(np.flipud(self.VpolNC_HF[it]), self.VpolNC_LF[it]),
            lw=1,
            label="NC poloidal rotation",
        )

        ax.set_xlabel("$\\rho_N$ extended")
        ax.set_ylabel("km/s")
        ax.set_title("Midplane HF to LF rotation speed")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_xlim([-1, 1])
        ax.axvline(x=0.0, ls="--", c="k", lw=1)

        GRAPHICStools.addDenseAxis(ax)

        ax = ax2
        ax.plot(self.x_lw, self.VtorkHz[it], c="r", lw=3, label="$\\omega$")
        ax.plot(self.x_lw, self.VtorkHz_nc[it], c="b", lw=2, label="$\\omega_{nc}$")
        ax.plot(
            self.x_lw, self.VtorkHz_data[it], c="g", lw=2, label="$\\omega_{INPUT}$"
        )

        ax.plot(
            self.xb_lw,
            self.VtorkHz_check[it],
            lw=1,
            ls="--",
            c="m",
            label="check ($dV_r/d\\psi$)",
        )
        ax.plot(
            self.xb_lw,
            self.VtorkHz_rot_check[it],
            lw=1,
            ls="--",
            c="y",
            label="check ($dV_{r,rot}/d\\psi$)",
        )
        ax.plot(
            self.xb_lw,
            self.VtorkHz_nc_check[it],
            lw=1,
            ls="--",
            c="c",
            label="check ($dV_{r,nc}/d\\psi$)",
        )

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("kHz")
        ax.set_title("Angular rotation")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        ax = ax3
        ax.plot(
            np.append(-np.flipud(self.x_lw), self.x_lw),
            np.append(np.flipud(self.Mach_HF[it]), self.Mach_LF[it]),
            lw=2,
            label="Mach number",
        )

        ax.set_xlabel("$\\rho_N$ extended")
        ax.set_ylabel("Mach number")
        ax.set_title("Mach number")
        # ax.legend(loc='best',prop={'size':self.mainLegendSize})
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, 0.3])

        GRAPHICStools.addDenseAxis(ax)

        ax = ax4
        ax.plot(self.t, self.Mach_LF[:, 0], lw=2, label="Mach number")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mach number")
        ax.set_title("Central Mach number")
        # ax.legend(loc='best',prop={'size':self.mainLegendSize})
        ax.set_ylim([0, 0.3])

        GRAPHICStools.addDenseAxis(ax)

    def plotFundamental(self, fig=None, time=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0])
        ax3 = fig.add_subplot(grid[0, 1])
        ax4 = fig.add_subplot(grid[1, 1])
        ax5 = fig.add_subplot(grid[0, 2])
        ax6 = fig.add_subplot(grid[1, 2])

        ax = ax1
        ax.plot(self.x_lw, self.rhos[it] * 1e3, lw=2, label="$\\rho_s$")
        ax.plot(self.x_lw, self.rhoi[it] * 1e3, lw=2, label="$\\rho_i$")
        ax.plot(self.x_lw, self.rhoe[it] * 1e3, lw=2, label="$\\rho_e$")
        ax.plot(self.x_lw, self.lD[it] * 1e3, lw=2, label="$\\lambda_{Debye}$")

        ax.set_yscale("log")
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("Length (mm)")
        ax.set_title("Spatial Quantities (LF)")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        ax = ax2
        ax.plot(
            self.x_lw, self.wpe[it] / (2 * np.pi) * 1e-9, lw=2, label="$\\omega_{pe}$"
        )
        ax.plot(
            self.x_lw,
            self.Oce[it] / (2 * np.pi) * 1e-9,
            lw=2,
            label="$\\omega_{ce,LF}$",
        )
        ax.plot(
            self.x_lw,
            self.Oci[it] / (2 * np.pi) * 1e-9,
            lw=2,
            label="$\\omega_{ci,LF}$",
        )

        ax.plot(self.x_lw, self.nu_ei[it] * 1e-9, lw=2, label="$\\nu_{ei}$")

        ax.set_yscale("log")
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_title("Frequencies (LF)")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        ax = ax3
        ax.plot(self.x_lw, self.cs[it], lw=2, label="$c_s$")
        ax.plot(self.x_lw, self.vA[it], lw=2, label="$v_A$")
        ax.plot(self.x_lw, self.vTe[it], lw=2, label="$v_{Te}$")
        ax.plot(self.x_lw, self.vTi[it], lw=2, label="$v_{Ti}$")

        ax.set_yscale("log")
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title("Speed (LF)")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        ax = ax4
        ax.plot(self.x_lw, self.nu_eff[it], lw=2, c="r", label="$\\nu_{eff}$")
        ax.plot(self.x_lw, self.nu_star[it], lw=2, c="b", label="$\\nu_*$")
        ax.plot(self.x_lw, self.nu_norm[it], lw=2, c="g", label="$\\hat{\\nu}$")
        ax.plot(self.x_lw, self.nuste[it], lw=2, label="$\\nu_{e,TRANSP}$")

        ax.axhline(y=self.nu_eff_avol[it], lw=2, c="r", alpha=0.3)
        ax.axhline(y=self.nu_star_avol[it], lw=2, c="b", alpha=0.3)
        ax.axhline(y=self.nu_norm_avol[it], lw=2, c="g", alpha=0.3)

        ax.set_yscale("log")
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("Collisionality")
        ax.set_title("Collisionalities")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        # ax.set_ylim([0,1.5*np.max([self.nu_eff_avol[it],self.nu_star_avol[it],self.nu_norm_avol[it]])])

        ax = ax5
        ax.plot(self.x_lw, self.LambdaCoul_e[it], lw=2, c="r", label="$log\\Lambda_e$")
        ax.plot(
            self.x_lw,
            self.LambdaCoul_e_TRANSP[it],
            lw=1,
            c="r",
            ls="--",
            label="$log\\Lambda_{e,TRANSP}$",
        )
        ax.plot(self.x_lw, self.LambdaCoul_i[it], lw=2, c="b", label="$log\\Lambda_i$")
        ax.plot(
            self.x_lw,
            self.LambdaCoul_i_TRANSP[it],
            lw=1,
            c="b",
            ls="--",
            label="$log\\Lambda_{i,TRANSP}$",
        )

        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        # axl = ax.twinx()
        # dif = np.abs(self.LambdaCoul_e[it]-self.LambdaCoul_e_TRANSP[it])/self.LambdaCoul_e_TRANSP[it]*100
        # axl.plot(self.x_lw,dif,lw=1,ls='-.',c='r')
        # dif = np.abs(self.LambdaCoul_i[it]-self.LambdaCoul_i_TRANSP[it])/self.LambdaCoul_i_TRANSP[it]*100
        # axl.plot(self.x_lw,dif,lw=1,ls='-.',c='b')
        # axl.set_ylim([0,50]); axl.set_ylabel('Error (%)')

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("Coulomb Logarithm")
        ax.set_title("Collisions")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_ylim(bottom=0)

        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

    def plotFast(self, fig=None, time=None, rhos=[0, 0.25, 0.5, 0.75, 0.95, 1.0]):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)

        # _________________________________
        ax = fig.add_subplot(grid[0, 0])

        ax.plot(self.x_lw, self.Wperp_x[it, :], lw=3, c="r", label="$W_{\\perp}$")
        ax.plot(
            self.x_lw,
            self.Wpar_x[it, :],
            lw=3,
            c="r",
            ls="--",
            label="$W_{\\parallel}$",
        )

        ax.plot(
            self.x_lw,
            self.Wperpx_mini[it, :],
            c="b",
            ls="-",
            lw=2,
            label="$W_{\\perp,mini}$",
        )
        ax.plot(
            self.x_lw,
            self.Wparx_mini[it, :],
            c="b",
            ls="--",
            lw=2,
            label="$W_{\\parallel,mini}$",
        )

        ax.plot(
            self.x_lw,
            self.Wperpx_fus[it, :],
            c="m",
            ls="-",
            lw=2,
            label="$W_{\\perp,fus}$",
        )
        ax.plot(
            self.x_lw,
            self.Wparx_fus[it, :],
            c="m",
            ls="--",
            lw=2,
            label="$W_{\\parallel,fus}$",
        )

        ax.plot(
            self.x_lw,
            self.Wperpx_b[it, :],
            c="g",
            ls="-",
            lw=2,
            label="$W_{\\perp,beam}$",
        )
        ax.plot(
            self.x_lw,
            self.Wparx_b[it, :],
            c="g",
            ls="--",
            lw=2,
            label="$W_{\\parallel,beam}$",
        )

        ax.plot(
            self.x_lw,
            self.Wperpx_fus[it, :] + self.Wperpx_mini[it, :] + self.Wperpx_b[it, :],
            c="y",
            ls="--",
            lw=1,
            label="check (sum $\\perp$)",
        )
        ax.plot(
            self.x_lw,
            self.Wparx_fus[it, :] + self.Wparx_mini[it, :] + self.Wparx_b[it, :],
            c="y",
            ls="-.",
            lw=1,
            label="check (sum $\\parallel$)",
        )

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("W ($MJ/m^3$)")
        ax.set_title("Fast ion stored energy profiles")
        ax.set_ylim(bottom=0)
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)

        # __________
        ax = fig.add_subplot(grid[1, 0])

        ax.plot(self.x_lw, self.pFast[it, :], lw=3, c="r", label="$p_{fast}$")
        ax.plot(self.x_lw, self.pFast_fus[it, :], lw=2, c="m", label="$p_{fast,fus}$")
        ax.plot(self.x_lw, self.pFast_mini[it, :], lw=2, c="b", label="$p_{fast,mini}$")
        ax.plot(self.x_lw, self.pFast_b[it, :], lw=2, c="g", label="$p_{fast,beam}$")

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("p (MPa)")
        ax.set_title("Fast ion pressure profiles")
        ax.set_ylim(bottom=0)
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)

        # _________________________________
        ax = fig.add_subplot(grid[0, 1], sharex=ax)

        ax.plot(self.x_lw, self.nFast[it, :], lw=3, c="r", label="$n_{fast}$")
        ax.plot(self.x_lw, self.nmini[it, :], lw=2, c="b", label="$n_{fast,mini}$")
        ax.plot(self.x_lw, self.nfus[it, :], lw=2, c="m", label="$n_{fast,fus}$")
        ax.plot(
            self.x_lw,
            self.nfusHe4[it, :],
            lw=1,
            c="m",
            ls="--",
            label="$n_{fast,fus, He4}$",
        )
        ax.plot(self.x_lw, self.nb[it, :], lw=2, c="g", label="$n_{fast,beam}$")

        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("n ($10^{20}m^{-3}$)")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Density Profiles")

        GRAPHICStools.addDenseAxis(ax)

        # _________________________________
        ax = fig.add_subplot(grid[1, 1], sharex=ax)

        ax.plot(self.x_lw, self.Tfast[it, :], lw=3, c="r", label="$T_{fast}$")
        ax.plot(self.x_lw, self.Tmini[it, :], lw=2, c="b", label="$T_{fast,mini}$")
        ax.plot(
            self.x_lw,
            self.Tmini_check[it, :],
            lw=1,
            c="y",
            ls="--",
            label="$T_{fast,mini}=2/3\\cdot (W_{\\perp}+W_{\\parallel})/n_{fast}$",
        )
        ax.plot(
            self.x_lw,
            self.Tmini_perp[it, :],
            lw=2,
            c="b",
            ls="--",
            label="$T_{fast,mini,\\perp}=W_{\\perp}/n_{fast}$",
        )
        ax.plot(
            self.x_lw,
            self.Tmini_par[it, :],
            lw=2,
            c="b",
            ls="-.",
            label="$T_{fast,mini,\\parallel}=2\\cdot W_{\\parallel}/n_{fast}$",
        )
        ax.plot(self.x_lw, self.Tfus[it, :], lw=2, c="m", label="$T_{fast,fus}$")
        ax.plot(self.x_lw, self.Tb[it, :], lw=2, c="g", label="$T_{fast,beam}$")
        ax.plot(self.x_lw, self.Ti[it, :], lw=1, c="k", label="$T_i$")

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("T (keV)")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Effective Temperature Profiles")
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

        # _________________________________
        ax = fig.add_subplot(grid[0, 2])

        for i in rhos:
            ir = np.argmin(np.abs(self.x_lw - i))
            ax.plot(
                self.t,
                self.pFast[:, ir],
                lw=2,
                label="$p_{fast}$ @ $\\rho_N$=" + str(i),
            )

        ax.plot(self.t, self.pFast_avol, lw=4, label="$\\langle p_{fast}\\rangle$")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("p (MPa)")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Pressure Traces")
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

        # _________________________________
        ax = fig.add_subplot(grid[1, 2])

        it1 = self.ind_saw
        it2 = self.ind_saw_after

        ax.plot(self.x_lw, self.Wperp_x[it1, :], lw=2, c="r", label="$p_{\\perp}$")
        ax.plot(
            self.x_lw,
            self.Wpar_x[it1, :],
            lw=2,
            c="b",
            ls="-",
            label="$p_{\\parallel}$",
        )
        ax.plot(self.x_lw, self.Wperp_x[it2, :], lw=2, c="r", ls="--")
        ax.plot(self.x_lw, self.Wpar_x[it2, :], lw=2, c="b", ls="--")

        ax.plot(
            self.x_lw,
            self.pFast[it1, :],
            lw=2,
            c="m",
            label="$p_{\\perp}+p_{\\parallel}$",
        )
        ax.plot(self.x_lw, self.pFast[it2, :], lw=2, c="m", ls="--")
        ax.axhline(y=self.pFast_avol[it1], lw=2, c="m", alpha=0.3)
        ax.axhline(y=self.pFast_avol[it2], lw=2, c="m", ls="--", alpha=0.3)

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("p (MPa)")
        ax.set_title("Pressure Profiles (saw)")
        ax.set_ylim(bottom=0)

        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        for i in rhos:
            ax.axvline(x=i, c="k", ls="--", lw=1)

        GRAPHICStools.addDenseAxis(ax)

    def plotFast2(self, fig=None, time=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 3, hspace=0.2, wspace=0.4)

        # _________________________________
        ax0 = fig.add_subplot(grid[0, 0])
        ax = ax0

        grad = self.aLTmini[it, :] * self.Rmajor[it] / self.a[it]
        ax.plot(self.x_lw, grad, lw=2, c="r", label="$R/L_{Tmini}$")
        ax.plot(self.x_lw, -grad, lw=2, c="r", ls="--")
        ax1 = ax0.twinx()
        rat = self.Tmini[it, :] / self.Te[it, :]
        ax1.plot(self.x_lw, rat, lw=2, c="b", label="$T_{mini}/T_e$")

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$R/L_{Tmini}$")
        ax1.set_ylabel("$T_{mini}/T_e$")
        ax.set_title("Fast ion stabilization metrics")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})
        ax1.legend(loc="upper right", prop={"size": self.mainLegendSize})

        ax.axvline(x=self.x_lw[np.argmax(rat)], ls="--", c="m", lw=0.5)
        ax.axvline(
            x=self.x_lw[np.argmax(grad[: np.argmin(np.abs(self.x_lw - 0.8))])],
            ls="--",
            c="m",
            lw=0.5,
        )

        ax1.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

        # _________________________________
        ax0 = fig.add_subplot(grid[0, 1])
        ax = ax0

        grad = self.aLTmini[it, :] * self.Rmajor[it] / self.a[it]
        ax.plot(self.x_lw, grad, lw=2, c="r", label="$R/L_{Tmini}$")
        ax.plot(self.x_lw, -grad, lw=2, c="r", ls="--")
        ax1 = ax0.twinx()
        rat = self.Tmini[it, :]
        ax1.plot(self.x_lw, rat, lw=2, c="b", label="$T_{mini}$")

        ax.set_xlim([0, 1])
        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("$R/L_{Tmini}$")
        ax1.set_ylabel("$T_{mini}$")
        ax.set_title("Fast ion stabilization metrics")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})
        ax1.legend(loc="upper right", prop={"size": self.mainLegendSize})

        ax.axvline(x=self.x_lw[np.argmax(rat)], ls="--", c="m", lw=0.5)
        ax.axvline(
            x=self.x_lw[np.argmax(grad[: np.argmin(np.abs(self.x_lw - 0.8))])],
            ls="--",
            c="m",
            lw=0.5,
        )

        # ax1.axhline(y=0,ls='--',c='k',lw=0.5)
        ax1.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

    def plotParticleBalance(self, fig=None, time=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0])
        ax3 = fig.add_subplot(grid[0, 1])
        ax4 = fig.add_subplot(grid[1, 1])

        # Electrons
        ax = ax1
        ax.plot(self.x_lw, self.ne_source[it, :], lw=2, label="$S_e$")
        ax.plot(
            self.x_lw, self.ne_divGamma[it, :], lw=2, label="$\\nabla\\cdot\\Gamma_e$"
        )
        ax.plot(self.x_lw, self.ne_dt[it, :], lw=2, label="$dn_e/dt$")

        ax.plot(
            self.x_lw,
            self.ne_source[it, :] - self.ne_divGamma[it, :],
            lw=1,
            c="y",
            ls="--",
            label="check",
        )  # ($S_e$-$\\nabla\\cdot\\Gamma_e$)')

        ax.set_title("Electron Particle Balance")
        ax.set_ylabel("$10^{20}m^{-3}/s$")
        ax.set_xlabel("$\\rho_N$")
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})
        ax.set_xlim([0, 0.95])

        ir = np.argmin(np.abs(0.95 - self.x_lw))
        max_Se = np.max(
            [
                self.ne_divGamma[it, :ir].max(),
                self.ne_source[it, :ir].max(),
                self.ne_dt[it, :ir].max(),
                self.nmain_source[it, :ir].max(),
            ]
        )
        min_Se = np.min(
            [
                0,
                self.ne_divGamma[it, :ir].min(),
                self.ne_source[it, :ir].min(),
                self.ne_dt[it, :ir].min(),
                self.nmain_source[it, :ir].min(),
            ]
        )

        ax.set_ylim([min_Se, max_Se])
        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        # Electron Source
        ax = ax2
        ax.plot(self.x_lw, self.ne_source[it, :], lw=3, label="$S_e$ (sum)")
        ax.plot(
            self.x_lw, self.ne_source_wall[it, :], lw=2, c="g", label="$S_{e,wall}$"
        )
        ax.plot(self.x_lw, self.ne_source_vol[it, :], lw=2, c="r", label="$S_{e,vol}$")
        ax.plot(self.x_lw, self.ne_source_imp[it, :], lw=2, c="k", label="$S_{e,imp}$")
        ax.plot(self.x_lw, self.ne_source_fi[it, :], lw=2, c="c", label="$S_{e,fi}$")

        ax.set_title("Electron Sources")
        ax.set_ylabel("$10^{20}m^{-3}/s$")
        ax.set_xlabel("$\\rho_N$")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax.set_ylim([min_Se, max_Se])
        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        # Electron wall Source
        ax = ax3

        ax.plot(
            self.x_lw,
            self.ne_source_wall[it, :] + self.ne_source_vol[it, :],
            lw=3,
            c="g",
            label="$S_{e,wall}+S_{e,vol}$",
        )
        ax.plot(self.x_lw, self.ne_source_gf[it, :], lw=2, c="r", label="$S_{e,gf}$")
        ax.plot(
            self.x_lw, self.ne_source_rcy[it, :], lw=2, c="orange", label="$S_{e,rcy}$"
        )

        ax.plot(
            self.x_lw,
            self.ne_source_rcy[it, :] + self.ne_source_gf[it, :],
            lw=1,
            ls="--",
            c="y",
            label="check ($S_{e,rcy}$+$S_{e,gf}$)",
        )

        ax.plot(self.x_lw, self.ne_source_gfD[it, :], lw=1, c="r", label="$S_{e,gf D}$")
        ax.plot(
            self.x_lw,
            self.ne_source_gfT[it, :],
            lw=1,
            c="r",
            ls="--",
            label="$S_{e,gf T}$",
        )
        ax.plot(
            self.x_lw,
            self.ne_source_gfHe4[it, :],
            lw=1,
            c="r",
            ls=":",
            label="$S_{e,gf He4}$",
        )
        ax.plot(
            self.x_lw,
            self.ne_source_rcyD[it, :],
            lw=1,
            c="orange",
            label="$S_{e,rcy D}$",
        )
        ax.plot(
            self.x_lw,
            self.ne_source_rcyT[it, :],
            lw=1,
            c="orange",
            ls="--",
            label="$S_{e,rcy T}$",
        )
        ax.plot(
            self.x_lw,
            self.ne_source_rcyHe4[it, :],
            lw=1,
            c="orange",
            ls=":",
            label="$S_{e,rcy He4}$",
        )

        ax.set_title("Electron Sources (division 2)")
        ax.set_ylabel("$10^{20}m^{-3}/s$")
        ax.set_xlabel("$\\rho_N$")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax.set_ylim([min_Se, max_Se])
        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        # DT sources
        ax = ax4

        ax.plot(self.x_lw, self.nD_source[it], lw=2, c="b", label="$S_{D}$")
        ax.plot(self.x_lw, self.nD_source_wall[it], lw=2, c="r", label="$S_{D,wall}$")
        ax.plot(self.x_lw, self.nD_source_vol[it], lw=2, c="g", label="$S_{D,vol}$")

        ax.plot(
            self.x_lw,
            self.nD_source_beams[it],
            lw=1,
            c="k",
            ls="--",
            label="$S_{D,vol,beam}$",
        )
        ax.plot(
            self.x_lw,
            self.nD_source_halo[it],
            lw=1,
            c="c",
            ls="--",
            label="$S_{D,vol,halo}$",
        )

        ax.plot(
            self.x_lw,
            self.nD_source_beams[it] + self.nD_source_halo[it],
            lw=1,
            c="y",
            ls="--",
            label="beam+halo, check",
        )

        if self.nT_source[it].max() > 1e-10:
            ax.plot(
                self.x_lw, self.nT_source[it], lw=2, c="b", ls="--", label="$S_{T}$"
            )
            ax.plot(
                self.x_lw,
                self.nT_source_wall[it],
                lw=2,
                c="r",
                ls="--",
                label="$S_{T,wall}$",
            )
            ax.plot(
                self.x_lw,
                self.nT_source_vol[it],
                lw=2,
                c="g",
                ls="--",
                label="$S_{T,vol}$",
            )

            ax.plot(self.x_lw, self.nmain_source[it], lw=1, c="m", label="$S_{D+T}$")

        ax.set_title("DT Sources")
        ax.set_ylabel("$10^{20}m^{-3}/s$")
        ax.set_xlabel("$\\rho_N$")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        ax.set_ylim([min_Se, max_Se])
        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

    def plotIonsBalance(self, time=None, fig=None, label=""):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(3, 3, hspace=0.4, wspace=0.4)

        # ~~~~~ Balance

        # Deuterium
        ax = fig.add_subplot(grid[0, 0])
        ax.plot(self.x_lw, self.nD_source[it, :], lw=2, label="$S_D$")
        ax.plot(
            self.x_lw, self.nD_divGamma[it, :], lw=2, label="$\\nabla\\cdot\\Gamma_D$"
        )
        ax.plot(self.x_lw, self.nD_dt[it, :], lw=2, label="$dn_D/dt$")

        ax.plot(
            self.x_lw,
            self.nD_source[it, :] - self.nD_divGamma[it, :],
            lw=1,
            c="y",
            ls="--",
            label="check",
        )  # ($S_e$-$\\nabla\\cdot\\Gamma_e$)')

        ax.set_title("Deuterium Balance")
        ax.set_ylabel("$10^{20}m^{-3}/s$")
        ax.set_xlabel("$\\rho_N$")
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})

        ax.set_xlim([0, 0.95])

        GRAPHICStools.addDenseAxis(ax)

        # ax1 = ax.twinx()
        # ax1.plot(self.x_lw,self.nD[it,:],lw=5,alpha=0.2,c='k',label='$n_D$')
        # ax1.legend(loc='upper left',prop={'size':self.mainLegendSize})
        # ax1.set_ylabel('$n_D$ ($10^{20}m^{-3}$)')

        # Tritium
        ax = fig.add_subplot(grid[0, 1])
        ax.plot(self.x_lw, self.nT_source[it, :], lw=2, label="$S_T$")
        ax.plot(
            self.x_lw, self.nT_divGamma[it, :], lw=2, label="$\\nabla\\cdot\\Gamma_T$"
        )
        ax.plot(self.x_lw, self.nT_dt[it, :], lw=2, label="$dn_T/dt$")

        ax.plot(
            self.x_lw,
            self.nT_source[it, :] - self.nT_divGamma[it, :],
            lw=1,
            c="y",
            ls="--",
            label="check",
        )  # ($S_e$-$\\nabla\\cdot\\Gamma_e$)')

        ax.set_title("Tritium Balance")
        ax.set_ylabel("$10^{20}m^{-3}/s$")
        ax.set_xlabel("$\\rho_N$")
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})

        ax.set_xlim([0, 0.95])

        GRAPHICStools.addDenseAxis(ax)

        # ax1 = ax.twinx()
        # ax1.plot(self.x_lw,self.nT[it,:],lw=5,alpha=0.2,c='k',label='$n_T$')
        # ax1.legend(loc='upper left',prop={'size':self.mainLegendSize})
        # ax1.set_ylabel('$n_T$ ($10^{20}m^{-3}$)')

        # He4
        ax = fig.add_subplot(grid[0, 2])
        ax.plot(self.x_lw, self.nHe4_source[it, :], lw=2, label="$S_{He4}$")
        ax.plot(
            self.x_lw,
            self.nHe4_divGamma[it, :],
            lw=2,
            label="$\\nabla\\cdot\\Gamma_{He4}$",
        )
        ax.plot(self.x_lw, self.nHe4_dt[it, :], lw=2, label="$dn_{He4}/dt$")

        ax.plot(
            self.x_lw,
            self.nHe4_source[it, :] - self.nHe4_divGamma[it, :],
            lw=1,
            c="y",
            ls="--",
            label="check",
        )  # ($S_e$-$\\nabla\\cdot\\Gamma_e$)')

        ax.set_title("He4 ash Balance")
        ax.set_ylabel("$10^{20}m^{-3}/s$")
        ax.set_xlabel("$\\rho_N$")
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})

        ax.set_xlim([0, 0.95])

        GRAPHICStools.addDenseAxis(ax)

        # ax1 = ax.twinx()
        # ax1.plot(self.x_lw,self.nHe4[it,:],lw=5,alpha=0.2,c='k',label='$n_{He4}$')
        # ax1.legend(loc='upper left',prop={'size':self.mainLegendSize})
        # ax1.set_ylabel('$n_{He4}$ ($10^{20}m^{-3}$)')

        # ~~~~~ Total

        # Deuterium
        ax = fig.add_subplot(grid[1, 0])
        ax.plot(self.t, self.nD_tot, lw=2)

        ax.set_ylabel("Total particles in plasma $10^{20}$")
        ax.set_xlabel("Time (s)")

        GRAPHICStools.addDenseAxis(ax)

        ax1 = ax.twinx()
        ax1.plot(self.t, self.LD, lw=2, alpha=0.5, c="k")
        ax1.set_ylabel("Total outflux LCFS $10^{20}/s$")

        # Tritium
        ax = fig.add_subplot(grid[1, 1])
        ax.plot(self.t, self.nT_tot, lw=2)

        ax.set_ylabel("Total particles in plasma $10^{20}$")
        ax.set_xlabel("Time (s)")

        GRAPHICStools.addDenseAxis(ax)

        ax1 = ax.twinx()
        ax1.plot(self.t, self.LT, lw=2, alpha=0.5, c="k")
        ax1.set_ylabel("Total outflux LCFS $10^{20}/s$")

        # He4
        ax = fig.add_subplot(grid[1, 2])
        ax.plot(self.t, self.nHe4_tot, lw=2)

        ax.set_ylabel("Total particles in plasma $10^{20}$")
        ax.set_xlabel("Time (s)")

        GRAPHICStools.addDenseAxis(ax)

        ax1 = ax.twinx()
        ax1.plot(self.t, self.LHe4, lw=2, alpha=0.5, c="k")
        ax1.set_ylabel("Total outflux LCFS $10^{20}/s$")

        # ~~~~~ Transport

        # Deuterium
        ax = fig.add_subplot(grid[2, 0])
        ax.plot(self.x_lw, self.Deff_D[it], lw=2, label="$D_{eff}$")

        ax.set_ylabel("$m^2/s$, $m/s$")
        ax.set_xlabel("$\\rho_N$")
        ax.legend()

        GRAPHICStools.addDenseAxis(ax)

        # Tritium
        ax = fig.add_subplot(grid[2, 1])
        ax.plot(self.x_lw, self.Deff_T[it], lw=2, label="$D_{eff}$")

        ax.set_ylabel("$m^2/s$")
        ax.set_xlabel("$\\rho_N$")
        ax.legend()

        GRAPHICStools.addDenseAxis(ax)

        # Tritium
        ax = fig.add_subplot(grid[2, 2])
        ax.plot(self.x_lw, self.Deff_He4[it], lw=2, label="$D_{eff}$")
        ax.plot(self.x_lw, self.D_He4[it], lw=2, label="$D$")
        ax.plot(self.x_lw, self.V_He4[it], lw=2, label="$V$")

        Deff_check = self.D_He4[it] + self.V_He4[it] * self.a[it] / self.aLnHe4_gR[it]
        ax.plot(self.x_lw, Deff_check, lw=1, c="y", ls="--", label="check $D_{eff}$")

        ax.set_ylabel("$m^2/s$, $m/s$")
        ax.set_xlabel("$\\rho_N$")
        ax.legend()

        GRAPHICStools.addDenseAxis(ax)

    def plotGS(self, time=None, fig=None, label=""):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 4, hspace=0.3, wspace=0.5)

        meanbefore = 0.1
        it2 = np.argmin(np.abs(self.t - (self.t[it] - meanbefore)))

        # -----------

        x0 = np.mean(self.psin[it2:it], axis=0)
        y0 = np.mean(self.q[it2:it], axis=0)
        y1 = np.mean(self.q_check[it2:it], axis=0)
        y2 = np.mean(self.q_MHD[it2:it], axis=0)

        ax = fig.add_subplot(grid[0, 0])
        ax.plot(x0, y0, lw=3, c="b", label="$q$")
        ax.plot(x0, y1, c="r", lw=2, label="$q_{check}$")
        ax.plot(x0, y2, c="k", lw=2, label="$q_{MHD}$")
        if self.isolver is not None and self.isolver.q_diffusion is not None:
            y3 = np.mean(self.isolver.q_diffusion[it2:it], axis=0)
            ax.plot(x0, y3, c="g", lw=2, ls="-", label="$q_{ISOLVER}$")
        else:
            y3 = None

        ax.set_title(f"q-profile (t={self.t[it2]:.3f}-{self.t[it]:.3f})")
        ax.set_ylabel("")
        ax.set_xlabel("$\\psi_n$")
        ax.legend(loc="upper center")
        ax.set_ylim([0, np.max(self.q[it])])
        ax.set_xlim([0, 1])
        GRAPHICStools.addDenseAxis(ax)

        ax1 = ax.twinx()
        ax1.plot(x0, np.abs(y0 - y1) / y0 * 100.0, c="r", lw=1, ls="--")
        ax1.plot(x0, np.abs(y0 - y2) / y0 * 100.0, c="k", lw=1, ls="--")
        if y3 is not None:
            ax1.plot(x0, np.abs(y0 - y3) / y0 * 100.0, c="g", lw=1, ls="--")
        ax1.set_ylabel("Relative Error (%)")
        ax1.set_ylim([0, 50.0])

        # -----------
        ax = fig.add_subplot(grid[0, 1])
        ax.plot(self.t, self.Ip, lw=2, c="b", label="$I_p$")
        ax.plot(self.t, self.Ip_eq, lw=2, c="r", label="$I_{p,check}$")
        if self.isolver is not None and self.isolver.Ip_Anom is not None:
            Ip_red = self.Ip - self.isolver.Ip_Anom
            ax.plot(self.t, Ip_red, lw=2, c="g", label="$I_{p}-I_{ISOLVER err}$")
        else:
            Ip_red = None

        ax.set_title("Plasma Current")
        ax.set_ylabel("$MA$")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper center")
        ax.set_ylim([self.Ip[it] * 0.5, self.Ip[it] * 1.5])
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)

        ax1 = ax.twinx()
        ax1.plot(
            self.t, np.abs(self.Ip - self.Ip_eq) / self.Ip * 100.0, c="r", lw=1, ls="--"
        )
        if Ip_red is not None:
            ax1.plot(
                self.t, np.abs(self.Ip - Ip_red) / self.Ip * 100.0, c="g", lw=1, ls="--"
            )
        ax1.set_ylabel("Relative Error (%)")
        ax1.set_ylim([0, 10.0])

        # -----------

        y0 = np.mean(self.phi[it2:it], axis=0)
        y1 = np.mean(self.phi_check[it2:it], axis=0)

        ax = fig.add_subplot(grid[0, 2])
        ax.plot(x0, y0, lw=2, c="b", label="$\\phi$")
        ax.plot(x0, y1, c="r", lw=2, label="$\\phi_{check}$")

        ax.set_title(f"Toroidal Flux (t={self.t[it2]:.3f}-{self.t[it]:.3f})")
        ax.set_ylabel("Wb")
        ax.set_xlabel("$\\psi_n$")
        ax.legend(loc="upper center")
        ax.set_ylim([0, np.max(self.phi[it])])
        ax.set_xlim([0, 1])
        GRAPHICStools.addDenseAxis(ax)

        ax1 = ax.twinx()
        ax1.plot(x0, np.abs(y0 - y1) / y0 * 100.0, c="r", lw=2, ls="--")
        ax1.set_ylabel("Relative Error (%)")
        ax1.set_ylim([0, 2.0])

        # -----------

        y0 = np.mean(self.p_kin[it2:it], axis=0)
        y1 = np.mean(self.p_mhd_in[it2:it], axis=0)
        y2 = np.mean(self.p_mhd_check[it2:it], axis=0)

        ax = fig.add_subplot(grid[1, 0])

        ax.plot(x0, y0, c="b", lw=2, label="$P_{kin}$")
        ax.plot(x0, y1, c="r", ls="-", lw=2, label="$P_{MHD,in}$")
        ax.plot(x0, y2, c="g", lw=2, label="$P_{check}$")

        ax.set_title(f"Pressure (t={self.t[it2]:.3f}-{self.t[it]:.3f})")
        ax.set_ylabel("atm")
        ax.set_xlabel("$\\psi_n$")
        ax.legend(loc="upper center")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)

        ax1 = ax.twinx()
        ax1.plot(x0, np.abs(y0 - y1) / y0 * 100.0, c="r", lw=1, ls="--")
        ax1.plot(x0, np.abs(y0 - y2) / y0 * 100.0, c="g", lw=1, ls="--")
        ax1.set_ylabel("Relative Error (%)")
        ax1.set_ylim([0, 50.0])

        # -----------
        ax = fig.add_subplot(grid[1, 1])
        ax.set_title("Global")
        ax.plot(self.t, self.GS_error, lw=1, c="b", label="$GS_{error}$")
        ax.ticklabel_format(axis="y", useMathText=True)
        ax.axhline(y=0.03, c="b", ls="--", lw=1, label="$GS_{error}$ max")
        ax.set_ylabel("GS Error")
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})
        ax.set_xlabel("Time (s)")
        ax.set_ylim([0, 0.04])
        GRAPHICStools.addDenseAxis(ax)

        ax1 = ax.twinx()
        ax1.plot(self.t, self.TEQ_error, c="r", lw=1, label="$TEQ_{resid}$")
        ax1.axhline(y=1e-6, c="r", ls="--", lw=1, label="$TEQ_{resid}$ goal")
        ax1.set_ylabel("TEQ residue")
        ax1.legend(loc="upper right", prop={"size": self.mainLegendSize})
        mm = 1e-6  # np.max(self.TEQ_error)
        ax1.set_ylim([0, mm * 1.5])

        # -----------

        ax = fig.add_subplot(grid[1, 2])

        y0 = np.mean(self.F[it2:it], axis=0)
        ax.plot(x0, y0, c="b", lw=2, ls="-", label="$R\\cdot B_\\phi$")

        y1 = np.mean(self.Magnetism[it2:it], axis=0) * np.mean(
            self.RBt_vacuum[it2:it], axis=0
        )
        ax.plot(x0, y1, lw=2, c="r", label="$M * (R\\cdot B_\\phi)_v$")

        y2 = np.mean(self.Magnetism_c[it2:it], axis=0) * np.mean(
            self.RBt_vacuum[it2:it], axis=0
        )
        ax.plot(x0, y2, c="g", lw=2, label="$M_c * (R\\cdot B_\\phi)_v$")

        ax.set_title(
            f"F function , $R\\cdot B_\\phi$ (t={self.t[it2]:.3f}-{self.t[it]:.3f})"
        )
        ax.set_ylabel("")
        ax.set_xlabel("$\\psi_n$")
        ax.legend(loc="upper center")
        GRAPHICStools.addDenseAxis(ax)

        ax1 = ax.twinx()
        ax1.plot(x0, np.abs(y2 - y0) / y0 * 100.0, c="r", lw=2, ls="--")
        ax1.plot(x0, np.abs(y2 - y1) / y1 * 100.0, c="g", lw=2, ls="--")
        ax1.set_ylabel("Relative Error (%)")
        ax1.set_ylim([0, 2.0])

        ax.set_xlim([0, 1])

        # -----------
        ax = fig.add_subplot(grid[0, 3])
        ax.plot(self.t, self.q95, lw=3, c="c", label="$q_{95}$")

        z = self.q
        q95_check = []
        for it in range(len(self.t)):
            q95_check.append(np.interp(0.95, self.psin[it], z[it]))
        ax.plot(self.t, q95_check, lw=2, c="b", label="$q$ @ $\\psi_n=0.95$")

        z = self.q_check
        q95_check = []
        for it in range(len(self.t)):
            q95_check.append(np.interp(0.95, self.psin[it], z[it]))
        ax.plot(self.t, q95_check, lw=2, c="r", label="$q_{check}$ @ $\\psi_n=0.95$")

        z = self.q_MHD
        q95_check = []
        for it in range(len(self.t)):
            q95_check.append(np.interp(0.95, self.psin[it], z[it]))
        ax.plot(self.t, q95_check, lw=2, c="k", label="$q_{MHD}$ @ $\\psi_n=0.95$")

        if self.isolver is not None and self.isolver.q_diffusion is not None:
            z = self.isolver.q_diffusion
            q95_check = []
            for it in range(len(self.t)):
                q95_check.append(np.interp(0.95, self.psin[it], z[it]))
            ax.plot(
                self.t, q95_check, lw=2, c="g", label="$q_{ISOLVER}$ @ $\\psi_n=0.95$"
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$q_{95}$")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_title("Evaluations of $q_{95}$")

        lastval = self.q95[self.ind_saw]
        ax.set_ylim([lastval - 0.5, lastval + 0.5])

        GRAPHICStools.addDenseAxis(ax)

        # -----------
        axa = fig.add_subplot(grid[1, 3])
        ax = axa
        ax.plot(x0, -self.pprime[it], lw=2, c="b", label="$-p'$")
        ax.plot(
            x0, -1 / self.mu0 * self.FFprime[it], lw=2, c="r", label="$-FF'/\\mu_0$"
        )

        ax.set_xlim([0, 1])
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.axhline(y=0, ls="--", lw=0.5)
        ax.set_ylabel("$p'$ (Pa/(Wb/rad)), $FF'$ $(T*m)^2/(Wb/rad)$")
        ax.set_xlabel("$\\psi_n$")

        GRAPHICStools.addDenseAxis(ax)

    def plotGEO(self, time=None, fig=None, label=""):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        time = self.t[it]

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(nrows=2, ncols=4, hspace=0.3, wspace=0.3)

        ax0 = fig.add_subplot(grid[:, :2])
        ax1 = fig.add_subplot(grid[0, 2])
        ax2 = fig.add_subplot(grid[1, 2], sharex=ax1)
        ax3 = fig.add_subplot(grid[0, 3])
        ax4 = fig.add_subplot(grid[1, 3], sharex=ax3)

        ax = ax0
        _ = self.plotGeometry(
            ax=ax,
            time=time,
            color="b",
            plotComplete=False,
            rhoS=[1 - 1e-6],
            plotSurfs=True,
            plotStructures=False,
            Aspect=False,
            labelS="TRANSP $\\rho=1.0$",
            label="TRANSP bound",
        )

        ax.axhline(y=self.Ymag[it], c="c", ls="--", lw=1)
        ax.axvline(x=self.Rmag[it], c="c", ls="--", lw=1, label="$R_{mag}$")
        ax.axvline(x=self.Rmajor[it], c="y", ls="--", lw=1, label="$R_{major}$")

        ax.axvline(x=self.Rmajor[it] + self.a[it], c="k", ls="--", lw=1, label="a")
        ax.axvline(x=self.Rmajor[it] - self.a[it], c="k", ls="--", lw=1)

        ax.axhline(
            y=self.Ymag[it] + self.b[it], c="m", ls="--", lw=1, label="$\\kappa$"
        )
        ax.axhline(y=self.Ymag[it] - self.b[it], c="m", ls="--", lw=1)

        ax.axvline(
            x=self.Rmajor[it] - self.d[it], c="g", ls="--", lw=1, label="$\\delta$"
        )

        if hasattr(self, "isolver") and self.isolver is not None:
            self.isolver.plotSurfaces(
                ax,
                time,
                levels=[1.0],
                colorBoundary="r",
                colorSurfs="r",
                lw=[0.5, 0.5],
                label="ISOLVER",
            )

        ax.set_xlim(
            [(self.Rmajor[it] - self.a[it]) * 0.8, (self.Rmajor[it] + self.a[it]) * 1.2]
        )
        ax.set_ylim(
            [(self.Ymag[it] - self.b[it]) * 1.2, (self.Ymag[it] + self.b[it]) * 1.2]
        )

        ax.legend(loc="lower left", fontsize=8)
        # ax.set_aspect('equal')

        ax = ax1
        ax.plot(
            self.t, self.kappa, c="b", lw=2, ls="-", label="$\\kappa_{TRANSP,sep}$ "
        )
        ax.plot(
            self.t,
            self.kappa_995,
            c="b",
            lw=1,
            ls="-",
            label="$\\kappa_{TRANSP,99.5\\%}$",
        )
        ax.plot(
            self.t,
            self.kappa_950,
            c="b",
            lw=1,
            ls="--",
            label="$\\kappa_{TRANSP,95\\%}$",
        )
        if hasattr(self, "isolver") and self.isolver is not None:
            kappa_isolver = (
                (
                    np.abs(self.isolver.xpoints[0][1])
                    + np.abs(self.isolver.xpoints[1][1])
                )
                / 2
                / self.a
            )
            ax.plot(
                self.t, kappa_isolver, c="r", lw=2, ls="-", label="$\\kappa_{ISOLVER}$"
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Elongation")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        GRAPHICStools.addDenseAxis(ax)

        ax = ax2
        ax.plot(self.t, self.delta, c="b", lw=2, ls="-", label="$\\delta_{TRANSP,sep}$")
        ax.plot(
            self.t,
            self.delta_995,
            c="b",
            lw=1,
            ls="-",
            label="$\\delta_{TRANSP,99.5\\%}$",
        )
        ax.plot(
            self.t,
            self.delta_950,
            c="b",
            lw=1,
            ls="--",
            label="$\\delta_{TRANSP,95\\%}$",
        )
        if hasattr(self, "isolver") and self.isolver is not None:
            deltaUp_isolver = (self.Rmajor - self.isolver.xpoints[0][0]) / self.a
            deltaLow_isolver = (self.Rmajor - self.isolver.xpoints[1][0]) / self.a
            delta_isolver = 0.5 * (deltaLow_isolver + deltaUp_isolver)
            ax.plot(
                self.t, delta_isolver, c="r", lw=2, ls="-", label="$\\delta_{ISOLVER}$"
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Triangularity")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        GRAPHICStools.addDenseAxis(ax)

        ax = ax3
        ax.plot(self.x_lw, self.dvol[it], "-o", c="b", lw=1, markersize=3)
        ax.axvline(x=0, ls="-", lw=1, c="b")
        ax.axvline(x=1, ls="-", lw=1, c="b")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Zone volume ($m^{3}$)")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)

        ax = ax4
        ax.plot(self.xb_lw, self.S_x[it], "-o", c="b", lw=1, markersize=3)
        ax.axvline(x=0, ls="-", lw=1, c="b")
        ax.axvline(x=1, ls="-", lw=1, c="b")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Surface area ($m^{3}$)")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)

    def plotNuclear(self, time=None, fig=None, label=""):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 2, hspace=0.4, wspace=0.5)

        # -----------
        ax = fig.add_subplot(grid[0, 0])
        ax.plot(self.t, self.nT_tot_mg, lw=3, label="$mg$ Tritium")
        ax.plot(self.t, self.nHe4_tot_mg, lw=3, label="$mg$ He4 ash")

        ax.set_title("Plasma Store")
        ax.set_ylabel("Total content ($mg$)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)

        # -----------
        ax = fig.add_subplot(grid[0, 1])
        ax.plot(self.t, self.neutrons * 1e20, lw=3, label="Total")
        ax.plot(self.t, self.neutrons_thr * 1e20, lw=2, label="Thermal")

        check_neutrons_thermal = self.neutrons_thr * 0.0
        if self.neutrons_thrDT.max() > self.eps00:
            check_neutrons_thermal += self.neutrons_thrDT
            ax.plot(self.t, self.neutrons_thrDT * 1e20, lw=1, label="Thermal DT")

        if self.neutrons_thrDD.max() > self.eps00:
            check_neutrons_thermal += self.neutrons_thrDD
            ax.plot(self.t, self.neutrons_thrDD * 1e20, lw=1, label="Thermal DD")

        if self.neutrons_thrTT.max() > self.eps00:
            check_neutrons_thermal += self.neutrons_thrTT
            ax.plot(self.t, self.neutrons_thrTT * 1e20, lw=1, label="Thermal TT")

        if check_neutrons_thermal.max() > self.eps00:
            ax.plot(
                self.t,
                check_neutrons_thermal * 1e20,
                lw=1,
                c="y",
                ls="--",
                label="check thermal",
            )

        if self.neutrons_beamtarget.max() > self.eps00:
            ax.plot(self.t, self.neutrons_beamtarget * 1e20, lw=2, label="beam-target")
            ax.plot(self.t, self.neutrons_beambeam * 1e20, lw=2, label="beam-beam")
            ax.plot(
                self.t,
                (self.neutrons_beambeam + self.neutrons_beamtarget + self.neutrons_thr)
                * 1e20,
                lw=1,
                c="y",
                ls="--",
                label="check",
            )

        if self.neutrons_m.max() > 0:
            ax.plot(self.t, self.neutrons_m * 1e20, lw=1, ls="--", label="Measured")

        ax.set_title("Neutron rate")
        ax.set_ylabel("Neutron rate ($/s$)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

        # -----------
        ax = fig.add_subplot(grid[1, 0])
        ax.plot(self.x_lw, self.neutrons_x[it], lw=3, label="Total")
        ax.plot(self.x_lw, self.neutrons_thr_x[it], lw=2, label="Thermal")
        if self.neutrons_beamtarget.max() > self.eps00:
            ax.plot(
                self.x_lw, self.neutrons_beamtarget_x[it], lw=2, label="beam-target"
            )
            ax.plot(self.x_lw, self.neutrons_beambeam_x[it], lw=2, label="beam-beam")

        ax.set_title("Neutron rate density")
        ax.set_ylabel("Neutron rate ($10^{20}/s/m^{3}$)")
        ax.set_xlabel("$\\rho_N$")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)

        # -----------
        ax = fig.add_subplot(grid[1, 1])
        ax.bar(
            self.x_lw,
            self.neutrons_x_cum[it],
            width=(self.x_lw[1] - self.x_lw[0]) * 0.8,
            label="Total",
        )
        ax.bar(
            self.x_lw,
            self.neutrons_thr_x_cum[it],
            width=(self.x_lw[1] - self.x_lw[0]) * 0.5,
            label="Thermal",
        )
        if self.neutrons_beamtarget.max() > self.eps00:
            ax.bar(
                self.x_lw,
                self.neutrons_beamtarget_x_cum[it],
                width=(self.x_lw[1] - self.x_lw[0]) * 0.3,
                label="beam-target",
            )
            ax.bar(
                self.x_lw,
                self.neutrons_beambeam_x_cum[it],
                width=(self.x_lw[1] - self.x_lw[0]) * 0.2,
                label="beam-beam",
            )

        ax.set_title("Neutron rate per zone-volume")
        ax.set_ylabel("Neutron rate ($10^{20}/s$)")
        ax.set_xlabel("$\\rho_N$")
        ax.set_xlim([0, 1])
        ax.legend(loc="upper right", prop={"size": self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)

    def plotDivertor(self, time=None, fig=None, label=""):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 2, hspace=0.4, wspace=0.5)

        # -----------
        ax = fig.add_subplot(grid[0, 0])
        ax.plot(self.t, self.P_LCFS, lw=2, label="$P_{SOL}$")

        ax.set_title("Power to the LCFS")
        ax.set_ylabel("Power ($MW$)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_ylim(bottom=0.0)

        GRAPHICStools.addDenseAxis(ax)

        # -----------
        ax = fig.add_subplot(grid[1, 0])
        ax.plot(self.t, self.Energy_LCFS, lw=2, label="$\\int P_{SOL}dt$")

        ax.set_title("Accumulated Energy ($MJ$)")
        ax.set_ylabel("Energy ($MJ$)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_ylim(bottom=0.0)

        GRAPHICStools.addDenseAxis(ax)

        # -----------
        ax = fig.add_subplot(grid[0, 1])
        ax.plot(self.t, self.Lambda_q_Eich14, lw=2, label="$\\lambda_q$ Eich 1")
        ax.plot(self.t, self.Lambda_q_Eich15, lw=2, label="$\\lambda_q$ Eich 2")
        ax.plot(self.t, self.Lambda_q_Brunner, lw=2, label="$\\lambda_q$ Brunner")
        ax.plot(self.t, self.Lambda_q_Goldston, lw=2, label="$\\lambda_q$ Goldston")

        ax.set_title("$\\lambda_q$")
        ax.set_ylabel("$\\lambda_q$ (mm)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_ylim(bottom=0.0)

        GRAPHICStools.addDenseAxis(ax)

        # -----------
        ax = fig.add_subplot(grid[1, 1])
        ax.plot(self.t, self.Te_u_Brunner, lw=2, c="r", label="$T_{e,u}$ Brunner")
        ax.plot(
            self.t, self.Ti_u_Brunner, lw=2, c="r", ls="--", label="$T_{i,u}$ Brunner"
        )
        ax.plot(self.t, self.Te_u_Eich14, lw=1, c="b", label="$T_{e,u}$ Eich 1")
        ax.plot(self.t, self.Te_u_Eich15, lw=1, c="g", label="$T_{e,u}$ Eich 2")
        ax.plot(self.t, self.Te_u_Goldston, lw=1, c="m", label="$T_{e,u}$ Goldston")
        ax.plot(self.t, self.Te[:, -1], lw=1, c="y", label="$T_{e,LCFS}$ check")
        ax.plot(
            self.t, self.Ti[:, -1], lw=1, ls="--", c="y", label="$T_{i,LCFS}$ check"
        )

        ax.set_title("Upstream Temperatures")
        ax.set_ylabel("$T_{upstream}$ (keV)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_ylim(bottom=0.0)

        GRAPHICStools.addDenseAxis(ax)

    def plotSlowDown(self, fig=None, time=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        time = self.t[it]

        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.2)

        # _________________________________

        ax = fig.add_subplot(grid[0, 0])
        ax.plot(
            self.x_lw,
            self.neutrons_thrDT_x[it],
            lw=2,
            label="$S_{fast,\\alpha}$",
            c="r",
        )
        ax.plot(self.x_lw, self.nHe4_source[it], lw=2, label="$S_{thr,\\alpha}$", c="b")

        ax.set_title("Source Density")
        ax.set_ylabel("$10^{20}/s/m^3$")
        ax.set_xlabel("$\\rho_N$")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.legend()

        GRAPHICStools.addDenseAxis(ax)

        #
        ax = fig.add_subplot(grid[0, 1])
        ax.plot(self.t, self.neutrons_thrDT, lw=2, label="$S_{fast,\\alpha}$", c="r")
        ax.plot(self.t, self.nHe4_sourceT, lw=2, label="$S_{thr,\\alpha}$", c="b")

        ax.set_title("Total source")
        ax.set_ylabel("Total $10^{20}/s$")
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

        ax = fig.add_subplot(grid[1, 0])
        i1, i2 = prepareTimingsSaw(time, self)
        ax.plot(
            self.x_lw,
            self.nfusHe4[i1],
            lw=2,
            label=f"$n_{{fast,\\alpha}}$, t = {self.t[i1]:.3f}s",
            c="r",
        )
        ax.plot(self.x_lw, self.nHe4[i1], lw=2, label="$n_{thr,\\alpha}$", c="b")
        ax.plot(
            self.x_lw,
            self.nfusHe4[i2],
            lw=2,
            c="r",
            ls="--",
            label=f"$n_{{fast,\\alpha}}$, t = {self.t[i2]:.3f}s",
        )
        ax.plot(
            self.x_lw, self.nHe4[i2], lw=2, c="b", ls="--", label="$n_{thr,\\alpha}$"
        )

        ax.set_title("Particle Density")
        ax.set_ylabel("$10^{20}/m^3$")
        ax.set_xlabel("$\\rho_N$")
        ax.set_xlim([0, 1])
        ax.legend()
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

        #
        ax = fig.add_subplot(grid[1, 1])
        ax.plot(self.t, self.nfusHe4_avol, lw=2, label="$n_{fast,\\alpha}$", c="r")
        ax.plot(self.t, self.nHe4_avol, lw=2, label="$n_{thr,\\alpha}$", c="b")
        if self.taup_He4.max() > 1e-10:
            for i in [1, 2, 3]:
                GRAPHICStools.drawLineWithTxt(
                    ax,
                    self.t[0] + self.taup_He4[it] / 1000.0 * i,
                    label=str(i) + "x $\\tau_{p,He4}$",
                    orientation="vertical",
                    color="m",
                    lw=5,
                    ls="-",
                    alpha=0.2,
                    fontsize=10,
                )

        ax.set_title("Volume average density and concentration")
        ax.set_ylabel("$10^{20}/m^3$")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})
        maxy = np.max([np.max(self.nfusHe4_avol), np.max(self.nHe4_avol)])
        ax.set_ylim([0, maxy * 1.2])

        GRAPHICStools.addDenseAxis(ax)

        ax = ax.twinx()
        ax.plot(
            self.t,
            self.ffusHe4_avol * 100.0,
            lw=2,
            ls="--",
            label="$f_{fast,\\alpha}$",
            c="r",
        )
        ax.plot(
            self.t,
            self.fHe4_avol * 100.0,
            lw=2,
            ls="--",
            label="$f_{thr,\\alpha}$",
            c="b",
        )
        ax.set_ylabel("Concentration $n/n_e$ (%)")
        ax.legend(loc="center left")
        maxy = np.max([np.max(self.ffusHe4_avol), np.max(self.fHe4_avol)]) * 100.0
        ax.set_ylim([0, maxy * 1.5])

    def plotFastTransport(self, fig=None, time=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        time = self.t[it]

        grid = plt.GridSpec(2, 2, hspace=0.3, wspace=0.2)

        # _________________________________

        ax = fig.add_subplot(grid[0, 0])
        ax1 = ax.twinx()
        ax.plot(self.x_lw, self.D_f[it], lw=2, label="$D_{anom,fast}$", c="r")
        ax1.plot(self.x_lw, self.V_f[it], lw=2, label="$V_{anom,fast}$", c="b")

        ax.set_title("Anomalous Diffusion")
        ax.set_ylabel("$m^2/s$")
        ax1.set_ylabel("$m/s$")
        ax.set_xlabel("$\\rho_N$")
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})
        ax1.legend(loc="upper right", prop={"size": self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.addDenseAxis(ax1)

        ax = fig.add_subplot(grid[0, 1])
        ax.plot(self.x_lw, self.Jr_anom[it] * 1e6, lw=2, c="r")

        ax.set_title("Anomalous radial current density")
        ax.set_ylabel("$J_r$ ($A/m^2$)")
        ax.set_xlabel("$\\rho_N$")
        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        ax = fig.add_subplot(grid[1, 0])
        ax.plot(self.t, self.Pf_loss_orbit_He4, lw=2, label="$P_{loss,orbit}$", c="r")
        ax.plot(self.t, self.Pf_loss_cx_He4, lw=2, label="$P_{loss,cx}$", c="b")

        ax.set_title("Fast He4 Power")
        ax.set_ylabel("$MW$")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})
        GRAPHICStools.addDenseAxis(ax)

        ax = fig.add_subplot(grid[1, 1])
        ax.plot(self.t, self.Gf_loss_orbit_He4, lw=2, label="$S_{loss,orbit}$", c="r")
        ax.plot(
            self.t,
            self.Gf_loss_orbitPrompt_He4,
            lw=2,
            label="$S_{loss,orbit,prompt}$",
            c="r",
            ls="--",
        )

        ax.set_title("Fast He4 Particles")
        ax.set_ylabel("$10^{20}n/s$")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", prop={"size": self.mainLegendSize})

        GRAPHICStools.addDenseAxis(ax)

        # self.Pf_loss_orbit 		= self.f['BPLIM'] * 1E-6 # in MW
        # #self.Pf_loss_orbitPrompt 	= self.Pf_loss_orbit * self.f['BSORBPR'][:]

        # self.Gf_loss_cx = self.f['SBCXX'][:] * 1E-20 # in 1E20 particles/s

        # # ~~~~ He4

        # self.GainHe4fast = self.f['DNFDT_4'][:]
        # self.Pf_loss_orbit_He4 	= self.f['BFLIM_4'] * 1E-6 # in MW

        # self.Pf_loss_cx_He4 	= ( self.f['BFCXI_4'][:]+self.f['BFCXX_4'][:] ) * 1E-6 # in MW
        # self.Pf_thermalized_He4 = self.f['BFTH_4'][:] * 1E-6 # in MW
        # self.Pf_stored_He4 		= self.f['BFST_4'][:] * 1E-6 # in MW

        # self.Gf_loss_orbit_He4 			= self.f['FSORB_4'][:] * 1E-20 # in 1E20 particles/s
        # self.Gf_loss_orbitPrompt_He4 	= self.f['FSORBPR_4'][:] * self.Gf_loss_orbit

    def plotNeutrals(self, fig=None, time=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 2, hspace=0.2, wspace=0.2)

        # _________________________________

        ax = fig.add_subplot(grid[0, 0])
        if self.nT0[it, :].max() > 1e-10:
            ax.plot(self.x_lw, self.nmain0[it, :], lw=3, label="$n^0_{D+T}$", c="r")

        ax.plot(self.x_lw, self.nD0[it, :], lw=2, label="$n^0_{D}$", c="b", ls="-")
        ax.plot(
            self.x_lw, self.nD0_vol[it, :], lw=2, label="$n^0_{D,vol}$", c="g", ls="-"
        )
        ax.plot(
            self.x_lw, self.nD0_wall[it, :], lw=2, label="$n^0_{D,wall}$", c="m", ls="-"
        )

        ax.plot(
            self.x_lw, self.nD0_recy[it, :], lw=2, label="$n^0_{D,recy}$", c="c", ls="-"
        )
        ax.plot(
            self.x_lw,
            self.nD0_gasf[it, :],
            lw=2,
            label="$n^0_{D,gas}$",
            c="orange",
            ls="-",
        )

        if self.nT0[it, :].max() > 1e-10:
            ax.plot(self.x_lw, self.nT0[it, :], lw=2, label="$n^0_{T}$", c="b", ls="--")
            ax.plot(
                self.x_lw,
                self.nT0_vol[it, :],
                lw=2,
                label="$n^0_{T,vol}$",
                c="g",
                ls="--",
            )
            ax.plot(
                self.x_lw,
                self.nT0_wall[it, :],
                lw=2,
                label="$n^0_{T,wall}$",
                c="m",
                ls="--",
            )

            ax.plot(
                self.x_lw,
                self.nT0_recy[it, :],
                lw=2,
                label="$n^0_{T,recy}$",
                c="c",
                ls="--",
            )
            ax.plot(
                self.x_lw,
                self.nT0_gasf[it, :],
                lw=2,
                label="$n^0_{T,gas}$",
                c="orange",
                ls="--",
            )

        setlim = 0.8
        maxy = 1.2 * np.max(self.nmain0[it, : np.argmin(np.abs(self.x_lw - setlim))])

        GRAPHICStools.addLegendApart(ax, ratio=0.7, withleg=True)
        ax.set_ylim([0, maxy])
        ax.set_ylabel("$n$ ($10^{20}m^{-3}$)")
        ax.set_xlabel("$\\rho_N$")
        ax.set_title("Neutral Density")
        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        ax = fig.add_subplot(grid[1, 0])

        ax.plot(self.x_lw, self.Ti[it, :], lw=2, label="$T_{i}$", c="k", ls="-")

        ax.plot(
            self.x_lw, self.TD0_vol[it, :], lw=2, label="$T^0_{D,vol}$", c="g", ls="-"
        )
        ax.plot(
            self.x_lw, self.TD0_wall[it, :], lw=2, label="$T^0_{D,wall}$", c="m", ls="-"
        )

        if self.nT0[it, :].max() > 1e-10:
            ax.plot(
                self.x_lw,
                self.TT0_vol[it, :],
                lw=2,
                label="$T^0_{T,vol}$",
                c="g",
                ls="--",
            )
            ax.plot(
                self.x_lw,
                self.TT0_wall[it, :],
                lw=2,
                label="$T^0_{T,wall}$",
                c="m",
                ls="--",
            )

        GRAPHICStools.addLegendApart(ax, ratio=0.7, withleg=True)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("$T$ (keV)")
        ax.set_xlabel("$\\rho_N$")
        ax.set_title("Neutral Temperature")
        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

        # Sources

        # ax = fig.add_subplot(grid[0,1])

        # ax.plot(self.x_lw,self.pmain0[it,:]*1E6,lw=5,label='$p^0_{D+T}$',c='r')

        # ax.legend(loc='best',prop={'size':self.mainLegendSize})
        # ax.set_ylim(bottom=0)
        # ax.set_ylabel('$p$ (Pa)')
        # ax.set_xlabel('$\\rho_N$')
        # ax.set_title('Sources')
        # ax.set_xlim([0,1])

        # PRESSURE

        ax = fig.add_subplot(grid[1, 1])

        if self.nT0[it, :].max() > 1e-10:
            ax.plot(
                self.x_lw, self.pmain0[it, :] * 1e6, lw=3, label="$p^0_{D+T}$", c="r"
            )

        ax.plot(
            self.x_lw,
            self.pD0_vol[it, :] * 1e6,
            lw=2,
            label="$p^0_{D,vol}$",
            c="g",
            ls="-",
        )
        ax.plot(
            self.x_lw,
            self.pD0_wall[it, :] * 1e6,
            lw=2,
            label="$p^0_{D,wall}$",
            c="m",
            ls="-",
        )

        if self.nT0[it, :].max() > 1e-10:
            ax.plot(
                self.x_lw,
                self.pT0_vol[it, :] * 1e6,
                lw=2,
                label="$p^0_{T,vol}$",
                c="g",
                ls="--",
            )
            ax.plot(
                self.x_lw,
                self.pT0_wall[it, :] * 1e6,
                lw=2,
                label="$p^0_{T,wall}$",
                c="m",
                ls="--",
            )

        GRAPHICStools.addLegendApart(ax, ratio=0.7, withleg=True)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("$p$ (Pa)")
        ax.set_xlabel("$\\rho_N$")
        ax.set_title("Neutral Pressure")
        ax.set_xlim([0, 1])

        GRAPHICStools.addDenseAxis(ax)

    def plotPerformance(self, fig=None, time=None):
        if time is None:
            it = self.ind_saw
        else:
            it = np.argmin(np.abs(self.t - time))

        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(2, 2, hspace=0.2, wspace=0.3)

        # _________________________________

        ax = fig.add_subplot(grid[0, 0])
        ax.plot(self.t, self.Q, lw=2, label="$Q_{plasma}$")
        ax.plot(self.t, self.Q_corrected_dWdt, lw=0.2, label="$Q_{plasma,-dWdt}$")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        if self.Q[-1] > 0.1:
            ax.axhline(y=1.0, c="k", ls="--")
            ax.text(
                self.t[0] + 0.1,
                1.2,
                "Breakeven",
                color="k",
                fontsize=10,
                horizontalalignment="left",
                verticalalignment="center",
            )  # , transform=ax.transAxes)
            ax.axhline(y=2.0, c="k", ls="--")
            ax.text(
                self.t[0] + 0.1,
                2.2,
                "SPARC Mission",
                color="k",
                fontsize=10,
                horizontalalignment="left",
                verticalalignment="center",
            )  # , transform=ax.transAxes)
            ax.axhline(y=5.0, c="k", ls="--")
            ax.text(
                self.t[0] + 0.1,
                5.2,
                "Burning Plasma",
                color="k",
                fontsize=10,
                horizontalalignment="left",
                verticalalignment="center",
            )  # , transform=ax.transAxes)
            ax.axhline(y=10.0, c="k", ls="--")
            ax.text(
                self.t[0] + 0.1,
                10.2,
                "ITER Goal",
                color="k",
                fontsize=10,
                horizontalalignment="left",
                verticalalignment="center",
            )  # , transform=ax.transAxes)
            ax.axhline(y=11.0, c="k", ls="--")
            ax.text(
                self.t[0] + 0.1,
                11.2,
                "",
                color="k",
                fontsize=10,
                horizontalalignment="left",
                verticalalignment="center",
            )  # , transform=ax.transAxes)
            ax.set_ylim([0, np.max([12.0, self.Q[-1] * 1.1])])

        ax.set_ylabel("$Q$")
        ax.set_xlabel("Time (s)")

        GRAPHICStools.addDenseAxis(ax)

        ax = fig.add_subplot(grid[1, 0], sharex=ax)
        ax.plot(self.t, self.Pout, lw=2, label="$P_{out}$")
        ax.axhline(y=16.0, c="k", ls="--")
        if self.Pout[-1] > 1:
            ax.text(
                self.t[0] + 0.1,
                16.2,
                "Fusion record (JET)",
                color="k",
                fontsize=10,
                horizontalalignment="left",
                verticalalignment="center",
            )  # , transform=ax.transAxes)
        ax.plot(self.t, self.utilsiliarPower, lw=3, label="$P_{in}$")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_ylabel("$P$ (MW)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(bottom=0)

        GRAPHICStools.addDenseAxis(ax)

        ax = fig.add_subplot(grid[0, 1], sharex=ax)
        ax.plot(self.t, self.H98y2_check, lw=2, c="r", label="$H_{98,y2}$")
        ax.plot(
            self.t,
            self.H98y2_tot_check,
            lw=1,
            c="r",
            ls="--",
            label="$H_{98,y2}$ w/fast",
        )
        ax.plot(self.t, self.H98y2, lw=1, c="y", ls="--", label="$H_{98,y2}$ (TRANSP)")

        ax.plot(self.t, self.H89p_check, lw=2, c="g", label="$H_{89p}$")
        ax.plot(self.t, self.H89p, lw=1, c="g", ls="--", label="$H_{89p}$ (TRANSP)")
        ax.plot(self.t, self.HNA, lw=2, c="k", label="$H_{Neo-Alc}$")

        GRAPHICStools.drawSpaningWithTxt(
            ax,
            1.15,
            0.85,
            labelU="H = 1.15 ($+1\\sigma$)",
            labelD="H = 0.85 ($-1\\sigma$)",
            orientation="horizontal",
            color="m",
            alpha=0.1,
            ls="--",
            lw=2,
            fontsize=10,
            fromtop=0.7,
            centralLine=True,
            extra=0.05,
        )

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_ylabel("$H$")
        ax.set_xlabel("Time (s)")
        try:
            ax.set_ylim([0, np.max([2, self.H89p_check[-1] * 1.1])])
        except:
            pass

        GRAPHICStools.addLegendApart(ax, ratio=0.7, withleg=True)

        GRAPHICStools.addDenseAxis(ax)

        ax = fig.add_subplot(grid[1, 1], sharex=ax)
        ax.plot(self.t, self.taue * 1e-3, lw=2, c="b", label="$\\tau_E$")
        ax.plot(
            self.t,
            self.taue_check * 1e-3,
            lw=1,
            ls="--",
            c="y",
            label="$\\tau_{E}$ check",
        )
        ax.plot(
            self.t, self.taueTot * 1e-3, ls="--", c="b", lw=2, label="$\\tau_{E,tot}$"
        )
        ax.plot(
            self.t,
            self.taueTot_check * 1e-3,
            c="y",
            lw=1,
            ls="--",
            label="$\\tau_{E,tot}$ check",
        )
        ax.plot(self.t, self.tau98y2_check * 1e-3, c="m", lw=2, label="$\\tau_{98,y2}$")
        ax.plot(
            self.t,
            self.tau98y2 * 1e-3,
            lw=1,
            c="m",
            ls="--",
            label="$\\tau_{98,y2}$ (TRANSP)",
        )
        ax.plot(self.t, self.tau89p_check * 1e-3, c="g", lw=2, label="$\\tau_{89p}$")
        ax.plot(
            self.t,
            self.tau89p * 1e-3,
            lw=1,
            c="g",
            ls="--",
            label="$\\tau_{89p}$ (TRANSP)",
        )
        if self.taup_He4[-1] > 0.0:
            ax.plot(self.t, self.taup_He4 * 1e-3, lw=2, c="c", label="$\\tau_{p,He4}$")

        ax.legend(loc="best", prop={"size": self.mainLegendSize})
        ax.set_ylabel("Confinement time (s)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim([0, 1])

        GRAPHICStools.addLegendApart(ax, ratio=0.7, withleg=True, extraPad=0.2)

        GRAPHICStools.addDenseAxis(ax)

        axs = ax.twinx()
        ax = axs
        ax.plot(self.t, self.nTtau, lw=2, c="c", label="$n_{i,0}T_{i,0}\\tau_E$")
        ax.set_ylabel("$n_{i,0}T_{i,0}\\tau_E$ ($10^{21}keVm^{-3}s$)")
        ax.legend(loc="best", prop={"size": self.mainLegendSize})

        GRAPHICStools.addLegendApart(ax, ratio=0.7, withleg=False)

    def plot(self, fn=None, time=None, timesAv=None, plot_analysis=False, tab_color=0):
        if time is None:
            time = self.t[self.ind_saw]

        name = f"MITIM Notebook, run #{self.nameRunid}, profiles at time t={time:.3f}s"
        fn_color = tab_color if tab_color > 0 else None

        if fn is None:
            self.fn = FigureNotebook(name)
        else:
            self.fn = fn

        # Machine
        fig = self.fn.add_figure(tab_color=fn_color, label="Machine")
        self.plotMachine(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Equil
        fig = self.fn.add_figure(tab_color=fn_color, label="Equilibrium")
        self.plotEquilParams(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # GS
        fig = self.fn.add_figure(tab_color=fn_color, label="Grad-Shafranov")
        self.plotGS(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Geometry
        fig = self.fn.add_figure(tab_color=fn_color, label="Geometry")
        self.plotGEO(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Profiles
        fig = self.fn.add_figure(tab_color=fn_color, label="Profiles")
        self.plotProfiles(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # PRESSURES
        fig = self.fn.add_figure(tab_color=fn_color, label="Pressure")
        self.plotPressures(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Systems
        fig = self.fn.add_figure(tab_color=fn_color, label="Power (Auxiliary)")
        self.plotSeparateSystems(fig=fig)
        GRAPHICStools.adjust_figure_layout(fig)

        # Heating
        fig = self.fn.add_figure(tab_color=fn_color, label="Power (Total)")
        self.plotHeating(fig=fig)
        GRAPHICStools.adjust_figure_layout(fig)

        # Radial Powe3
        fig = self.fn.add_figure(tab_color=fn_color, label="Power (Radial)")
        fig2 = self.fn.add_figure(tab_color=fn_color, label="Power (Cumul.)")
        self.plotRadialPower(time=time, fig=fig, figCum=fig2)
        GRAPHICStools.adjust_figure_layout(fig)
        GRAPHICStools.adjust_figure_layout(fig2)

        # ICRF
        if np.sum(self.PichT) > 0.0 + self.eps00 * (len(self.t) + 1):
            fig = self.fn.add_figure(tab_color=fn_color, label="ICRF (Total)")
            self.plotICRF_t(fig=fig)
            GRAPHICStools.adjust_figure_layout(fig)
            fig = self.fn.add_figure(tab_color=fn_color, label="ICRF (Radial)")
            self.plotICRF(fig=fig, time=time)
            GRAPHICStools.adjust_figure_layout(fig)

        # ECRF
        if np.sum(self.PechT) > 0.0 + self.eps00 * (len(self.t) + 1):
            fig = self.fn.add_figure(tab_color=fn_color, label="ECRF")
            self.plotECRF(fig=fig, time=time)
            GRAPHICStools.adjust_figure_layout(fig)

        # NBI
        if np.sum(self.PnbiT) > 0.0 + self.eps00 * (len(self.t) + 1):
            fig = self.fn.add_figure(tab_color=fn_color, label="NBI")
            self.plotNBI(fig=fig, time=time)
            GRAPHICStools.adjust_figure_layout(fig)

        # LH
        if np.sum(self.PlhT) > 0.0 + 2 * self.eps00 * (len(self.t) + 1):
            fig = self.fn.add_figure(tab_color=fn_color, label="LowerHyb")
            self.plotLowerHybrid(fig=fig, time=time)
            GRAPHICStools.adjust_figure_layout(fig)

        # Transport
        fig = self.fn.add_figure(tab_color=fn_color, label="Transport")
        self.plotTransport(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Derivatives
        fig = self.fn.add_figure(tab_color=fn_color, label="Gradients")
        self.plotDerivatives(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Porcelli
        fig = self.fn.add_figure(tab_color=fn_color, label="Sawtooth Trigger")
        self.plotSawtooth(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Around sawtooth
        # try:
        fig = self.fn.add_figure(tab_color=fn_color, label="Sawtooth Effect")
        try:
            self.plotAroundSawtoothQuantities(fig=fig)
            GRAPHICStools.adjust_figure_layout(fig)
        except:
            print("\t* Could not plot plotAroundSawtoothQuantities", typeMsg="w")

        fig = self.fn.add_figure(tab_color=fn_color, label="Sawtooth Mixing")
        self.plotSawtoothMixing(fig=fig)
        GRAPHICStools.adjust_figure_layout(fig)

        # Electric Field
        fig = self.fn.add_figure(tab_color=fn_color, label="Current Diffusion")
        self.plotEM(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        fig = self.fn.add_figure(tab_color=fn_color, label="Poynting")
        self.plotUmag(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Fundamental
        fig = self.fn.add_figure(tab_color=fn_color, label="Fundamental Plasma")
        self.plotFundamental(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Stability
        fig = self.fn.add_figure(tab_color=fn_color, label="MHD Stability")
        self.plotStability(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Electric Field
        fig = self.fn.add_figure(tab_color=fn_color, label="Electric Field")
        self.plotElectricField(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Rotation
        fig = self.fn.add_figure(tab_color=fn_color, label="Rotation")
        self.plotRotation(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Time scales
        try:
            fig = self.fn.add_figure(tab_color=fn_color, label="Time Scales")
            self.plotTimeScales(fig=fig, time=time)
            GRAPHICStools.adjust_figure_layout(fig)
        except:
            pass

        fig = self.fn.add_figure(tab_color=fn_color, label="Averaging")
        self.plotTimeAverages(fig=fig, times=timesAv)
        GRAPHICStools.adjust_figure_layout(fig)

        # Impurities
        fig = self.fn.add_figure(tab_color=fn_color, label="Impurities")
        self.plotImpurities(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Radiation
        fig = self.fn.add_figure(tab_color=fn_color, label="Radiation")
        self.plotRadiation(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Convergence
        fig = self.fn.add_figure(tab_color=fn_color, label="Flux Matching")
        self.checkRun(fig=fig, time=time, printYN=False)
        GRAPHICStools.adjust_figure_layout(fig)

        # LH
        fig = self.fn.add_figure(tab_color=fn_color, label="LH Transition")
        self.plotLH(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Particle Balance
        fig = self.fn.add_figure(tab_color=fn_color, label="Particle Balance")
        self.plotParticleBalance(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        fig = self.fn.add_figure(tab_color=fn_color, label="Ions Balance")
        self.plotIonsBalance(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # SLow down
        if self.neutrons_thrDT[-1] > self.eps00:
            fig = self.fn.add_figure(tab_color=fn_color, label="Slow Down")
            self.plotSlowDown(fig=fig, time=time)
            GRAPHICStools.adjust_figure_layout(fig)

        # Fast
        fig = self.fn.add_figure(tab_color=fn_color, label="Fast (Radial)")
        self.plotFast(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Fast
        fig = self.fn.add_figure(tab_color=fn_color, label="Fast (Stabilization)")
        self.plotFast2(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # SLow down
        if self.neutrons_thrDT[-1] > self.eps00:
            fig = self.fn.add_figure(tab_color=fn_color, label="Fast (Transport)")
            self.plotFastTransport(fig=fig, time=time)
            GRAPHICStools.adjust_figure_layout(fig)

        # Neutrals
        fig = self.fn.add_figure(tab_color=fn_color, label="Neutrals")
        self.plotNeutrals(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Performance
        fig = self.fn.add_figure(tab_color=fn_color, label="Performance")
        self.plotPerformance(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Nuclear
        fig = self.fn.add_figure(tab_color=fn_color, label="Neutrons")
        self.plotNuclear(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # Boundary
        fig = self.fn.add_figure(tab_color=fn_color, label="Boundary")
        self.plotDivertor(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # ----------- DIAGRAMS

        # Species
        self.diagramSpecies(time=time, fn=self.fn, label="Species", fn_color=fn_color)

        # Flows
        timeStr = f" (@{time:.3f}s)" if time is not None else f"(@{self.t[self.ind_saw]:.3f}s)"
        self.diagramFlows(time=time, fn=self.fn, label=f"Flows{timeStr}", fn_color=fn_color)

        # CPU
        fig = self.fn.add_figure(tab_color=fn_color, label="CPU usage")
        self.plotCPUperformance(fig=fig, time=time)
        GRAPHICStools.adjust_figure_layout(fig)

        # ----------- EXTRA

        # TORIC
        for i, toric in enumerate(self.torics):
            fig = self.fn.add_figure(tab_color=fn_color, label=f"TORIC #{i+1}")
            self.plotTORIC(fig=fig, position=i)
            GRAPHICStools.adjust_figure_layout(fig)

        # FBM

        if self.fbm_He4_gc is not None:
            fig = self.fn.add_figure(tab_color=fn_color, label="FBM He4 GC")
            self.plotFBM(
                fig=fig, particleFile=self.fbm_He4_gc, finalProfile=self.nfusHe4
            )
            GRAPHICStools.adjust_figure_layout(fig)
            fig = self.fn.add_figure(tab_color=fn_color, label="FBM He4 PO")
            self.plotFBM(
                fig=fig, particleFile=self.fbm_He4_po, finalProfile=self.nfusHe4
            )
            GRAPHICStools.adjust_figure_layout(fig)

        if self.fbm_Dbeam_gc is not None:
            fig = self.fn.add_figure(tab_color=fn_color, label="FBM Dbeam GC")
            self.plotFBM(fig=fig, particleFile=self.fbm_Dbeam_gc, finalProfile=self.nbD)
            GRAPHICStools.adjust_figure_layout(fig)
            fig = self.fn.add_figure(tab_color=fn_color, label="FBM Dbeam PO")
            self.plotFBM(fig=fig, particleFile=self.fbm_Dbeam_po, finalProfile=self.nbD)
            GRAPHICStools.adjust_figure_layout(fig)

            if self.fbm_Dbeam_gc.birth is not None:
                fig = self.fn.add_figure(tab_color=fn_color, label="BIRTH D")
                self.plotBirth(particleFile=self.fbm_Dbeam_gc, fig=fig)
                GRAPHICStools.adjust_figure_layout(fig)

        if self.fbm_T_gc is not None:
            fig = self.fn.add_figure(tab_color=fn_color, label="FBM T GC")
            self.plotFBM(fig=fig, particleFile=self.fbm_T_gc, finalProfile=self.nfusT)
            GRAPHICStools.adjust_figure_layout(fig)
            fig = self.fn.add_figure(tab_color=fn_color, label="FBM T PO")
            self.plotFBM(fig=fig, particleFile=self.fbm_T_po, finalProfile=self.nfusT)
            GRAPHICStools.adjust_figure_layout(fig)

        # TGLF
        if hasattr(self, "TGLF") and self.TGLF is not None:
            figGR = self.fn.add_figure(tab_color=fn_color, label="TGLF1")
            figFL = self.fn.add_figure(tab_color=fn_color, label="TGLF2")
            self.plotTGLF(figGR=figGR, figFL=figFL)
            GRAPHICStools.adjust_figure_layout(figGR)
            GRAPHICStools.adjust_figure_layout(figFL)

        # ~~~~~~~~~~~~~ Comparisons
        if hasattr(self, "exp") and self.exp is not None:
            fig = self.fn.add_figure(tab_color=fn_color, label="EXP")
            self.plotComparison(fig=fig)
            GRAPHICStools.adjust_figure_layout(fig)

        # ~~~~~~~~~~~~~ Comparisons
        if hasattr(self, "isolver") and self.isolver is not None:
            self.plotISOLVER(fn=self.fn, time=time)
            GRAPHICStools.adjust_figure_layout(fig)

        # ~~~~~~~~~~~~~ Final g-file
        # 	Here I make the exception of reading it at plotting, because I may have generated it since loading the class

        self.gfile_out = None
        fileG = self.folderWork / f"RELEASE_folder" / f"TRANSPrun.geq"
        if self.readGEQDSK and fileG.exists():
            try:
                self.gfile_out = GEQtools.MITIMgeqdsk(fileG)
            except:
                print("Could not plot geqdsk")

        if self.gfile_out is not None:
            if self.gfile_in is None:
                ax_plasma = self.gfile_out.plot(fn=self.fn, extraLabel="G_out - ")
            else:
                ax_plasma, fnG = GEQplotting.compareGeqdsk(
                    [self.gfile_in, self.gfile_out],
                    fn=self.fn,
                    labelsGs=["G_in", "G_out"],
                )

            it = np.argmin(np.abs(self.t - time))
            ax = ax_plasma[0]
            ax.plot(
                self.x_lw,
                self.p_kin[it, :],
                "-s",
                c="g",
                lw=0.5,
                markersize=1,
                label="TRANSP $p_{kin}$",
            )
            ax.legend()

            ax = ax_plasma[4]
            ax.plot(
                self.xb_lw,
                self.q[it, :],
                "-s",
                c="g",
                lw=0.5,
                markersize=1,
                label="TRANSP q",
            )
            ax.legend()

            ax = ax_plasma[5]
            ax.plot(
                self.x_lw,
                self.j[it, :],
                "-s",
                c="g",
                lw=0.5,
                markersize=1,
                label="TRANSP",
            )
            ax.legend()

            ax = ax_plasma[6]
            ax.plot(
                self.Rmaj[it, :],
                np.abs(self.Bt_plasma[it, :]),
                "-s",
                c="g",
                lw=0.5,
                markersize=1,
                label="TRANSP Bt",
            )
            ax.plot(
                self.Rmaj[it, :],
                np.abs(self.Bp_ext[it, :]),
                "-s",
                c="c",
                lw=0.5,
                markersize=1,
                label="TRANSP Bp",
            )
            ax.legend()

        # ----------------------------------------------------------------
        # Other Analysis
        # ----------------------------------------------------------------

        if plot_analysis:
            # Pulse
            try:
                fig = self.fn.add_figure(
                    tab_color=fn_color, label="ANALYSIS - Heat Pulse"
                )
                self.plotPulse(fig=fig)
            except:
                pass

            fig = self.fn.add_figure(tab_color=fn_color, label="ANALYSIS - initial")
            self.analyze_initial(fig=fig)
            GRAPHICStools.adjust_figure_layout(fig)

            if len(self.tlastsawU) > 1:
                fig = self.fn.add_figure(
                    tab_color=fn_color, label="ANALYSIS - sawtooth"
                )
                self.analyze_sawtooth(fig=fig)

    # --------------------------------------
    # Additional analysis
    # --------------------------------------

    def getSpecies(self):

        self.Species = {
            "e": {
                "name": "e",
                "type": "thermal",
                "m": self.me,
                "Z": -1*self.t,
                "n": self.ne,
                "T": self.Te,
            }
        }

        # ~~~~~~ Background ions
        if self.nD_avol.max() > 1e-5:
            self.Species["D"] = {
                "name": "D",
                "type": "thermal",
                "m": self.mD,
                "Z": 1*np.ones(len(self.t)),
                "n": self.nD,
                "T": self.Ti,
            }
        if self.nT_avol.max() > 1e-5:
            self.Species["T"] = {
                "name": "T",
                "type": "thermal",
                "m": self.mT,
                "Z": 1*np.ones(len(self.t)),
                "n": self.nT,
                "T": self.Ti,
            }
        if self.nHe4_avol.max() > 1e-5:
            self.Species["He4_ash"] = {
                "name": "He",
                "type": "thermal",
                "m": self.mHe4,
                "Z": 2*np.ones(len(self.t)),
                "n": self.nHe4,
                "T": self.Ti,
            }

        # ~~~~~~ Impurities
        for cont,i in enumerate(self.nZs):

            foundImpurity = False
            if self.LocationNML is not None:
                try:
                    mass = IOtools.findValue(self.LocationNML, f"aimps({cont+1})", "=")
                    foundImpurity = True
                except:
                    pass
                
            if not foundImpurity:
                print(f"\t- Could not find mass for impurity {i} in namelist. Using default value of 2*Zave", typeMsg="w")
                mass =self.fZs_avol[i]['Zave'][self.ind_saw]*2

            self.Species[i+"_imp"] = {
                "name": i,
                "type": "thermal",
                "m": mass  * self.u,
                "Z": self.fZs_avol[i]['Zave'],
                "n": self.nZs[i]["total"],
                "T": self.Ti,
            }

        # ~~~~~~ Minorities
        if self.nminiH.max() > 1e-5:
            self.Species["H_mini"] = {
                "name": "H",
                "type": "fast",
                "m": self.mH,
                "Z": 1*np.ones(len(self.t)),
                "n": self.nminiH,
                "T": self.Tmini,
            }
        if self.nminiHe3.max() > 1e-5:
            self.Species["He3_mini"] = {
                "name": "He",
                "type": "fast",
                "m": self.mHe3,
                "Z": 2*np.ones(len(self.t)),
                "n": self.nminiHe3,
                "T": self.Tmini,
            }

        # ~~~~~~ Fusion
        if self.nfusT.max() > 1e-5:
            self.Species["T_fus"] = {
                "name": "T",
                "type": "fast",
                "m": self.mT,
                "Z": 1*np.ones(len(self.t)),
                "n": self.nfusT,
                "T": self.Tfus,
            }
        if self.nfusHe4.max() > 1e-5:
            self.Species["He4_fus"] = {
                "name": "He",
                "type": "fast",
                "m": self.mHe4,
                "Z": 2*np.ones(len(self.t)),
                "n": self.nfusHe4,
                "T": self.Tfus,
            }
        if self.nfusHe3.max() > 1e-5:
            self.Species["He3_fus"] = {
                "name": "He",
                "type": "fast",
                "m": self.mHe3,
                "Z": 2*np.ones(len(self.t)),
                "n": self.nfusHe3,
                "T": self.Tfus,
            }

    # --------------------------- Convergence ------------------

    def fullMITIMconvergence(
        self,
        minWait,
        timeDifference,
        convCriteria,
        tstart=0,
        phasetxt="",
        checkOnlyOnce=False,
    ):
        numSaw = len(self.tlastsawU)
        currentTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        Rate = self.cptim[-1] / (self.t[-1] - self.t[0])
        RateTXT = IOtools.createTimeTXT(Rate * 3600, until=1)
        MaxTXT = IOtools.createTimeTXT(self.cptim[-1] * 3600, until=1)

        timeTot = self.t[-1] - self.t[0]

        print(
            f"\n----------- SUMMARY FOR RUN {self.nameRunid} ({currentTime}) -----------"
        )
        print(">> Run info:")
        print(f" \t* MITIM phase: {phasetxt}")
        print(
            f" \t* Initial time t={self.t[0]:.3f}s, last time run t={self.t[-1]:.3f}s"
        )
        print(
            f" \t* Run time t={timeTot*1000.0:.0f}ms (took {MaxTXT}; {timeDifference*1000.0:.0f}ms since last check {minWait:.0f}min ago -> rate: {RateTXT} / plasma-sec)"
        )
        print(f" \t* Number of sawtooth crashes: {numSaw}")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~ Define different convergence metrics
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ConvergedByNumSaw = None
        for iC in convCriteria:
            try:
                # If numbercheck<0, convergence is based on how many sawteeth have passed (requires -numbercheck sawteeth to run)
                if convCriteria[iC]["numbercheck"] < 0:
                    ConvergedByNumSaw = -convCriteria[iC]["numbercheck"]
                # If numbercheck>0, convergence is based on variation over this number of sawteetth (requires numbercheck+1 sawteeth to run)
                else:
                    ConvergedByNumSaw = None
            except:
                pass

        # Min time convergence and Variable convergence
        ConvergedRun, ConvergedRun_minTime = self.runConvergenceTest(
            convCriteria, tstart=tstart
        )

        ConvergedRun = ConvergedRun or checkOnlyOnce

        if ConvergedByNumSaw is not None:
            # Does the last sawtooth crash happens after the minimum requested time?
            if len(self.tlastsawU) > 0:
                LastSawtoothAfterTmin = self.tlastsawU[-1] > (
                    self.t[0] + convCriteria["timerunMIN"]
                )
            else:
                LastSawtoothAfterTmin = False
            # Number of sawteeth is enough?]
            EnoughSawteeth = numSaw >= ConvergedByNumSaw

            ConvergedRun = (
                EnoughSawteeth and ConvergedRun_minTime and LastSawtoothAfterTmin
            )
            print(
                f">> Convergence based on running {ConvergedByNumSaw} sawteeth and minimum time (last sawteeth after this time)... {ConvergedRun}"
            )

        print(
            " -----------------------------------------------------------------------------\n"
        )
        if ConvergedRun:
            print("This run is converged")

        return ConvergedRun

    def runConvergenceTest(self, convCriteria, tstart=0.0):
        print(">> Convergence metrics:")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check whether it has run at least the minimum time requested
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        minTime = convCriteria["timerunMIN"]
        Conv_minTime = (self.t[-1] - self.t[0]) > minTime
        print(
            f" \t* Simulation has run for more than at least the minimum requested time ({minTime:.3f}s)? {Conv_minTime}"
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check convergence per variable
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        numSaw, Conv_variables = 0, True
        for iC in convCriteria:
            # ~~~~~ Check variable evolution

            if iC != "timerun" and iC != "timerunMIN" and iC != "transportsolver":
                if convCriteria[iC]["numbercheck"] == 0:
                    numbercheck = -1
                else:
                    numbercheck = convCriteria[iC]["numbercheck"]

                conv, numSaw = self.convergedVariable(
                    iC,
                    numbercheck=numbercheck,
                    tolerance=convCriteria[iC]["tolerance"],
                    sawSmooth=convCriteria[iC]["sawSmooth"],
                    radius=convCriteria[iC]["radius"],
                    tstart=tstart,
                )

                Conv_variables = Conv_variables and conv

            # ~~~~~ Check convergence of transport solver

            elif iC == "transportsolver":
                for rad in convCriteria[iC]["radii"]:
                    conv = self.convergenceSolver(
                        tolerance=convCriteria[iC]["tolerance"],
                        sawSmooth=convCriteria[iC]["sawSmooth"],
                        radius=rad,
                    )

                    Conv_variables = Conv_variables and conv

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Final Convergence
        # 	Note:   Even if variables have not converged,
        # 			but always having covered at least two sawteeth, to avoid that next cold_started run starts again from the beginning!
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ConvergedRun = (Conv_minTime and Conv_variables) and numSaw > 1

        return ConvergedRun, Conv_minTime

    def convergedVariable(
        self,
        var,
        numbercheck=2,
        tolerance=5.0e-2,
        sawSmooth=True,
        radius=None,
        tstart=0.0,
    ):
        t = self.t
        tlastsaw = self.tlastsaw

        # -------- Standard TRANSP variable (only in time)

        if "VarMITIM" not in var:
            if radius is None:
                val = self.f[var][:]
            else:
                rho = self.x_lw
                indexRho = np.argmin([abs(i - radius) for i in rho])
                val = self.f[var][:][:, indexRho]

        # -------- Specially-built variable

        else:
            val = self.SpecialVariables[var]

        # -------- Consider only those times after tstart

        it = 0
        if tstart < t[-2]:
            it = np.argmin(np.abs(t - tstart))

        t, val, tlastsaw = t[it:], val[it:], tlastsaw[it:]

        # ---------------------------------------------------------

        if sawSmooth:
            val, t = MATHtools.smoothThroughSawtooth(t, val, tlastsaw, 1)

        if len(t) > np.abs(numbercheck):
            # differences = np.abs(np.diff(val)) # [ np.abs( (i-j)/j ) for i,j in zip(val[1:],val[0:-1]) ]
            differences = np.abs((val[:-1] - val[-1]) / val[-1])

            (
                ConvergedRun,
                ConvergenceText,
            ) = (
                True,
                f" \t* Variable {var} has changed",
            )
            for i in range(numbercheck):
                ConvergedRun = ConvergedRun and differences[-(i + 1)] < tolerance
                ConvergenceText += " {0:.2f}% (point {1})".format(
                    differences[-(i + 1)] * 100.0, -(i + 1)
                )

            ConvergenceText += f"... all <{tolerance * 100.0:.1f}%? {ConvergedRun}"

            print(ConvergenceText)

        else:
            print(f"\t- Not run for enough time to compare variables {var}")
            ConvergedRun = False

        return ConvergedRun, len(t)

    def convergenceSolver(self, tolerance=5.0e-2, sawSmooth=True, radius=0.5):
        t = self.t
        rho = self.x_lw
        indexRho = np.argmin([abs(i - radius) for i in rho])
        qe_obs = self.qe_obs[:, indexRho]
        qe_model = self.qe_tr[:, indexRho]
        qi_obs = self.qi_obs[:, indexRho]
        qi_model = self.qi_tr[:, indexRho]

        if sawSmooth:
            tlastsaw = self.tlastsaw

            qe_obs, _ = MATHtools.smoothThroughSawtooth(t, qe_obs, tlastsaw, 1)
            qi_obs, _ = MATHtools.smoothThroughSawtooth(t, qi_obs, tlastsaw, 1)
            qe_model, _ = MATHtools.smoothThroughSawtooth(t, qe_model, tlastsaw, 1)
            qi_model, t = MATHtools.smoothThroughSawtooth(t, qi_model, tlastsaw, 1)
            textt = "Sawteeth"
        else:
            textt = "Raw"

        i = -1
        differenceQe = np.abs((qe_obs[i] - qe_model[i]) / qe_obs[i])
        ConvergedQe = differenceQe < tolerance
        print(
            ">> {0} times: {1}, point {2} Qe-solver convergence at rho {3} = {4:.2f}%, satisfied (<{5:.1f}%)? {6}".format(
                textt,
                len(t),
                i,
                radius,
                differenceQe * 100.0,
                tolerance * 100.0,
                ConvergedQe,
            )
        )

        differenceQi = np.abs((qi_obs[i] - qi_model[i]) / qi_obs[i])
        ConvergedQi = differenceQi < tolerance
        print(
            ">> {0} times: {1}, point {2} Qi-solver convergence at rho {3} = {4:.2f}%, satisfied (<{5:.1f}%)? {6}".format(
                textt,
                len(t),
                i,
                radius,
                differenceQi * 100.0,
                tolerance * 100.0,
                ConvergedQi,
            )
        )

        return ConvergedQe and ConvergedQi

    # -----------------------------------------------------------------------------------------------
    # 	Running standalone TGLF from TRANSP outputs
    # -----------------------------------------------------------------------------------------------

    def runTGLFstandalone(
        self,
        time=None,
        avTime=0.0,
        rhos=np.linspace(0.3, 0.9, 11),
        cold_startPreparation=False,
        plotCompare=True,
        extraflag="",
        onlyThermal_TGYRO=False,
        forceIfcold_start=True,
        **kwargs_TGLFrun,
    ):
        """
        Note: If this plasma had fast paricles but not at the time I'm running TGLF, then it will fail if I
        set onlyThermal_TGYRO=False because at that time the particles are zero
        """

        if time is None:
            time = self.t[self.ind_saw]
        elif time == -1:
            time = self.t[-1]

        nameF = int(time * 1000)  # Name it with the original time, not the modified one

        time = self.t[np.argmin(np.abs(self.t - time))]

        for i in range(len(rhos)):
            rhos[i] = round(rhos[i], 2)

        # --------------------------------------------
        #  	PRF workflow to run TGLF with TGLF class
        # --------------------------------------------

        folderGACODE = (
            self.FolderCDF / f'FolderGACODE_{f"{nameF:.0f}".zfill(5)}ms{extraflag}'
        )

        if int(time * 1000) not in self.TGLFstd:
            self.TGLFstd[nameF] = TGLFtools.TGLF(
                cdf=self.LocationCDF, time=time, avTime=avTime, rhos=rhos
            )

        cdf = self.TGLFstd[nameF].prep(
            folderGACODE,
            cold_start=cold_startPreparation,
            onlyThermal_TGYRO=onlyThermal_TGYRO,
            cdf_open=self,
            forceIfcold_start=forceIfcold_start,
        )

        labelTGLF = kwargs_TGLFrun.get("label", "tglf1")

        self.TGLFstd[nameF].run(
            subFolderTGLF=labelTGLF,
            forceIfcold_start=forceIfcold_start,
            **kwargs_TGLFrun,
        )

        self.TGLFstd[nameF].read(label=labelTGLF)

        del self.TGLFstd[nameF].convolution_fun_fluct

        # -----------------------------------------------------------------------------------
        # Compare with TRANSP fluxes
        # -----------------------------------------------------------------------------------

        if plotCompare:
            self.fn_std = FigureNotebook(
                "TGLF-TRANSP Notebook", geometry="1500x900", vertical=True
            )

            self.TGLFstd[nameF].plot(labels=[labelTGLF], fn=self.fn_std)

            fig1 = self.fn_std.add_figure(label="Comparison Flux")
            self.plotStdTRANSP(fig=fig1, tglfRun=labelTGLF, time=time)
            fig2 = self.fn_std.add_figure(label="Comparison GR")
            self.plotGRTRANSP(fig=fig2, tglfRun=labelTGLF, time=time)
            fig3 = self.fn_std.add_figure(label="Comparison FL")
            self.plotFLTRANSP(fig=fig3, tglfRun=labelTGLF, time=time)

        return self.TGLFstd[nameF]

    def transportAnalysis(
        self,
        typeAnalysis="CHIPERT",
        quantity="QE",
        rho=0.50,
        time=None,
        avTime=0.0,
        TGLFsettings=1,
        d_perp_cm=None,
    ):
        if time is None:
            time = self.t[self.ind_saw]

        # -----------------------------------------
        # ~~~~~~~ PRF workflow to run workflows
        # -----------------------------------------

        self.TGLFstd[int(time * 1000)] = TGLFtools.TGLF(
            cdf=self.LocationCDF, time=time, avTime=avTime, rhos=[rho]
        )

        self.TGLFstd[int(time * 1000)].prep(
            self.FolderCDF / f"FolderGACODE_{'{0:.0f}'.format(time * 1000).zfill(5)}ms"
        )

        # ~~~~~~~ Perturbative diffusivity

        if typeAnalysis == "CHIPERT":
            self.TGLFstd[int(time * 1000)].runAnalysis(
                subFolderTGLF="chi_per",
                label="chi_pert",
                analysisType="e",
                TGLFsettings=TGLFsettings,
            )

            value = self.TGLFstd[int(time * 1000)].scans["chi_pert"]["chi_inc"][0]

        if typeAnalysis == "D" or typeAnalysis == "V" or typeAnalysis == "VoD":
            if quantity == "W":
                addTrace = [40, 173]

            self.TGLFstd[int(time * 1000)].runAnalysis(
                subFolderTGLF="impurity",
                label="impurity",
                analysisType="Z",
                TGLFsettings=TGLFsettings,
                trace=addTrace,
            )

            if typeAnalysis == "D":
                value = self.TGLFstd[int(time * 1000)].scans["impurity"]["DZ"][0]
            elif typeAnalysis == "V":
                value = self.TGLFstd[int(time * 1000)].scans["impurity"]["VZ"][0]
            elif typeAnalysis == "VoD":
                value = self.TGLFstd[int(time * 1000)].scans["impurity"]["VoD"][0]

        if "FLUC" in typeAnalysis:
            self.TGLFstd[int(time * 1000)].run(
                subFolderTGLF="fluctuations",
                TGLFsettings=TGLFsettings,
                forceIfcold_start=True,
            )

            self.TGLFstd[int(time * 1000)].read(
                label="fluctuations", d_perp_cm=d_perp_cm
            )

            if typeAnalysis[4:] == "TE":
                fluct = self.TGLFstd[int(time * 1000)].results["fluctuations"][
                    "TeFluct"
                ][0]
            elif typeAnalysis[4:] == "NE":
                fluct = self.TGLFstd[int(time * 1000)].results["fluctuations"][
                    "neFluct"
                ][0]

            value = fluct

        # Store pickle
        # self.TGLFstd[int(time*1000)]

        return value

    def plotStdTRANSP(self, tglfRun="tglf1", fig=None, label="", time=None, leg=True):
        if time is not None:
            it = np.argmin(np.abs(self.t - time))
        else:
            it = self.ind_saw

        if fig is None:
            fig = plt.figure(figsize=(12, 5))

        grid = plt.GridSpec(1, 3, hspace=1.0, wspace=0.2)
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[0, 2])

        if int(time * 1000) not in self.TGLFstandalone:
            print(
                'Cannot plot standalone TGLF because this time was not computed, please run "runTGLFstandalone"',
                typeMsg="w",
            )
            TGLFstd_x, TGLFstd_Qe, TGLFstd_Qi = 0, 0, 0
        else:
            TGLFstd_x = self.TGLFstd[int(time * 1000)].results[tglfRun]["x"]
            TGLFstd_Qe, TGLFstd_Qi = [], []
            for i in range(
                len(self.TGLFstd[int(time * 1000)].results[tglfRun]["TGLFout"])
            ):
                TGLFstd_Qe.append(
                    self.TGLFstd[int(time * 1000)].results[tglfRun]["TGLFout"][i].Qe_unn
                )
                TGLFstd_Qi.append(
                    self.TGLFstd[int(time * 1000)].results[tglfRun]["TGLFout"][i].Qi_unn
                )

            TGLFstd_Qe, TGLFstd_Qi = np.array(TGLFstd_Qe), np.array(TGLFstd_Qi)

        ax1.plot(
            self.xb_lw,
            self.qe_tr_GACODE[it],
            "r",
            ls="-",
            lw=2,
            label="$\\langle \\vec{q}_e^{tr}\\cdot\\nabla r\\rangle$ TRANSP",
        )
        ax1.plot(
            self.xb_lw,
            self.qe_obs_GACODE[it],
            "r",
            ls="--",
            lw=2,
            label="$\\langle \\vec{q}_e^{obs}\\cdot\\nabla r\\rangle$ TRANSP",
        )
        ax1.plot(self.xb_lw, self.qe_obs[it], "g", ls="-", lw=2, label="$q_e$ TRANSP")
        ax1.plot(
            TGLFstd_x,
            TGLFstd_Qe,
            "-o",
            c="b",
            lw=2,
            label="$\\langle \\vec{q}_e\\cdot\\nabla r\\rangle$ TGLF",
            markersize=5,
        )

        ax2.plot(
            self.xb_lw,
            self.qi_obs_GACODE[it],
            "r",
            ls="--",
            lw=2,
            label="$\\langle \\vec{q}_i^{obs}\\cdot\\nabla r\\rangle$ TRANSP",
        )
        ax2.plot(
            self.xb_lw,
            self.qi_tr_GACODE[it],
            "r",
            ls="-",
            lw=2,
            label="$\\langle \\vec{q}_i^{tr}\\cdot\\nabla r\\rangle$ TRANSP",
        )
        ax2.plot(self.xb_lw, self.qi_obs[it], "g", ls="-", lw=2, label="$q_i$ TRANSP")
        ax2.plot(
            TGLFstd_x,
            TGLFstd_Qi,
            "-o",
            c="b",
            lw=3,
            label="$Q_i$ (std TGLF)",
            markersize=5,
        )

        if leg:
            ax1.legend(loc="best", prop={"size": self.mainLegendSize})
            # ax2.legend(loc='best',prop={'size':self.mainLegendSize})
            ax1.set_title("Electrons")
            ax2.set_title("Ions")

        ax1.set_ylabel("Heat Flux ($MW/m^2$)")

        ax1.set_xlabel("$\\rho_N$")
        ax2.set_xlabel("$\\rho_N$")

        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)

        # TRANSP ones in gacode grid
        z = np.interp(TGLFstd_x, self.xb_lw, self.qe_tr_GACODE[it])
        if z.sum() == 0:
            z = np.interp(TGLFstd_x, self.xb_lw, self.qe_obs_GACODE[it])
        ax3.plot(
            TGLFstd_x,
            TGLFstd_Qe / z,
            "m",
            ls="-",
            lw=3,
            label="$\\langle \\vec{q}_e\\cdot\\nabla r\\rangle$ TGLF / $\\langle \\vec{q}_e^{tr}\\cdot\\nabla r\\rangle$ TRANSP",
        )
        z = np.interp(TGLFstd_x, self.xb_lw, self.qi_tr_GACODE[it])
        if z.sum() == 0:
            z = np.interp(TGLFstd_x, self.xb_lw, self.qi_obs_GACODE[it])
        ax3.plot(
            TGLFstd_x,
            TGLFstd_Qi / z,
            "c",
            ls="-",
            lw=3,
            label="$\\langle \\vec{q}_i\\cdot\\nabla r\\rangle$ TGLF / $\\langle \\vec{q}_i^{tr}\\cdot\\nabla r\\rangle$ TRANSP",
        )

        ax3.set_title("Ratios")
        ax3.set_xlabel("$\\rho_N$")
        ax3.legend(loc="best", prop={"size": self.mainLegendSize})
        ax3.set_ylim([0, (TGLFstd_Qi / z).max() * 2.0])
        ax3.axhline(y=1.0, ls="--", c="k", lw=1)

    def plotGRTRANSP(self, tglfRun="tglf1", fig=None, label="", time=None):
        if time is not None:
            it = np.argmin(np.abs(self.t - time))
        else:
            it = self.ind_saw

        numpos = len(self.TGLFstd[int(time * 1000)].results[tglfRun]["x"])

        if fig is None:
            fig = plt.figure(figsize=(12, 5))

        grid = plt.GridSpec(numpos, 2, hspace=1.0, wspace=0.2)
        axT = []
        for i in range(numpos):
            axT0 = []
            for j in range(2):
                axT0.append(fig.add_subplot(grid[i, j]))
            axT.append(axT0)

        axT = np.array(axT)

        if int(time * 1000) not in self.TGLFstandalone:
            print(
                'Cannot plot standalone TGLF because this time was not computed, please run "runTGLFstandalone"',
                typeMsg="w",
            )
            TGLFstd_ky, TGLFstd_freq, TGLFstd_gamma = 0, 0, 0
        else:
            TGLFstd_ky, TGLFstd_gamma, TGLFstd_freq = [], [], []
            for i in range(
                len(self.TGLFstd[int(time * 1000)].results[tglfRun]["TGLFout"])
            ):
                TGLFstd_ky.append(
                    self.TGLFstd[int(time * 1000)].results[tglfRun]["TGLFout"][i].ky
                )
                TGLFstd_gamma.append(
                    self.TGLFstd[int(time * 1000)].results[tglfRun]["TGLFout"][i].g[0]
                )
                TGLFstd_freq.append(
                    self.TGLFstd[int(time * 1000)].results[tglfRun]["TGLFout"][i].f[0]
                )

            TGLFstd_ky, TGLFstd_gamma, TGLFstd_freq = (
                np.array(TGLFstd_ky),
                np.array(TGLFstd_gamma),
                np.array(TGLFstd_freq),
            )

        coeff = 0
        for i in range(len(self.TGLFstd[int(time * 1000)].results[tglfRun]["x"])):
            if numpos > 1:
                ax1 = axT[i, 0]
                ax2 = axT[i, 1]
            else:
                ax1 = axT[0]
                ax2 = axT[1]

            # --------------------
            # ------ Growth rate
            # --------------------

            # Standalone
            GACODEplotting.plotTGLFspectrum(
                ax1,
                TGLFstd_ky[i],
                TGLFstd_gamma[i],
                coeff=coeff,
                c="b",
                ls="-",
                lw=3,
                markersize=20,
                label="$\\gamma$ (std TGLF)",
                ylabel=False,
            )
            ax1.set_title(
                "Growth rate @ rho={0:.2f}".format(
                    self.TGLFstd[int(time * 1000)].results[tglfRun]["x"][i]
                )
            )

            # TRANSP
            try:
                ir = np.argmin(
                    np.abs(
                        self.xb_lw
                        - self.TGLFstd[int(time * 1000)].results[tglfRun]["x"][i]
                    )
                )
                GACODEplotting.plotTGLFspectrum(
                    ax1,
                    self.TGLF.kys[it, ir],
                    self.TGLF.grates[it, ir],
                    coeff=coeff,
                    c="r",
                    ls="-",
                    lw=3,
                    markersize=20,
                    ylabel=False,
                    label="$\\gamma$ (TRANSP)",
                )
            except:
                pass

            ax1.set_ylabel("Growth Rate")

            # --------------------
            # ------ Frequency
            # --------------------

            # Standalone
            GACODEplotting.plotTGLFspectrum(
                ax2,
                TGLFstd_ky[i],
                TGLFstd_freq[i],
                coeff=coeff,
                c="b",
                ls="-",
                lw=3,
                markersize=20,
                label="$\\omega$ (std TGLF)",
                ylabel=False,
            )

            ax2.set_title(
                "Frequency @ rho={0:.2f}".format(
                    self.TGLFstd[int(time * 1000)].results[tglfRun]["x"][i]
                )
            )

            # TRANSP
            try:
                GACODEplotting.plotTGLFspectrum(
                    ax2,
                    self.TGLF.kys[it, ir],
                    self.TGLF.freqs[it, ir],
                    coeff=coeff,
                    c="r",
                    ls="-",
                    lw=3,
                    markersize=20,
                    ylabel=False,
                    label="$\\omega$ (TRANSP)",
                )
            except:
                pass

            # ax2.set_yscale('symlog',thr=1E-1)

            if i == 0:
                ax1.legend(loc="best", prop={"size": self.mainLegendSize})

            ax2.set_ylabel("Frequency")

            ax1.set_xlim([0.05, 35.0])

        GRAPHICStools.adjust_figure_layout(fig)

    def plotFLTRANSP(self, tglfRun="tglf1", fig=None, label="", time=None):
        if time is not None:
            it = np.argmin(np.abs(self.t - time))
        else:
            it = self.ind_saw

        numpos = len(self.TGLFstd[int(time * 1000)].results[tglfRun]["x"])

        if fig is None:
            fig = plt.figure(figsize=(12, 5))

        grid = plt.GridSpec(numpos, 2, hspace=1.0, wspace=0.2)
        axT = []
        for i in range(numpos):
            axT0 = []
            for j in range(2):
                axT0.append(fig.add_subplot(grid[i, j]))
            axT.append(axT0)

        axT = np.array(axT)

        if int(time * 1000) not in self.TGLFstandalone:
            print(
                'Cannot plot standalone TGLF because this time was not computed, please run "runTGLFstandalone"',
                typeMsg="w",
            )
            TGLFstd_ky, TGLFstd_te, TGLFstd_ne = 0, 0, 0
        else:
            TGLFstd_ky, TGLFstd_te, TGLFstd_ne = [], [], []
            for i in range(
                len(self.TGLFstd[int(time * 1000)].results[tglfRun]["TGLFout"])
            ):
                TGLFstd_ky.append(
                    self.TGLFstd[int(time * 1000)].results[tglfRun]["TGLFout"][i].ky
                )
                TGLFstd_te.append(
                    self.TGLFstd[int(time * 1000)]
                    .results[tglfRun]["TGLFout"][i]
                    .AmplitudeSpectrum_Te
                )
                TGLFstd_ne.append(
                    self.TGLFstd[int(time * 1000)]
                    .results[tglfRun]["TGLFout"][i]
                    .AmplitudeSpectrum_ne
                )

            TGLFstd_ky, TGLFstd_te, TGLFstd_ne = (
                np.array(TGLFstd_ky),
                np.array(TGLFstd_te),
                np.array(TGLFstd_ne),
            )

        coeff = 0
        for i in range(len(self.TGLFstd[int(time * 1000)].results[tglfRun]["x"])):
            if numpos > 1:
                ax1 = axT[i, 0]
                ax2 = axT[i, 1]
            else:
                ax1 = axT[0]
                ax2 = axT[1]

            # --------------------
            # ------ Te Fluctuations
            # --------------------

            # Standalone
            GACODEplotting.plotTGLFspectrum(
                ax1,
                TGLFstd_ky[i],
                TGLFstd_te[i],
                coeff=coeff,
                c="b",
                ls="-",
                lw=3,
                markersize=20,
                label="",
                ylabel=False,
                titles=[
                    "Te Fluct @ rho={0:.2f}".format(
                        self.TGLFstd[int(time * 1000)].results[tglfRun]["x"][i]
                    )
                ],
            )

            # TRANSP
            try:
                ir = np.argmin(
                    np.abs(
                        self.xb_lw
                        - self.TGLFstd[int(time * 1000)].results[tglfRun]["x"][i]
                    )
                )
                GACODEplotting.plotTGLFspectrum(
                    ax1,
                    self.TGLF.kys[it, ir],
                    self.TGLF.TeFluct[it, ir],
                    coeff=coeff,
                    c="r",
                    ls="-",
                    lw=3,
                    markersize=20,
                    label="",
                    ylabel=False,
                )
            except:
                pass

            ax1.set_yscale("linear")
            ax1.set_ylim(bottom=0)
            # --------------------
            # ------ ne Fluctuations
            # --------------------

            # Standalone
            GACODEplotting.plotTGLFspectrum(
                ax2,
                TGLFstd_ky[i],
                TGLFstd_ne[i],
                coeff=coeff,
                c="b",
                ls="-",
                lw=3,
                markersize=20,
                label="",
                ylabel=False,
                titles=[
                    "ne Fluct @ rho={0:.2f}".format(
                        self.TGLFstd[int(time * 1000)].results[tglfRun]["x"][i]
                    )
                ],
            )

            # TRANSP
            try:
                GACODEplotting.plotTGLFspectrum(
                    ax2,
                    self.TGLF.kys[it, ir],
                    self.TGLF.neFluct[it, ir],
                    coeff=coeff,
                    c="r",
                    ls="-",
                    lw=3,
                    markersize=20,
                    label="",
                    ylabel=False,
                )
            except:
                pass

            # ax2.set_yscale('symlog',thr=1E-1)

            if i == 0:
                ax1.legend(loc="best", prop={"size": self.mainLegendSize})

            ax2.set_yscale("linear")
            ax2.set_ylim(bottom=0)

            ax1.set_ylabel("Te fluct")
            ax2.set_ylabel("ne fluct")

        GRAPHICStools.adjust_figure_layout(fig)

    # --------

    def getChiPertPulse(
        self, pulseTimes=None, rhoRange=[0.3, 0.7], timeRange=0.1, axs=None
    ):
        # Last sawtooth
        if pulseTimes is None:
            print("Calculating chi_pert from second to last sawteeth crash")
            pulseTimes = [self.t[self.ind_sawAll[1] - 2]]
            rhoRange = [self.x_q1[self.ind_sawAll[1] - 2] + 0.1, 0.8]
            timeRange = self.t[self.ind_sawAll[0]] - self.t[self.ind_sawAll[1]] - 0.005

        # Creely Method

        self.chiCalc = PRIMAtools.chiPertCalculator(
            self, pulseTimes, rhoRange, timeRange
        )
        self.chiCalc.prepareData(axs=axs)
        self.chiCalc.calculateShotChiPert()
        self.Creely_ChiPert = self.chiCalc.printOutputs()

    def compareChiPert(
        self,
        time=None,
        rhoRange=[0.4, 0.8],
        timeRange=0.5,
        TGLFsettings=1,
        cold_start=False,
        plotYN=True,
    ):
        if time is None:
            try:
                time = self.t[self.ind_sawAll[1]]
            except:
                time = self.t[self.ind_saw]

        it = np.argmin(np.abs(self.t - time))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get ChiPert from TGLF
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        num = 5
        rhos = np.linspace(rhoRange[0], rhoRange[1], num)

        self.ChiPert_tglf = TGLFtools.TGLF(cdf=self.LocationCDF, time=time, rhos=rhos)
        self.ChiPert_tglf.prep(self.FolderCDF / "chi_per_calc", cold_start=cold_start)
        self.ChiPert_tglf.runAnalysis(
            subFolderTGLF="chi_per",
            label="chi_pert",
            analysisType="e",
            TGLFsettings=TGLFsettings,
            cold_start=cold_start,
            cdf_open=self,
        )

        self.TGLF_ChiPert = np.array(self.ChiPert_tglf.scans["chi_pert"]["chi_inc"])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get ChiPert from Creely's
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.getChiPertPulse(pulseTimes=[time], rhoRange=rhoRange, timeRange=timeRange)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if plotYN:
            self.fnChiPert = FigureNotebook(f"mitim Notebook, run #{self.nameRunid}")
            fig1 = self.fnChiPert.add_figure(label="Compare")
            fig2 = self.fnChiPert.add_figure(label="Pulse")
            fig3 = self.fnChiPert.add_figure(label="TGLF")

            grid = plt.GridSpec(2, 2, hspace=0.6, wspace=0.2)
            ax00 = fig1.add_subplot(grid[0, 0])
            ax10 = fig1.add_subplot(grid[1, 0])

            ms = 30
            maxx = np.nanmax([np.nanmax(self.TGLF_ChiPert), self.Creely_ChiPert])

            ax = ax00
            ax.plot(
                np.linspace(0, maxx * 1.5, 100),
                np.linspace(0, maxx * 1.5, 100),
                "k",
                ls="-.",
                lw=2,
            )
            colors = GRAPHICStools.listColors()
            for i in range(len(rhos)):
                ax.scatter(
                    [self.Creely_ChiPert],
                    [self.TGLF_ChiPert[i]],
                    s=ms,
                    marker="o",
                    color=colors[i],
                    label=f"$\\rho_N$={rhos[i]:.2f}",
                )

            ax.set_xlabel("$\\chi_e^{pert}$ Pulse Analysis ($m^2/s$)")
            ax.set_ylabel("$\\chi_e^{pert}$ TGLF ($m^2/s$)")
            ax.set_xlim([0, maxx * 1.2])
            ax.set_ylim([0, maxx * 1.2])
            ax.legend(loc="best", fontsize=10)

            ax = ax10

            ax.plot(
                np.linspace(0, maxx * 1.5, 100),
                np.linspace(0, maxx * 1.5, 100),
                "k",
                ls="-.",
                lw=2,
            )
            # ~~~~~~~~~~ Average chi
            ax.scatter(
                [self.Creely_ChiPert],
                np.mean(self.TGLF_ChiPert),
                s=ms,
                marker="s",
                color=colors[i + 1],
                label="$<\\chi_e^{pert}>$",
            )
            # ~~~~~~~~~~ Chi interpolated at the average radius
            ax.scatter(
                [self.Creely_ChiPert],
                np.interp([np.mean(rhoRange)], rhos, self.TGLF_ChiPert),
                s=ms,
                marker="s",
                color=colors[i + 2],
                label="$\\chi_e^{pert} @ <\\rho_N>$",
            )

            ax.set_xlabel("$\\chi_e^{pert}$ Pulse Analysis ($m^2/s$)")
            ax.set_ylabel("$\\chi_e^{pert}$ TGLF ($m^2/s$)")
            ax.set_xlim([0, maxx * 1.2])
            ax.set_ylim([0, maxx * 1.2])
            ax.legend(loc="best", fontsize=10)

            # Plot pulse analysis details
            self.plotPulse(fig=fig2, time=time, rhoRange=rhoRange)

            # Plot TGLF calculation details
            self.ChiPert_tglf.plotAnalysis(labels=["chi_pert"], fig=fig3)

    # --------

    def getGFILE(self):

        print("\t- Looking for equilibrium file in CDF folder...")
        for extension in ["geqdsk", "geq", "gfile", "eqdsk"]:
            for folder in ["EQ_folder", ""]:
                gf = IOtools.findFileByExtension(self.FolderCDF / folder, extension, ForceFirst=True)
                if gf is not None:
                    print("\t\t- Reference gfile found in folder")
                    self.gfile_in = GEQtools.MITIMgeqdsk(self.FolderCDF / folder / gf)
                    break
        if gf is None:
            print("\t\t- Reference g-file associated to this run could not be found",typeMsg="w")

        # Try to read boundary too
        if (self.FolderCDF / "MIT12345.RFS").exists():
            self.bound_R, self.bound_Z = TRANSPhelpers.readBoundary(self.FolderCDF / "MIT12345.RFS", self.FolderCDF / "MIT12345.ZFS")

    def getICRFantennas(self, namelist):
        nicha = int(IOtools.findValue(namelist, "nicha", "="))

        if nicha == 1:
            ex = IOtools.findValue(namelist, f"rgeoant_a(1)", "=", raiseException=False)
            ex1 = IOtools.findValue(namelist, f"rgeoant", "=", raiseException=False)
        else:
            ex = IOtools.findValue(
                namelist, f"rgeoant_a(1,1)", "=", raiseException=False
            )
            ex1 = IOtools.findValue(namelist, f"rgeoant", "=", raiseException=False)

        self.R_ant, self.Z_ant = [], []

        # ---------------------------------------------------------------------------------------------------
        # Antennas specified with rgeoant_a
        # ---------------------------------------------------------------------------------------------------

        if ex is not None:
            print(
                f"\t- Detected {nicha} ICRF antenna(s), searching for rgeoant_a and ygeoant_a in namelist"
            )

            if nicha == 1:
                self.R_ant0, self.Z_ant0 = [], []
                for ip in range(30):
                    try:
                        self.R_ant0.append(
                            IOtools.findValue(namelist, f"rgeoant({ip+1})", "=")
                        )
                        self.Z_ant0.append(
                            IOtools.findValue(namelist, f"ygeoant({ip+1})", "=")
                        )
                    except:
                        break
                self.R_ant.append(self.R_ant0)
                self.Z_ant.append(self.Z_ant0)

            else:
                for icha in range(nicha):
                    self.R_ant0, self.Z_ant0 = [], []
                    for ip in range(30):
                        try:
                            self.R_ant0.append(
                                IOtools.findValue(
                                    namelist, f"rgeoant_a({ip+1},{icha+1})", "="
                                )
                            )
                            self.Z_ant0.append(
                                IOtools.findValue(
                                    namelist, f"ygeoant_a({ip+1},{icha+1})", "="
                                )
                            )
                        except:
                            break
                    self.R_ant.append(self.R_ant0)
                    self.Z_ant.append(self.Z_ant0)

        # ---------------------------------------------------------------------------------------------------
        # Antennas specified with rgeoant
        # ---------------------------------------------------------------------------------------------------

        elif ex1 is not None:
            print(
                f"\t\t- Detected {nicha} ICRF antenna(s), searching for rgeoant and ygeoant in namelist"
            )

            self.R_ant0 = [
                float(i)
                for i in IOtools.findValue(
                    namelist, f"rgeoant", "=", isitArray=True
                ).split(",")
            ]
            self.Z_ant0 = [
                float(i)
                for i in IOtools.findValue(
                    namelist, f"ygeoant", "=", isitArray=True
                ).split(",")
            ]

            self.R_ant.append(self.R_ant0)
            self.Z_ant.append(self.Z_ant0)

        # ---------------------------------------------------------------------------------------------------
        # Antennas specified with rmjicha
        # ---------------------------------------------------------------------------------------------------

        else:
            print(
                f"\t- Detected {nicha} ICRF antenna(s), searching for rmjicha in namelist"
            )

            if nicha == 1:
                R_ant = IOtools.findValue(namelist, "RMJICHA", "=")
                r_ant = IOtools.findValue(namelist, "RMNICHA", "=")
                t_ant = IOtools.findValue(namelist, "THICHA", "=")

                self.R_ant0, self.Z_ant0 = TRANSPhelpers.reconstructAntenna(
                    R_ant, r_ant, t_ant
                )

                self.R_ant.append(self.R_ant0)
                self.Z_ant.append(self.Z_ant0)

            else:
                R_ant = [
                    float(i)
                    for i in IOtools.findValue(namelist, "RMJICHA", "=").split(",")
                ]
                r_ant = [
                    float(i)
                    for i in IOtools.findValue(namelist, "RMNICHA", "=").split(",")
                ]
                t_ant = [
                    float(i)
                    for i in IOtools.findValue(namelist, "THICHA", "=").split(",")
                ]

                for icha in range(nicha):
                    self.R_ant0, self.Z_ant0 = TRANSPhelpers.reconstructAntenna(
                        R_ant[icha], r_ant[icha], t_ant[icha]
                    )

                    self.R_ant.append(self.R_ant0)
                    self.Z_ant.append(self.Z_ant0)

        self.R_ant, self.Z_ant = (
            np.array(self.R_ant) * 1e-2,
            np.array(self.Z_ant) * 1e-2,
        )

        print(
            f" \t\t- Gathered ICRF antenna ({self.R_ant.shape[0]}) structure from namelist"
        )

    def getStructures(self):
        try:
            NML = IOtools.findFileByExtension(self.FolderCDF, "TR.DAT")
        except:
            NML = IOtools.findFileByExtension(self.FolderCDF, "TR.DAT", ForceFirst=True)

        if NML is not None:
            namelist = NML

            if np.sum(self.PichT) > 0.0 + self.eps00 * (1 + len(self.t)):
                self.getICRFantennas(namelist)

            try:
                # ECRF stuff
                if np.sum(self.PechT) > 0.0 + self.eps00 * (1 + len(self.t)):
                    self.R_gyr = (
                        IOtools.cleanArray(
                            IOtools.findValue(namelist, "XECECH", "=", isitArray=True)
                        )
                        * 1e-2
                    )
                    self.Z_gyr = (
                        IOtools.cleanArray(
                            IOtools.findValue(namelist, "ZECECH", "=", isitArray=True)
                        )
                        * 1e-2
                    )
                    self.F_gyr = (
                        IOtools.cleanArray(
                            IOtools.findValue(namelist, "FREQECH", "=", isitArray=True)
                        )
                        * 1e-9
                    )

                    print("\t- Gathered ECRH gyrotrons structure from namelist")
            except:
                pass

            try:
                VVr, VVz = [], []
                for i in range(20):
                    try:
                        VVr.append(IOtools.findValue(namelist, f"VVRmom({i + 1})", "="))
                        VVz.append(IOtools.findValue(namelist, f"VVZmom({i + 1})", "="))
                    except:
                        break

                self.R_vv, self.Z_vv = TRANSPhelpers.reconstructVV(VVr, VVz)
                self.R_vv = self.R_vv * 1e-2
                self.Z_vv = self.Z_vv * 1e-2

                print("\t\t- Gathered vacuum vessel structure from namelist")
            except:
                pass

        else:
            print(
                "\t\t- Namelist associated to this run could not be found", typeMsg="w"
            )

        try:
            LIM = IOtools.findFileByExtension(self.FolderCDF, ".LIM")
            if LIM is not None:
                ufile = self.FolderCDF + LIM + ".LIM"
                uf = UFILEStools.UFILEtransp()
                uf.readUFILE(ufile)
                self.R_lim = uf.Variables["X"]
                self.Z_lim = uf.Variables["Z"]
                print("\t\t- Gathered limiters structure from LIM Ufile")
        except:
            pass

        if NML is not None:
            print("\t- Looking for information on beam trajectories")
            try:
                self.beam_trajectories = getBeamTrajectories(namelist)
                print("\t\t- Gathered beam trajectories from namelist post-processing")
            except:
                pass

            print("\t- Looking for information on ECH trajectories")
            try:
                self.ECRH_trajectories = getECRHTrajectories(
                    namelist,
                    self.Theta_gyr[:, self.ind_saw],
                    self.Phi_gyr[:, self.ind_saw],
                )
                print(
                    "\t\t- Gathered ECRH trajectories from namelist and CDF post-processing"
                )
            except:
                pass

    def getEstimatedMachineCost(self):
        self.cost = 0.7266 * self.Bt**2 * self.Rmajor**3 + self.PichT
        self.cost = self.cost * 0.0123

    def writePickle(
        self,
        folderWork,
        time=None,
        avTime=0.0,
        name="results.pkl",
        removeFirstPoint=True,
    ):
        # Selection of time output

        if time is None:
            it = self.ind_saw
        else:
            if time > 0.0:
                it = np.argmin(np.abs(self.t - time))
            if time < 0.0:
                it = np.argmin(np.abs(self.t - (self.t[self.ind_saw] + time)))

        time = self.t[it]

        if avTime > 0:
            it1 = np.argmin(np.abs(self.t - (time - avTime)))
            it2 = np.argmin(np.abs(self.t - (time + avTime)))
        else:
            it1, it2 = it, it + 1

        # ~~~~~~~~~~~~~~~ Standard variables (given in zone centers)

        """
		Because the TRANSP grid starts with a boundary point, the first point of the array, when 
		interpolating quantities from XB to X, it goes outside. If this option is enabled, then 
		all arrays start "inside" of the XB and X ranges
		"""
        if removeFirstPoint:
            startP = 1
        else:
            startP = 0

        # In Zone centers
        variables = {
            "rho_tor": self.x[:, startP:],
            "vol_zone": self.dvol[:, startP:],
            "te": self.Te[:, startP:],
            "ti": self.Ti[:, startP:],
            "ne": self.ne[:, startP:],
            "Zeff": self.Zeff[:, startP:],
            "nD": self.nD[:, startP:],
            "nT": self.nT[:, startP:],
            "nZ": self.nZAVE[:, startP:],
            "nZ_zave_vol": self.fZAVE_Z[:, startP:],
            "nHe3": self.nminiHe3[:, startP:],
            "tHe3": self.Tmini[:, startP:],
            "nHe4_fast": self.nfusHe4[:, startP:],
            "nHe4_thermal": self.nHe4[:, startP:],
            "WHe3_perp": self.Wperpx_mini[:, startP:],
            "WHe3_par": self.Wparx_mini[:, startP:],
            "WHe4_perp": self.Wperpx_fus[:, startP:],
            "WHe4_par": self.Wparx_fus[:, startP:],
            "FastAlpha_source": self.neutrons_thrDT_x[:, startP:],
            "Pich_min": self.Pich_min[:, startP:],
            "chi_e": self.Chi_e[:, startP:],
            "chi_i": self.Chi_i[:, startP:],
            "jtor": self.j[:, startP:],
            "jtor_b": self.jB[:, startP:],
            "vol_zone": self.dvol[:, startP:],
            "Piche": self.Peich[:, startP:],
            "Pichi": self.Piich[:, startP:],
            "Pei": self.Pei[:, startP:],
        }

        # Other variables
        variables.update(
            {
                "Bt_ext": self.Bt_ext,
                "Bt_plasma": self.Bt_plasma,
                "Rmajor": self.Rmaj,
                "timeExtracted": self.t,
                "P_LCFS": self.P_LCFS,
                "W_th": self.Wth,
                "W_tot": self.Wtot,
                "li1": self.Li1,
                "li3": self.Li3,
            }
        )

        # Time average all
        dictPKL = {}
        for ikey in variables:
            dictPKL[ikey] = np.mean(variables[ikey][it1:it2], axis=0)

        # ~~~~~~~~~~~~~~~ Variables given in zone boundaries
        variablesInBoundaryZones = {
            "rho_pol": self.xpol,
            "pol_flux": self.psi,
            "q": self.q,
            "shat": self.shat,
            "rmin": self.rmin,
            "Rmax": self.Rmax,
            "Rmin": self.Rmin,
            "Zmax": self.Zmax,
            "Zmin": self.Zmin,
        }

        for ikey in variablesInBoundaryZones:
            dictPKL[ikey] = np.interp(
                np.mean(self.x[it1:it2], axis=0),
                np.mean(self.xb[it1:it2], axis=0),
                np.mean(variablesInBoundaryZones[ikey][it1:it2], axis=0),
            )[startP:]

        # ~~~~~~~~~~~~~~~ Other variables

        dictPKL["timeLastSawtooth"] = self.t[self.ind_saw]
        dictPKL["timeRangeExtracted"] = [self.t[it1], self.t[it2]]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print(f" ~~~~~~ Writing output pickle file with plasma at t={time:.3f}s")
        folderWork = IOtools.expandPath(folderWork)
        file = folderWork / f"{name}.pkl"
        with open(file, "wb") as handle:
            pickle.dump(dictPKL, handle, protocol=4)


    # ---------------------------------------------------------------------------------------------------------
    # Code conversions
    # ---------------------------------------------------------------------------------------------------------

    def produceTGYROfiles(
        self, folderWork="~/scratch/outputsMITIM/", time=-0.06, avTime=0.05
    ):
        folderWork = IOtools.expandPath(folderWork)

        print(f' ~~~~~~ Working on folder "{folderWork}"')

        if time is None:
            it = self.ind_saw_before
        else:
            if time > 0.0:
                it = np.argmin(np.abs(self.t - time))
            if time < 0.0:
                it = np.argmin(np.abs(self.t - (self.t[self.ind_saw] + time)))
        time = self.t[it]

        # gridsTRXPL = [151,101,101]
        # gridsTRXPL = [150,100,100]
        gridsTRXPL = [300, 257, 257]

        # ---- TGYRO workflow to generate plasmastate, geqdsk and profiles

        self.tgyro = TGYROtools.TGYRO(self.LocationCDF, time=time, avTime=avTime)
        self.tgyro.prep(
            folderWork, cold_start=True, BtIp_dirs=[0, 0], gridsTRXPL=gridsTRXPL
        )

    def writeOutput(self, folderWork=None, time=-0.06, avTime=0.05):
        """
        VIP: folderWork will be cleaned!

        call: c.writeOutput()
        """

        IOtools.askNewFolder(self.FolderCDF / f"RELEASE_folder", force=True)

        # ---- Perform TGYRO operations fully in folderWork (scratch) and then move

        # IOtools.askNewFolder('{0}/TGYROprep_folder/'.format(self.FolderCDF),force=True)
        if folderWork is None:
            folderWork = self.FolderCDF / "TGYROprep_folder" / "scratch"

        self.produceTGYROfiles(folderWork=folderWork, time=time, avTime=avTime)

        for item in folderWork.glob("*"):
            if item.is_file():
                shutil.copy2(item, self.FolderCDF / "TGYROprep_folder")
            elif item.is_dir():
                shutil.copytree(item, self.FolderCDF / "TGYROprep_folder" / item.name)

        # ---- Organize relevant things from TGYRO folder

        shutil.copy2(folderWork / '10001.geq', self.FolderCDF / 'RELEASE_folder' / 'TRANSPrun.geq')
        shutil.copy2(folderWork / '10001.cdf', self.FolderCDF / 'RELEASE_folder' / 'TRANSPrun.cdf')
        shutil.copy2(folderWork / 'input.gacode', self.FolderCDF / 'RELEASE_folder' / 'TRANSPrun.input.gacode')

        print(
            "\n~~~~~~ Simulation results ready at t={0:.3f}s +- {3:0.3f}s (last sawtooth={2:.3f}s) in folder {1}/RELEASE_folder/\n".format(
                time, self.FolderCDF, self.t[self.ind_saw], avTime
            )
        )

        # ---- Pickle file

        self.writePickle(
            self.FolderCDF / "RELEASE_folder",
            time=time,
            avTime=avTime,
            name="TRANSPrun",
        )

        # ---- Zip at this stage

        os.chdir(self.FolderCDF)
        os.system("tar -czvf TRANSPrun.tar RELEASE_folder")
        IOtools.shutil_rmtree(self.FolderCDF / 'RELEASE_folder')
        (self.FolderCDF / 'TRANSPrun.tar').replace(self.FolderCDF / 'RELEASE_folder')

    def to_transp(self, folder = '~/scratch/', shot = '12345', runid = 'P01', times = [0.0,1.0], time_extraction = -1):

        print("\t- Converting to TRANSP")
        folder = IOtools.expandPath(folder)
        folder.mkdir(parents=True, exist_ok=True)

        transp = TRANSPhelpers.transp_run(folder, shot, runid)
        for time in times:
            transp.populate_time.from_cdf(time, self, time_extraction=time_extraction)

        transp.write_ufiles()

        return transp

    def to_profiles(self, time_extraction = None):

        if time_extraction is None:
            time_extraction = self.t[self.ind_saw]
        elif time_extraction < 0:
            time_extraction = self.t[-1] + time_extraction

        it = np.argmin(np.abs(self.t - time_extraction))
        
        print(f"\t- Converting to input.gacode class, extracting at t={time_extraction:.3f}s")
        print("\t\t* Ignoring rotation and no-ICRF auxiliary sources",typeMsg='w')
        print("\t\t* Extrapolating using cubic spline",typeMsg='w')
        print("\t\t* Not time averaging yet",typeMsg='w')

        #TODO: I should be looking at the extrapolated quantities in TRANSP?
        from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as extrapolation_routine

        # -------------------------------------------------------------------------------------------------------
        # Main structure
        # -------------------------------------------------------------------------------------------------------

        profiles = {}

        # Radial grid
        rho_grid = self.xb[it]

        # Info
        nion = len(self.Species) - 1
        nexp = rho_grid.shape[0]
        profiles['nexp'] = np.array([f'{nexp}'])
        profiles['nion'] = np.array([f'{nion}'])

        profiles['shot'] = np.array(['12345'])

        # -------------------------------------------------------------------------------------------------------
        # Species
        # -------------------------------------------------------------------------------------------------------

        profiles['name'] = []
        profiles['type'] = []
        profiles['masse'] = []
        profiles['mass'] = []
        profiles['ze'] = []
        profiles['z'] = []
        mass_ref = 2.0
        for specie in self.Species:
            if specie == 'e':
                profiles['masse'].append(self.Species[specie]['m']/self.mD * mass_ref)
                profiles['ze'].append(-1.0)
            else:
                profiles['name'].append(self.Species[specie]['name'])
                profiles['mass'].append(self.Species[specie]['m']/self.mD * mass_ref)
                profiles['z'].append(self.Species[specie]['Z'][it])
                if self.Species[specie]['type'] == 'thermal':
                    profiles['type'].append('[therm]')
                else:
                    profiles['type'].append('[fast]')
        profiles['name'] = np.array(profiles['name'])
        profiles['type'] = np.array(profiles['type'])
        profiles['masse'] = np.array(profiles['masse'])
        profiles['mass'] = np.array(profiles['mass'])
        profiles['ze'] = np.array(profiles['ze'])
        profiles['z'] = np.array(profiles['z'])

        # -------------------------------------------------------------------------------------------------------
        # Global equilibrium
        # -------------------------------------------------------------------------------------------------------

        profiles['torfluxa(Wb/radian)'] = np.array([self.phi_bnd[it] / (2*np.pi)])
        profiles['rcentr(m)'] = np.array([self.Rmajor[it]])
        profiles['bcentr(T)'] = np.array([self.Bt_vacuum[it]])
        profiles['current(MA)'] = np.array([self.Ip[it]])

        # -------------------------------------------------------------------------------------------------------
        # Equilibrium profiles
        # -------------------------------------------------------------------------------------------------------

        profiles['rho(-)'] = rho_grid

        profiles['polflux(Wb/radian)'] = self.psi[it,:]
        profiles['q(-)'] = self.q[it,:]
        
        # -------------------------------------------------------------------------------------------------------
        # Flux surfaces
        # -------------------------------------------------------------------------------------------------------

        coeffs_MXH = 7

        Rs, Zs = [], []
        for rho in profiles['rho(-)']:
            R, Z = getFluxSurface(self.f, time_extraction, rho, rhoPol=False, sqrt=True)
            Rs.append(R)
            Zs.append(Z)
        Rs = np.array(Rs)
        Zs = np.array(Zs)

        surfaces = GEQtools.mitim_flux_surfaces()
        surfaces.reconstruct_from_RZ(Rs, Zs)
        surfaces._to_mxh(n_coeff=coeffs_MXH)

        for i in range(coeffs_MXH):
            profiles[f'shape_cos{i}(-)'] = surfaces.cn[:,i]
            if i > 2:
                profiles[f'shape_sin{i}(-)'] = surfaces.sn[:,i]
        
        profiles['kappa(-)'] = surfaces.kappa
        profiles['delta(-)'] = np.sin(surfaces.sn[:,1])
        profiles['zeta(-)'] = -surfaces.sn[:,2]
        profiles['rmin(m)'] = surfaces.a
        profiles['rmaj(m)'] = surfaces.R0
        profiles['zmag(m)'] = surfaces.Z0

        # -------------------------------------------------------------------------------------------------------
        # Kinetic profiles
        # -------------------------------------------------------------------------------------------------------

        profiles['ni(10^19/m^3)'] = []
        profiles['ti(keV)'] = []
        for specie in self.Species:
            if specie == 'e':
                profiles['te(keV)'] = self.Te[it,:]
                profiles['ne(10^19/m^3)'] =self.ne[it,:]*1E1
            else:
                profiles['ni(10^19/m^3)'].append(self.Species[specie]['n'][it,:]*1E1)
                profiles['ti(keV)'].append(self.Species[specie]['T'][it,:])
        profiles['ni(10^19/m^3)'] = np.array(profiles['ni(10^19/m^3)']).T
        profiles['ti(keV)'] = np.array(profiles['ti(keV)']).T

        # Power profiles
        profiles['qei(MW/m^3)'] = self.Pei[it,:]
        profiles['qrfe(MW/m^3)'] = self.Peich[it,:]
        profiles['qrfi(MW/m^3)'] = self.Piich[it,:]
        profiles['qbrem(MW/m^3)'] = self.Prad_b[it,:]
        profiles['qsync(MW/m^3)'] = self.Prad_c[it,:]
        profiles['qline(MW/m^3)'] = self.Prad_l[it,:]
        profiles['qohme(MW/m^3)'] = self.Poh[it,:]
        profiles['qfuse(MW/m^3)'] = self.Pfuse[it,:]
        profiles['qfusi(MW/m^3)'] = self.Pfusi[it,:]

        # -------------------------------------------------------------------------------------------------------
        # Postprocessing: Interpolate from xb to x (boundary to center quantities)
        # -------------------------------------------------------------------------------------------------------

        def grid_interpolation_method_to_one(x,y,x_new):
            return extrapolation_routine(x_new, x, y)

        keys_in_x = ['te(keV)', 'ne(10^19/m^3)', 'ni(10^19/m^3)', 'ti(keV)', 'qei(MW/m^3)', 'qrfe(MW/m^3)', 'qrfi(MW/m^3)', 'qbrem(MW/m^3)', 'qsync(MW/m^3)', 'qline(MW/m^3)', 'qohme(MW/m^3)', 'qfuse(MW/m^3)', 'qfusi(MW/m^3)']
        for key in keys_in_x:
            if (profiles[key].ndim == 1):
                profiles[key] = grid_interpolation_method_to_one(self.x[it], profiles[key],profiles['rho(-)'])
            elif (profiles[key].ndim == 2):
                profiles[key] = np.vstack([grid_interpolation_method_to_one(self.x[it], profiles[key][:,i],profiles['rho(-)']) for i in range(profiles[key].shape[1])]).T

        # -------------------------------------------------------------------------------------------------------
        # Postprocessing: Add zero at the beginning
        # -------------------------------------------------------------------------------------------------------

        def grid_interpolation_method_to_zero(x,y):
            return extrapolation_routine(np.append(0.0, x), x, y)

        for key in profiles:
            if (profiles[key].ndim == 1) and (profiles[key].shape[0] == nexp) and (key != 'rho(-)'):
                profiles[key] =  grid_interpolation_method_to_zero(profiles['rho(-)'],profiles[key])
            elif (profiles[key].ndim == 2):
                profiles[key] = np.vstack([grid_interpolation_method_to_zero(profiles['rho(-)'],profiles[key][:,i]) for i in range(profiles[key].shape[1])]).T

        profiles['rho(-)'] = np.append(0.0, profiles['rho(-)'])

        # -------------------------------------------------------------------------------------------------------
        # Load class
        # -------------------------------------------------------------------------------------------------------
        
        # Ensure positive values, non-zero for some
        minimum = 1E-9
        for key in ['ne(10^19/m^3)', 'ni(10^19/m^3)', 'te(keV)', 'ti(keV)', 'rmin(m)']:
            profiles[key] = profiles[key].clip(min=minimum)

        from mitim_tools.gacode_tools import PROFILEStools
        p = PROFILEStools.PROFILES_GACODE.scratch(profiles)

        return p


    # ---------------------------------------------------------------------------------------------------------
    # -------- Outputs
    # ---------------------------------------------------------------------------------------------------------

    def writeResults_TXT(self, file, ensureBackUp=True):
        file = IOtools.expandPath(file)
        if ensureBackUp:
            IOtools.loopFileBackUp(file)

        ind = self.ind_saw

        lines = [
            "\n",
            " MITIM framework (P. Rodriguez-Fernandez, 2020)",
            " CDF-reader to analyze TRANSP simulation has been launched ({0})".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            "\n",
        ]

        with open(file, "w") as f:
            f.write("\n".join(lines))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        lines = [
            "",
            "--------------------------",
            "TRANSP settings and observations",
            "--------------------------",
            "",
            "{1} radial zones used, leading to a resolution in sqrt root of normalized toroidal flux of Delta_x = {0:.3f}.".format(
                self.x_lw[-1] - self.x_lw[-2], len(self.x_lw)
            ),
            "Simulation time starts at t = {0:.3f}s (simulation has run until t = {1:.3f}s)".format(
                self.timeOri, self.timeFin
            ),
            "This simulation has been run for {0:.3f}s, consuming {1:.1f}h of wall-time.".format(
                self.timeFin - self.timeOri, self.cptim[-1]
            ),
            "\n",
        ]

        with open(file, "a") as f:
            f.write("\n".join(lines))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        lines = [
            "",
            "--------------------------",
            "Error levels",
            "--------------------------",
            "",
            "Error in Calculated, Equilibrium and Integrated currents:          {:.1f}%, {:.1f}%, {:.1f}%".format(
                np.abs(self.Ip[ind] - self.Ip_eq[ind]) / self.Ip[ind] * 100.0,
                np.abs(self.Ip[ind] - self.Ip_j[ind]) / self.Ip[ind] * 100.0,
                np.abs(self.Ip[ind] - self.Ip_eq[ind]) / self.Ip[ind] * 100.0,
            ),
            "Relative equilibrium Grad-Shafranov Error:                         {:.1f}%".format(
                self.GS_error[ind] * 100.0
            ),
            "Zeff check, difference from real Zeff and thermal-species Zeff:    {0:.1e}%".format(
                self.Zeff_error[ind] * 100.0
            ),
            "Quasineutrality error:                                             {0:.1e}%".format(
                self.QN_error[ind] * 100.0
            ),
            "\n",
        ]

        with open(file, "a") as f:
            f.write("\n".join(lines))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.numSaw is not None:
            txtall = "Times when plasma sawtoothed (only the last 3 times):"
            for i in range(np.min([3, self.numSaw])):
                txtall += f" {self.tlastsawU[-i - 1]:.3f}s,"
            if self.numSaw == 0:
                txtall = ""
            else:
                txtall = txtall[:-1] + "..."
            txtsaw = "Sawtooth trigger model is enabled in this plasma. So far, {0} sawtooth crashes have taken place.\n{1}\n".format(
                self.numSaw, txtall
            )
        else:
            txtsaw = "\n"

        lines = [
            "",
            "--------------------------",
            "Results",
            "--------------------------",
            ""
            "In the following, simulation results are indicated, extracted at t={0:.3f}s".format(
                self.t[ind]
            ),
            txtsaw,
        ]

        with open(file, "a") as f:
            f.write("\n".join(lines))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if self.gfile_in is not None:
            rmajor, epsilon, kappa, delta, zeta, z0 = self.gfile_in.Rmajor, self.gfile_in.eps, self.gfile_in.kappa, self.gfile_in.delta, self.gfile_in.zeta, self.gfile_in.Zmag
            extrakappa = f"\t(gfile kappa = {kappa:.2f})"
            extradelta = f"\t(gfile delta = {delta:.2f})"
        else:
            extrakappa, extradelta = "", ""

        lines = [
            "",
            "Basic machine parameters",
            "--------------------------",
            ""
            "Geometric axis position:               R0         = {0:.2f} m".format(
                self.Rmajor[ind]
            ),
            f"Half-width of midplane intercept:      a          = {self.a[ind]:.2f} m",
            "Inverse aspect ratio:                  epsilon    = {0:.3f}".format(
                self.epsilon[ind]
            ),
            f"External magnetic field on axis (@R0): Bt         = {self.Bt[ind]:.2f} T",
            "Total plasma current:                  Ip         = {0:.2f} MA".format(
                self.Ip[ind]
            ),
            "TRANSP-separatrix elongation:          kappa_sep  = {0:.2f}{1}".format(
                self.kappa[ind], extrakappa
            ),
            "TRANSP-separatrix triangularity:       delta_sep  = {0:.2f}{1}".format(
                self.delta[ind], extradelta
            ),
            "\n",
        ]

        with open(file, "a") as f:
            f.write("\n\t\t".join(lines))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        lines = [
            "",
            "Geometric parameters",
            "--------------------------",
            f"Shafranov shift:           Asaf        = {self.ShafShift[ind]:.2f} m",
            f"Magnetic axis position:    Rmag        = {self.Rmag[ind]:.2f} m",
            f"99.5% elongation:          kappa_99.5  = {self.kappa_995[ind]:.2f}",
            f"99.5% triangularity:       delta_99.5  = {self.delta_995[ind]:.2f}",
            "\n",
        ]

        with open(file, "a") as f:
            f.write("\n\t\t".join(lines))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        lines = [
            "",
            "Energy content & Power flow",
            "--------------------------",
            ""
            "Total stored energy:                           Wtot       = {0:.2f} MJ".format(
                self.Wtot[ind]
            ),
            "Fraction of thermal to total stored energy:    Wth/Wtot   = {0:.0f}%".format(
                self.Wth_frac[ind] * 100.0
            ),
            "Total power through LCFS:                      Psol       = {0:.2f} MW".format(
                self.P_LCFS[ind]
            ),
            "Total radiated power:                          Prad       = {0:.2f} MW ({1:.0f}% of total power)".format(
                self.PradT[ind],
                100.0 * self.PradT[ind] / (self.PradT[ind] + self.P_LCFS[ind]),
            ),
            "\n",
        ]

        with open(file, "a") as f:
            f.write("\n\t\t".join(lines))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        lines = [
            "",
            "ICRF",
            "--------------------------",
            ""
            "Fraction of ICRF to minorities:                Fmini  = {0:.0f}%".format(
                self.PichT_min_frac[ind] * 100.0
            ),
            "Fraction of ICRF to electrons (direct):        Fe     = {0:.0f}%".format(
                self.PeichT_dir_frac[ind] * 100.0
            ),
            "\n",
        ]

        with open(file, "a") as f:
            f.write("\n\t\t".join(lines))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        lines = [
            "",
            "Plasma Performance",
            "--------------------------",
            ""
            "Fusion gain:                                     Q       = {0:.2f}".format(
                self.Q[ind]
            ),
            "Volume-averaged ion temperature:                 <Ti>    = {0:.2f} keV".format(
                self.Ti_avol[ind]
            ),
            "Volume-averaged electron temperature:            <Te>    = {0:.2f} keV".format(
                self.Te_avol[ind]
            ),
            "Volume-averaged electron density:                <ne>    = {0:.2f} 1E20m^-3 (fGv = {1:.2f})".format(
                self.ne_avol[ind], self.fGv[ind]
            ),
            "Ion temperature peaking factor (@rho=0):         nu_Ti   = {0:.2f}".format(
                self.Ti_peakingX[ind, 0]
            ),
            "Ion temperature peaking factor (@rhop=0.2):      nu_Ti02 = {0:.2f}".format(
                self.Ti_peakingX[ind, np.argmin(np.abs(self.xpol[ind] - 0.2))]
            ),
            "Electron temperature peaking factor (@rho=0):    nu_Te   = {0:.2f}".format(
                self.Te_peakingX[ind, 0]
            ),
            "Electron temperature peaking factor (@rhop=0.2): nu_Te02 = {0:.2f}".format(
                self.Te_peakingX[ind, np.argmin(np.abs(self.xpol[ind] - 0.2))]
            ),
            "Electron density peaking factor (@rho=0):        nu_ne   = {0:.2f}".format(
                self.ne_peakingX[ind, 0]
            ),
            "Electron density peaking factor (@rhop=0.2):     nu_ne02 = {0:.2f}".format(
                self.ne_peakingX[ind, np.argmin(np.abs(self.xpol[ind] - 0.2))]
            ),
            "Thermal energy confinement time:                 taue    = {0:.0f} ms".format(
                self.taue[ind]
            ),
            "H98y2 factor (check):                            H98y2   = {0:.2f}".format(
                self.H98y2_check[ind]
            ),
            "H89 factor:                                      H89     = {0:.2f}".format(
                self.H89p[ind]
            ),
            "Total fusion power:                              Pfus    = {0:.1f} MW".format(
                self.Pout[ind]
            ),
            "Fraction of thermal to total neutrons:           fth     = {0:.2f}%".format(
                self.neutrons_thr_frac[ind] * 100.0
            ),
            "Fraction of DT to thermal neutrons:              fDT     = {0:.2f}%".format(
                self.neutrons_thrDT_frac[ind] * 100.0
            ),
            "LH Power Threshold:							  Plh     = {0:.1f} MW (fLH = {1:.2f})".format(
                self.Plh[ind], self.Plh_ratio[ind]
            ),
            "\n",
        ]

        with open(file, "a") as f:
            f.write("\n\t\t".join(lines))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        lines = [
            "",
            "Plasma Content",
            "--------------------------",
            ""
            "Fuel Mix:                                      {:.1f}% D and {:.1f}% T".format(
                self.nD_frac[ind] * 100.0, self.nT_frac[ind] * 100.0
            ),
            "Volume-average concentration of fast alphas:   falpha      = {0:.2f}%".format(
                self.ffus_avol[ind] * 100.0
            ),
            "Central (@rho=0) concentration of fast alphas: falpha0     = {0:.2f}%".format(
                self.ffus[ind, 0] * 100.0
            ),
            "Volume-average concentration of He4 ash:       fHe4        = {0:.2f}%".format(
                self.fHe4_avol[ind] * 100.0
            ),
            "Central (@rho=0) concentration of He4 ash:     fHe40       = {0:.2f}%".format(
                self.fHe4[ind, 0] * 100.0
            ),
            "Volume-average effective ion charge:           Zeff        = {0:.2f}".format(
                self.Zeff_avol[ind]
            ),
            "Volume-average concentration of impurities:    fimp        = {0:.2f}%".format(
                self.fZ_avol[ind] * 100.0
            ),
            "Average charge and associated concentration:   Zave, fave  = {0:.1f}, {1:.2f}%".format(
                self.fZ_avolAVE_Z[ind], self.fZ_avolAVE[ind] * 100.0
            ),
            "\n",
        ]

        with open(file, "a") as f:
            f.write("\n\t\t".join(lines))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        lines = [
            "",
            "Stability",
            "--------------------------",
            ""
            "Volume-averaged Greenwald fraction:            fGv     = {0:.2f}".format(
                self.fGv[ind]
            ),
            "Line-averaged (thru Rmag) Greenwald fraction:  fGl     = {0:.2f} ({1:.1f}% above fGv)".format(
                self.fGl[ind], 100 * (self.fGl[ind] - self.fGv[ind]) / self.fGv[ind]
            ),
            "Edge (@ rho=0.95) Greenwald fraction:          fGp     = {0:.2f}".format(
                self.fG[ind, np.argmin(np.abs(self.x_lw - 0.95))]
            ),
            "Safety factor at 95% flux surface:             q95     = {0:.2f}".format(
                self.q95[ind]
            ),
            "Normalized Beta:                               BetaN   = {0:.2f}".format(
                self.BetaN[ind]
            ),
            "Bootstrap Fraction:                            fB      = {0:.1f}%".format(
                self.IpB_fraction[ind] * 100.0
            ),
            "\n",
        ]

        with open(file, "a") as f:
            f.write("\n\t\t".join(lines))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        lines = [
            "",
            "Times",
            "--------------------------",
            f"Thermal energy confinement time:   taue     = {self.taue[ind]:.0f} ms",
            "Current Diffusion Time:            tau_CD   = {0:.1f} s".format(
                self.tau_c[ind] / 1000.0
            ),
            "Sawtooth Period:                   tau_saw  = {0:.1f} s".format(
                self.tau_saw[ind] / 1000.0
            ),
            "\n",
        ]

        with open(file, "a") as f:
            f.write("\n\t\t".join(lines))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        dif = self.GainT[ind] - (self.PtotExtra[ind] - self.Losses[ind])
        difPerc = np.abs(dif) / self.Ptot[ind] * 100.0

        lines = [
            "",
            "Main power flows",
            "--------------------------",
            "",
            ">>>> Losses:",
            "ELECTRONS Exhausted power through LCFS:    P = {0:.2f} MW".format(
                self.Pe_LCFS[ind]
            ),
            f"ELECTRONS Radiated power:                  P = {self.PradT[ind]:.2f} MW",
            "ELECTRONS Losses:                          P = {0:.2f} MW".format(
                self.Losses_elec[ind]
            ),
            "",
            "IONS Exhausted power through LCFS:         P = {0:.2f} MW".format(
                self.Pi_LCFS[ind]
            ),
            f"IONS CX power:                             P = {self.PcxT[ind]:.2f} MW",
            "IONS Losses:                               P = {0:.2f} MW".format(
                self.Losses_ions[ind]
            ),
            "",
            f"TOTAL Exhausted power through LCFS:        P = {self.P_LCFS[ind]:.2f} MW",
            f"TOTAL Losses:                              P = {self.Losses[ind]:.2f} MW",
            "",
            ">>>> Heating:",
            f"ELECTRONS ICRF power:                      P = {self.PeichT[ind]:.2f} MW",
            f"ELECTRONS ECRF power:                      P = {self.PechT[ind]:.2f} MW",
            f"ELECTRONS NBI power:                       P = {self.PnbieT[ind]:.2f} MW",
            f"ELECTRONS Ohmic power:                     P = {self.PohT[ind]:.2f} MW",
            f"ELECTRONS Alpha power:                     P = {self.PfuseT[ind]:.2f} MW",
            "ELECTRONS Total power:                     P = {0:.2f} MW".format(
                self.PowerToElec[ind]
            ),
            "",
            f"IONS ICRF power:                           P = {self.PiichT[ind]:.2f} MW",
            f"IONS NBI power:                            P = {self.PnbiiT[ind]:.2f} MW",
            f"IONS Alpha power:                          P = {self.PfusiT[ind]:.2f} MW",
            "IONS Total power:                          P = {0:.2f} MW".format(
                self.PowerToIons[ind]
            ),
            "",
            f"TOTAL ICRF power:                          P = {self.PichT[ind]:.2f} MW",
            f"TOTAL ECRF power:                          P = {self.PechT[ind]:.2f} MW",
            f"TOTAL NBI power:                           P = {self.PnbiT[ind]:.2f} MW",
            f"TOTAL Alpha power:                         P = {self.PfusT[ind]:.2f} MW",
            f"TOTAL Power in:                            P = {self.Ptot[ind]:.2f} MW",
            "",
            ">>>> Balance:",
            "ELECTRONS Power_in - Power_out - QIE:      P = {0:.2f} MW".format(
                self.PowerToElec[ind] - (self.Losses_elec[ind] + self.PeiT[ind])
            ),
            "IONS Power_in - Power_out + QIE:           P = {0:.2f} MW".format(
                self.PowerToIons[ind] - (self.Losses_ions[ind]) + self.PeiT[ind]
            ),
            "TOTAL Power_in - Power_out:                P = {0:.2f} MW".format(
                self.Ptot[ind] - self.Losses[ind]
            ),
            "",
            f"ELECTRONS dW/dt:                           G = {self.GaineT[ind]:.2f} MW",
            f"IONS dW/dt:                                G = {self.GainiT[ind]:.2f} MW",
            f"TOTAL dW/dt:                               G = {self.GainT[ind]:.2f} MW",
            "",
            ">>>> Extra:",
            f"IONS THERMALIZED NBI power:                P = {self.PnbihT[ind]:.2f} MW",
            f"IONS ROTATION NBI power:                   P = {self.PnbirT[ind]:.2f} MW",
            "EXTRA Total power balance:                 P = {0:.2f} MW".format(
                self.PowerIn_other[ind]
            ),
            "",
            "~~~ Difference (including EXTRA):          P = {0:.1f}kW ({1:.1f}% compared to total heating)".format(
                dif * 1000.0, difPerc
            ),
            "\n",
        ]

        with open(file, "a") as f:
            f.write("\n\t\t".join(lines))

        lines = [
            "",
            "-----------------------------------------------------------------------------------------------------",
        ]
        with open(file, "a") as f:
            f.write("\n".join(lines))

    def writeResults_XLSX(self, file="results.xlsx"):
        variables = IOtools.OrderedDict(
            {
                "Bt (T)": self.Bt,
                "Ip (MA)": self.Ip,
                "Picrf (MW)": self.PichT,
                "mass (u)": self.mi / self.u * np.ones(len(self.t)),
                "ne_top/nG": self.fGh,  #'fW (1E-5)' :self.fZs_avol['W']['total']*1E5,
                "Zeff": self.Zeff_avol,
            }
        )

        self.dataSet = {}
        for ikey in variables:
            self.dataSet[ikey] = variables[ikey][self.ind_saw]

        IOtools.addRowToExcel(
            file, self.dataSet, row_name=self.nameRunid, repeatIfIndexExist=False
        )

    def videoMain1(
        self, MovieFile="~/movie.mp4", trange=[0.0, 100.0], numTimes=10, secondsMovie=5
    ):
        MovieFile = IOtools.expandPath(MovieFile)

        # ~~~~~~~~~~~~ Variables to include

        Te = GRAPHICStools.reduceVariable(self.Te, numTimes, t=self.t, trange=trange)
        Ti = GRAPHICStools.reduceVariable(self.Ti, numTimes, t=self.t, trange=trange)
        ne = GRAPHICStools.reduceVariable(self.ne, numTimes, t=self.t, trange=trange)
        nmain = GRAPHICStools.reduceVariable(
            self.nmain, numTimes, t=self.t, trange=trange
        )
        nHe4 = GRAPHICStools.reduceVariable(
            self.nHe4, numTimes, t=self.t, trange=trange
        )
        x = GRAPHICStools.reduceVariable(self.x, numTimes, t=self.t, trange=trange)
        time = GRAPHICStools.reduceVariable(self.t, numTimes, t=self.t, trange=trange)
        Q = GRAPHICStools.reduceVariable(self.Q, numTimes, t=self.t, trange=trange)
        q95 = GRAPHICStools.reduceVariable(self.q95, numTimes, t=self.t, trange=trange)
        qstar_sep = GRAPHICStools.reduceVariable(
            self.qstar_sep, numTimes, t=self.t, trange=trange
        )
        kappa_995 = GRAPHICStools.reduceVariable(
            self.kappa_995, numTimes, t=self.t, trange=trange
        )
        delta_995 = GRAPHICStools.reduceVariable(
            self.delta_995, numTimes, t=self.t, trange=trange
        )

        # ~~~~~~~~~~~~ Plots

        size = 9
        plt.rc("axes", labelsize=size)
        plt.rc("xtick", labelsize=size)
        plt.rc("ytick", labelsize=size)
        fig = plt.figure(figsize=(10, 5))
        grid = plt.GridSpec(4, 3, hspace=0.2, wspace=0.2)

        ax0 = fig.add_subplot(grid[1:, 0])
        ax5 = fig.add_subplot(grid[0, 0])

        ax1 = fig.add_subplot(grid[:2, 1])
        ax2 = fig.add_subplot(grid[2:, 1])

        ax3 = fig.add_subplot(grid[:2, 2])
        ax4 = fig.add_subplot(grid[2:, 2])
        ax6 = ax4.twinx()

        axs = [ax0, ax1, ax2, ax3, ax4, ax5, ax6]

        def plottingFunction(axs, i):
            # ~~~~~~~~~~ Subplot
            ax = axs[0]
            self.plotGeometry(ax=ax, color="b", plotComplete=True, time=time[i])
            ax.set_ylim([-1.2, 1.2])

            ax = axs[5]
            self.plotGeometry(
                ax=ax,
                color="b",
                plotComplete=True,
                time=time[i],
                Aspect=False,
                rhoS=np.linspace(0.90, 1, 30),
            )
            ax.set_ylim([0.9, 1.15])
            ax.set_aspect("equal")

            # ~~~~~~~~~~ Subplot
            ax = axs[1]
            ax.plot(x[i], Te[i], lw=3, c="r", label="$T_e$ ($keV$)")
            ax.plot(x[i], Ti[i], lw=3, c="b", label="$T_i$ ($keV$)")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 25])
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.legend(loc="upper right", prop={"size": size})

            # ~~~~~~~~~~ Subplot
            ax = axs[2]
            ax.plot(x[i], ne[i], lw=3, c="r", label="$n_{e}$ ($10^{20}m^{-3}$)")
            ax.plot(x[i], nmain[i], lw=3, c="b", label="$n_{DT}$ ($10^{20}m^{-3}$)")
            # ax.plot(x[i],nHe4[i]*10.0,lw=3,c='m',label='$n_{He4}x10$ ($10^{20}m^{-3}$)')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 7])
            ax.set_xlabel("$\\rho_N$")
            ax.legend(loc="upper right", prop={"size": size})

            # ~~~~~~~~~~ Subplot
            ax = axs[3]
            ax.plot(
                time[: i + 1], Ti[: i + 1, 0], lw=3, c="b", label="$T_{e,0}$ ($keV$)"
            )
            ax.plot(
                time[: i + 1], Te[: i + 1, 0], lw=3, c="r", label="$T_{i,0}$ ($keV$)"
            )
            ax.plot(time[: i + 1], Q[: i + 1], lw=3, c="m", label="$Q$")
            ax.set_xlim([time[0], time[-1]])
            ax.set_ylim([0, 25])
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.legend(loc="lower left", prop={"size": size})

            # ~~~~~~~~~~ Subplot
            ax = axs[4]
            ax.plot(
                time[: i + 1], kappa_995[: i + 1], lw=2, c="b", label="$\\kappa_{99.5}$"
            )
            ax.plot(
                time[: i + 1], delta_995[: i + 1], lw=2, c="r", label="$\\delta_{99.5}$"
            )
            ax.set_xlim([time[0], time[-1]])
            ax.set_ylim([0, 2.2])
            ax.set_xlabel("Time (s)")
            ax.legend(loc="lower left", prop={"size": size})

            ax = axs[6]
            ax.plot(time[: i + 1], q95[: i + 1], lw=2, c="g", label="$q_{95}$")
            ax.plot(time[: i + 1], qstar_sep[: i + 1], lw=2, c="m", label="$q_{*}$")
            ax.set_xlim([time[0], time[-1]])
            ax.set_ylim([0, 5])
            ax.set_xlabel("Time (s)")
            ax.legend(loc="lower right", prop={"size": size})

        GRAPHICStools.animageFunction(
            plottingFunction,
            axs,
            fig,
            MovieFile,
            len(x),
            framePS=round(numTimes / secondsMovie),
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def writeAuxiliarFiles(
        self, loc="~/mitim_runs/data/SPARC/features", name=None
    ):
        loc = IOtools.expandPath(loc)
        if name is None:
            name = f"Ev_{self.LocationCDF.name.split('.')[-2][5:]}"

        folder = loc / f"{name}"

        folder.mkdir(parents=True, exist_ok=True)

        # Impurities
        for cont, key in enumerate(self.nZs):
            dictPKL = {"rho": self.x_lw, "nimp": self.nZs[key]["total"][self.ind_saw]}
            file = folder / f"nimp{cont + 1}.pkl"
            with open(file, "wb") as handle:
                pickle.dump(dictPKL, handle, protocol=4)
            print(f" --> Written {file}")

        # Minorities
        dictPKL = {"rho": self.x_lw, "n": self.nmini[self.ind_saw]}
        file = folder / f"nmini.pkl"
        with open(file, "wb") as handle:
            pickle.dump(dictPKL, handle, protocol=4)
        print(f" --> Written {file}")

# ---------------------------------------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------------------------------------


def LH_Martin1(n, Bt, S, nmin=0):
    return PLASMAtools.LHthreshold_Martin1(n, Bt, S, nmin=nmin)


def LH_Martin2(n, Bt, a, Rmajor, nmin=0):
    return PLASMAtools.LHthreshold_Martin2(n, Bt, a, Rmajor, nmin=nmin)


def LH_Schmid1(n, Bt, S, nmin=0):
    return PLASMAtools.LHthreshold_Schmid1(n, Bt, S, nmin=nmin)


def LH_Schmid2(n, Bt, S, nmin=0):
    return PLASMAtools.LHthreshold_Schmid2(n, Bt, S, nmin=nmin)


def LH_Martin1_low(n, Bt, S, nmin=0):
    return PLASMAtools.LHthreshold_Martin1_low(n, Bt, S, nmin=nmin)


def LH_Martin1_up(n, Bt, S, nmin=0):
    return PLASMAtools.LHthreshold_Martin1_up(n, Bt, S, nmin=nmin)


def getFluxSurface(f, t0, x0, thetap=2e3, rhoPol=False, sqrt=True):
    if not rhoPol:
        x = f["XB"][:][-1, :]
        if not sqrt:
            x = x**2
    else:
        if sqrt:
            x = np.sqrt(f["PLFLX"][:][-1, :] / f["PLFLX"][:][-1, -1])
        else:
            x = f["PLFLX"][:][-1, :] / f["PLFLX"][:][-1, -1]
    t = f["TIME"][:]

    if x0 < 1.0:
        RMC, YMC = constructMoments(f, t0, x0, x, t, thetap=thetap)
    else:
        RMC, YMC = constructMomentsBoundary(f, t0, t, thetap=thetap)

    return RMC, YMC


def constructMoments(f, t0, x0, x, t, thetap=2e3):
    index_x, index_t = np.argmin(np.abs(x - x0)), np.argmin(np.abs(t - t0))

    theta = np.linspace(0, 2 * np.pi, int(thetap))
    RMC = f["RMC00"][:][index_t, index_x] * 1e-2
    YMC = f["YMC00"][:][index_t, index_x] * 1e-2

    for i in range(100):
        try:
            RMC1 = f[f"RMC{str(i + 1).zfill(2)}"][:][index_t, index_x] * 1e-2
            RMS1 = f[f"RMS{str(i + 1).zfill(2)}"][:][index_t, index_x] * 1e-2
            YMC1 = f[f"YMC{str(i + 1).zfill(2)}"][:][index_t, index_x] * 1e-2
            YMS1 = f[f"YMS{str(i + 1).zfill(2)}"][:][index_t, index_x] * 1e-2

            RMC += RMC1 * np.cos(theta * (i + 1)) + RMS1 * np.sin(theta * (i + 1))
            YMC += YMC1 * np.cos(theta * (i + 1)) + YMS1 * np.sin(theta * (i + 1))
        except:
            # print('\t~ Max moment for flux surface at x={0}: {1}'.format(x0,i))
            break

    return RMC, YMC


def constructMomentsBoundary(f, t0, t, thetap=2e3):
    index_t = np.argmin(np.abs(t - t0))

    theta = np.linspace(0, 2 * np.pi, int(thetap))
    RMC = f["RMCB0"][:][index_t] * 1e-2
    YMC = f["YMCB0"][:][index_t] * 1e-2

    for i in range(100):
        try:
            RMC1 = f[f"RMCB{str(i + 1)}"][:][index_t] * 1e-2
            RMS1 = f[f"RMSB{str(i + 1)}"][:][index_t] * 1e-2
            YMC1 = f[f"YMCB{str(i + 1)}"][:][index_t] * 1e-2
            YMS1 = f[f"YMSB{str(i + 1)}"][:][index_t] * 1e-2

            RMC += RMC1 * np.cos(theta * (i + 1)) + RMS1 * np.sin(theta * (i + 1))
            YMC += YMC1 * np.cos(theta * (i + 1)) + YMS1 * np.sin(theta * (i + 1))
        except:
            # print('\t~ Max moment for boundary: {0}'.format(i))
            break
    return RMC, YMC


# def sawtoothAverage(self,var):

# 	last 			= self.ind_sawAll[0]
# 	secondToLast 	= self.ind_sawAll[1]

# 	var = np.mean(var[secondToLast:last],0)

# 	return var


def sumBetas(pol, tor):
    return 1.0 / (1.0 / pol + 1.0 / tor)


def findLastSawtoothIndex(cdf, howmanybefore=1, positionInChain=-1):
    numSawteeth = len(cdf.tlastsawU)

    # If no sawteeth, return last time
    if numSawteeth == 0:
        time = cdf.t[-1]
    # If sawteeth, return the position I requested
    else:
        time = cdf.tlastsawU[positionInChain]

    tin = np.argmin(np.abs(time - cdf.t))

    return tin - howmanybefore


def calculateFlux(var, f):
    q = volumeIntegral(f, var)
    Surf = f["SURF"][:] * 1e-4  # m^2

    return q / Surf


def plasmaSurface(f):
    # Plasma surface
    return f["SURF"][:][:, -1]


def plasmaXSarea(f):
    return np.sum(f["DAREA"][:], axis=1)


def volumeAverage(f, var, rangex=None):
    # Volume average of quantity
    return volumeIntegralTot(f, var, rangex=rangex) / plasmaVolume(f, rangex=rangex)


def plasmaVolume(f, rangex=None):
    # Plasma volume up to rho=1
    if rangex is None:
        volint = np.cumsum(f["DVOL"][:], axis=1)[:, -1]
    else:
        volint = np.cumsum(f["DVOL"][:], axis=1)[:, rangex]
    return volint


def volumeIntegralTot(f, var, rangex=None):
    # Calculates volume integral up to rho=1.0, in time
    if rangex is None:
        volint = volumeIntegral(f, var)[:, -1]
    else:
        volint = volumeIntegral(f, var)[:, rangex]
    return volint


def volumeIntegral(f, var):
    # Calculates volume integral up to RHO, in time
    return np.cumsum(f["DVOL"][:] * f[var][:], axis=1)  # Watch out for X or XB grid!!!!


def volumeAverage_var(f, var, rangex=None):
    return volumeIntegralTot_var(f, var, rangex=rangex) / plasmaVolume(f, rangex=rangex)


def volumeIntegralTot_var(f, var, rangex=None):
    # Calculates volume integral up to rho=1.0, in time
    if rangex is None:
        volint = volumeIntegral_var(f, var)[:, -1]
    else:
        volint = volumeIntegral_var(f, var)[:, rangex]
    return volint


def volumeIntegral_var(f, var):
    # Calculates volume integral up to RHO, in time
    return np.cumsum(f["DVOL"][:] * var, axis=1)  # Watch out for X or XB grid!!!!


def volumeMultiplication(f, var):
    # Multiplies zone volume and quantity, to transform MW/m^3 -> MW at each zone
    return f["DVOL"][:] * f[var][:]


def surfaceIntegralTot(f, var):
    # Calculates volume integral up to rho=1.0, in time
    return surfaceIntegral(f, var)[:, -1]


def surfaceIntegralTot_var(f, var):
    # Calculates volume integral up to rho=1.0, in time
    return surfaceIntegral_var(f, var)[:, -1]


def surfaceIntegral(f, var):
    # Calculates surface integral up to RHO, in time
    return np.cumsum(
        f["DAREA"][:] * f[var][:], axis=1
    )  # Watch out for X or XB grid!!!!


def surfaceIntegral_var(f, var):
    # Calculates surface integral up to RHO, in time
    return np.cumsum(f["DAREA"][:] * var, axis=1)  # Watch out for X or XB grid!!!!


def gradNorm(CDFc, varData, specialDerivative=None):
    """
    This gives:
            a/var * dvar/dr
    """

    grad = derivativeVar(CDFc, varData, specialDerivative=specialDerivative)

    Ld_inv = -1.0 * grad / varData

    aLd = []
    for it in range(len(CDFc.t)):
        aLd.append(CDFc.a[it] * Ld_inv[it, :])

    return np.array(aLd)


def derivativeVar(CDFc, varData, specialDerivative=None, onlyOneTime=False):
    if specialDerivative is None:
        specialDerivative = CDFc.rmin

    if not onlyOneTime:
        grad = []
        for it in range(len(CDFc.t)):
            grad.append(MATHtools.deriv(specialDerivative[it, :], varData[it, :]))
        grad = np.array(grad)
    else:
        grad = MATHtools.deriv(specialDerivative, varData)

    return grad


def projectGyrotron(x0, y0, polar, azym, time, rho, f):
    t = np.arange(0, 10, 0.1)

    R = x0 - t * np.cos(polar)
    Z = y0 - t * np.sin(polar)

    ix = np.argmin(np.abs(rho - 1.0))
    RMC, YMC = getFluxSurface(f, time, ix)
    Z = Z[R > RMC.min()]
    R = R[R > RMC.min()]

    return R, Z


def mapRmajorToX(cdf, var, originalCoordinate="RMAJM", interpolateToX=True):
    var_HFx, var_LFx = [], []
    for i in range(cdf.nt):
        # Divide in LH and HF (but if it's RMJSYM, first convert to RMAJM, which correspond to XB)
        if originalCoordinate == "RMJSYM":
            varZ = np.interp(cdf.Rmaj[i], cdf.Rmaj2[i], var[i])
        elif originalCoordinate == "RMAJM":
            varZ = var[i]

        # Divide vector in the two parts (everything in XB here), flip the HF so that it starts near axis
        hf = len(varZ) // 2
        var_HF = np.flipud(varZ[:hf])
        var_LF = varZ[hf + 1 :]

        # Perform interpolation to X from XB
        if interpolateToX:
            var_HF = np.interp(cdf.x_lw, cdf.xb_lw, var_HF)
            var_LF = np.interp(cdf.x_lw, cdf.xb_lw, var_LF)

        var_HFx.append(var_HF)
        var_LFx.append(var_LF)

    var_HFx = np.array(var_HFx)
    var_LFx = np.array(var_LFx)

    return var_LFx, var_HFx


# -------------- PLOTTING


def prepareTimingsSaw(time, cdf):
    if time is None or time == cdf.t[cdf.ind_saw]:
        i1, i2 = cdf.ind_saw, cdf.ind_saw_after
    else:
        i1, i2 = np.argmin(np.abs(cdf.t - time)), np.argmin(np.abs(cdf.t - time))

    return i1, i2


def addDetailsRho(ax, label="", title="", legend=True, size=None):
    ax.set_xlim([0, 1])
    ax.autoscale(enable=True, axis="y", tight=False)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("$\\rho_N$", size=size)
    if label is not None:
        if size is not None:
            ax.set_ylabel(label, size=size)
        else:
            ax.set_ylabel(label)
    else:
        ax.yaxis.tick_right()
    if size is not None:
        ax.tick_params(axis="both", which="major", labelsize=size)
        ax.set_title(title, size=size)
    else:
        ax.tick_params(axis="both", which="major")
        ax.set_title(title)

    if legend:
        GRAPHICStools.addLegendApart(ax, ratio=0.7)


def preparePlotTime(rhos, cdf, ax):
    i = []
    for ir in rhos:
        i.append(np.argmin(np.abs(cdf.x_lw - ir)))
    i = np.array(i)

    if ax is None:
        fig, ax = plt.subplots()

    return ax, i


def addDetailsTime(ax, label="", title="", legend=True, size=None):
    ax.autoscale(enable=True, axis="x", tight=True)
    ax.autoscale(enable=True, axis="y", tight=False)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time (s)", size=size)
    if label is not None:
        if size is not None:
            ax.set_ylabel(label, size=size)
        else:
            ax.set_ylabel(label)
    else:
        ax.yaxis.tick_right()
    if size is not None:
        ax.tick_params(axis="both", which="major", labelsize=size)
        ax.set_title(title, size=size)
    else:
        ax.tick_params(axis="both", which="major")
        ax.set_title(title)

    if legend:
        GRAPHICStools.addLegendApart(ax, ratio=0.7)


def correctMasked(x):
    xn = copy.deepcopy(x)

    for i in range(len(x)):
        if len(x.shape) > 1:
            for j in range(len(x[i])):
                if np.ma.is_masked(x[i, j]):
                    xn[i, j] = 0
                else:
                    xn[i, j] = x[i, j]
        else:
            if np.ma.is_masked(x[i]):
                xn[i] = 0
            else:
                xn[i] = x[i]
    return xn


def getRunMetaInfo(cdf):
    net = netCDF4.Dataset(cdf)

    try:
        dictVar = {
            "version_TRANSP": "{0} ({1}, {2})".format(
                net.TRANSP_version, net.TRANSP_DOI, net.BUILD_date
            ),
            "version_TGLF": net.TGLF_hash,
            "date_run": net.CDF_date,
        }
    except:
        print("\t ! Could not parse meta data", typeMsg="w")
        dictVar = {}

    return dictVar


def constant_radius(vect, lenX=100):
    return np.transpose(np.tile(vect, (lenX, 1)))


def timeAverage(t, var):
    dt = np.append(np.diff(t), [0])
    dtn = []
    for i in range(len(dt)):
        if i == 0:
            dtn.append(dt[i])
        elif i == len(dt) - 1:
            dtn.append(dt[i - 1])
        else:
            dtn.append(dt[i - 1] / 2 + dt[i] / 2)
    dtn = np.array(dtn)

    if np.sum(dtn) == 0:
        dtn = [1]

    z = np.average(var, weights=dtn, axis=0)

    return z


def sawtoothAverage(t, var, tsaws, time=100.0):
    closestSawtooth = np.argmin(np.abs(time - tsaws))

    try:
        it1 = np.argmin(np.abs(t - tsaws[closestSawtooth - 1]))
        it2 = np.argmin(np.abs(t - tsaws[closestSawtooth]))
        return timeAverage(t[it1:it2], var[it1:it2])
    except:
        return timeAverage(t[-2:], var[-2:])


def profilePower(x, volumes, TotalPower, mixRadius, blend=0.05):
    """
    Constant h=1 until x1 and parabola until 0 at x1+w
    """

    x1 = mixRadius - blend
    w = blend * 2

    # Normalized function
    ix = np.argmin(np.abs(x - (x1 + w)))

    if ix > 1:

        y1 = PLASMAtools.fitTANHPedestal(
            w=w, xgrid=np.linspace(0, 1, len(x[:ix])), perc=[0.01, 0.01]
        )

    # Is it's a point, just give zero, not worry about calculating
    else:
        y1 = np.zeros(len(x[:ix]))

    y2 = np.zeros(len(x[ix:]))

    y = np.append(y1, y2)

    y = y / np.sum(y * volumes)

    # Multiply to give total required energy

    y = y * TotalPower

    return y

def penaltyFunction(x, valuemin=-1e9, valuemax=1e9):
    if valuemax < 1e9:
        fac = 100 / valuemax
    else:
        fac = 100 / valuemin

    y = (1 - np.exp(fac * (x - valuemax))) * (1 - np.exp(-fac * (x - valuemin)))

    return y


def definePenalties(q95, fG, kappa, BetaN, maxKappa=1.8):
    # Penalty on q95
    penalty_q95 = penaltyFunction(q95, valuemin=3.0)

    # Penalty on fG
    penalty_fG = penaltyFunction(fG, valuemax=0.9)

    # Penalty on kappa
    penalty_kappa = penaltyFunction(kappa, valuemax=maxKappa)

    # Penalty on beta
    penalty_beta = penaltyFunction(BetaN, valuemax=2.8)

    return np.max([penalty_q95 * penalty_fG * penalty_kappa * penalty_beta, 0.0])


def getBeamTrajectories(namelist):
    try:
        from trgui_fbm import plot_aug
    except ImportError:
        print(
            "\t- TRANSP tools external modules are not available. Please ensure it is installed and accessible.",
            typeMsg="i",
        )

    xlin, ylin, rlin, zlin = plot_aug.nbi_plot(
        nbis=[1, 2, 3, 4, 5, 6, 7, 8], runid=namelist[:-6]
    )

    beam_trajectories = {
        "xlin": np.array(xlin) * 1e-2,
        "ylin": np.array(ylin) * 1e-2,
        "rlin": np.array(rlin) * 1e-2,
        "zlin": np.array(zlin) * 1e-2,
    }

    return beam_trajectories


def getECRHTrajectories(namelist, Theta_gyr, Phi_gyr):
    try:
        from trgui_fbm import los
    except ImportError:
        print(
            "\t- TRANSP tools external modules are not available. Please ensure it is installed and accessible.",
            typeMsg="i",
        )

    xsrc = np.array(
        [
            float(i)
            for i in IOtools.findValue(namelist, "XECECH", "=", isitArray=True)
            .split("\n")[0]
            .split(",")
        ]
    )
    ysrc = np.zeros(8)
    xybsca = np.array(
        [
            float(i)
            for i in IOtools.findValue(namelist, "ZECECH", "=", isitArray=True)
            .split("\n")[0]
            .split(",")
        ]
    )

    xlin, ylin, rlin, zlin, nbis = [], [], [], [], [1, 2, 3, 4, 5, 6, 7, 8]
    for jnbi in nbis:
        jnb = jnbi - 1

        aug_geo = {}
        aug_geo["xend"] = 100
        aug_geo["x0"] = xsrc[jnb]
        aug_geo["y0"] = ysrc[jnb]
        aug_geo["z0"] = xybsca[jnb]
        aug_geo["theta"] = (90 - Theta_gyr[jnb]) * np.pi / 180.0
        aug_geo["phi"] = (Phi_gyr[jnb]) * np.pi / 180.0

        geom = los.PHILOS(aug_geo)
        xlin0 = [geom.xline[0], geom.xline[-1]]
        ylin0 = [geom.yline[0], geom.yline[-1]]
        rlin0 = geom.rline
        zlin0 = geom.zline

        xlin.append(xlin0)
        ylin.append(ylin0)
        rlin.append(rlin0)
        zlin.append(zlin0)

    ECRH_trajectories = {
        "xlin": np.array(xlin) * 1e-2,
        "ylin": np.array(ylin) * 1e-2,
        "rlin": np.array(rlin) * 1e-2,
        "zlin": np.array(zlin) * 1e-2,
    }

    return ECRH_trajectories
