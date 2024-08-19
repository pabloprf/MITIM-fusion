import os
import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools.utils import GACODEdefaults
from mitim_tools.misc_tools.IOtools import printMsg as print
from mitim_tools import __version__
from IPython import embed

'''
Example usage:

    # Initialize and define some timings
    t = NMLtools.transp_nml( shotnum = 12345, inputdir = "folder1/",
        timings     = {
            "time_start": 0.0,
            "time_current_diffusion": 0.11,
            "time_end": 1.0,
        } )

    # Grab structure from a given prescribed machine
    t.define_machine('SPARC')

    # Populate with default excepts the values passed
    t.populate(
        MCparticles = 1e4,
        Pich=True,
        )

    # Write the namelist
    t.write('Z99')

'''

class transp_nml:
    def __init__(
        self, 
        inputdir=None, 
        shotnum=175844, 
        pservers = {'nbi':1,'toric':1,'ptr':0}, 
        timings = {}
        ):

        self.inputdir = inputdir
        self.shotnum = shotnum

        if self.inputdir is not None and not os.path.exists(self.inputdir):
            os.makedirs(self.inputdir)

        # Until machine is defined, these are None
        self.ICRFantennas = None
        self.ECRFgyrotrons = None
        self.NBIbeams = None
        self.defineTRANSPnmlStructures = None
        self.defineISOLVER = None

        self.pservers = pservers

        self.tinit =timings.get("time_start",0.0) 
        self.ftime = timings.get("time_end",100.0) 
        self.time_current_diffusion = timings.get("time_current_diffusion",self.tinit) 
        self.timeSaw = timings.get("time_sawtooth",self.time_current_diffusion)
        self.timeAC = timings.get("time_extraction",None)

        self.contents_ptr_glf23 = None
        self.contents_ptr_tglf = None
        self.contents_ptr_ptsolver = None

    def define_machine(self, tokamak = "SPARC"):

        if tokamak == "SPARC" or tokamak == "ARC":
            from mitim_tools.experiment_tools.SPARCtools import ICRFantennas, defineTRANSPnmlStructures, defineISOLVER
            ECRFgyrotrons, NBIbeams = None, None
        elif tokamak == "CMOD":
            from mitim_tools.experiment_tools.CMODtools import ICRFantennas, defineTRANSPnmlStructures
            ECRFgyrotrons, NBIbeams, defineISOLVER = None, None, None
        elif tokamak == "AUG":
            from mitim_tools.experiment_tools.AUGtools import ECRFgyrotrons, NBIbeams, defineTRANSPnmlStructures
            ICRFantennas, defineISOLVER = None, None
        elif tokamak == "D3D":
            from mitim_tools.experiment_tools.DIIIDtools import ECRFgyrotrons, NBIbeams
            ICRFantennas, defineTRANSPnmlStructures, defineISOLVER = None, None, None

        self.ICRFantennas = ICRFantennas
        self.ECRFgyrotrons = ECRFgyrotrons
        self.NBIbeams = NBIbeams
        self.defineTRANSPnmlStructures = defineTRANSPnmlStructures
        self.defineISOLVER = defineISOLVER

    def _default_params(self,**transp_params):

        # -----------------------------------------------------------------------------------
        # Default values: Plasma
        # -----------------------------------------------------------------------------------

        self.DTplasma = transp_params.get("DTplasma",True)
        self.AddHe4ifDT = transp_params.get("AddHe4ifDT",False)
        self.Minorities = transp_params.get("Minorities",[2,3,0.05])
        self.zlump = transp_params.get("zlump",[
            [74.0, 183.0, 0.000286],
            [9.0, 18.0, 0.1]
            ] )

        self.taupD = transp_params.get("taupD",3.0)
        self.taupZ = transp_params.get("taupZ",3.0)
        self.taupmin = transp_params.get("taupmin",3.0)        
        
        self.rotating_impurity = transp_params.get("rotating_impurity",[self.zlump[-1][0],self.zlump[-1][1]])
        self.B_ccw = transp_params.get("B_ccw",False)
        self.Ip_ccw = transp_params.get("Ip_ccw",False)

        Ufiles = transp_params.get("Ufiles",["qpr","mry","cur","vsf","ter","ti2","ner","rbz","df4","vc4","lim","zf2","gfd"])

        potential_ufiles = {
            "qpr": ["QPR",-5],
            "mry": ["MRY",None],
            "cur": ["CUR",None],
            "vsf": ["VSF",None],
            "ter": ["TEL",-5],
            "ti2": ["TIO",-5],
            "ner": ["NEL",-5],
            "rbz": ["RBZ",None],
            "df4": ["DHE4",-5],
            "vc4": ["VHE4",-5],
            "lim": ["LIM",None],
            "zf2": ["ZF2",-5],
            "gfd": ["GFD",None],
            }

        self.Ufiles = {}
        for key in Ufiles:
            self.Ufiles[key] = potential_ufiles[key]

        # -----------------------------------------------------------------------------------
        # Default values: Settings
        # -----------------------------------------------------------------------------------

        self.Pich = transp_params.get("Pich",False)
        self.Pech = transp_params.get("Pech",False)
        self.Pnbi = transp_params.get("Pnbi",False)
        self.isolver = transp_params.get("isolver",False)

        self.LimitersInNML = transp_params.get("LimitersInNML",False)
        self.UFrotation = transp_params.get("UFrotation",False)

        self.dtHeating_ms = transp_params.get("dtHeating_ms",5.0)
        self.dtOut_ms = transp_params.get("dtOut_ms",1)
        self.dtIn_ms = transp_params.get("dtIn_ms",1)
        self.nzones = transp_params.get("nzones",100)
        self.nzones_energetic = transp_params.get("nzones_energetic",50)
        self.nzones_distfun = transp_params.get("nzones_distfun",25)
        self.nzones_frantic = transp_params.get("nzones_frantic",20)
        self.gridsMHD = transp_params.get("gridsMHD",[151,127])
        self.MCparticles = transp_params.get("MCparticles",1e6)
        self.useNUBEAMforAlphas = transp_params.get("useNUBEAMforAlphas",True)
        self.toric_ntheta = transp_params.get("toric_ntheta",128)  # 128 int(2**7), default: 64
        self.toric_nrho = transp_params.get("toric_nrho",320)  # 320, default: 128

        self.coronal = transp_params.get("coronal",True)
        self.nteq_mode = transp_params.get("nteq_mode",5)
        self.smoothMHD = transp_params.get("smoothMHD",[-1.5,0.15])
        
        self.coeffsSaw = transp_params.get("coeffsSaw",[1.0,3.0,1.0,0.4])
        self.ReconnectionFraction = transp_params.get("ReconnectionFraction",0.37)
        self.predictRad = transp_params.get("predictRad",True)
        self.useBootstrapSmooth = transp_params.get("useBootstrapSmooth",None)

        # -----------------------------------------------------------------------------------
        # Default values: Predictive
        # -----------------------------------------------------------------------------------

        self.PTsolver = transp_params.get("PTsolver",False)
        self.xbounds = transp_params.get("xbounds",[0.95,0.95,0.95])
        self.xminTrick = transp_params.get("xminTrick",0.2)
        self.xmaxTrick = transp_params.get("xmaxTrick",1.0)
        self.grTGLF = transp_params.get("grTGLF",False)
        self.Te_edge = transp_params.get("Te_edge",80.0)
        self.Ti_edge = transp_params.get("Ti_edge",80.0)
        self.TGLFsettings = transp_params.get("TGLFsettings",5)

    def populate(self, **transp_params):

        # Default values
        self._default_params(**transp_params)
        # -----------------------------------

        self.contents = ""

        self.addHeader()
        self.addTimings()
        self.addPB()
        self.addSpecies()
        self.addParticleBalance()
        self.addRadiation()
        self.addNCLASS()
        self.addMHDandCurrentDiffusion()
        if self.isolver: self.addISOLVER(defineISOLVER_method=self.defineISOLVER)
        self.addSawtooth()
        self.addFusionProducts()
        self.addRotation()
        self.addVessel(defineTRANSPnmlStructures_method=self.defineTRANSPnmlStructures)
        self.addUFILES()
        if self.Pich:    self.addICRF(ICRFantennas_method=self.ICRFantennas)
        if self.Pech:    self.addECRF(ECRFgyrotrons_method=self.ECRFgyrotrons)
        if self.Pnbi:    self.addNBI(NBIbeams_method=self.NBIbeams)
        
        if self.PTsolver:   self.addPTSOLVER()

        if self.timeAC is not None: self.addAC()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Methods to add different sections to the namelist
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def addHeader(self):
        lines = [
            f"! TRANSP Namelist generated by MITIM (version {__version__})",
            "",
            "!==============================================================",
            "! General settings & Output options",
            "!==============================================================",
            "",
            f"NSHOT={self.shotnum}",
            "",
            "mrstrt = -120     ! Frequency of restart records (-, means x wall clock min)",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

        if self.inputdir is not None:
            lines = [f'inputdir="{self.inputdir}"']
            self.contents += "\n".join(lines) + "\n"

    def addTimings(self):
        lines = [
            "",
            "!----- Time and Spatial ranges",
            "",
            f"tinit = {self.tinit:.3f} ! Start time",
            f"ftime = {self.ftime:.3f} ! End time",
            "xtend = 0.050 ! Distance beyond LCFS to extend analysis ",
            "",
            "!----- Time buffers",
            "",
            "tlim1 = 0.0    ! Ignore UF data before this time, extrapolate flat backward in time",
            "tlim2 = 1000.0 ! Ignore UF data after this time, extrapolate flat forward in time",
            "",
            "!----- Spatial resolution",
            "",
            f"nzones   = {self.nzones} ! Number of radial zones in 1D transport equations",
            f"nzone_nb = {self.nzones_energetic} ! Number of zones in NUBEAM for beams and alphas (default 20)",
            f"nzone_fp = {self.nzones_energetic} ! Number of zones in the FPPMOD for ICRF minority (default 20)",
            f"nzone_fb = {self.nzones_distfun} ! Number of zone rows in fast ion distr function (default 10)",
            "",
            "!----- Temporal resolution",
            "",
            "! * Geometry (MHD equilibrium)",
            f"dtming = {1.0e-5}  ! Minimum timestep",
            f"dtmaxg = {1.0e-3}  ! Maximum timestep (default 1.0e-2)",
            "",
            "! * Particle and energy balance (transport)",
            f"dtinit = {1.0e-3}  ! Initial timestep (default 1.0e-3)",
            f"dtmint = {1.0e-7}  ! Minimum timestep (default 1.0e-7)",
            f"dtmaxt = {2.0e-3}  ! Maximum timestep (default 2.0e-3)",
            "",
            "! * Poloidal field diffusion",
            f"dtminb = {1.0e-7}  ! Minimum timestep (default 1.0e-7)",
            f"dtmaxb = {2.0e-3}  ! Maximum timestep (default 2.0e-3)",
            "",
            "! * Heating and current drive",
            f"dticrf = {self.dtHeating_ms*1E-3} ! Timestep step for ICRF/TORIC (default 5.0e-3)",
            f"dtech  = {self.dtHeating_ms*1E-3} ! Timestep step for ECH (no default, it is needed)",
            f"dtlh   = {self.dtHeating_ms*1E-3} ! Timestep step for LH (default 5.0e-3)",
            f"dtbeam = {self.dtHeating_ms*1E-3} ! Timestep step for NBI (default 5.0e-3)",
            "",
            "! * Outputs Resolution",
            f"sedit  = {self.dtOut_ms*1E-3} ! Control of time resolution of scalar output",
            f"stedit = {self.dtOut_ms*1E-3} ! Control of time resolution of profile output",
            "",
            "! * Inputs Resolution",
            f"tgrid1 = {self.dtIn_ms*1E-3}  ! Control of time resolution of 1D input data",
            f"tgrid2 = {self.dtIn_ms*1E-3}  ! Control of time resolution of 2D input data",
            "",
            "!----- MPI Settings",
            "",
            f"nbi_pserve     = {self.pservers['nbi']} ! enable mpi for nubeam, nbi and fusion products",
            f"ntoric_pserve  = {self.pservers['toric']} ! enable mpi toric, icrf",
            f"nptr_pserve    = {self.pservers['ptr']} ! enable mpi ptransp (tglf only)",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addPB(self):
        lines = [
            "!=============================================================",
            "! Power balance",
            "!=============================================================",
            "",
            "!------ Electron Power Balance",
            "",
            "alph0i  = 0.6    ! Sets ion convection power loss coeff to 3/2",
            "",
            "!------ Ion Power Balance",
            "",
            "alph0e  = 0.6    ! Sets electron convection power loss coeff to 3/2",
            "",
            "nlti2   = T      ! Use Ti input data profiles to determine Xi",
            "nltipro = T      ! Use Ti input data directly, no predictive calculation",
            "",
            "nslvtxi = 2      ! 2: Ti data is average temperature (TIAV)",
            "",
            "tixlim0 = 0.0    ! min xi (r/a) of validity of the Ti data",
            "tixlim  = 1.0    ! max xi (r/a) of validity of the Ti data",
            "tidxsw  = 0.05   ! Continuous linear transition",
            "tifacx  = 1.0    ! edge Ti/Te factor",
            "",
            "!------ Ti not available",
            "",
            "giefac = 1.0     ! Ti = giefac * Te ",
            "fiefac = 1.0     ! Ti = (1-fiefac)*(giefac*Te)+fiefac*Ti(data)",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addSpecies(self):
        lines = [
            "!=============================================================",
            "! Plasma Species",
            "!=============================================================",
            "",
            "!----- Background Species",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

        if self.DTplasma:
            if self.AddHe4ifDT:
                lines = [
                    "ng       = 3       ! Number of INITIAL background species   ",
                    "ngmax   = 3  	    ! Maximum FUTURE background species (Pellets,NBI,recy)",
                    "",
                    "backz(1)  = 1.0   ! Charge of background species",
                    "aplasm(1) = 2.0   ! Atomic mass of background species",
                    "frac(1)   = 0.5   ! Background gas fractions",
                    "",
                    "backz(2)  = 1.0",
                    "aplasm(2) = 3.0 ",
                    "frac(2)   = 0.5 ",
                ]

            else:
                lines = [
                    "ng	      = 2         ! Number of INITIAL background species   ",
                    "ngmax   = 2  	      ! Maximum FUTURE background species (Pellets,NBI,recy)",
                    "",
                    "backz(1)   = 1.0    ! Charge of background species",
                    "aplasm(1)  = 2.0    ! Atomic mass of background species",
                    "frac(1)    = 0.5    ! Background gas fractions",
                    "",
                    "backz(2)   = 1.0",
                    "aplasm(2)  = 3.0",
                    "frac(2)    = 0.5",
                ]
        else:
            lines = [
                "ng        = 1      ! Number of INITIAL background species   ",
                "ngmax	   = 1      ! Maximum FUTURE background species (Pellets,NBI,recy)",
                "",
                "backz(1)   = 1.0 	! Charge of background species",
                "aplasm(1)  = 2.0 	! Atomic mass of background species",
                "frac(1)    = 1.0  ! Background gas fractions",
                "",
            ]

        self.contents += "\n".join(lines) + "\n"

        if self.DTplasma and self.AddHe4ifDT:
            lines = [
                "backz(3)   = 2.0 ",
                "aplasm(3)   = 4.0 ",
                "frac(3)   = 1.0E-10 ",
                "",
            ]

            self.contents += "\n".join(lines) + "\n"

        # -------
        lines = [
            "!----- Impurities",
            "",
            "!xzeffi   = 1.5  ! Zeff (if assumed constant in r and t)",
            "nlzfin    = F    ! Use 1D Zeff input",
            "nlzfi2    = T    ! Use 2D Zeff input (this is MITIM preferred way)",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

        for i in range(len(self.zlump)):
            lines = [
                f"xzimps({i+1}) = {self.zlump[i][0]} ! Charge of impurity species",
                f"aimps({i+1})  = {self.zlump[i][1]} ! Atomic mass of impurity species",
                f"densim({i+1}) = {self.zlump[i][2]} ! Impurity densities (n_imp/n_e)",
            ]

            self.contents += "\n".join(lines)

            # Possibility for densima
            if len(self.zlump[i]) > 3:
                lines = [
                    f"densima({i+1}) = {self.zlump[i][3]} ! Impurity densities (n_imp/n_e)",
                    "",
                ]
            else:
                lines = ["", ""]

            self.contents += "\n".join(lines)

            if self.coronal:
                self.contents += "\n".join([f"nadvsim({i + 1}) = 1"]) + "\n"

        if self.Minorities is not None and self.Pich is not None:
            lines = [
                "",
                "! ----- Minorities",
                "",
                f"xzmini  = {self.Minorities[0]:.2f} ! Charge of minority species (can be list)",
                f"amini   = {self.Minorities[1]:.2f} ! Atomic mass of minority species (can be list)",
                f"frmini  = {self.Minorities[2]} ! Minority density (n_min/n_e)",
                "rhminm  = 1.0E-12   ! Minimum density of a gas present (fraction of ne)",
                "",
            ]

            self.contents += "\n".join(lines) + "\n"

    def addParticleBalance(self):

        AddHe4 = self.AddHe4ifDT and self.DTplasma

        lines = [
            "!=============================================================",
            "! Particle Balance",
            "!=============================================================",
            "",
            "!-----  Particle balance model",
            "",
            "ndefine(1) = 0  ! Density profile (0=nmodel, 1= D-V from UF, 2= n from UF)",
            "ndefine(2) = 0",
            "ndefine(3)	  = 1" if AddHe4 else "",
            "",
            "nmodel = 1  ! Mixing Model: 1 = same shape, 2 = same Vr, 4 = mixed model",
            "ndiffi = 5  ! Input Diffusivity (for nmodel=4)",
            "",
            "!----- Particle Source: Recycling model",
            "",
            "nrcyopt = 0 ! Recycling model (0=by species,1=mixed model, 2=)",
            "nltaup  = F ! (T) UFile, (F) Namelist tauph taupo",
            "",
            "taupmn   = 0.001     ! Minimum allowed confinement time (s)",
            f"tauph(1) = {self.taupD}   ! Main (hydrogenic) species particle confinement time",
            f"tauph(2) = {self.taupD}",
            f"tauph(3) = {self.taupD}",
            f"taupo    = {self.taupZ}   ! Impurity particle confinement time",
            f"taumin   = {self.taupmin} ! Minority confinement time",
            "",
            "nlrcyc     = F",
            "nlrcyx     = F     ! Enforce for impurities same taup as main ions",
            '!rfrac(1)	 = 0.5   ! Recyling fractions (fraction of "limiter outgas" flux)',
            "!rfrac(2)	 = 0.5",
            "!rfrac(3)	 = 0.0",
            '!recych(1) = 0.5   ! Fraction of ion outflux of species that is "promptly" recycled',
            "!recych(2) = 0.5",
            "!recych(3) = 0.0",
            "",
            "!----- Particle Source: Gas Flow",
            "",
            "!gfrac(1) = 1.0 ! gas flow ratio for each species",
            "!gfrac(2) = 0.5",
            "!gfrac(3) = 0.5",
            "",
            "!----- Neutrals Transport",
            "",
            "nsomod          = 1   ! FRANTIC analytic neutral transport model",
            "!nldbg0          = T   ! More FRANTIC outputs",
            f"nzones_frantic  = {self.nzones_frantic} ! Number of radial zones in FRANTIC model",
            "mod0ed          = 1   ! 1 = Use time T0 or TIEDGE",
            "ti0frc          = 0.0333 ! If mod0ed=2, Ti(0) multiplier" "",
            "fh0esc          = 0.3 ! Fraction of escaping neutrals to return as warm (rest as cold)",
            "nlreco          = T   ! Include recombination volume neutral source ",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

        num_species = 2 + int(AddHe4)

        # Neutrals tempereatures
        lines, cont = [], 0
        for i in range(num_species):
            cont += 1
            lines.append(f"e0in({cont}) = 10.0 ! T0 for warm, specie {i}")
            cont += 1
            lines.append(f"e0in({cont}) =  3.0 ! T0 for vol sce, specie {i}")
            cont += 1
            lines.append(f"e0in({cont}) =  3.0 ! T0 for cold, specie {i}")
            lines.append("")

        self.contents += "\n".join(lines)

    def addRadiation(self):

        lines = [
            "",
            "!=============================================================",
            "! Radiation Model (Bremsstrahlung, Line and Cyclotron)",
            "!=============================================================",
            "",
            f"nprad    = {2 if self.predictRad else 0} ! Radiative power calculation (1 = U-F or Theory, 2 = Always Theory)",
            "nlrad_br = T     ! Calculate BR by impurities (Total is always calculated)",
            "nlrad_li = T     ! Calculate impurity line radiation ",
            "nlrad_cy = T     ! Calculate cyclotron radiation",
            "vref_cy  = 0.9   ! Wall reflectivity used in cyclotron radiation formula",
            "prfac    = 0.0   ! For nprad=-1, corrects BOL input",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addNCLASS(self):
        lines = [
            "!===========================================================",
            "! NCLASS neoclassical Model",
            "!===========================================================",
            "",
            "ncmodel  = 2   ! 2: Houlbergs NCLASS 2004",
            "ncmulti  = 2   ! 2= Always use multiple impurities when present",
            "nltinc   = T   ! Use average Ti for calculations",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addMHDandCurrentDiffusion(self):
        if self.isolver:
            levgeo = 12
            nlqlim0 = "False"
        else:
            levgeo = 11
            nlqlim0 = "True"

        lines = [
            "!=============================================================",
            "! MHD equilibrium and Current Diffusion",
            "!=============================================================",
            "",
            "!----- MHD Geometry",
            "",
            f"levgeo	       = {levgeo}    ! 11 = TEQ model from LLNL, 12 = ISOLVER",
            "",
            f"nteq_mode     = {self.nteq_mode}   ! Free param to be matched (5= Q,F_edge & loop for Ip)",
            "nteq_stretch  = 0   ! Radial grid distribution",
            f"nteq_nrho     = {self.gridsMHD[0]}   ! Radial points in TEQ (default 71)",
            f"nteq_ntheta   = {self.gridsMHD[1]}   ! Poloidal points in TEQ (default 127)",
            f"teq_smooth    = {self.smoothMHD[0]}  ! Smoothing half-width (negative means -val/nzones) (default -1.5)",
            f"teq_axsmooth  = {self.smoothMHD[1]} ! Smoothing half-width near-axis as val*min(1,2-x/val) (default 0.05)",
            "softteq       = 0.3   ! Maximum allowed GS average error",
            "",
            "!------ Poloidal field (current) diffusion",
            "",
            "nlmdif	  = T    ! Solve poloidal diffusion equation ",
            "nlpcur	  = T    ! Match total plasma current",
            "nlvsur	  = F    ! Match surface voltage",
            "",
            f"nlbccw = {self.B_ccw}    ! Is Bt counter-clockwise (CCW) seen from above?",
            f"nljccw = {self.Ip_ccw}    ! Is Ip counter-clockwise (CCW) seen from above?",
            "",
            "!----- Initialization of EM fields",
            "nefld     = 7    ! 7 = q-profile from QPR, 3 = parametrized V(x) with qefld, rqefld",
            "qefld     = 0.0  ! If nefld=3, value of q to match",
            "rqefld    = 0.0  ! If nefld=3, location to match",
            "xpefld    = 2.0  ! If nefld=3, quartic correction",
            "",
            "!------ q-profile",
            "",
            f"nlqlim0  = {nlqlim0}  ! Place a limit on the max q value possible",
            "qlim0	  = 5.0  ! Set the max possible q value",
            "nlqdata  = F    ! Use input q data (cannot be set with nlmdif=T as well)",
            "",
            "nqmoda(1) = 4     ! 4: nlqdata=T, 1: nlmdif=T",
            "nqmodb(1) = 1",
            f"tqmoda(1) = {self.time_current_diffusion} ! Time to change to next stage",
            "",
            "nqmoda(2) = 1     ! 4: nlqdata=T, 1: nlmdif=T",
            "nqmodb(2) = 1",
            f"tqmoda(2) = {self.ftime+100.0} ! Time to change to next stage",
            "",
            "tauqmod   = 1.0E-2 ! Transition window ",
            "",
            "!------ Resitivity Model",
            "",
            "nmodpoh  = 1		! 1: Standard Poh from EM equations, 2: Poh = eta*j^2",
            "",
            "nletaw	  	     = F    ! Use NCLASS resistivity",
            "nlspiz	  	     = F    ! Use Spitzer resistivity ",
            "nlrestsc             = F    ! Use neoclass as in TSC",
            "nlres_sau 	     = T    ! Use Sauter",
            "",
            "!sauter_nc_adjust(1) = 1.0  ! Factor that multiplies Sauter resistivity",
            "xl1nceta  	     = 0.0  ! Do not trust inside this r/a (extrapolate to 0)",
            "nlrsq1	  	     = F  	! Flatten resistivity profile inside q=1",
            "nlsaws	  	     = F  	! Flatten resistivity profile inside q=qsaw",
            "qsaw	  	     = 1.0  ! Specification of q to flatten resistivity inside",
            "xsaw_eta  	     = 0.0  ! Flattening factor",
            "nlresis_flatten      = F  	 ! Flatten within xresis_flatten",
            "xresis_flatten       = 0.0  ! Flattening factor",
            "",
            "bpli2max  = 20.0 ! The maximum value of Lambda = li/2 + Beta_p allowed",
            "",
            "!=============================================================",
            "! Current drive",
            "!=============================================================",
            "",
            "! ---- Current Drive",
            "",
            "nmcurb   = 4    ! Beam current drive model (0 = none, 4 = most recent)",
            "nllh   = F    ! Include LH currents",
            "",
            "!=============================================================",
            "! Boostrap current model",
            "!=============================================================",
            "",
            "nlboot	    = T     ! Use bootstrap currents in poloidal field eqs",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

        if self.useBootstrapSmooth is None:
            lines = [
                "",
                "nlbootw     = F    ! Use NCLASS bootstrap current",
                "nlboot_sau  = F    ! Use Sauter bootstrap current",
                "nlboothager = T    ! Use Hager bootstrap current (modification to Sauter)",
                "nlbootneo   = F    ! Use NEO bootstrap current",
                "",
                "xbstrap     = 1.0   ! Anomaly factor",
                "!xl1ncjbs   = 0.3   ! Do not trust inside this r/a (extrapolate to 0)",
                "njsmboot    = 0    ! JB moothing Number of zones for sliding triangular weighted average",
                "",
            ]

        else:
            lines = [
                "",
                "nlbootw	 = T    ! Use NCLASS bootstrap current",
                "nlboot_sau  = F    ! Use Sauter bootstrap current",
                "nlboothager = F    ! Use Hager bootstrap current (modification to Sauter)",
                "nlbootneo   = F    ! Use NEO bootstrap current",
                "",
                "xbstrap    = 1.0   ! Anomaly factor",
                f"xl1ncjbs   = {self.useBootstrapSmooth}   ! Do not trust inside this r/a (extrapolate to 0)",
                "njsmboot    = 0    ! JB moothing Number of zones for sliding triangular weighted average",
                "",
            ]

        self.contents += "\n".join(lines) + "\n"

    def addISOLVER(self,defineISOLVER_method):


        ISOLVERdiffusion = "T"

        isolver_file, pfcs = defineISOLVER_method()

        lines = [
            "!=============================================================",
            "! ISOLVER",
            "!=============================================================",
            "",
            "nlpfc_circuit = T 		! PFC currents in UFILES are circuit currents",
        ]

        self.contents += "\n".join(lines) + "\n"

        linesISOLVER = []
        for cont, i in enumerate(pfcs):
            linesISOLVER.append(f"pfc_names({cont+1}) = {i}\n")
            linesISOLVER.append(f"pfc_sigma({cont+1}) = {pfcs[i][0]}\n")
            linesISOLVER.append(f"pfc_cur({cont+1})   = {pfcs[i][1]}\n")
            linesISOLVER.append("\n")

        self.contents += "\n".join(lines) + "\n"

        lines = [
            "",
            "! ---- Flux diffusion",
            f"nlisodif = {ISOLVERdiffusion} 			! Use ISOLVER flux diffusion instead of nlmdif",
            "!niso_ndif = 			 ! Number of internal flux diffusion points",
            "niso_difaxis = 1		! Distribution of internal flux diffusion points (1=uniform)",
            "niso_qlim0 = 0 		! Way to limit on axis q (0=no limit, 2=force, 3=current drive)",
            "nisorot = -1 			! Rotation treatment in GS (-1=no rotation)",
            "nisorot_maximp = 3     ! Max number charge state for impurity to enter as rotating",
            "xisorot_scale = 1.0 	! Rotation scale factor",
            "",
            "! ---- Machine and equilibrium",
            f'iso_config = "{isolver_file}"		! Tokamak configuration',
            "neq_ilim = 0			! Limiter info (0= try ufile first, then internal)",
            'probe_names = "@all"   ! List of probes',
            "neq_xguess = 2 		! Guess x-points (2-both x-points)",
            "!pass_names = '@pass'",
            "",
            "! ---- Solution Method",
            "niso_mode = 102		! Use in solution: 102 (default) - p, q; 101 - p, <J*B>",
            "niso_qedge = 2			! How to handle matching q if niso=102 (2=Mod to match Ip)",
            "eq_x_weight = 1.0		! Weight for x-point location matching",
            "eq_ref_weight = 1.0	! Weight for prescribed boundary matching",
            "eq_cur_weight = 1.0	! Weight for experimental currents matching",
            "",
            "! ---- Convergence and errors",
            "niso_maxiter = 250 	! Number of iterations with fixed parameters (then, 8 tries changing stuff)",
            "niso_sawtooth_mode = 0 ! In sawtooth, keep continuity of coil/vessel (0) poloidal flux or (1) current",
            "xiso_omega = 0.5	    ! Relaxation between iterations (0.5 for q-mode, 0.2 for <J*B>-mode",
            "xiso_omega_dif = 0.6   ! Relaxation for flux diffusion boundary condition",
            "xiso_error = 1.e-6		! Convergence error criteria",
            "xiso_warn_ratio = 50.0	! Accept error relaxed by this factor if struggling",
            "xiso_last_ratio = -1	! Ratio applied to last retry",
            "xbdychk = 0.05 		! Maximum allowable difference between prescribed and LCFS at midplane",
            "!dtfirstg = 			 ! Time step for first flux diffusion (hardest)",
            "!niso_njpre = 4		 ! Adds a 4 point interation filter to Jphi",
            "",
            "! ---- Resolution",
            "niso_nrho = 100		! Radial surfaces to use in ISOLVER",
            "niso_ntheta = 121 		! Number of poloidal rays to use in ISOLVER",
            "niso_axis = 2 			! Distribution of radial surfaces (2: sin(pi/2*uniform))",
            "niso_nrzgrid = 129,129 ! Points in R,Z direction",
            "!xiso_smooth = 		 ! Smoothing of profiles prior to ISOLVER",
            "xiso_pcurq_smooth = -2 ! Smoothing of current in q mode",
            "eq_crat = 0.08			! Minimum curvature of boundary derived from psiRZ solution",
            "eq_bdyoff = 0.007		! Boundary offset to meet eq_crat requirement",
            "",
            "! ---- Other",
            "xeq_p_edge = 0			 ! 0 = force P_edge=0, 1 = shift down first",
            "",
            "! ---- Circuit mode",
            "!fb_names = 			 ! Feedback controllers (this turns on the CIRCUIT mode)",
            "",
            "!fb_feedback = 		 ! ",
            "!fb_feedomega = 		 ! ",
            "!fb_lim_feedback = 		 ! ",
            "!tinit_precoil = 		 ! ",
            "!xiso_relbal = 		 ! ",
            "!xiso_relbal_dt = 		 ! ",
            "!neq_redo = 		 ! ",
            "!niso0_circ = 		 ! ",
            "!niso_jit_psifix = 		 ! ",
            "!pfc_jit_voltage = 		 ! ",
            "!npfc_jit = 			 ! Way to use just-in-time for next step currents",
            "!xiso_jit_relax =",
            "!pfc__jit_sigma = ",
            "!xiso_def_jit_sigma = ",
            "!xiso_jit_sigma_pcur = ",
            "!niso_def_drive_type = -2 ! -2: drive with current sources",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addSawtooth(self):

        lines = [
            "!===============================================================",
            "!Sawtooth Modeling",
            "!===============================================================",
            "",
            "!---- Sawtooth crash/redistribution model",
            "",
            "nlsaw   = T       ! Sawooth for bulk plasma",
            "nlsawic = T       ! Sawooth for ICRF minority fast ions",
            "nlsawe  = T       ! Sawooth for electrons",
            "nlsawi  = T       ! Sawooth for ions",
            "nlsawb  = T       ! Sawooth for fast ions",
            "nlsawf  = T       ! Sawooth for fusion products",
            "nlsawd  = F       ! For clean sawtooth crashes (U-Files smooth through sawtooth)",
            "dtsawd  = 0.002   ! Interval around sawtooth to extrapolate input data",
            "",
            "nmix_kdsaw     = 4       ! 1= Std Kadomtsev, 3(4) = Porcelli 2 islands with 2(1) energy mixing region",
            f"fporcelli	   = {1 - self.ReconnectionFraction} 	  ! Porcelli island width fraction (1 is q=1 to axis, so it is 1-Porcelli definition)",
            "xswid1         = 0.0     ! Conserve total ion energy (0.0 =yes) in Kad or Por",
            "xswid2 	    = 0.0     ! Conserve current & q profiles (1.0=yes; 0.0=no, mix them)",
            "xswidq         = 0.05    ! Finite thickness in x of current sheet width to avoid solver crash ",
            "xswfrac_dub_te = 1.0 	  ! Fraction of change in Bp energy to assign to electrons",
            "",
            "!---- Sawtooth triggering model",
            "",
            "nlsaw_trigger    = T 		! trigger sawtooth crashes using model",
            "nlsaw_diagnostic = F		! diagnose sawtooth conditions but do not crash",
            "model_sawtrigger = 2      ! 0=specify below,1=Park-monticello, 2=Porcelli",
            f"t_sawtooth_on    = {self.timeSaw}	    ! Parameter changed ",
            "t_sawtooth_off   = 1.0E3  ! Last sawtooth crash time",
            "sawtooth_period  = 1.0    ! Sawtooth period in seconds (if model_sawtrigger = 0)",
            "",
            "l_sawtooth(1)    = -1     ! 0 = Do not crash if multiple q=1",
            "xi_sawtooth_min  = 0.0    ! Smallest radius that trigger q=1 sawtooth",
            "c_sawtooth(2)    = 0.1    ! Impose a minimum sawtooth period (as fraction of Park-Monticello model)",
            "c_sawtooth(20)   = 1.0    ! Coefficient for d beta_fast / d r",
            "",
            "l_sawtooth(32)   = 1   	 ! 1 = Use c_sawtooth(25:29) from namelist",
            "c_sawtooth(25)   = 0.1    ! shear_minimum (default 0.1)",
            f"c_sawtooth(26)   = {self.coeffsSaw[0]}    ! ctrho in Eq 15a             (default porcelli = 1.0)",
            f"c_sawtooth(27)   = {self.coeffsSaw[1]}    ! ctstar in Eq 15b            (default porcelli = 3.0)",
            f"c_sawtooth(28)   = {self.coeffsSaw[2]}    ! cwrat  in Eq B.8 for dWfast (default porcelli = 1.0)",
            f"c_sawtooth(29)   = {self.coeffsSaw[3]}    ! chfast in Eq 13             (default porcelli = 0.4)",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addFusionProducts(self):

        useNUBEAM = self.DTplasma and self.useNUBEAMforAlphas

        lines = [
            "!==============================================================================",
            "! Fusion products, reactions and slowing down",
            "!==============================================================================",
            "",
            "!----- General Model",
            "",
            f"nalpha  				   = {int(not useNUBEAM)}     ! Fusion products model (0=MC for alphas, 1= fast model)",
            f"nptclf  				   = {int(self.MCparticles)} ! Number of monte carlo particles for fusion product calcs",
            "nlfatom 				   = T     ! Include atomic physics effects on products (e.g. CX)",
            "nl_ignore_mini_conflicts    = T		! Ignore issue with He3 product coinciden with ICRF minority",
            "",
            "!----- Reactions",
            "",
            f"nlfhe4  = {useNUBEAM}       ! Turn on MC slowing-down of He4 from D+T reactions",
            "plfhe4  = 1.0E2   ! Source power threshold to run MC (in W)",
            "",
            "nlfst   = F       ! Turn on MC slowing-down of T from D+D reactions (D+D=T+p)",
            "plfst   = 1.0E0	! Source power threshold to run MC NLFST",
            "",
            "nlfsp   = F       ! Turn on MC slowing-down of protons from D+D reactions (D+D=T+p)",
            "plfsp   = 1.0E0	! Source power threshold to run MC for NLFST",
            "",
            "nlfhe3  = F       ! Turn on MC slowing-down of He3 from D+D reactions (D+D=He3+n)",
            "plfhe3  = 1.0E0   ! Source power threshold to run MC for NLFHE3",
            "",
            "!----- From U-File",
            "",
            "nlusfa   = F	  ! He4",
            "nlusft   = F	  ! T from DD",
            "nlusfp   = F	  ! p from DD",
            "nlusf3   = F	  ! He3 from DD",
            "",
            "!----- Smoothing",
            "",
            "dxbsmoo = 0.05	 ! Profile smoothing half-width",
            "",
            "!----- Anomalous transport",
            "",
            "nmdifb = 0 	! Anomalous diffusion (0=none, 3=Ufiles D2F & V2F)" "",
            "nrip   = 0 	! Ripple loss model (1=old, 2=less old)",
            "nlfbon = F 	! Fishbone loss model",
            "!nlfi_mcrf = T   ! Account for wave field caused by ICRF antenna",
            "",
            "!----- Orbit physics",
            "",
            "nlbflr    = T 	! Plasma interacts with T=gyro, F=guiding center",
            "nlbgflr    = F 	! Model for gyro v.s. guiding displacement",
            "",
            "!----- MonteCarlo controls",
            "",
            "!nlseed  =  		! Random number seed for MC",
            "!goocon = 10 		! Numeric Goosing",
            "!dtn_nbi_acc = 1.0e-3 ! Orbit timestep control",
            "!nptcls = 1000 	! Constant census number of MC ions to retain",
            "!nlfbm = T 		! Calculate beam distribution function",
            "!ebdmax = 1e9 		 Maximum energy tracked in distribution function",
            "!wghta = 1.0 		! MC profile statistics control",
            "!nznbme = 800 ! Number of energy zones in beam distribution function ",
            "!ebdfac = 6.0D0 ! Max energy factor",
            "!nbbcal = 1100000000 ! Number of beam D-D collisions to calculate",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addRotation(self):

        lines = [
            "!===========================================================",
            "! Momentum Transport -> Rotation",
            "!===========================================================",
            "",
            f"nlvphi	   = {'T' if self.UFrotation else 'F'}    ! Rotation Moldeing using U-File",
            "nlomgvtr  = F 	    ! T: Impurity rotation is provided, F: Bulk plasma rotation is provided",
            "ngvtor    = 0 	    ! 0: Toroidal rotation species given by nvtor_z, xvtor_a",
            f"nvtor_z   = {int(self.rotating_impurity[0])}	! Charge of toroidal rotation species",
            f"xvtor_a   = {self.rotating_impurity[1]}	! Mass of toroidal rotation species",
            "xl1ncvph  = 0.10   ! Minimum r/a",
            "xl2ncvph  = 0.85   ! Maximum r/a",
            "",
            "nlivpo    = F	    ! Radial electrostatic potential and field from U-File",
            "nlvwnc    = F      ! Compute NCLASS radial electrostatic potential profile",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addVessel(self, defineTRANSPnmlStructures_method):

        limiters, VVmoms = defineTRANSPnmlStructures_method()

        if limiters is not None and self.LimitersInNML:
            alnlmr = str(limiters[0][0])
            alnlmy = str(limiters[0][1])
            alnlmt = str(limiters[0][2])
            for i in range(len(limiters) - 1):
                alnlmr += f", {str(limiters[i + 1][0])}"
                alnlmy += f", {str(limiters[i + 1][1])}"
                alnlmt += f", {str(limiters[i + 1][2])}"

            lines = [
                "! -----Limiters ",
                "",
                f"nlinlm        = {len(limiters)}",
                f"alnlmr 	   = {alnlmr}",
                f"alnlmy 	   = {alnlmy}",
                f"alnlmt 	   = {alnlmt}",
                "",
            ]

            self.contents += "\n".join(lines) + "\n"

        if VVmoms is not None:
            lines = [
                "! ----- Vaccum Vessel (reflecting boundary for the wave field)",
                "",
                f"VVRmom = {', '.join([f'{x:.8e}' for x in np.transpose(VVmoms)[0]])}",
                f"VVZmom = {', '.join([f'{x:.8e}' for x in np.transpose(VVmoms)[1]])}",
                "",
            ]

            self.contents += "\n".join(lines) + "\n"

    def addUFILES(self):
        lines = [
            "\n!=========================================================================",
            "! Included Input Data Files (Files, coordinate and asymmetry)",
            "!=========================================================================",
            "!Note: NRIxxx: -5=sqrt tor flux (rho?), -4=r/a",
            "",
            "levtrk	 = 2   ! Limiter locations from lim ufile",
            "lfixup   = 2   ! Correct U-Files units and axis (=2 according to labels)",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

        if self.Pich and 'rfp' not in self.Ufiles:
            lines = [
                'preRFP	 = "PRF"',
                'extRFP	 = "RFP"',
            ]
            self.contents += "\n".join(lines) + "\n"

        if self.Pech:
            lines = [
                'preECP	 = "PRF"',
                'extECP	 = "ECH"',
                'preECA	 = "PRF"',
                'extECA	 = "THE"',
                'preECB	 = "PRF"',
                'extECB	 = "PHI"',
            ]
            self.contents += "\n".join(lines) + "\n"

        if self.Pnbi:
            lines = [
                'preNB2	 = "PRF"',
                'extNB2	 = "NB2"',
            ]
            self.contents += "\n".join(lines) + "\n"

        for i in self.Ufiles:
            lines = [
                f'pre{i}	 = "PRF" ',
                f'ext{i}	 = "{self.Ufiles[i][0]}"',
                "",
            ]
            self.contents += "\n".join(lines)

            if self.Ufiles[i][1] is not None:
                lines = [f"nri{i}	 = {self.Ufiles[i][1]}", f"nsy{i}	 = 0", ""]
                self.contents += "\n".join(lines) + "\n"

    def addICRF(self, ICRFantennas_method=None):

        lines = [
            "!===========================================================",
            "! Ion Cyclotron Resonance (ICRF)",
            "!===========================================================",
            "",
            "! ----- Main Controls",
            "",
            "nlicrf = T   ! Enable ICRF simulation",
            "nicrf  = 8   ! 8: Modern TORIC",
            "",
            "! ----- Resolution",
            "",
            f"nichchi    = {self.toric_ntheta}   ! Number of poloidal grid points, powers of 2 (default 64)",
            f"nmdtoric   = {int(self.toric_ntheta/2-1)}   ! Number of poloidal modes (recommended: NichChi/2-1)",
            f"nichpsi    = {self.toric_nrho}   ! Number of radial grid points (default 128)",
            "",
            "! ----- Model options",
            "",
            "ANTLCTR    = 1.6   ! Effective antenna propagation constant",
            "NFLRTR     = 1	    ! Include ion FLR contribution",
            "NBPOLTR    = 1     ! Include poloidal magnetic field",
            "NQTORTR    = 1     ! Include toroidal broadening of the plasma dispersion",
            "NCOLLTR    = 0     ! Include collisional contribution to  plasma dispersion",
            "ENHCOLTR   = 1.0   ! Electron collision enhancement factor with NCOLL",
            "ALFVNTR(1) = 0.0   ! Included (= 1.0) or ignored (= 0.0) collisional broadening of Alfven res",
            "ALFVNTR(2) = 0.1   ! Enhancement factor (~ 0.1)",
            "ALFVNTR(3) = 3.0   ! Value of ABS((n//^2 - S)/R) below which damping added",
            "ALFVNTR(4) = 5.0   ! Value of ABS(w/(k//*v_te)) below which damping calc",
            "",
        ]
        self.contents += "\n".join(lines) + "\n"

        if ICRFantennas_method is not None:
            self.contents += ICRFantennas_method() + "\n"

    def addECRF(self, ECRFgyrotrons_method=None, TORAY=False):
        lines = [
            "!===========================================================",
            "! Electron Cyclotron Resonance (ECRF)",
            "!===========================================================",
            "",
        ]
        self.contents += "\n".join(lines) + "\n"

        if TORAY:
            lines = [
                "",
                "! ~~~~~~~~~~ TORAY parameters",
                "",
                "NLTORAY = T",
                "",
                "NPROFTOR = 41",
                "DTTOR = 0.05",
                "EFFECH   = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
                "NRAYECH  = 30, 30, 30, 30, 30, 30, 30, 30",
                "BSRATECH = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
                "BHALFECH = 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7",
                "NDAMPECH = 2, 2, 2, 2, 2, 2, 2, 2",
                "",
                "NGAFITTOR = 1",
                "SMAXTOR = 330.0",
                "DSTOR = 0.5",
                "DSMINTOR = 0.25",
                "FDOUTTOR = 5.0",
                "POWINCTOR = 1.0e13",
                "RELERRTOR = 5.0e-5",
                "ABSERRTOR = 5.0e-5",
                "NHARMTOR = 2",
                "MN0TOR = 32",
                "NGZNTOR = 4",
                "MRAYTOR = 1, 5, 12, 12, 20, 24, 26",
                "CRTOR = 0.0, 0.1, 0.05, -0.05, 0.05, -0.025, 0.025",
                "MODELCTOR = 4",
                "PWRFMNTOR = 0.001",
                "NLCQLDAT = .F.  ",
                "",
            ]
        else:
            lines = [
                "",
                "! ~~~~~~~~~~ TORBEAM Parameters",
                "",
                "nltorbeam 		   = T",
                "nltorbeam_verbose    = T ! Generate extra files",
                "",
                "NTOB_NPROFVW    = 25",
                "NTOB_MMAX 	   = 150",
                "TOB_XZEFF 	   = -1",
                "",
            ]

        self.contents += "\n".join(lines) + "\n"

        if ECRFgyrotrons_method is not None:
            self.contents += ECRFgyrotrons_method() + "\n"

    def addNBI(self, NBIbeams_method=None):
        lines = [
            "!===============================================",
            "!           NEUTRAL BEAMS                      !",
            "!===============================================",
            "",
            "nlbeam   = T 		! NUBEAM model on",
            "nlbfpp   = F 		! FPP model on",
            "",
            "lev_nbidep = 2 	! Ground state dep model (2=ADAS atomic physics)",
            "nsigexc     = 1		! Simple excited states deposition model",
            "",
            "nlbdat = T 		! Time-dependent UFILE with Power",
            "",
            "! ~~ Model settings",
            "nlbflr   = T 		! Force plasma interactions on the gyro",
            "nlbeamcx = T",
            "nptcls  = 50000	! Number of Monte Carlo ions",
            "dxbsmoo = 0.05		! Radial smoothing half-width (in r/a space)",
            "ndep0  = 500		! Min. number of deposition tracks to trace per step",
            "goocon = 10.		! Numeric goosing (10 is default precision)",
            "goomax = 500.",
            "ebdmax = 150E+03 	! Max energy tracked in distribution function",
            "nlfbon = F 		! Fishbone losses",
            "nlbout = F",
            "",
        ]
        self.contents += "\n".join(lines) + "\n"

        if NBIbeams_method is not None:
            self.contents += NBIbeams_method() + "\n"

    def addPTSOLVER(self):

        self.addPedestal()
        self.addPredictive()
        self.addGLF23()
        self.addTGLF()

    def addPredictive(self):
        lines = [
            "!==================================================================",
            "! Predictive TRANSP",
            "!==================================================================",
            "",
            "lpredictive_mode = 3   ! Use PT_SOLVER",
            "pt_template      = 'tmp_folder/ptsolver_namelist.dat'",
            "",
            "!------ Equation selection",
            "",
            "lpredict_te      = 0    ! Predict Te",
            "lpredict_ti      = 0    ! Predict Ti",
            "lpredict_pphi    = 0    ! Predict Rotation",
            "lpredict_ne      = 0    ! Predict ne",
            "lpredict_nmain   = 0    ! Predict n_main",
            "lpredict_nimp    = 0    ! Predict n_imp",
            "",
            "!------ Region boundaries (in rho)",
            "",
            f"ximin_conf = {self.xminTrick:.3f}	 ! Max rho for AXIAL",
            f"ximax_conf = {self.xmaxTrick:.3f}   ! Min rho for EDGE (after this, EDGE until X*BOUND)",
            "",
            "!------ Boundary conditions (rho > x*bound -> experimental)",
            "",
            f"xibound    = {self.xbounds[0]:.3f}	 ! Te, Ti",
            f"xnbound    = {self.xbounds[1]:.3f}	 ! ne, ni",
            f"xphibound  = {self.xbounds[2]:.3f}  ! rotation",
            "",
            '!------ "USER" transport model',
            "",
            "nchan     =      1,      1,    0,    0,    0,    0      ! Channels to implement axial values (Te, Ti, Phi, ni, nimp, ne)",
            "user_chi0 =  0.1e4,  0.1e4,  0.0,  0.0,  0.0,  0.0      ! Diffusivities on axis (cm^2/s)",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

        lines = [
            "&pt_controls",
            " ",
            " !------ Solver Options",
            " ",
            " newton_iterates       = 1000 ! Number of newton iterations (default = 10)",
            " theta_implicit        = -1   ! Implicitness param in time diff (negative = exponential average method)",
            " !xrelfac              = 0.1  ! Relaxation factor to mix diffusion coefficients",
            " ",
            " !------ Residuals",
            " ",
            " pt_residual%res_te       = 1D-3",
            " pt_residual%res_ti       = 1D-3",
            " pt_residual%res_ne       = 1D-3",
            " pt_residual%res_nimp     = 1D-3",
            " pt_residual%res_nmain    = 1D-3",
            " pt_residual%res_pphi     = 1D-3",
            " ",
            " !------ Numerical diffusivities (factors and minimum values for the Peclet numbers)"
            " ",
            " pt_num_diffusivity%pt_chie_fact   = 500D0",
            " pt_num_diffusivity%pt_chie_min    = 100D0",
            " pt_num_diffusivity%pt_chii_fact   = 100D0",
            " pt_num_diffusivity%pt_chii_min    = 100D0",
            " pt_num_diffusivity%pt_chine_fact  = 200D0",
            " pt_num_diffusivity%pt_chine_min   = 50D0",
            " pt_num_diffusivity%pt_chiphi_fact = 200D0",
            " pt_num_diffusivity%pt_chiphi_min  = 50D0",
            " ",
            " !------ ExB flow shear model: 1 (NCEXB), 2 (DMEXB), 3 (TGYROEXB), 4 (TRANSPEXB)",
            " ",
            " pt_axial%exb%version        = 3",
            " pt_confinement%exb%version  = 3",
            " pt_edge%exb%version         = 3",
            " ",
            " !------ Transport models for each region",
            " ",
            " ! ~~~ GLF23",
            " pt_axial%glf23%active          = T",
            " pt_axial%glf23%xanom           = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  ! Te, Ti, Pphi, Ni, Nz, Ne",
            " ",
            " pt_confinement%glf23%active    = T",
            " pt_confinement%glf23%xanom     = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            " ",
            " pt_edge%glf23%active           = T",
            " pt_edge%glf23%xanom            = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            "",
            " ! ~~~ TGLF",
            " pt_axial%tglf%active           = F",
            " pt_axial%tglf%xanom            = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  ! Te, Ti, Pphi, Ni, Nz, Ne",
            " ",
            " pt_confinement%tglf%active     = F",
            " pt_confinement%tglf%xanom      = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            " ",
            " pt_edge%tglf%active            = F",
            " pt_edge%tglf%xanom             = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            "",
            " ! ~~~ User",
            " pt_axial%user%active           = F",
            " pt_axial%user%xanom            = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  ! Te, Ti, Pphi, Ni, Nz, Ne",
            " ",
            " pt_confinement%user%active     = F",
            " pt_confinement%user%xanom      = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            " ",
            " pt_edge%user%active            = F",
            " pt_edge%user%xanom             = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            " ",
            " ! ~~~ Neoclassical: Chang-Hilton model",
            " pt_axial%neoch%active          = T",
            " pt_axial%neoch%xanom           = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            " ",
            " pt_confinement%neoch%active    = T",
            " pt_confinement%neoch%xanom     = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            " ",
            " pt_edge%neoch%active           = T",
            " pt_edge%neoch%xanom            = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            " ",
            " !------ Sawtooth model (for interchange instability)",
            " ",
            " !pt_sawtooth%active         = T                               ! Activate PTSAW",
            " !pt_sawtooth%xanom          = 1e3, 1e3, 1e3, 1e3, 1e3, 1e3    ! Transport enhancement factor (Te, Ti, Pphi, Ni, Nz, Ne)",
            " !pt_sawtooth%model          = 1                               ! 1 (q<1), 2 (interchange; TGLF only), 3 (both; TGLF only), 4 (USER)",
            " !pt_sawtooth%xsaw_bound     =                                 ! Sawtooth radius (if model = 4)",
            " ",
        ]

        # lines.append('/') # I do it later, in defineRunParams

        self.contents_ptr_ptsolver = "\n".join(lines) + "\n"

    def addPedestal(self):
        lines = [
            "!===============================================================",
            "! Pedestal and Edge Model -> Boundary condition for X > XBOUND",
            "!===============================================================",
            "",
            "!------ General: Te, Ti, T0, Rotation",
            "",
            "modeedg  = 3        ! 2 = Use TEEDGE, 3 = Exp Te, 5 = NTCC ",
            f"teedge   = {self.Te_edge}     ! Specify Te (eV) if MODEEDG = 2",
            "teped    = 3050     ! Electron pedestal temperature in eV",
            "tepedw   = -0.0591  ! Electron pedestal width in cm(+) or x(-)",
            "",
            "modiedg  = 4        ! 1 = Same as neutrals, 2 = TIEDGE, 3 = Exp Te, 4 = Exp Ti, 5 = NTC ",
            f"tiedge   = {self.Ti_edge}     ! Specify Ti (eV) if MODIEDG = 2 and for FRANTIC if MOD0ED=1",
            "tiped    = 3050     ! Ion pedestal temperature in eV",
            "tipedw   = -0.0643  ! Ion pedestal width in cm(+) or x(-)",
            "",
            "modnedg  = 3",
            "xneped   = 3.65E14  ! Electron pedestal density in cm^-3",
            "xnepedw  = -0.0591  ! Electron density pedestal width in cm(+) or x(-)",
            "",
            "modomedg = 3        ! Rotation, 2 = use OMEDG (constant), 3 = use OMEGA input data",
            "omedge   = 1E4      !",
            "",
            "nmodel_ped_height = 0 ! Ped Height model choice  (1=NTCC, 0=namelist)",
            "nmodel_ped_width  = 0 ! Ped Width model choice  (1=NTCC, 0=namelist)",
            "",
            "!------ L-mode, H-mode transitions",
            "",
            "nmodel_l2h_trans = 1     ! Model choice (1=NTCC (exp if not enough power), 0=namelist)",
            "!time_l2h        = 2.0",
            "!time_h2l        = 3.0",
            "!tau_lh_trans_x2 = 2.0",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addGLF23(self):
        lines = [
            "!------ GLF23 namelist",
            "",
            "glf23_template = 'tmp_folder/glf23_namelist.dat'",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

        lines = [
            "&glf23_control",
            " ",
            " version                    = 2       ! 1 - original; 2 - v1.61; 3 - renormalized and real geometry fit",
            " lprint                     = 0       ! get output files with lprint=1",
            " use_adiabatic_electrons    = F",
            " bt_flag                    = 1       ! 0 - use B_T; 1 - use Bt_eff",
            " alpha_p_mult               = 1.0",
            " alpha_quench_mult          = 1.0",
            " cxnu                       = 1.0     ! scale factor for collisionality",
            " cbetae                     = 1.0     ! factor for plasma beta",
            " ",
        ]

        # lines.append('/')#I do it later, in defineRunParams

        self.contents_ptr_glf23 = "\n".join(lines) + "\n"

    def addTGLF(self):
        TGLFoptions, label = GACODEdefaults.TGLFinTRANSP(self.TGLFsettings)
        print(
            f"\t- Adding TGLF control parameters with TGLFsettings = {self.TGLFsettings} ({label})"
        )

        lines = [
            "!------ TGLF namelist",
            "",
            "tglf_template   =   'tmp_folder/tglf_namelist.dat'",
            f"nky             =   {TGLFoptions['NKY']}",
            f"nlgrowth_tglf   =   {self.grTGLF}   ! Output TGLF information (growth rates, freqs...)",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"

        lines = ["&tglf_control"]
        for ikey in TGLFoptions:
            if ikey not in ["NKY"]:
                lines.append(f" {ikey.ljust(21)} = {TGLFoptions[ikey]}")
        # lines.append('/')# I do it later, in defineRunParams

        self.contents_ptr_tglf = "\n".join(lines) + "\n"

    def addAC(self, avgtim=0.05):
        
        lines = [
            "!------ File Extraction",
            "",
            "mthdavg         = 2",
            f"avgtim          = {avgtim}        ! Average before",
            f"outtim          = {self.timeAC}",
            f"fi_outtim       = {self.timeAC}",
            f"fe_outtim       = {self.timeAC}",
            "nldep0_gather   = T",
            "",
        ]

        self.contents += "\n".join(lines) + "\n"


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Updates
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_updates(self, time, quantities={}):

        lines = [
            "",
            "! ================== UPDATES ==================",
            "",
            f"~update_time={time}",
            "",
        ]

        for quantity in quantities:
            lines.append(f'{quantity} = {quantities[quantity]}')

        self.contents += "\n".join(lines) + "\n"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write to file
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def write(self, runid = 'Z99', file = None):

        if file is not None:
            self.file = file
            self.inputdir = os.path.dirname(file)
        else:
            self.file = f"{self.inputdir}/{self.shotnum}{runid}TR.DAT"

        print(f"\t- Writing main namelist to {IOtools.clipstr(self.file)}")
        with open(self.file, "w") as file:
            file.write(self.contents)

        if self.contents_ptr_ptsolver is not None:
            print(f"\t- Writing PT_SOLVER namelist to {IOtools.clipstr(self.file)}")
            with open(f"{self.inputdir}/ptsolver_namelist.dat", "w") as file:
                file.write(self.contents_ptr_ptsolver)
        if self.contents_ptr_glf23 is not None:
            print(f"\t- Writing GLF23 namelist to {IOtools.clipstr(self.file)}")
            with open(f"{self.inputdir}/glf23_namelist.dat", "w") as file:
                file.write(self.contents_ptr_glf23)
        if self.contents_ptr_tglf is not None:
            print(f"\t- Writing TGLF namelist to {IOtools.clipstr(self.file)}")
            with open(f"{self.inputdir}/tglf_namelist.dat", "w") as file:
                file.write(self.contents_ptr_tglf)

def adaptNML(FolderTRANSP, runid, shotnumber, FolderRun):
    '''
    This ensures that the folders are adapted to the, e.g., remote execution setup
    '''

    nml_file = f"{FolderTRANSP}/{runid}TR.DAT"

    # Change inputdir
    IOtools.changeValue(
        nml_file, "inputdir", f"'{FolderRun}'", [""], "=", CommentChar=None
    )

    # Change PTR templates, if they are aleady there
    if (
        IOtools.findValue(
            nml_file, "pt_template", SplittingChar="=", raiseException=False
        )
        is not None
    ):
        IOtools.changeValue(
            nml_file,
            "pt_template",
            f"'{os.path.abspath(FolderRun)}/ptsolver_namelist.dat'",
            [""],
            "=",
            CommentChar=None,
        )
    if (
        IOtools.findValue(
            nml_file, "tglf_template", SplittingChar="=", raiseException=False
        )
        is not None
    ):
        IOtools.changeValue(
            nml_file,
            "tglf_template",
            f"'{os.path.abspath(FolderRun)}/tglf_namelist.dat'",
            [""],
            "=",
            CommentChar=None,
        )
    if (
        IOtools.findValue(
            nml_file, "glf23_template", SplittingChar="=", raiseException=False
        )
        is not None
    ):
        IOtools.changeValue(
            nml_file,
            "glf23_template",
            f"'{os.path.abspath(FolderRun)}/glf23_namelist.dat'",
            [""],
            "=",
            CommentChar=None,
        )

    # Change shot number
    IOtools.changeValue(nml_file, "nshot", shotnumber, [""], "=", CommentChar=None)

    # Correct namelist just in case
    IOtools.correctNML(nml_file)

