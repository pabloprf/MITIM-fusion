import os
from mitim_tools.misc_tools import IOtools, FARMINGtools
from mitim_tools.gacode_tools.aux import GACODEdefaults
from mitim_tools.misc_tools import CONFIGread
from mitim_tools.misc_tools.IOtools import printMsg as print
from IPython import embed


def adaptNML(FolderTRANSP, runid, shotnumber, FolderRun):
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


def interpret_trdat(file):
    if not os.path.exists(file):
        print("TRDAT was not generated. It will likely fail!", typeMsg="q")
    else:
        with open(file, "r") as f:
            aux = f.readlines()

        file_plain = "".join(aux)

        errors = [pos for pos, char in enumerate(file_plain) if char == "?"]

        errors_clean = []

        truly_error = False
        for cont, i in enumerate(errors):
            if (i == 0 or errors[cont - 1] < i - 1) and file_plain[
                i : i + 4
            ] != "????":  # Because that's an TOK error, it would be fine
                truly_error = True

        if truly_error:
            print('\t- Detected "?" in TRDAT output, printing tr_dat:', typeMsg="w")
            print("".join(aux), typeMsg="w")
            print("Do you wish to continue? It will likely fail! (c)", typeMsg="q")
        else:
            print("\t- TRDAT output did not show any error", typeMsg="i")


"""
TRANSP nml constructor
----------------------
This routine constructs default namelist based on a tokamak.
Once written, it can be modified with changeValues, which is something that MITIM will do anyway
"""


class default_nml:
    def __init__(
        self,
        shotnum,
        tok,
        PlasmaFeatures={
            "ICH": True,
            "ECH": False,
            "NBI": False,
            "ASH": False,
            "Fuel": 2.5,
        },
        pservers=[1, 1, 0],
        TGLFsettings=5,
    ):
        coeffsSaw = [1.0, 3.0, 1.0, 0.4]
        useMMX = False

        Pich, Pech, Pnbi, AddHe4ifDT, Fuel = (
            PlasmaFeatures["ICH"],
            PlasmaFeatures["ECH"],
            PlasmaFeatures["NBI"],
            PlasmaFeatures["ASH"],
            PlasmaFeatures["Fuel"],
        )

        DTplasma = Fuel == 2

        Ufiles = {
            "lim": ["LIM", None],
            f"qpr": ["QPR", -5],
            "cur": ["CUR", None],
            "vsf": ["VSF", None],
            "rbz": ["RBZ", None],
            f"ter": ["TEL", -5],
            "ti2": ["TIO", -5],
            "ner": ["NEL", -5],
            "zf2": ["ZF2", -5],
            f"gfd": ["GFD", None],
        }

        if useMMX:
            Ufiles["mmx"] = ["MMX", None]
        else:
            Ufiles["mry"] = ["MRY", None]

        # ~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~ TGLF
        # ~~~~~~~~~~~~~~~~~~~~~~~

        isolver = False

        grTGLF = False  # Disable by default because it takes disk space and time... enable for 2nd preditive outside of this routine

        # ~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~ Differences between tokamaks
        # ~~~~~~~~~~~~~~~~~~~~~~~

        if tok == "SPARC" or tok == "ARC":
            Ufiles["df4"], Ufiles["vc4"] = ["DHE4", -5], ["VHE4", -5]

            timeStep = 1e-2  # Time step for DTMAXG and the heating modules
            msOut = 1e0
            nteq_mode = 2
            taupD, taupZ, taupmin = 3.0, 3.0, 3.0
            tokred = "sprc"
            UFrotation = False
            Te_edge = 80.0
            Ti_edge = 80.0

        if tok == "AUG":
            timeStep = 1e-2
            msOut = 1e-1
            nteq_mode = 2
            taupD, taupZ, taupmin = 3.0, 3.0, 3.0
            tokred = "aug"
            UFrotation = True
            Te_edge = 80.0
            Ti_edge = 80.0

        if tok == "CMOD":
            timeStep = 1e-3
            msOut = 1.0  # 1E-1
            nteq_mode = 5
            taupD, taupZ, taupmin = 30e-3, 20e-3, 1e6
            tokred = "cmod"
            UFrotation = True
            Te_edge = 80.0
            Ti_edge = 80.0

            coeffsSaw[1] = 2.0  # If not, 15 condition triggered too much

        # ~~~~~~~~~~~~~~~~~~~~~~
        # Primary namelist
        # ~~~~~~~~~~~~~~~~~~~~~~

        nzones = 100  # multiple of nzones_fast
        nzones_codes = 50  # multiple of nzones_fast
        nzones_fast = 25

        self.nml = TRANSPnml_General(
            shotnum=shotnum,
            tokamak=tok,
            tok=tokred,
            nzones=nzones,
            nzones_codes=nzones_codes,
            nzones_fast=nzones_fast,
            Pich=Pich,
            Pech=Pech,
            Pnbi=Pnbi,
            pservers=pservers,
            DTplasma=DTplasma,
            AddHe4ifDT=AddHe4ifDT,
            Ufiles=Ufiles,
            UFrotation=UFrotation,
            msOut=msOut,
            timeStep=timeStep,
            nteq_mode=nteq_mode,
            taupD=taupD,
            taupZ=taupZ,
            taupmin=taupmin,
            isolver=isolver,
        )

        # ~~~~~~~~~~~~~~~~~~~~~~
        # Predictive
        # ~~~~~~~~~~~~~~~~~~~~~~

        self.nml = TRANSPnml_Predictive(
            self.nml.contents,
            TGLFsettings=TGLFsettings,
            grTGLF=grTGLF,
            Te_edge=Te_edge,
            Ti_edge=Ti_edge,
        )

        # ~~~~~~~~~~~~~~~~~~~~~~
        # Heating Namelists
        # ~~~~~~~~~~~~~~~~~~~~~~

        # ICRF
        if Pich:
            self.nml_ich = TRANSPnml_Heating(
                Pich=True, tokamak=tok, timeStep=timeStep
            ).contents
        else:
            self.nml_ich = ""

        # ECRF
        if Pech:
            self.nml_ech = TRANSPnml_Heating(
                Pech=True, tokamak=tok, timeStep=timeStep
            ).contents
        else:
            self.nml_ech = ""

        # NBI
        if Pnbi:
            self.nml_nbi = TRANSPnml_Heating(
                Pnbi=True, tokamak=tok, timeStep=timeStep
            ).contents
        else:
            self.nml_nbi = ""

    def write(self, BaseFile="./10000TR.DAT"):
        self.BaseFile = BaseFile

        with open(self.BaseFile, "w") as f:
            f.write(self.nml.contents)
        IOtools.correctNML(self.BaseFile)

        with open(self.BaseFile + "_ICRF", "w") as f:
            f.write(self.nml_ich)
        with open(self.BaseFile + "_ECRF", "w") as f:
            f.write(self.nml_ech)
        with open(self.BaseFile + "_NBI", "w") as f:
            f.write(self.nml_nbi)
        with open(self.BaseFile + "_LH", "w") as f:
            f.write("")

        a, b = IOtools.reducePathLevel(self.BaseFile)
        if self.nml.contents_ptr_ptsolver is not None:
            with open(a + "ptsolver_namelist.dat", "w") as f:
                f.write(self.nml.contents_ptr_ptsolver)
        if self.nml.contents_ptr_tglf is not None:
            with open(a + "tglf_namelist.dat", "w") as f:
                f.write(self.nml.contents_ptr_tglf)
        if self.nml.contents_ptr_glf23 is not None:
            with open(a + "glf23_namelist.dat", "w") as f:
                f.write(self.nml.contents_ptr_glf23)

    def appendNMLs(self):
        possibleAppends = ["ICRF", "ECRF", "NBI", "LH"]

        with open(self.BaseFile, "a") as fo:
            for i in possibleAppends:
                try:
                    with open(self.BaseFile + "_" + i, "r") as fi:
                        fo.write(fi.read())
                    os.system("rm " + self.BaseFile + "_" + i)
                except:
                    print(f"\t- Could not append {i} file")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NAMELIST CONSTRUCTORS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class TRANSPnml_General:
    def __init__(
        self,
        tokamak="SPARC",
        contents="",
        LimitersInNML=False,
        timeStep=0.001,
        msOut=1,
        msIn=1,
        nzones=100,
        nzones_codes=50,
        nzones_fast=25,
        nzones_frantic=20,
        shotnum=175844,
        inputdir=None,
        pservers=[1, 1, 0],
        tinit=0.0,
        ftime=100.0,
        taupD=5.0,
        taupZ=0.4,
        taupmin=3.0,
        zeff=2.0,
        zlump=[[6.0, 12.0, 0.018], [74.0, 183.0, 1.5e-5]],
        DTplasma=True,
        AddHe4ifDT=True,
        Minorities=[2, 3, 0.05],
        nteq_mode=5,
        currDiff_phases=[[4, 4.85], [1, 100.0]],
        gridsMHD=[151, 127],
        smoothMHD=[-1.5, 0.15],
        UFrotation=False,
        impurity=[11, 22],
        Pich=False,
        Pech=False,
        Pnbi=False,
        timeSaw=0.1,
        coeffsSaw=[1.0, 3.0, 1.0, 0.4],
        ReconnectionFraction=0.37,
        predictRad=True,
        MCparticles=1e6,
        useBootstrapSmooth=None,
        B_ccw=False,
        Ip_ccw=False,
        tok="SPRC",
        isolver=False,
        Ufiles={
            "qpr": ["QPR", -5],
            f"mry": ["MRY", None],
            f"cur": ["CUR", None],
            f"vsf": ["VSF", None],
            f"ter": ["TEL", -5],
            f"ti2": ["TIO", -5],
            f"ner": ["NEL", -5],
            f"rbz": ["RBZ", None],
            f"df4": ["DHE4", -5],
            f"vc4": ["VHE4", -5],
        },
    ):
        self.contents = contents

        self.addHeader(shotnum, inputdir=inputdir)
        self.addTimings(
            tinit,
            ftime,
            pservers,
            msOut=msOut,
            msIn=msIn,
            nzones=nzones,
            nzones_codes=nzones_codes,
            nzones_fast=nzones_fast,
            timeStep=timeStep,
        )
        self.addPB()
        if not Pich:
            Minorities = None
        self.addSpecies(zeff, zlump, DTplasma, Minorities, AddHe4ifDT=AddHe4ifDT)
        self.addParticleBalance(
            taupD=taupD,
            taupZ=taupZ,
            taupmin=taupmin,
            AddHe4=AddHe4ifDT and DTplasma,
            nzones_frantic=nzones_frantic,
        )
        self.addRadiation(predictRad=predictRad)
        self.addNCLASS()
        self.addMHDandCurrentDiffusion(
            nteq_mode,
            tokamak,
            currDiff_phases,
            B_ccw=B_ccw,
            Ip_ccw=Ip_ccw,
            useBootstrapSmooth=useBootstrapSmooth,
            grids=gridsMHD,
            smooth=smoothMHD,
            isolver=isolver,
        )
        self.addSawtooth(
            timeSaw, coeffs=coeffsSaw, ReconnectionFraction=ReconnectionFraction
        )
        self.addFusionProducts(DTplasma, MCparticles=MCparticles)
        self.addRotation(UFrotation, impurity)
        self.addVessel(tokamak, LimitersInNML=LimitersInNML)
        self.addUFILES(Ufiles, Pich=Pich, Pech=Pech, Pnbi=Pnbi)

    def addHeader(self, shotnum, inputdir=None):
        lines = [
            "! Authors: P. Rodriguez-Fernandez and N.T. Howard, ",
            "",
            f"!==============================================================",
            f"! General settings & Output options",
            f"!==============================================================",
            f"",
            f"NSHOT={0}".format(shotnum),
            f"",
            f"mrstrt = -120     ! Frequency of restart records (-, means x wall clock min)",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

        if inputdir is not None:
            lines = [f'inputdir="{inputdir}"']
            self.contents += "\n".join(lines) + "\n"

    def addTimings(
        self,
        tinit,
        ftime,
        pservers,
        msOut=1,
        msIn=1,
        nzones=100,
        nzones_codes=20,
        nzones_fast=10,
        timeStep=0.001,
    ):
        lines = [
            "!----- Time and Spatial ranges",
            f"",
            f"tinit = {tinit:.3f} ! Start time",
            f"ftime = {ftime:.3f} ! End time",
            f"xtend = 0.050 ! Distance beyond LCFS to extend analysis ",
            f"",
            f"!----- Time buffers",
            f"",
            f"tlim1 = 0.0    ! Ignore UF data before this time, extrapolate flat backward in time",
            f"tlim2 = 1000.0 ! Ignore UF data after this time, extrapolate flat forward in time",
            f"",
            f"!----- Spatial resolution",
            f"",
            f"nzones   = {nzones} ! Number of radial zones in 1D transport equations",
            f"nzone_nb = {nzones_codes} ! Number of zones in NUBEAM (beams and alphas)",
            f"nzone_fp = {nzones_codes} ! Number of zones in the FPPMOD (minority ICRF)",
            f"nzone_fb = {nzones_fast} ! Number of zone rows in fast ion distr function (must be divider of all other)",
            f"",
            f"!----- Temporal resolution",
            f"",
            f"dtmaxg = {timeStep}    ! Max time step for MHD",
            f"dtinit = 1.0e-6        ! Initial timestep to obtain power balance (default=1e-6)",
            f"",
            f"!----- I/O resolution",
            f"sedit    = {msOut*1E-3} ! Control of time resolution of scalar output",
            f"stedit   = {msOut*1E-3} ! Control of time resolution of profile output",
            f"tgrid1   = {msIn*1E-3}  ! Control of time resolution of 1D input data",
            f"tgrid2   = {msIn*1E-3}  ! Control of time resolution of 2D input data",
            f"",
            f"!----- MPI Settings",
            f"",
            f"nbi_pserve     = {pservers[0]} ! enable mpi for nubeam, nbi and fusion products",
            f"ntoric_pserve  = {pservers[1]} ! enable mpi toric, icrf",
            f"nptr_pserve    = {pservers[2]} ! enable mpi ptransp (tglf only)",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addPB(self):
        lines = [
            "!=============================================================",
            f"! Power balance",
            f"!=============================================================",
            f"",
            f"!------ Electron Power Balance",
            f"",
            f"alph0i  = 0.6    ! Sets ion convection power loss coeff to 3/2",
            f"",
            f"!------ Ion Power Balance",
            f"",
            f"alph0e  = 0.6    ! Sets electron convection power loss coeff to 3/2",
            f"",
            f"nlti2   = T      ! Use Ti input data profiles to determine Xi",
            f"nltipro = T      ! Use Ti input data directly, no predictive calculation",
            f"",
            f"nslvtxi = 2      ! 2: Ti data is average temperature (TIAV)",
            f"",
            f"tixlim0 = 0.0    ! min xi (r/a) of validity of the Ti data",
            f"tixlim  = 1.0    ! max xi (r/a) of validity of the Ti data",
            f"tidxsw  = 0.05   ! Continuous linear transition",
            f"tifacx  = 1.0    ! edge Ti/Te factor",
            f"",
            f"!------ Ti not available",
            f"",
            f"giefac = 1.0     ! Ti = giefac * Te ",
            f"fiefac = 1.0     ! Ti = (1-fiefac)*(giefac*Te)+fiefac*Ti(data)",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addSpecies(
        self, zeff, zlump, DTplasma, Minorities, coronal=True, AddHe4ifDT=False
    ):
        lines = [
            "!=============================================================",
            f"! Plasma Species",
            f"!=============================================================",
            f"",
            f"!----- Background Species",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

        if DTplasma:
            if AddHe4ifDT:
                lines = [
                    "ng       = 3       ! Number of INITIAL background species   ",
                    f"ngmax   = 3  	    ! Maximum FUTURE background species (Pellets,NBI,recy)",
                    f"",
                    f"backz(1)  = 1.0   ! Charge of background species",
                    f"aplasm(1) = 2.0   ! Atomic mass of background species",
                    f"frac(1)   = 0.5   ! Background gas fractions",
                    f"",
                    f"backz(2)  = 1.0",
                    f"aplasm(2) = 3.0 ",
                    f"frac(2)   = 0.5 ",
                ]

            else:
                lines = [
                    "ng	      = 2         ! Number of INITIAL background species   ",
                    f"ngmax   = 2  	      ! Maximum FUTURE background species (Pellets,NBI,recy)",
                    f"",
                    f"backz(1)   = 1.0    ! Charge of background species",
                    f"aplasm(1)  = 2.0    ! Atomic mass of background species",
                    f"frac(1)    = 0.5    ! Background gas fractions",
                    f"",
                    f"backz(2)   = 1.0",
                    f"aplasm(2)  = 3.0",
                    f"frac(2)    = 0.5",
                ]
        else:
            lines = [
                "ng        = 1      ! Number of INITIAL background species   ",
                f"ngmax	   = 1      ! Maximum FUTURE background species (Pellets,NBI,recy)",
                f"",
                f"backz(1)   = 1.0 	! Charge of background species",
                f"aplasm(1)  = 2.0 	! Atomic mass of background species",
                f"frac(1)    = 1.0  ! Background gas fractions",
                f"",
            ]

        self.contents += "\n".join(lines) + "\n"

        if DTplasma and AddHe4ifDT:
            lines = [
                "backz(3)   = 2.0 ",
                f"aplasm(3)   = 4.0 ",
                f"frac(3)   = 1.0E-10 ",
                f"",
            ]

            self.contents += "\n".join(lines) + "\n"

        # -------
        lines = [
            "!----- Impurities",
            f"",
            f"!xzeffi   = {zeff} ! Zeff (if assumed constant in r and t)",
            f"nlzfin    = F    ! Use 1D Zeff input",
            f"nlzfi2    = T    ! Use 2D Zeff input",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

        for i in range(len(zlump)):
            lines = [
                f"xzimps({i+1}) = {zlump[i][0]} ! Charge of impurity species",
                f"aimps({i+1})  = {zlump[i][1]} ! Atomic mass of impurity species",
                f"densim({i+1}) = {zlump[i][2]} ! Impurity densities (n_imp/n_e)",
            ]

            self.contents += "\n".join(lines)

            # Possibility for densima
            if len(zlump[i]) > 3:
                lines = [
                    f"densima({i+1}) = {zlump[i][3]} ! Impurity densities (n_imp/n_e)",
                    f"",
                ]
            else:
                lines = ["", ""]

            self.contents += "\n".join(lines)

            if coronal:
                self.contents += "\n".join([f"nadvsim({i + 1}) = 1"]) + "\n"

        if Minorities is not None:
            lines = [
                "",
                f"! ----- Minorities",
                f"",
                f"xzmini  = {Minorities[0]:.2f} ! Charge of minority species (can be list)",
                f"amini   = {Minorities[1]:.2f} ! Atomic mass of minority species (can be list)",
                f"frmini  = {Minorities[2]} ! Minority density (n_min/n_e)",
                f"rhminm  = 1.0E-12   ! Minimum density of a gas present (fraction of ne)",
                f"",
            ]

            self.contents += "\n".join(lines) + "\n"

    def addParticleBalance(
        self, taupD=5.0, taupZ=0.4, taupmin=3.0, AddHe4=False, nzones_frantic=20
    ):
        if AddHe4:
            strt = "ndefine(3)	  = 1"
        else:
            strt = ""

        lines = [
            "!=============================================================",
            f"! Particle Balance",
            f"!=============================================================",
            f"",
            f"!-----  Particle balance model",
            f"",
            f"ndefine(1) = 0  ! Density profile (0=nmodel, 1= D-V from UF, 2= n from UF)",
            f"ndefine(2) = 0",
            strt,
            f"",
            f"nmodel = 1  ! Mixing Model: 1 = same shape, 2 = same Vr, 4 = mixed model",
            f"ndiffi = 5  ! Input Diffusivity (for nmodel=4)",
            f"",
            f"!----- Particle Source: Recycling model",
            f"",
            f"nrcyopt = 0 ! Recycling model (0=by species,1=mixed model, 2=)",
            f"nltaup  = F ! (T) UFile, (F) Namelist tauph taupo",
            f"",
            f"taupmn   = 0.001     ! Minimum allowed confinement time (s)",
            f"tauph(1) = {taupD}   ! Main (hydrogenic) species particle confinement time",
            f"tauph(2) = {taupD}",
            f"tauph(3) = {taupD}",
            f"taupo    = {taupZ}   ! Impurity particle confinement time",
            f"taumin   = {taupmin} ! Minority confinement time",
            f"",
            f"nlrcyc     = F",
            f"nlrcyx     = F     ! Enforce for impurities same taup as main ions",
            f'!rfrac(1)	 = 0.5   ! Recyling fractions (fraction of "limiter outgas" flux)',
            f"!rfrac(2)	 = 0.5",
            f"!rfrac(3)	 = 0.0",
            f'!recych(1) = 0.5   ! Fraction of ion outflux of species that is "promptly" recycled',
            f"!recych(2) = 0.5",
            f"!recych(3) = 0.0",
            f"",
            f"!----- Particle Source: Gas Flow",
            f"",
            f"!gfrac(1) = 1.0 ! gas flow ratio for each species",
            f"!gfrac(2) = 0.5",
            f"!gfrac(3) = 0.5",
            f"",
            f"!----- Neutrals Transport",
            f"",
            f"nsomod          = 1   ! FRANTIC analytic neutral transport model",
            f"!nldbg0          = T   ! More FRANTIC outputs",
            f"nzones_frantic  = {nzones_frantic} ! Number of radial zones in FRANTIC model",
            f"mod0ed          = 1   ! 1 = Use time T0 or TIEDGE",
            f"ti0frc          = 0.0333 ! If mod0ed=2, Ti(0) multiplier" f"",
            f"fh0esc          = 0.3 ! Fraction of escaping neutrals to return as warm (rest as cold)",
            f"nlreco          = T   ! Include recombination volume neutral source ",
            f"",
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

    def addRadiation(self, predictRad=True):
        if predictRad:
            num = 2
        else:
            num = 0

        lines = [
            f"",
            "!=============================================================",
            f"! Radiation Model (Bremsstrahlung, Line and Cyclotron)",
            f"!=============================================================",
            f"",
            f"nprad    = {num} ! Radiative power calculation (1 = U-F or Theory, 2 = Always Theory)",
            f"nlrad_br = T     ! Calculate BR by impurities (Total is always calculated)",
            f"nlrad_li = T     ! Calculate impurity line radiation ",
            f"nlrad_cy = T     ! Calculate cyclotron radiation",
            f"vref_cy  = 0.9   ! Wall reflectivity used in cyclotron radiation formula",
            f"prfac    = 0.0   ! For nprad=-1, corrects BOL input",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addNCLASS(self):
        lines = [
            "!===========================================================",
            f"! NCLASS neoclassical Model",
            f"!===========================================================",
            f"",
            f"ncmodel  = 2   ! 2: Houlbergs NCLASS 2004",
            f"ncmulti  = 2   ! 2= Always use multiple impurities when present",
            f"nltinc   = T   ! Use average Ti for calculations",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addMHDandCurrentDiffusion(
        self,
        nteq_mode,
        tokamak,
        currDiff_phases,
        B_ccw=False,
        Ip_ccw=False,
        useBootstrapSmooth=None,
        nmodpoh=1,
        grids=[151, 127],
        smooth=[-1.5, 0.15],
        isolver=False,
    ):
        if isolver:
            levgeo = 12
            NLQLIM0 = "False"
        else:
            levgeo = 11
            NLQLIM0 = "True"

        lines = [
            "!=============================================================",
            f"! MHD equilibrium and Current Diffusion",
            f"!=============================================================",
            f"",
            f"!----- MHD Geometry",
            f"",
            f"levgeo	       = {levgeo}    ! 11 = TEQ model from LLNL, 12 = ISOLVER",
            f"",
            f"nteq_mode     = {nteq_mode}   ! Free param to be matched (5= Q,F_edge & loop for Ip)",
            f"nteq_stretch  = 0   ! Radial grid (0 is default)",
            f"nteq_nrho     = {grids[0]}   ! Radial points in TEQ",
            f"nteq_ntheta   = {grids[1]}   ! Poloidal points in TEQ",
            f"teq_smooth    = {smooth[0]}  ! Smoothing half-width (negative means -val/nzones)",
            f"teq_axsmooth  = {smooth[1]} ! Smoothing half-width near-axis as val*min(1,2-x/val)",
            f"softteq       = 0.3   ! Maximum allowed GS average error",
            f"",
            f"!------ Poloidal field (current) diffusion",
            f"",
            f"nlmdif	  = F    ! Solve poloidal diffusion equation ",
            f"nlpcur	  = T    ! Match total plasma current",
            f"nlvsur	  = F    ! Match surface voltage",
            f"",
            f"nlbccw = {B_ccw}    ! Is Bt counter-clockwise (CCW) seen from above?",
            f"nljccw = {Ip_ccw}    ! Is Ip counter-clockwise (CCW) seen from above?",
            f"",
            f"!----- Initialization of EM fields",
            f"nefld     = 7    ! 7 = q-profile from QPR, 3 = parametrized V(x) with qefld, rqefld",
            f"qefld     = 0.0  ! If nefld=3, value of q to match",
            f"rqefld    = 0.0  ! If nefld=3, location to match",
            f"xpefld    = 2.0  ! If nefld=3, quartic correction",
            f"",
            f"!------ q-profile",
            f"",
            f"nlqlim0  = {NLQLIM0}  ! Place a limit on the max q value possible",
            f"qlim0	  = 5.0  ! Set the max possible q value",
            f"nlqdata  = T    ! Use input q data (cannot be set with nlmdif=T as well)",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

        for i in range(len(currDiff_phases)):
            lines = [
                f"nqmoda({i+1}) = {currDiff_phases[i][0]} ! 4: nlqdata=T, 1: nlmdif=T",
                f"nqmodb({i+1}) = 1",
                f"tqmoda({i+1}) = {currDiff_phases[i][1]} ! Time to change to next stage",
                f"",
            ]

            self.contents += "\n".join(lines) + "\n"

        lines = [
            "tauqmod   = 1.0E-2 ! Transition window ",
            f"",
            f"!------ Resitivity Model",
            f"",
            f"nmodpoh  = {nmodpoh}		! 1: Standard Poh from EM equations, 2: Poh = eta*j^2",
            f"",
            f"nletaw	  	     = F    ! Use NCLASS resistivity",
            f"nlspiz	  	     = F    ! Use Spitzer resistivity ",
            f"nlrestsc             = F    ! Use neoclass as in TSC",
            f"nlres_sau 	     = T    ! Use Sauter",
            f"",
            f"!sauter_nc_adjust(1) = 1.0  ! Factor that multiplies Sauter resistivity",
            f"xl1nceta  	     = 0.0  ! Do not trust inside this r/a (extrapolate to 0)",
            f"nlrsq1	  	     = F  	! Flatten resistivity profile inside q=1",
            f"nlsaws	  	     = F  	! Flatten resistivity profile inside q=qsaw",
            f"qsaw	  	     = 1.0  ! Specification of q to flatten resistivity inside",
            f"xsaw_eta  	     = 0.0  ! Flattening factor",
            f"nlresis_flatten      = F  	 ! Flatten within xresis_flatten",
            f"xresis_flatten       = 0.0  ! Flattening factor",
            f"",
            f"bpli2max  = 20.0 ! The maximum value of Lambda = li/2 + Beta_p allowed",
            f"",
            f"!=============================================================",
            f"! Current drive",
            f"!=============================================================",
            f"",
            f"! ---- Current Drive",
            f"",
            f"nmcurb   = 4    ! Beam current drive model (0 = none, 4 = most recent)",
            f"nllh   = F    ! Include LH currents",
            f"",
            f"!=============================================================",
            f"! Boostrap current model",
            f"!=============================================================",
            f"",
            f"nlboot	    = T     ! Use bootstrap currents in poloidal field eqs",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

        if useBootstrapSmooth is None:
            lines = [
                "",
                f"nlbootw     = F    ! Use NCLASS bootstrap current",
                f"nlboot_sau  = F    ! Use Sauter bootstrap current",
                f"nlboothager = T    ! Use Hager bootstrap current (modification to Sauter)",
                f"nlbootneo   = F    ! Use NEO bootstrap current",
                f"",
                f"xbstrap     = 1.0   ! Anomaly factor",
                f"!xl1ncjbs   = 0.3   ! Do not trust inside this r/a (extrapolate to 0)",
                f"njsmboot    = 0    ! JB moothing Number of zones for sliding triangular weighted average",
                f"",
            ]

        else:
            lines = [
                "",
                f"nlbootw	 = T    ! Use NCLASS bootstrap current",
                f"nlboot_sau  = F    ! Use Sauter bootstrap current",
                f"nlboothager = F    ! Use Hager bootstrap current (modification to Sauter)",
                f"nlbootneo   = F    ! Use NEO bootstrap current",
                f"",
                f"xbstrap    = 1.0   ! Anomaly factor",
                f"xl1ncjbs   = {useBootstrapSmooth}   ! Do not trust inside this r/a (extrapolate to 0)",
                f"njsmboot    = 0    ! JB moothing Number of zones for sliding triangular weighted average",
                f"",
            ]

        self.contents += "\n".join(lines) + "\n"

        if isolver:
            if tokamak == "SPARC":
                from mitim_tools.experiment_tools.SPARCtools import defineISOLVER

            ISOLVERdiffusion = "T"

            isolver_file, pfcs = defineISOLVER()

            lines = [
                "!=============================================================",
                f"! ISOLVER",
                f"!=============================================================",
                f"",
                f"nlpfc_circuit = T 		! PFC currents in UFILES are circuit currents",
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
                f"! ---- Flux diffusion",
                f"nlisodif = {ISOLVERdiffusion} 			! Use ISOLVER flux diffusion instead of nlmdif",
                f"!niso_ndif = 			 ! Number of internal flux diffusion points",
                f"niso_difaxis = 1		! Distribution of internal flux diffusion points (1=uniform)",
                f"niso_qlim0 = 0 		! Way to limit on axis q (0=no limit, 2=force, 3=current drive)",
                f"nisorot = -1 			! Rotation treatment in GS (-1=no rotation)",
                f"nisorot_maximp = 3     ! Max number charge state for impurity to enter as rotating",
                f"xisorot_scale = 1.0 	! Rotation scale factor",
                f"",
                f"! ---- Machine and equilibrium",
                f'iso_config = "{isolver_file}"		! Tokamak configuration',
                f"neq_ilim = 0			! Limiter info (0= try ufile first, then internal)",
                f'probe_names = "@all"   ! List of probes',
                f"neq_xguess = 2 		! Guess x-points (2-both x-points)",
                "!pass_names = '@pass'",
                f"",
                f"! ---- Solution Method",
                f"niso_mode = 102		! Use in solution: 102 (default) - p, q; 101 - p, <J*B>",
                f"niso_qedge = 2			! How to handle matching q if niso=102 (2=Mod to match Ip)",
                f"eq_x_weight = 1.0		! Weight for x-point location matching",
                f"eq_ref_weight = 1.0	! Weight for prescribed boundary matching",
                f"eq_cur_weight = 1.0	! Weight for experimental currents matching",
                f"",
                f"! ---- Convergence and errors",
                f"niso_maxiter = 250 	! Number of iterations with fixed parameters (then, 8 tries changing stuff)",
                f"niso_sawtooth_mode = 0 ! In sawtooth, keep continuity of coil/vessel (0) poloidal flux or (1) current",
                f"xiso_omega = 0.5	    ! Relaxation between iterations (0.5 for q-mode, 0.2 for <J*B>-mode",
                f"xiso_omega_dif = 0.6   ! Relaxation for flux diffusion boundary condition",
                f"xiso_error = 1.e-6		! Convergence error criteria",
                f"xiso_warn_ratio = 50.0	! Accept error relaxed by this factor if struggling",
                f"xiso_last_ratio = -1	! Ratio applied to last retry",
                f"xbdychk = 0.05 		! Maximum allowable difference between prescribed and LCFS at midplane",
                f"!dtfirstg = 			 ! Time step for first flux diffusion (hardest)",
                f"!niso_njpre = 4		 ! Adds a 4 point interation filter to Jphi",
                f"",
                f"! ---- Resolution",
                f"niso_nrho = 100		! Radial surfaces to use in ISOLVER",
                f"niso_ntheta = 121 		! Number of poloidal rays to use in ISOLVER",
                f"niso_axis = 2 			! Distribution of radial surfaces (2: sin(pi/2*uniform))",
                f"niso_nrzgrid = 129,129 ! Points in R,Z direction",
                f"!xiso_smooth = 		 ! Smoothing of profiles prior to ISOLVER",
                f"xiso_pcurq_smooth = -2 ! Smoothing of current in q mode",
                f"eq_crat = 0.08			! Minimum curvature of boundary derived from psiRZ solution",
                f"eq_bdyoff = 0.007		! Boundary offset to meet eq_crat requirement",
                f"",
                f"! ---- Other",
                f"xeq_p_edge = 0			 ! 0 = force P_edge=0, 1 = shift down first",
                f"",
                f"! ---- Circuit mode",
                f"!fb_names = 			 ! Feedback controllers (this turns on the CIRCUIT mode)",
                f"",
                f"!fb_feedback = 		 ! ",
                f"!fb_feedomega = 		 ! ",
                f"!fb_lim_feedback = 		 ! ",
                f"!tinit_precoil = 		 ! ",
                f"!xiso_relbal = 		 ! ",
                f"!xiso_relbal_dt = 		 ! ",
                f"!neq_redo = 		 ! ",
                f"!niso0_circ = 		 ! ",
                f"!niso_jit_psifix = 		 ! ",
                f"!pfc_jit_voltage = 		 ! ",
                f"!npfc_jit = 			 ! Way to use just-in-time for next step currents",
                f"!xiso_jit_relax =",
                f"!pfc__jit_sigma = ",
                f"!xiso_def_jit_sigma = ",
                f"!xiso_jit_sigma_pcur = ",
                f"!niso_def_drive_type = -2 ! -2: drive with current sources",
                f"",
            ]

            self.contents += "\n".join(lines) + "\n"

    def addSawtooth(
        self, time_on, coeffs=[1.0, 3.0, 1.0, 0.4], ReconnectionFraction=0.37
    ):
        islandFract = 1 - ReconnectionFraction

        lines = [
            "!===============================================================",
            f"!Sawtooth Modeling",
            f"!===============================================================",
            f"",
            f"!---- Sawtooth crash/redistribution model",
            f"",
            f"nlsaw   = T       ! Sawooth for bulk plasma",
            f"nlsawic = T       ! Sawooth for ICRF minority fast ions",
            f"nlsawe  = T       ! Sawooth for electrons",
            f"nlsawi  = T       ! Sawooth for ions",
            f"nlsawb  = T       ! Sawooth for fast ions",
            f"nlsawf  = T       ! Sawooth for fusion products",
            f"nlsawd  = F       ! For clean sawtooth crashes (U-Files smooth through sawtooth)",
            f"dtsawd  = 0.002   ! Interval around sawtooth to extrapolate input data",
            f"",
            f"nmix_kdsaw     = 4       ! 1= Std Kadomtsev, 3(4) = Porcelli 2 islands with 2(1) energy mixing region",
            f"fporcelli	   = {islandFract} 	  ! Porcelli island width fraction (1 is q=1 to axis, so it is 1-Porcelli definition)",
            f"xswid1         = 0.0     ! Conserve total ion energy (0.0 =yes) in Kad or Por",
            f"xswid2 	    = 0.0     ! Conserve current & q profiles (1.0=yes; 0.0=no, mix them)",
            f"xswidq         = 0.05    ! Finite thickness in x of current sheet width to avoid solver crash ",
            f"xswfrac_dub_te = 1.0 	  ! Fraction of change in Bp energy to assign to electrons",
            f"",
            f"!---- Sawtooth triggering model",
            f"",
            f"nlsaw_trigger    = T 		! trigger sawtooth crashes using model",
            f"nlsaw_diagnostic = F		! diagnose sawtooth conditions but do not crash",
            f"model_sawtrigger = 2      ! 0=specify below,1=Park-monticello, 2=Porcelli",
            f"t_sawtooth_on    = {time_on}	    ! Parameter changed ",
            f"t_sawtooth_off   = 1.0E3  ! Last sawtooth crash time",
            f"sawtooth_period  = 1.0    ! Sawtooth period in seconds (if model_sawtrigger = 0)",
            f"",
            f"l_sawtooth(1)    = -1     ! 0 = Do not crash if multiple q=1",
            f"xi_sawtooth_min  = 0.0    ! Smallest radius that trigger q=1 sawtooth",
            f"c_sawtooth(2)    = 0.1    ! Impose a minimum sawtooth period (as fraction of Park-Monticello model)",
            f"c_sawtooth(20)   = 1.0    ! Coefficient for d beta_fast / d r",
            f"",
            f"l_sawtooth(32)   = 1   	 ! 1 = Use c_sawtooth(25:29) from namelist",
            f"c_sawtooth(25)   = 0.1    ! shear_minimum (default 0.1)",
            f"c_sawtooth(26)   = {coeffs[0]}    ! ctrho in Eq 15a             (default porcelli = 1.0)",
            f"c_sawtooth(27)   = {coeffs[1]}    ! ctstar in Eq 15b            (default porcelli = 3.0)",
            f"c_sawtooth(28)   = {coeffs[2]}    ! cwrat  in Eq B.8 for dWfast (default porcelli = 1.0)",
            f"c_sawtooth(29)   = {coeffs[3]}    ! chfast in Eq 13             (default porcelli = 0.4)",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addFusionProducts(self, completeMCmodel, MCparticles=1e6):
        lines = [
            "!==============================================================================",
            f"! Fusion products, reactions and slowing down",
            f"!==============================================================================",
            f"",
            f"!----- General Model",
            f"",
            f"nalpha  				   = {int(not completeMCmodel)}     ! Fusion products model (0=MC for alphas, 1= fast model)",
            f"nptclf  				   = {int(MCparticles)} ! Number of monte carlo particles for fusion product calcs",
            f"nlfatom 				   = T     ! Include atomic physics effects on products (e.g. CX)",
            f"nl_ignore_mini_conflicts    = T		! Ignore issue with He3 product coinciden with ICRF minority",
            f"",
            f"!----- Reactions",
            f"",
            f"nlfhe4  = {completeMCmodel}       ! Turn on MC slowing-down of He4 from D+T reactions",
            f"plfhe4  = 1.0E2   ! Source power threshold to run MC (in W)",
            f"",
            f"nlfst   = F       ! Turn on MC slowing-down of T from D+D reactions (D+D=T+p)",
            f"plfst   = 1.0E0	! Source power threshold to run MC NLFST",
            f"",
            f"nlfsp   = F       ! Turn on MC slowing-down of protons from D+D reactions (D+D=T+p)",
            f"plfsp   = 1.0E0	! Source power threshold to run MC for NLFST",
            f"",
            f"nlfhe3  = F       ! Turn on MC slowing-down of He3 from D+D reactions (D+D=He3+n)",
            f"plfhe3  = 1.0E0   ! Source power threshold to run MC for NLFHE3",
            f"",
            f"!----- From U-File",
            f"",
            f"nlusfa   = F	  ! He4",
            f"nlusft   = F	  ! T from DD",
            f"nlusfp   = F	  ! p from DD",
            f"nlusf3   = F	  ! He3 from DD",
            f"",
            f"!----- Smoothing",
            f"",
            f"dxbsmoo = 0.05	 ! Profile smoothing half-width",
            f"",
            f"!----- Anomalous transport",
            f"",
            f"nmdifb = 0 	! Anomalous diffusion (0=none, 3=Ufiles D2F & V2F)" f"",
            f"nrip   = 0 	! Ripple loss model (1=old, 2=less old)",
            f"nlfbon = F 	! Fishbone loss model",
            f"!nlfi_mcrf = T   ! Account for wave field caused by ICRF antenna",
            f"",
            f"!----- Orbit physics",
            f"",
            f"nlbflr    = T 	! Plasma interacts with T=gyro, F=guiding center",
            f"nlbgflr    = F 	! Model for gyro v.s. guiding displacement",
            f"",
            f"!----- MonteCarlo controls",
            f"",
            f"!nlseed  =  		! Random number seed for MC",
            f"!dtbeam = 0.01 	! Beam timestep",
            f"!goocon = 10 		! Numeric Goosing",
            f"!dtn_nbi_acc = 1.0e-3 ! Orbit timestep control",
            f"!nptcls = 1000 	! Constant census number of MC ions to retain",
            f"!nlfbm = T 		! Calculate beam distribution function",
            f"!ebdmax = 1e9 		 Maximum energy tracked in distribution function",
            f"!wghta = 1.0 		! MC profile statistics control",
            f"!nznbme = 800 ! Number of energy zones in beam distribution function ",
            f"!ebdfac = 6.0D0 ! Max energy factor",
            f"!nbbcal = 1100000000 ! Number of beam D-D collisions to calculate",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addRotation(self, UFrotation, impurity):
        if UFrotation:
            flag = "T"
        else:
            flag = "F"

        lines = [
            "!===========================================================",
            f"! Momentum Transport -> Rotation",
            f"!===========================================================",
            f"",
            f"nlvphi	   = {flag}    ! Rotation Moldeing using U-File",
            f"nlomgvtr  = F 	    ! T: Impurity rotation is provided, F: Bulk plasma rotation is provided",
            f"ngvtor    = 0 	    ! 0: Toroidal rotation species given by nvtor_z, xvtor_a",
            f"nvtor_z   = {impurity[0]:.2f}	! Charge of toroidal rotation species",
            f"xvtor_a   = {impurity[1]:.2f}	! Mass of toroidal rotation species",
            f"xl1ncvph  = 0.10   ! Minimum r/a",
            f"xl2ncvph  = 0.85   ! Maximum r/a",
            f"",
            f"nlivpo    = F	    ! Radial electrostatic potential and field from U-File",
            f"nlvwnc    = F      ! Compute NCLASS radial electrostatic potential profile",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addVessel(self, tokamak, LimitersInNML=False):
        if tokamak == "SPARC" or tokamak == "ARC":
            from mitim_tools.experiment_tools.SPARCtools import (
                defineTRANSPnmlStructures,
            )
        elif tokamak == "CMOD":
            from mitim_tools.experiment_tools.CMODtools import defineTRANSPnmlStructures
        elif tokamak == "AUG":
            from mitim_tools.experiment_tools.AUGtools import defineTRANSPnmlStructures

        limiters, VVmoms = defineTRANSPnmlStructures()

        if limiters is not None and LimitersInNML:
            alnlmr = str(limiters[0][0])
            alnlmy = str(limiters[0][1])
            alnlmt = str(limiters[0][2])
            for i in range(len(limiters) - 1):
                alnlmr += f", {str(limiters[i + 1][0])}"
                alnlmy += f", {str(limiters[i + 1][1])}"
                alnlmt += f", {str(limiters[i + 1][2])}"

            lines = [
                "! -----Limiters ",
                f"",
                f"nlinlm        = {len(limiters)}",
                f"alnlmr 	   = {alnlmr}",
                f"alnlmy 	   = {alnlmy}",
                f"alnlmt 	   = {alnlmt}",
                f"",
            ]

            self.contents += "\n".join(lines) + "\n"

        if VVmoms is not None:
            lines = [
                "! ----- Vaccum Vessel (reflecting boundary for the wave field)",
                f"",
            ]

            self.contents += "\n".join(lines) + "\n"

            for i in range(len(VVmoms)):
                lines = [
                    f"VVRmom({i+1}) = {VVmoms[i][0]}",
                    f"VVZmom({i+1}) = {VVmoms[i][1]}",
                ]

                self.contents += "\n".join(lines) + "\n"

    def addUFILES(self, Ufiles, Pich=False, Pech=False, Pnbi=False):
        lines = [
            "\n!=========================================================================",
            f"! Included Input Data Files (Files, coordinate and asymmetry)",
            f"!=========================================================================",
            f"!Note: NRIxxx: -5=sqrt tor flux (rho?), -4=r/a",
            f"",
            f"levtrk	 = 2   ! Limiter locations from lim ufile",
            f"lfixup   = 2   ! Correct U-Files units and axis (=2 according to labels)",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

        if Pich:
            lines = [
                'preRFP	 = "PRF"',
                f'extRFP	 = "RFP"',
            ]
            self.contents += "\n".join(lines) + "\n"

        if Pech:
            lines = [
                'preECP	 = "PRF"',
                f'extECP	 = "ECH"',
                f'preECA	 = "PRF"',
                f'extECA	 = "THE"',
                f'preECB	 = "PRF"',
                f'extECB	 = "PHI"',
            ]
            self.contents += "\n".join(lines) + "\n"

        if Pnbi:
            lines = [
                'preNB2	 = "PRF"',
                f'extNB2	 = "NB2"',
            ]
            self.contents += "\n".join(lines) + "\n"

        for i in Ufiles:
            lines = [
                f'pre{i}	 = "PRF" ',
                f'ext{i}	 = "{Ufiles[i][0]}"',
                f"",
            ]
            self.contents += "\n".join(lines)

            if Ufiles[i][1] is not None:
                lines = [f"nri{i}	 = {Ufiles[i][1]}", f"nsy{i}	 = 0", f""]
                self.contents += "\n".join(lines) + "\n"


class TRANSPnml_Heating:
    def __init__(
        self,
        contents="",
        Pich=False,
        Pech=False,
        Pnbi=False,
        tokamak="SPARC",
        timeStep=0.01,
    ):
        self.contents = contents

        if Pich:
            self.addICRF(tokamak, timeStep=timeStep)
        if Pech:
            self.addECRF(tokamak, Pech=Pech)
        if Pnbi:
            self.addNBI(tokamak, Pnbi=Pnbi, timeStep=timeStep)

    def addICRF(self, tokamak, timeStep=0.001):
        theta_points = int(2**7)  # 128, default: 64
        radial_points = 320  # 320, default: 128

        lines = [
            "!===========================================================",
            f"! Ion Cyclotron Resonance (ICRF)",
            f"!===========================================================",
            f"",
            f"! ----- Main Controls",
            f"",
            f"nlicrf = T   ! Enable ICRF simulation",
            f"nicrf  = 8   ! 8: Modern TORIC",
            f"",
            f"! ----- Resolution",
            f"",
            f"dticrf     = {timeStep}   ! Max time step for TORIC",
            f"NichChi    = {theta_points}   ! Number of poloidal grid points (power of 2)",
            f"NmdToric   = {int(theta_points/2-1)}   ! Number of poloidal modes (recommended: NichChi/2-1)",
            f"NichPsi    = {radial_points}   ! Number of radial grid points",
            f"",
            f"! ----- Model options",
            f"",
            f"ANTLCTR    = 1.6   ! Effective antenna propagation constant",
            f"NFLRTR     = 1	    ! Include ion FLR contribution",
            f"NBPOLTR    = 1     ! Include poloidal magnetic field",
            f"NQTORTR    = 1     ! Include toroidal broadening of the plasma dispersion",
            f"NCOLLTR    = 0     ! Include collisional contribution to  plasma dispersion",
            f"ENHCOLTR   = 1.0   ! Electron collision enhancement factor with NCOLL",
            f"ALFVNTR(1) = 0.0   ! Included (= 1.0) or ignored (= 0.0) collisional broadening of Alfven res",
            f"ALFVNTR(2) = 0.1   ! Enhancement factor (~ 0.1)",
            f"ALFVNTR(3) = 3.0   ! Value of ABS((n//^2 - S)/R) below which damping added",
            f"ALFVNTR(4) = 5.0   ! Value of ABS(w/(k//*v_te)) below which damping calc",
            f"",
        ]
        self.contents += "\n".join(lines) + "\n"

        if tokamak == "SPARC" or tokamak == "ARC":
            from mitim_tools.experiment_tools.SPARCtools import ICRFantennas

            MHz = 120.0
        elif tokamak == "CMOD":
            from mitim_tools.experiment_tools.CMODtools import ICRFantennas

            MHz = 80.0

        contentlines = ICRFantennas(MHz)

        self.contents += contentlines + "\n"

    def addECRF(self, tokamak, Pech=0.0, TORAY=False):
        lines = [
            "!===========================================================",
            f"! Electron Cyclotron Resonance (ECRF)",
            f"!===========================================================",
            f"",
        ]
        self.contents += "\n".join(lines) + "\n"

        if TORAY:
            lines = [
                "",
                f"! ~~~~~~~~~~ TORAY parameters",
                f"",
                f"NLTORAY = T",
                f"",
                f"NPROFTOR = 41",
                f"DTTOR = 0.05",
                f"EFFECH   = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
                f"NRAYECH  = 30, 30, 30, 30, 30, 30, 30, 30",
                f"BSRATECH = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
                f"BHALFECH = 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7",
                f"NDAMPECH = 2, 2, 2, 2, 2, 2, 2, 2",
                f"",
                f"NGAFITTOR = 1",
                f"SMAXTOR = 330.0",
                f"DSTOR = 0.5",
                f"DSMINTOR = 0.25",
                f"FDOUTTOR = 5.0",
                f"POWINCTOR = 1.0e13",
                f"RELERRTOR = 5.0e-5",
                f"ABSERRTOR = 5.0e-5",
                f"NHARMTOR = 2",
                f"MN0TOR = 32",
                f"NGZNTOR = 4",
                f"MRAYTOR = 1, 5, 12, 12, 20, 24, 26",
                f"CRTOR = 0.0, 0.1, 0.05, -0.05, 0.05, -0.025, 0.025",
                f"MODELCTOR = 4",
                f"PWRFMNTOR = 0.001",
                f"NLCQLDAT = .F.  ",
                f"",
            ]
        else:
            lines = [
                "",
                f"! ~~~~~~~~~~ TORBEAM Parameters",
                f"",
                f"nltorbeam 		   = T",
                f"nltorbeam_verbose    = T ! Generate extra files",
                f"",
                f"NTOB_NPROFVW    = 25",
                f"NTOB_MMAX 	   = 150",
                f"TOB_XZEFF 	   = -1",
                f"",
            ]

        self.contents += "\n".join(lines) + "\n"

        if tokamak == "AUG":
            from mitim_tools.experiment_tools.AUGtools import ECRFgyrotrons
        if tokamak == "D3D":
            from mitim_tools.experiment_tools.DIIIDtools import ECRFgyrotrons

        contentlines = ECRFgyrotrons()
        self.contents += contentlines + "\n"

    def addNBI(self, tokamak, Pnbi=0.0, timeStep=0.01):
        lines = [
            "!===============================================",
            f"!           NEUTRAL BEAMS                      !",
            f"!===============================================",
            f"",
            f"nlbeam   = T 		! NUBEAM model on",
            f"nlbfpp   = F 		! FPP model on",
            f"dtbeam   = {timeStep}		! Beam time-step",
            f"",
            f"lev_nbidep = 2 	! Ground state dep model (2=ADAS atomic physics)",
            f"nsigexc     = 1		! Simple excited states deposition model",
            f"",
            f"nlbdat = T 		! Time-dependent UFILE with Power",
            f"",
            f"! ~~ Model settings" f"",
            f"nlbflr   = T 		! Force plasma interactions on the gyro",
            f"nlbeamcx = T",
            f"nptcls  = 50000	! Number of Monte Carlo ions",
            f"dxbsmoo = 0.05		! Radial smoothing half-width (in r/a space)",
            f"ndep0  = 500		! Min. number of deposition tracks to trace per step",
            f"goocon = 10.		! Numeric goosing (10 is default precision)",
            f"goomax = 500.",
            f"ebdmax = 150E+03 	! Max energy tracked in distribution function",
            f"nlfbon = F 		! Fishbone losses",
            f"nlbout = F",
            f"",
        ]
        self.contents += "\n".join(lines) + "\n"

        if tokamak == "AUG":
            from mitim_tools.experiment_tools.AUGtools import NBIbeams
        if tokamak == "D3D":
            from mitim_tools.experiment_tools.DIIIDtools import NBIbeams

        contentlines = NBIbeams()
        self.contents += contentlines + "\n"


class TRANSPnml_Predictive:
    def __init__(
        self,
        contents,
        xbounds=[0.95, 0.95, 0.95],
        xminTrick=0.2,
        grTGLF=False,
        Te_edge=80.0,
        Ti_edge=80.0,
        TGLFsettings=5,
    ):
        self.contents = contents
        self.contents_ptr_glf23 = None
        self.contents_ptr_tglf = None
        self.contents_ptr_ptsolver = None

        self.addPedestal(Te_edge=Te_edge, Ti_edge=Ti_edge)
        self.addPredictive(xbounds, xminTrick=xminTrick)
        self.addGLF23()
        self.addTGLF(TGLFsettings, grTGLF=grTGLF)

    def addPredictive(self, xbounds, xminTrick=0.2, xmaxTrick=1.0):
        lines = [
            "!==================================================================",
            f"! Predictive TRANSP",
            f"!==================================================================",
            f"",
            f"lpredictive_mode = 3   ! Use PT_SOLVER",
            f"pt_template      = 'tmp_folder/ptsolver_namelist.dat'",
            f"",
            f"!------ Equation selection",
            f"",
            f"lpredict_te      = 0    ! Predict Te",
            f"lpredict_ti      = 0    ! Predict Ti",
            f"lpredict_pphi    = 0    ! Predict Rotation",
            f"lpredict_ne      = 0    ! Predict ne",
            f"lpredict_nmain   = 0    ! Predict n_main",
            f"lpredict_nimp    = 0    ! Predict n_imp",
            f"",
            f"!------ Region boundaries (in rho)",
            f"",
            f"ximin_conf = {xminTrick:.3f}	 ! Max rho for AXIAL",
            f"ximax_conf = {xmaxTrick:.3f}   ! Min rho for EDGE (after this, EDGE until X*BOUND)",
            f"",
            f"!------ Boundary conditions (rho > x*bound -> experimental)",
            f"",
            f"xibound    = {xbounds[0]:.3f}	 ! Te, Ti",
            f"xnbound    = {xbounds[1]:.3f}	 ! ne, ni",
            f"xphibound  = {xbounds[2]:.3f}  ! rotation",
            f"",
            f'!------ "USER" transport model',
            f"",
            f"nchan     =      1,      1,    0,    0,    0,    0      ! Channels to implement axial values (Te, Ti, Phi, ni, nimp, ne)",
            f"user_chi0 =  0.1e4,  0.1e4,  0.0,  0.0,  0.0,  0.0      ! Diffusivities on axis (cm^2/s)",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

        lines = [
            "&pt_controls",
            f" ",
            f" !------ Solver Options",
            f" ",
            f" newton_iterates       = 1000 ! Number of newton iterations (default = 10)",
            f" theta_implicit        = -1   ! Implicitness param in time diff (negative = exponential average method)",
            f" !xrelfac              = 0.1  ! Relaxation factor to mix diffusion coefficients",
            f" ",
            f" !------ Residuals",
            f" ",
            f" pt_residual%res_te       = 1D-3",
            f" pt_residual%res_ti       = 1D-3",
            f" pt_residual%res_ne       = 1D-3",
            f" pt_residual%res_nimp     = 1D-3",
            f" pt_residual%res_nmain    = 1D-3",
            f" pt_residual%res_pphi     = 1D-3",
            f" ",
            f" !------ Numerical diffusivities (factors and minimum values for the Peclet numbers)"
            f" ",
            f" pt_num_diffusivity%pt_chie_fact   = 500D0",
            f" pt_num_diffusivity%pt_chie_min    = 100D0",
            f" pt_num_diffusivity%pt_chii_fact   = 100D0",
            f" pt_num_diffusivity%pt_chii_min    = 100D0",
            f" pt_num_diffusivity%pt_chine_fact  = 200D0",
            f" pt_num_diffusivity%pt_chine_min   = 50D0",
            f" pt_num_diffusivity%pt_chiphi_fact = 200D0",
            f" pt_num_diffusivity%pt_chiphi_min  = 50D0",
            f" ",
            f" !------ ExB flow shear model: 1 (NCEXB), 2 (DMEXB), 3 (TGYROEXB), 4 (TRANSPEXB)",
            f" ",
            f" pt_axial%exb%version        = 3",
            f" pt_confinement%exb%version  = 3",
            f" pt_edge%exb%version         = 3",
            f" ",
            f" !------ Transport models for each region",
            f" ",
            f" ! ~~~ GLF23",
            f" pt_axial%glf23%active          = T",
            f" pt_axial%glf23%xanom           = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  ! Te, Ti, Pphi, Ni, Nz, Ne",
            f" ",
            f" pt_confinement%glf23%active    = T",
            f" pt_confinement%glf23%xanom     = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            f" ",
            f" pt_edge%glf23%active           = T",
            f" pt_edge%glf23%xanom            = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            f"",
            f" ! ~~~ TGLF",
            f" pt_axial%tglf%active           = F",
            f" pt_axial%tglf%xanom            = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  ! Te, Ti, Pphi, Ni, Nz, Ne",
            f" ",
            f" pt_confinement%tglf%active     = F",
            f" pt_confinement%tglf%xanom      = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            f" ",
            f" pt_edge%tglf%active            = F",
            f" pt_edge%tglf%xanom             = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            f"",
            f" ! ~~~ User",
            f" pt_axial%user%active           = F",
            f" pt_axial%user%xanom            = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  ! Te, Ti, Pphi, Ni, Nz, Ne",
            f" ",
            f" pt_confinement%user%active     = F",
            f" pt_confinement%user%xanom      = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            f" ",
            f" pt_edge%user%active            = F",
            f" pt_edge%user%xanom             = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            f" ",
            f" ! ~~~ Neoclassical: Chang-Hilton model",
            f" pt_axial%neoch%active          = T",
            f" pt_axial%neoch%xanom           = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            f" ",
            f" pt_confinement%neoch%active    = T",
            f" pt_confinement%neoch%xanom     = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            f" ",
            f" pt_edge%neoch%active           = T",
            f" pt_edge%neoch%xanom            = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0",
            f" ",
            f" !------ Sawtooth model (for interchange instability)",
            f" ",
            f" !pt_sawtooth%active         = T                               ! Activate PTSAW",
            f" !pt_sawtooth%xanom          = 1e3, 1e3, 1e3, 1e3, 1e3, 1e3    ! Transport enhancement factor (Te, Ti, Pphi, Ni, Nz, Ne)",
            f" !pt_sawtooth%model          = 1                               ! 1 (q<1), 2 (interchange; TGLF only), 3 (both; TGLF only), 4 (USER)",
            f" !pt_sawtooth%xsaw_bound     =                                 ! Sawtooth radius (if model = 4)",
            f" ",
        ]

        # lines.append('/') # I do it later, in defineRunParams

        self.contents_ptr_ptsolver = "\n".join(lines) + "\n"

    def addPedestal(self, Te_edge=80.0, Ti_edge=80.0):
        lines = [
            "!===============================================================",
            f"! Pedestal and Edge Model -> Boundary condition for X > XBOUND",
            f"!===============================================================",
            f"",
            f"!------ General: Te, Ti, T0, Rotation",
            f"",
            f"modeedg  = 3        ! 2 = Use TEEDGE, 3 = Exp Te, 5 = NTCC ",
            f"teedge   = {Te_edge}     ! Specify Te (eV) if MODEEDG = 2",
            f"teped    = 3050     ! Electron pedestal temperature in eV",
            f"tepedw   = -0.0591  ! Electron pedestal width in cm(+) or x(-)",
            f"",
            f"modiedg  = 4        ! 1 = Same as neutrals, 2 = TIEDGE, 3 = Exp Te, 4 = Exp Ti, 5 = NTC ",
            f"tiedge   = {Ti_edge}     ! Specify Ti (eV) if MODIEDG = 2 and for FRANTIC if MOD0ED=1",
            f"tiped    = 3050     ! Ion pedestal temperature in eV",
            f"tipedw   = -0.0643  ! Ion pedestal width in cm(+) or x(-)",
            f"",
            f"modnedg  = 3",
            f"xneped   = 3.65E14  ! Electron pedestal density in cm^-3",
            f"xnepedw  = -0.0591  ! Electron density pedestal width in cm(+) or x(-)",
            f"",
            f"modomedg = 3        ! Rotation, 2 = use OMEDG (constant), 3 = use OMEGA input data",
            f"omedge   = 1E4      !",
            f"",
            f"nmodel_ped_height = 0 ! Ped Height model choice  (1=NTCC, 0=namelist)",
            f"nmodel_ped_width  = 0 ! Ped Width model choice  (1=NTCC, 0=namelist)",
            f"",
            f"!------ L-mode, H-mode transitions",
            f"",
            f"nmodel_l2h_trans = 1     ! Model choice (1=NTCC (exp if not enough power), 0=namelist)",
            f"!time_l2h        = 2.0",
            f"!time_h2l        = 3.0",
            f"!tau_lh_trans_x2 = 2.0",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

    def addGLF23(self):
        lines = [
            f"!------ GLF23 namelist",
            f"",
            "glf23_template = 'tmp_folder/glf23_namelist.dat'",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

        lines = [
            f"&glf23_control",
            f" ",
            f" version                    = 2       ! 1 - original; 2 - v1.61; 3 - renormalized and real geometry fit",
            f" lprint                     = 0       ! get output files with lprint=1",
            f" use_adiabatic_electrons    = F",
            f" bt_flag                    = 1       ! 0 - use B_T; 1 - use Bt_eff",
            f" alpha_p_mult               = 1.0",
            f" alpha_quench_mult          = 1.0",
            f" cxnu                       = 1.0     ! scale factor for collisionality",
            f" cbetae                     = 1.0     ! factor for plasma beta",
            f" ",
        ]

        # lines.append('/')#I do it later, in defineRunParams

        self.contents_ptr_glf23 = "\n".join(lines) + "\n"

    def addTGLF(self, TGLFsettings, grTGLF=False):
        TGLFoptions, label = GACODEdefaults.TGLFinTRANSP(TGLFsettings)
        print(
            "\t- Adding TGLF control parameters with TGLFsettings = {0} ({1})".format(
                TGLFsettings, label
            )
        )

        lines = [
            f"!------ TGLF namelist",
            f"",
            f"tglf_template   =   'tmp_folder/tglf_namelist.dat'",
            f"nky             =   {TGLFoptions['NKY']}",
            f"nlgrowth_tglf   =   {grTGLF}   ! Output TGLF information (growth rates, freqs...)",
            f"",
        ]

        self.contents += "\n".join(lines) + "\n"

        lines = ["&tglf_control"]
        for ikey in TGLFoptions:
            if ikey not in ["NKY"]:
                lines.append(f" {ikey.ljust(21)} = {TGLFoptions[ikey]}")
        # lines.append('/')# I do it later, in defineRunParams

        self.contents_ptr_tglf = "\n".join(lines) + "\n"


def changeNamelist(
    namelistPath, nameBaseShot, TRANSPnamelist, FolderTRANSP, outtims=[]
):
    # Change shot number
    IOtools.changeValue(
        namelistPath, "nshot", nameBaseShot, None, "=", MaintainComments=True
    )

    # TRANSP fixed namelist + those parameters changed
    for itag in TRANSPnamelist:
        IOtools.changeValue(
            namelistPath, itag, TRANSPnamelist[itag], None, "=", MaintainComments=True
        )

    # Add inputdir to namelist
    with open(namelistPath, "a") as f:
        f.write("inputdir='" + os.path.abspath(FolderTRANSP) + "'\n")

    # Change PTR templates
    IOtools.changeValue(
        namelistPath,
        "pt_template",
        f'"{os.path.abspath(FolderTRANSP)}/ptsolver_namelist.dat"',
        None,
        "=",
        CommentChar=None,
    )
    IOtools.changeValue(
        namelistPath,
        "tglf_template",
        f'"{os.path.abspath(FolderTRANSP)}/tglf_namelist.dat"',
        None,
        "=",
        CommentChar=None,
    )
    IOtools.changeValue(
        namelistPath,
        "glf23_template",
        f'"{os.path.abspath(FolderTRANSP)}/glf23_namelist.dat"',
        None,
        "=",
        CommentChar=None,
    )

    # Add outtims
    if len(outtims) > 0:
        addOUTtimes(namelistPath, outtims)


def addOUTtimes(namelistPath, outtims, differenceBetween=0.0):
    strNBI, strTOR, strECH = "", "", ""
    for time in outtims:
        print(f"\t- Adding time (t={time:.3f}s) to output intermediate files (NUBEAM)")
        strNBI += f"{time:.3f},"

        print(f"\t- Adding time (t={time:.3f}s) to output intermediate files (TORIC)")
        strTOR += f"{time - differenceBetween:.3f},"

        print(f"\t- Adding time (t={time:.3f}s) to output intermediate files (TORBEAM)")
        strECH += f"{time - differenceBetween:.3f},"

    strNBI = strNBI[:-1]
    strTOR = strTOR[:-1]
    strECH = strECH[:-1]

    avgtim = 0.05  # Average before

    IOtools.changeValue(namelistPath, "mthdavg", 2, None, "=", MaintainComments=True)
    IOtools.changeValue(
        namelistPath, "avgtim", avgtim, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "outtim", strNBI, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "fi_outtim", strTOR, None, "=", MaintainComments=True
    )
    IOtools.changeValue(
        namelistPath, "fe_outtim", strECH, None, "=", MaintainComments=True
    )

    # To get birth deposition:
    IOtools.changeValue(
        namelistPath, "nldep0_gather", "T", None, "=", MaintainComments=True
    )


def appendUpdates(
    namelistPath,
    namelistPath_pt,
    TRANSPnamelist,
    MITIMparams,
    timeStartPrediction,
    TransportModel,
    useAxialTrick=True,
    timeLagDensity=0.0,
    predictQuantities=["ne", "te", "ti"],
    pedestalPred=False,
    rotationPred=False,
):
    densityTerms = []
    if "ne" in predictQuantities:
        densityTerms = ["ne"]
    TemperatureTerms = []
    if "te" in predictQuantities:
        TemperatureTerms.append("te")
    if "ti" in predictQuantities:
        TemperatureTerms.append("ti")

    # _________________________Compound axial trick________________________________

    Trick_Te, Trick_Ti, Trick_ne = MITIMparams["AxialDeffs"]

    useAxialTrick = useAxialTrick and (
        Trick_Te is not None or Trick_Ti is not None or Trick_ne is not None
    )

    Trick_ne_ptr = ""
    if Trick_ne is not None:
        Trick_ne_ptr = f", D_e={Trick_ne}"

    if useAxialTrick:
        print(
            f"\t- Effective diffusivities adjusted on axis: chi_e={Trick_Te*1E-4:.1f}m^2/s, chi_i={Trick_Ti*1E-4:.1f}m^2/s{Trick_ne_ptr}"
        )

        nchan_array, chi0_array = "", ""
        if Trick_Te is not None:
            nchan_array += "1,"
            chi0_array += f"{str(Trick_Te)},"
        else:
            nchan_array += "0,"
            chi0_array += "0,"
        if Trick_Ti is not None:
            nchan_array += "1,"
            chi0_array += f"{str(Trick_Ti)},"
        else:
            nchan_array += "0,"
            chi0_array += "0,"
        nchan_array += "0,0,0,"
        chi0_array += "0.0,0.0,0.0,"
        if Trick_ne is not None:
            nchan_array += "1"
            chi0_array += f"{str(Trick_ne)}"
        else:
            nchan_array += "0"
            chi0_array += "0"

        # if Trick_ne is not None: axialTrick += 'XNE_NC_AXIAL   = 0.0\n'
    # ______________________________________________________________________________

    with open(namelistPath, "a") as f:
        # ~~~~~ Temperature prediction
        f.write("\n! ================== UPDATES ==================\n\n")
        f.write(f"~update_time={timeStartPrediction}\n")
        for temperatureTerm in TemperatureTerms:
            f.write(f"lpredict_{temperatureTerm}=1\n")

        # ~~~~~ Density prediction
        if len(densityTerms) > 0:
            if timeLagDensity > 0.0:
                f.write(f"~update_time={timeStartPrediction+timeLagDensity}\n")
            if timeLagDensity > -1e-5:
                for densityTerm in densityTerms:
                    f.write(f"lpredict_{densityTerm}=1\n")

        # ~~~~~ Rotation prediction

        if rotationPred:
            timeLagRotation = 0.1

            f.write(f"~update_time={timeStartPrediction+timeLagRotation}\n")
            f.write("lpredict_pphi=1\n")

        if pedestalPred:
            f.write(f"~update_time={timeStartPrediction}\n")
            f.write("nmodel_ped_height = 1\n")
            f.write("nmodel_ped_width = 1\n")

    # Modify pt_namliest

    if TransportModel == "tglf":
        IOtools.changeValue(
            namelistPath_pt,
            "pt_axial%glf23%active",
            False,
            None,
            "=",
            MaintainComments=True,
        )
        IOtools.changeValue(
            namelistPath_pt,
            "pt_confinement%glf23%active",
            False,
            None,
            "=",
            MaintainComments=True,
        )
        IOtools.changeValue(
            namelistPath_pt,
            "pt_edge%glf23%active",
            False,
            None,
            "=",
            MaintainComments=True,
        )

        IOtools.changeValue(
            namelistPath_pt,
            "pt_axial%tglf%active",
            True,
            None,
            "=",
            MaintainComments=True,
        )
        IOtools.changeValue(
            namelistPath_pt,
            "pt_confinement%tglf%active",
            True,
            None,
            "=",
            MaintainComments=True,
        )
        IOtools.changeValue(
            namelistPath_pt,
            "pt_edge%tglf%active",
            True,
            None,
            "=",
            MaintainComments=True,
        )

    if useAxialTrick:
        IOtools.changeValue(
            namelistPath_pt,
            "pt_axial%glf23%active",
            False,
            None,
            "=",
            MaintainComments=True,
        )
        IOtools.changeValue(
            namelistPath_pt,
            "pt_axial%tglf%active",
            False,
            None,
            "=",
            MaintainComments=True,
        )

        IOtools.changeValue(
            namelistPath_pt,
            "pt_axial%user%active",
            True,
            None,
            "=",
            MaintainComments=True,
        )

        IOtools.changeValue(
            namelistPath, "nchan", nchan_array, None, "=", MaintainComments=True
        )
        IOtools.changeValue(
            namelistPath, "user_chi0", chi0_array, None, "=", MaintainComments=True
        )


def addParametersBeforeUpdate(file, TRANSPvars):
    with open(file, "r") as f:
        lines = f.readlines()

    lines_n = ""
    done = False
    for i in lines:
        if "~update" in i and not done:
            for ikey in TRANSPvars:
                lines_n += f"{ikey}={TRANSPvars[ikey]}\n"
            done = True
        lines_n += i

    with open(file, "w") as f:
        f.write(lines_n)

    print(f"--> Modified file {file}")
