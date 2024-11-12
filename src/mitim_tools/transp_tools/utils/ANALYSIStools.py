import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

from mitim_tools.misc_tools import GRAPHICStools, MATHtools, IOtools
from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as extrap
from mitim_tools.misc_tools.MATHtools import integrate_definite as integra


def plotFullCurrentDynamics(c, t1, t2, tmargin=0.0, fig=None, rho_plot_lim=None):
    if fig is None:
        fig = plt.figure()

    grid = plt.GridSpec(3, 3, hspace=0.3, wspace=0.3)
    ax00 = fig.add_subplot(grid[0, 0])
    ax01 = fig.add_subplot(grid[0, 1])
    ax02 = fig.add_subplot(grid[0, 2])
    ax10 = fig.add_subplot(grid[1, 0])
    ax11 = fig.add_subplot(grid[1, 1])
    ax12 = fig.add_subplot(grid[1, 2])
    ax20 = fig.add_subplot(grid[2, 0])
    ax21 = fig.add_subplot(grid[2, 1])
    ax22 = fig.add_subplot(grid[2, 2])

    it_saw = np.argmin(np.abs(c.t - t1))
    it1 = it_saw + 1
    it2 = np.argmin(np.abs(c.t - t2))

    it1m = np.argmin(np.abs(c.t - (t1 - tmargin))) + 1
    it2m = np.argmin(np.abs(c.t - (t2 + tmargin)))

    itBig = []  # it2,it1]

    lw, alpha = 0.5, 0.5

    # ----------------------------------------------------------------------
    # All radii
    # ----------------------------------------------------------------------

    if rho_plot_lim is not None:
        ix1 = np.argmin(np.abs(c.x_lw - rho_plot_lim))
        ixb1 = np.argmin(np.abs(c.xb_lw - rho_plot_lim))
    else:
        ixb1 = ix1 = -1

    ax = ax00
    cols = GRAPHICStools.plotRange(
        c.t,
        c.xb[:, :ixb1],
        c.q[:, :ixb1],
        ax=ax,
        it1=it1,
        it2=it2,
        itBig=itBig,
        colors=["b", "r"],
        colorsBig=["b", "r"],
        lw=lw,
        alpha=alpha,
    )
    ax.set_xlabel("$\\rho_N$")
    ax.set_xlim([0, 1])
    ax.set_ylabel("$q$")
    ax.axhline(y=1, ls="--", lw=1.0, c="k")
    GRAPHICStools.addDenseAxis(ax)

    ax.axvline(x=c.x_saw_inv[it_saw], ls="-.", lw=1.0, c="g")
    ax.axvline(x=c.x_saw_mix[it_saw], ls="-.", lw=1.0, c="c")

    ax = ax10
    _ = GRAPHICStools.plotRange(
        c.t,
        c.x,
        c.V,
        ax=ax,
        it1=it1,
        it2=it2,
        itBig=itBig,
        colors=["b", "r"],
        colorsBig=["b", "r"],
        lw=lw,
        alpha=alpha,
    )
    ax.set_xlabel("$\\rho_N$")
    ax.set_xlim([0, 1])
    ax.set_ylabel("$V$")
    GRAPHICStools.addDenseAxis(ax)

    ax.axvline(x=c.x_saw_inv[it_saw], ls="-.", lw=1.0, c="g")
    ax.axvline(x=c.x_saw_mix[it_saw], ls="-.", lw=1.0, c="c")

    ax = ax20
    _ = GRAPHICStools.plotRange(
        c.t,
        c.x,
        c.j,
        ax=ax,
        it1=it1,
        it2=it2,
        itBig=itBig,
        colors=["b", "r"],
        colorsBig=["b", "r"],
        lw=lw,
        alpha=alpha,
    )
    ax.set_xlabel("$\\rho_N$")
    ax.set_xlim([0, 1])
    ax.set_ylabel("$J$")
    GRAPHICStools.addDenseAxis(ax)

    ax.axvline(x=c.x_saw_inv[it_saw], ls="-.", lw=1.0, c="g")
    ax.axvline(x=c.x_saw_mix[it_saw], ls="-.", lw=1.0, c="c")

    ax = ax01
    _ = GRAPHICStools.plotRange(
        c.t,
        c.x,
        c.Poh,
        ax=ax,
        it1=it1,
        it2=it2,
        itBig=itBig,
        colors=["b", "r"],
        colorsBig=["b", "r"],
        lw=lw,
        alpha=alpha,
    )
    ax.set_xlabel("$\\rho_N$")
    ax.set_xlim([0, 1])
    ax.set_ylabel("$P_{OH}$")
    GRAPHICStools.addDenseAxis(ax)

    ax.axvline(x=c.x_saw_inv[it_saw], ls="-.", lw=1.0, c="g")
    ax.axvline(x=c.x_saw_mix[it_saw], ls="-.", lw=1.0, c="c")

    # ax = ax11
    # _ = GRAPHICStools.plotRange(c.t,c.x,c.eta,ax=ax,it1=it1,it2=it2,itBig=itBig,
    # 	colors=['b','r'],colorsBig=['b','r'],lw=lw,alpha=alpha)
    # ax.set_xlabel('$\\rho_N$')
    # ax.set_ylabel('$\\eta$')
    # ax.set_yscale('log')

    ax = ax11
    _ = GRAPHICStools.plotRange(
        c.t,
        c.x[:, :ix1],
        c.psi_heli[:, :ix1],
        ax=ax,
        it1=it1,
        it2=it2,
        itBig=itBig,
        colors=["b", "r"],
        colorsBig=["b", "r"],
        lw=lw,
        alpha=alpha,
    )
    ax.set_xlabel("$\\rho_N$")
    ax.set_xlim([0, 1])
    ax.set_ylabel("$\\psi^*$ (Wb/rad)")
    GRAPHICStools.addDenseAxis(ax)
    ax.axhline(y=0.0, ls="--", lw=1.0, c="k")

    ax.axvline(x=c.x_saw_inv[it_saw], ls="-.", lw=1.0, c="g")
    ax.axvline(x=c.x_saw_mix[it_saw], ls="-.", lw=1.0, c="c")

    ax = ax21
    _ = GRAPHICStools.plotRange(
        c.t,
        c.x,
        c.Umag_pol,
        ax=ax,
        it1=it1,
        it2=it2,
        itBig=itBig,
        colors=["b", "r"],
        colorsBig=["b", "r"],
        lw=lw,
        alpha=alpha,
    )
    ax.set_xlabel("$\\rho_N$")
    ax.set_xlim([0, 1])
    ax.set_ylabel("$U_{\\theta}$")
    GRAPHICStools.addDenseAxis(ax)

    ax.axvline(x=c.x_saw_inv[it_saw], ls="-.", lw=1.0, c="g")
    ax.axvline(x=c.x_saw_mix[it_saw], ls="-.", lw=1.0, c="c")

    # ----------------------------------------------------------------------
    # Times
    # ----------------------------------------------------------------------

    ax = ax02
    ax.plot(c.t[it1m:it2m], c.q[it1m:it2m, -1], "-o", lw=1, c="g", markersize=3)
    ax.scatter(
        c.t[it1:it2], c.q[it1:it2, -1], color=cols
    )  # ,'-o',lw=1,c='m',markersize=3)

    ax.plot(c.t[it1m:it2m], c.q[it1m:it2m, 0], "-o", lw=1, c="m", markersize=3)
    ax.scatter(
        c.t[it1:it2], c.q[it1:it2, 0], color=cols
    )  # ,'-o',lw=1,c='m',markersize=3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$q_0$, $q_1$")
    GRAPHICStools.addDenseAxis(ax)

    ax = ax12
    ax.plot(c.t[it1m:it2m], c.V[it1m:it2m, -1], "-o", lw=1, c="g", markersize=3)
    ax.scatter(
        c.t[it1:it2], c.V[it1:it2, -1], color=cols
    )  # ,'-o',lw=1,c='m',markersize=3)
    ax.plot(
        c.t[it1:it2], c.Vsurf[it1:it2], "-o", lw=1, c="c", markersize=2, label="$V$"
    )
    ax.plot(
        c.t[it1:it2], c.Vsurf_m[it1:it2], "-o", lw=1, c="k", markersize=2, label="$V_m$"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$V_1$")
    ax.legend()
    GRAPHICStools.addDenseAxis(ax)

    ax = ax22
    ax.plot(c.t[it1m:it2m], c.UmagT_pol[it1m:it2m], "-o", lw=1, c="g", markersize=3)
    ax.scatter(
        c.t[it1:it2], c.UmagT_pol[it1:it2], color=cols
    )  # ,'-o',lw=1,c='m',markersize=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$\\int U_\\theta$")
    ax.set_ylim(bottom=0)
    GRAPHICStools.addDenseAxis(ax)

    # # ----------------------------------------------------------------------
    # # Zoom-in
    # # ----------------------------------------------------------------------
    #
    # rhomin = 0.95
    # ix1 = np.argmin(np.abs(c.x_lw-rhomin))
    # ix2 = -1 #np.argmin(np.abs(c.x_lw-0.9))
    #
    #
    # ax = ax[2,2]
    # GRAPHICStools.plotRange(c.t,c.x[:,ix1:ix2],c.q[:,ix1:ix2],ax=ax,it1=it1,it2=it2,itBig=itBig,
    # 	colors=['b','r'],colorsBig=['r'],lw=lw,alpha=alpha)
    # ax.set_xlabel('$\\rho_N$')
    # ax.set_ylabel('$q$')


class Porcelli:
    def __init__(self, transp, resolution=1000):
        print(" ----->> Calculating internal parameters of Porcelli model...")

        from transp_tools import CDFtools as CDFt

        self.r1 = np.zeros(len(transp.t))
        self.r1av = np.zeros(len(transp.t))
        self.aav = np.zeros(len(transp.t))
        self.eps1 = np.zeros(len(transp.t))
        self.s1 = np.zeros(len(transp.t))
        self.s1_por = np.zeros(len(transp.t))
        self.s1_prf = np.zeros(len(transp.t))
        self.s1 = np.zeros(len(transp.t))
        self.BpC = np.zeros(len(transp.t))
        self.li1 = np.zeros(len(transp.t))
        self.Cmhd = np.zeros(len(transp.t))
        self.Bp1 = np.zeros(len(transp.t))
        self.Bp1s = np.zeros(len(transp.t))
        self.dWBussac = np.zeros(len(transp.t))
        self.Bpol1 = np.zeros(len(transp.t))
        self.Bpolsq1 = np.zeros(len(transp.t))
        self.p1 = np.zeros(len(transp.t))
        self.p1av = np.zeros(len(transp.t))
        self.k1 = np.zeros(len(transp.t))
        self.cel = np.zeros(len(transp.t))
        self.dWel = np.zeros(len(transp.t))
        self.cp = np.zeros(len(transp.t))
        self.dWko = np.zeros(len(transp.t))
        self.bi0 = np.zeros(len(transp.t))
        self.dWmhd = np.zeros(len(transp.t))
        self.dWcore = np.zeros(len(transp.t))
        self.A1 = np.zeros(len(transp.t))
        self.li1_check = np.zeros(len(transp.t))
        self.Bp1av = np.zeros(len(transp.t))
        self.wDa = np.zeros(len(transp.t))
        self.tA = np.zeros(len(transp.t))
        self.chtAwDa = np.zeros(len(transp.t))
        self.Efast = np.zeros(len(transp.t))
        self.Bfast = np.zeros(len(transp.t))
        self.dWfast = np.zeros(len(transp.t))
        self.Bfast_tot = np.zeros(len(transp.t))
        self.dWfast_tot = np.zeros(len(transp.t))

        # for it in [np.argmin(np.abs(transp.t-5.0))]:
        for it in range(len(transp.t)):
            # ix = np.argmin(np.abs(transp.q[it]-1.0))
            # ix = np.argmin(np.abs(transp.x_saw_inv[it]-transp.x_lw))
            # ix = np.argmin(np.abs(transp.rmin[it]-self.rq1[it]))

            x = transp.porcelli_rq1[it]
            xgridlong = np.linspace(transp.rmin[it, 0], transp.rmin[it, 1], resolution)
            xgridlong_red = np.linspace(transp.rmin[it, 0], x, resolution)

            # ===========================================================================
            # 		Parameters from TRANSP

            self.r1[it] = x
            self.k1[it] = extrap(x, transp.rmin[it], transp.kappaS[it])
            self.s1_prf[it] = (
                extrap(x, transp.rmin[it], transp.shat[it]) / transp.a[it] * self.r1[it]
            )
            self.s1_por[it] = transp.porcelli_s1[it]

            p = extrap(xgridlong_red, transp.rmin[it], transp.p_kin[it]) * 1e6
            dvol = extrap(xgridlong_red, transp.rmin[it], transp.f["DVOL"][:][it])
            self.p1av[it] = np.sum(dvol * p) / np.sum(dvol)
            # self.p1av[it] 		= volumeAverage_var(transp.f,transp.p_kin*1E6,rangex=ix)[it]

            self.p1[it] = extrap(x, transp.rmin[it], transp.p_kin[it]) * 1e6
            self.Bpol1[it] = extrap(
                x, transp.rmin[it], transp.Bp_LFx[it]
            )  # in Bateman implementation it says at outboard midplane
            self.bi0[it] = (
                transp.p_i[it, 0] * 1e6 / (transp.Bt[it] ** 2 / (2 * transp.mu0))
            )
            self.A1[it] = extrap(
                x,
                transp.rmin[it],
                CDFt.surfaceIntegral_var(
                    transp.f, np.ones([len(transp.t), len(transp.x_lw)])
                )[it],
            )
            self.Efast[it] = extrap(x, transp.rmin[it], transp.Efast[it])

            # Derived
            self.r1av[it] = np.sqrt(self.k1[it]) * self.r1[it]
            self.aav[it] = (
                np.sqrt(self.k1[it]) * transp.a[it]
            )  # Using k at q=1 instead separatrix because of Bateman implementation
            self.eps1[it] = self.r1av[it] / transp.Rmajor[it]
            self.s1[it] = np.sqrt(
                self.s1_por[it] ** 2 + 0.1**2
            )  # Normalization in Bateman implementation

            # ===========================================================================

            # ~~~~~~~ Bussac Term ~~~~~~~
            self.BpC[it] = 0.3 * (1 - 5 / 3 * self.r1av[it] / self.aav[it])
            self.Bp1av[it] = self.p1av[it] / (self.Bpol1[it] ** 2 / (2 * transp.mu0))
            self.Bp1[it] = self.p1[it] / (self.Bpol1[it] ** 2 / (2 * transp.mu0))
            self.Bp1s[it] = self.Bp1av[it] - self.Bp1[it]

            self.li1[it] = extrap(
                x,
                transp.rmin[it],
                CDFt.surfaceIntegral_var(transp.f, transp.Bp_LFx**2)[it],
            ) / (self.A1[it] * self.Bpol1[it] ** 2)
            # self.li1_check[it]  = 0.5 + (1-transp.q0[it])/3
            self.Cmhd[it] = 9 * np.pi / self.s1[it] * (self.li1[it] - 1 / 2)

            self.dWBussac[it] = (
                -self.Cmhd[it]
                * self.eps1[it] ** 2
                * (self.Bp1s[it] ** 2 - self.BpC[it] ** 2)
            )

            # ~~~~~~~ Elongation term ~~~~~~~
            self.cel[it] = 18 * np.pi / self.s1[it] * (self.li1[it] - 1 / 2) ** 3
            self.dWel[it] = -self.cel[it] * ((self.k1[it] - 1) / 2) ** 2

            # ~~~~~~~ Kruskal-Oberman term ~~~~~~~
            y = extrap(
                xgridlong_red, transp.rmin[it], transp.p_i[it] / transp.p_i[it, 0]
            )
            x = extrap(
                xgridlong_red,
                transp.rmin[it],
                transp.rmin[it] * np.sqrt(transp.kappaS[it]) / self.r1av[it],
            )
            integral01 = integra(x, x**1.5 * y)

            self.cp[it] = 5 / 2 * integral01
            self.dWko[it] = (
                0.6 * self.cp[it] * np.sqrt(self.eps1[it]) * self.bi0[it] / self.s1[it]
            )

            # Core
            self.dWmhd[it] = self.dWBussac[it] + self.dWel[it]
            self.dWcore[it] = self.dWmhd[it] + self.dWko[it]

            # ~~~~~~~ Fast ion stabilization ~~~~~~~

            self.tA[it] = (
                0.8e-6
                * transp.Rmajor[it]
                / transp.Bt[it]
                * np.sqrt(2.5 * transp.ne[it, 0])
            )
            self.wDa[it] = (
                500
                * self.Efast[it]
                / (2 * transp.Bt[it] * transp.Rmajor[it] * self.r1av[it])
            )
            self.chtAwDa[it] = 0.4 * self.tA[it] * self.wDa[it]

            # ~~~~~~~ dWfast term ~~~~~~~

            p = extrap(xgridlong_red, transp.rmin[it], transp.pFast_fus[it])
            y = (
                CDFt.derivativeVar(
                    self, p, specialDerivative=xgridlong_red, onlyOneTime=True
                )
                * 1e6
                * self.r1[it]
            )
            x = extrap(xgridlong_red, transp.rmin[it], transp.rmin[it] / self.r1[it])

            integral01 = integra(x, x**1.5 * y)
            self.Bfast[it] = -integral01 / (self.Bpol1[it] ** 2 / (2 * transp.mu0))
            self.dWfast[it] = 1.0 * self.eps1[it] ** 1.5 * self.Bfast[it] / self.s1[it]

            p = extrap(xgridlong_red, transp.rmin[it], transp.pFast[it])
            y = (
                CDFt.derivativeVar(
                    self, p, specialDerivative=xgridlong_red, onlyOneTime=True
                )
                * 1e6
                * self.r1[it]
            )

            integral01 = integra(x, x**1.5 * y)
            self.Bfast_tot[it] = -integral01 / (self.Bpol1[it] ** 2 / (2 * transp.mu0))
            self.dWfast_tot[it] = (
                1.0 * self.eps1[it] ** 1.5 * self.Bfast_tot[it] / self.s1[it]
            )

        print(" -----")


class transpISOLVER:
    def __init__(self, f, RlimZlim=None, folder_isolver_data=None, tok="sprc"):
        """
        folder_isolver_data
        """

        self.t = f["TIME"][:]
        self.x_lw = f["X"][:][-1, :]
        self.xb_lw = f["XB"][:][-1, :]

        self.RlimZlim = RlimZlim

        # ------- Coils -------

        self.gatherCoils(f)

        # ---------------------------------------
        # ---------- 2D Poloidal Grids ----------
        # ---------------------------------------

        # ------- 2D Poloidal Grids -------

        self.R = f["RGRID"][:] * 1e-2  # m
        self.Z = f["ZGRID"][:] * 1e-2  # m
        if "APSIRZ" in f:
            self.RelativeToMachine = True
        else:
            self.RelativeToMachine = False

        # ------- Relative to machine axis

        if self.RelativeToMachine:
            self.psiRZ = f["APSIRZ"][:].reshape(
                (self.R.shape[0], self.R.shape[1], self.Z.shape[1])
            )  # Wb/rad (TIME, R, Z)
            self.psi1 = f["ABPMHD"][:]
            self.psi0 = f["PSI0_ISO"][:] * 2 * np.pi

            self.psi0 = f["PSI0_ISO"][:] * 2 * np.pi

        # ------- Relative to magnetic axis
        else:
            self.psiRZ = f["PSIRZ"][:].reshape(
                (self.R.shape[0], self.R.shape[1], self.Z.shape[1])
            )  # Wb/rad
            self.psi1 = f["PBPMHD"][:]
            self.psi0 = 0 * self.psi1
            self.psi0_tomachine = f["PSI0_ISO"][:] * 2 * np.pi

        rhoPol = []
        for i in range(self.psiRZ.shape[0]):
            rhoPol.append(
                (self.psiRZ[i] - self.psi0[i]) / (self.psi1[i] - self.psi0[i])
            )
        self.rhoPol = np.array(rhoPol)

        # ---------------------------------------
        # ---------- Xppoints ----------
        # ---------------------------------------

        self.gatherXpoints(f)

        # Iteratons
        self.iterations = f["ISOITER"][:]
        self.retries = f["ISORETRY"][:]
        self.iterations_saw = f["ISOITERSAW"][:]

        self.offset_free = f["FBDY_FREE"][:]
        self.offset_ref = f["FBDY_REF"][:]

        self.psi0_err = f["PSI0ERR"][:] * 2 * np.pi

        # Diffusion (ISOLVER may be enabled but without current diffusion (using just TRANSP's))
        self.getDiffusion(f)

        self.psi_enclosed = f["PLFLXISO"][:] * 2 * np.pi

        self.inputs = None
        if folder_isolver_data is not None:
            try:
                self.inputs = ISOLVERinputs(folder_isolver_data, name=tok)
            except:
                print("could not read isolver input data")

    def gatherCoils(self, f):
        self.coils, st = {}, ""
        for ikey in f:
            if ikey[:3] == "CC_":  # and 'CC_'+ikey.strip('CC_') in f:
                coil_name = ikey[3:]
                self.coils[coil_name] = {
                    "Ic": f["CC_" + coil_name][:]
                    * 1e-6,  # total poloidal current in coil (MA-turns)
                    "Ik": f["KK_" + coil_name][:] * 1e-6,  # circuit current in coil
                    "Ic_check": f["CP_" + coil_name][:]
                    * 1e-6,  # comparison of Isolver and PFC coil currents if NLPFC_CIRCUIT=.FALSE.
                    "Ik_check": f["KP_" + coil_name][:]
                    * 1e-6,  # comparison of Isolver and PFC circuit currents if NLPFC_CIRCUIT=.TRUE.
                }

                if "PF_" + coil_name in f:
                    self.coils[coil_name]["Ipf"] = (
                        f["PF_" + coil_name][:] * 1e-6
                    )  # poloidal current in coil as input in PFC ufile
                else:
                    self.coils[coil_name]["Ipf"] = None

                if "VV_" + coil_name in f:
                    self.coils[coil_name]["VV"] = (
                        f["VV_" + coil_name][:] * 1e-6
                    )  # coil terminal voltage at middle of previous geometry time step
                    self.coils[coil_name]["VS"] = (
                        f["VS_" + coil_name][:] * 1e-6
                    )  # coil source voltage at middle of previous geometry time step
                else:
                    self.coils[coil_name]["VV"] = None
                    self.coils[coil_name]["VS"] = None

                st += coil_name + ", "

        if len(self.coils) > 0:
            print("\t\t--> Gathering ISOLVER data for machine with coils: " + st[:-2])

    def getDiffusion(self, f):
        # Voltage profiles
        self.Vloop_b = f["VISO"][:]
        self.Vloop_c = f["VISOZ"][:]
        try:
            self.Vloop_b_surf = f["VSURR"][:]
        except:
            self.Vloop_b_surf = self.t * 0.0

        try:
            self.Vloop_f = f["VDIF"][:]
        except:
            self.Vloop_f = self.Vloop_b * 0.0

        try:
            self.Vloop_mhd = f["VCHEKMHD"][:]
        except:
            self.Vloop_mhd = self.Vloop_b * 0.0
        try:
            self.Vloop_mhd_surf = f["VSURM"][:]
        except:
            self.Vloop_mhd_surf = self.t * 0.0
        try:
            self.Vloop_mhd_surf0 = f["VSURA"][:]
        except:
            self.Vloop_mhd_surf0 = self.t * 0.0

        try:
            self.q_diffusion = f["QDIF"][:]
        except:
            self.q_diffusion = None

        try:
            from transp_tools.CDFtools import surfaceIntegralTot_var

            self.jAnom = f["CURERR"][:] * 1e-6 * 1e4  # in MA/m^2
            self.Ip_Anom = surfaceIntegralTot_var(f, f["CURERR"][:]) * 1e-6  # in MA
            self.Ip_Anom_abs = (
                surfaceIntegralTot_var(f, np.abs(f["CURERR"][:])) * 1e-6
            )  # in MA
        except:
            self.jAnom = None
            self.Ip_Anom = None
            self.Ip_Anom_abs = None

        try:
            self.JdotB_Anom = f["PLJBERR"][:] * 1e-6 * 1e4  # in MA*T/m^2
        except:
            self.JdotB_Anom = None

    def gatherXpoints(self, f):
        # Act Xpoint
        xpoints = []
        for i in range(10):
            if f"ZXGUESS{i + 1}" in f and np.sum(f[f"RXPMHD{i + 1}"][:]) > 0.0:
                xpoint = [
                    f[f"RXPMHD{i + 1}"][:] * 1e-2,
                    f[f"ZXPMHD{i + 1}"][:] * 1e-2,
                ]
                xpoints.append(xpoint)
        self.xpoints_sol = np.array(xpoints)

        # Guessed Xpoint (WHERE THE LOCATION SHOULD BE BASED ON THE PRESCRIBED BOUNDARY)
        xpoints = []
        for i in range(10):
            if f"ZXGUESS{i + 1}" in f:
                xpoint = [
                    f[f"RXGUESS{i + 1}"][:] * 1e-2,
                    f[f"ZXGUESS{i + 1}"][:] * 1e-2,
                ]
                xpoints.append(xpoint)
        self.xpoints = np.array(xpoints)

        # Flux through x-point (resolution issues may give this a different than 1 value)
        self.xpoint_dominant = self.xpoints[0]
        self.xpoint_rhoPol = np.zeros(len(self.xpoint_dominant[0]))
        self.xpoint_psi = np.zeros(len(self.xpoint_dominant[0]))
        for it in range(len(self.xpoint_dominant[0])):
            self.xpoint_rhoPol[it] = MATHtools.interp2D(
                self.xpoint_dominant[0][it],
                self.xpoint_dominant[1][it],
                self.R[it],
                self.Z[it],
                self.rhoPol[it],
                kind="linear",
            )[0]
            self.xpoint_psi[it] = MATHtools.interp2D(
                self.xpoint_dominant[0][it],
                self.xpoint_dominant[1][it],
                self.R[it],
                self.Z[it],
                self.psiRZ[it],
                kind="linear",
            )[0]

    def strikes(self, time, thrFlux=1e-3):
        it = np.argmin(np.abs(self.t - time))

        self.RlimZlim_extended = MATHtools.upsampleCurve(
            self.RlimZlim[0], self.RlimZlim[1], extra_factor=20
        )
        R = self.RlimZlim_extended[0]
        Z = self.RlimZlim_extended[1]

        R_inter, Z_inter = [], []
        for i in range(len(R)):
            rhoLim = MATHtools.interp2D(
                R[i], Z[i], self.R[it], self.Z[it], self.rhoPol[it], kind="linear"
            )[0]

            if np.abs(rhoLim - 1) < thrFlux:
                R_inter.append(R[i])
                Z_inter.append(Z[i])

        return np.array(R_inter), np.array(Z_inter)

    # ---------------------------------------
    # ---------- Plotting
    # ---------------------------------------

    def plotSummary(self, time, fig=None, V=None):
        if fig is None:
            fig = plt.figure()

        it = np.argmin(np.abs(self.t - time))

        grid = plt.GridSpec(nrows=2, ncols=4, hspace=0.3, wspace=0.4)
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0])
        ax3 = fig.add_subplot(grid[0, 1])
        ax4 = fig.add_subplot(grid[1, 1])
        ax5 = fig.add_subplot(grid[0, 2])
        ax6 = fig.add_subplot(grid[1, 2])
        ax7 = fig.add_subplot(grid[0, 3])
        ax8 = fig.add_subplot(grid[1, 3])

        # ~~~~~~ Performance
        ax = ax1
        ax.plot(self.t, self.iterations, lw=2, label="Iterations")
        ax.plot(self.t, self.retries, lw=2, label="Retries")
        ax.plot(self.t, self.iterations_saw, lw=2, ls="--", label="Iterations Saw")
        ax.set_xlabel("Time (s)")
        ax.set_title("ISOLVER runs")
        ax.set_ylabel("#")
        ax.legend(loc="best")
        ax.set_ylim(bottom=0)

        # ~~~~~~ Offset
        ax = ax2
        ax.plot(self.t, self.offset_free, lw=3, label="Free bndry")
        ax.plot(self.t, self.offset_ref, ls="--", lw=3, label="Presc bndry")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$\\Delta\\psi_n$")
        ax.set_title("Average offests")
        ax.set_ylim(
            [0, 1.5 * np.max([0.01, np.max(self.offset_free), np.max(self.offset_ref)])]
        )
        ax.legend(loc="best")

        # # ~~~~~~ VLoop surf
        ax = ax3
        ax.plot(self.t, self.Vloop_b_surf, lw=3, label="$V_{\\eta J_{OH}}$")
        ax.plot(self.t, self.Vloop_mhd_surf, lw=3, label="$V_{flux,\\psi0 diff}$")
        ax.plot(self.t, self.Vloop_mhd_surf0, lw=3, label="$V_{flux,\\psi0}$")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("V")
        ax.set_title("Surface Voltage")
        ax.legend(loc="best")
        ax.axhline(y=0, ls="--", c="k", lw=0.5)

        # # ~~~~~~ VLoop
        ax = ax4
        ax.plot(self.x_lw, self.Vloop_b[it], lw=3, label="$V_{\\eta J_{OH}}$")
        ax.plot(self.x_lw, self.Vloop_mhd[it], lw=3, label="$V_{flux}$")
        ax.plot(self.x_lw, self.Vloop_f[it], lw=3, ls="-.", label="$V_{FD}$")
        if V is not None:
            ax.plot(self.x_lw, V[it], ls="--", lw=1, c="m", label="$V_{PRF}$")

        ax.set_xlabel("$\\rho_N$")
        ax.set_ylabel("V")
        ax.set_title("Loop Voltage")
        ax.set_xlim([0, 1])
        ax.legend(loc="best")
        ax.axhline(y=0, ls="--", c="k", lw=0.5)

        # ~~~~~~
        ax = ax5
        ax.plot(self.t, self.psi_enclosed, lw=3, c="b", label="$\\psi_{encl.}$")

        ax.plot(self.t, self.psi0_err, lw=2, ls="--", c="b", label="$\\psi_0$ error")
        ax.axhline(y=0, ls="--", c="k", lw=0.5)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$Wb$")
        ax.set_title("Enclosed flux")
        ax.legend(loc="best")

        # ~~~~~~
        ax = ax6
        ax.plot(self.t, self.Ip_Anom_abs, lw=3, c="b", label="")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Effective $MA$")
        ax.set_title("Surf. Int. Abs. Anom. Current")
        ax.set_ylim(bottom=0)

        # ~~~~~~
        ax = ax7
        ax.plot(self.t, self.psi0_tomachine, lw=3, c="b", label="$\\psi_0$ ISOLVER")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$Wb$")
        ax.set_title("Axis Flux rel. to machine axis")

        # ~~~~~~
        ax = ax8
        ax.plot(
            self.t,
            MATHtools.deriv(self.t, self.psi0_tomachine),
            lw=3,
            c="b",
            label="$\\partial \\psi_0/\\partial t$ ISOLVER",
        )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$V$")
        ax.set_title('Axis "voltage"')

        return ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8

    def plotSurfaces(
        self,
        ax,
        time,
        lw=[0.2, 1.0],
        levels=500,
        resol=1e-3,
        colorBoundary="r",
        colorSurfs=None,
        label="",
        rhoPol=True,
    ):
        it = np.argmin(np.abs(self.t - time))

        if rhoPol:
            z = self.rhoPol
            zx = self.xpoint_rhoPol
        else:
            z = self.psiRZ
            zx = self.xpoint_psi

        for xpoint in self.xpoints:
            ax.scatter([xpoint[0, it]], [xpoint[1, it]], 100, marker="*", facecolor="g")
        for xpoint in self.xpoints_sol:
            ax.scatter([xpoint[0, it]], [xpoint[1, it]], 80, marker="*", facecolor="r")
        cs = ax.contour(
            self.R[it],
            self.Z[it],
            z[it],
            levels,
            linewidths=lw[0],
            colors=colorSurfs,
            label=label,
        )
        _ = ax.contour(
            self.R[it],
            self.Z[it],
            z[it],
            levels=[zx[it]],
            colors=colorBoundary,
            linewidths=lw[1],
            label=label,
        )
        if resol > 0.0:
            _ = ax.contour(
                self.R[it],
                self.Z[it],
                z[it],
                levels=np.linspace(zx[it] - resol, zx[it] + resol, 10),
                colors=colorBoundary,
                linewidths=0.5,
            )

        return cs

    def plotEquilibria(self, time, fig=None):
        if fig is None:
            fig = plt.figure()

        grid = plt.GridSpec(nrows=2, ncols=4, hspace=0.3, wspace=0.6)
        ax1 = fig.add_subplot(grid[:, :2])
        ax3 = fig.add_subplot(grid[0, 2])
        ax4 = fig.add_subplot(grid[1, 2])
        ax5 = fig.add_subplot(grid[0, 3])
        ax6 = fig.add_subplot(grid[1, 3])

        it = np.argmin(np.abs(self.t - time))

        # ------------------------------------
        # --------- Flux surfaces
        # ------------------------------------

        resol = 1e-3
        ax = ax1
        cs = self.plotSurfaces(ax, time, resol=resol)
        ax.set_title(f"Norm. Poloidal Flux (LCFS+-{resol:.0e})")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal")

        if self.RlimZlim is not None:
            ax.plot(self.RlimZlim[0], self.RlimZlim[1], "-", c="k", lw=3)

            # r,z = self.strikes(time)
            # ax.plot(r,z,'o',c='r',markersize=5)

        # # ------------------------------------
        # # --------- Flux values
        # # ------------------------------------

        # ax2 = fig.add_subplot(grid[:,4:])
        # ax = ax2

        # cs =  ax.contourf(self.R[it],self.Z[it],self.psiRZ[it],1000)
        # _ =  ax.contour(self.R[it],self.Z[it],self.psiRZ[it],levels=[self.xpoint_psi[it]],colors='r',linewidths=1)
        # for xpoint in self.xpoints_sol:
        # 	ax.scatter([xpoint[0,it]],[xpoint[1,it]],100,marker='o',facecolor='w')

        # if self.RelativeToMachine:	title = 'machine'
        # else:						title = 'magn. axis'

        # ax.set_title('Poloidal Flux (relative to {0})'.format(title)); ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)')

        # ax.set_aspect('equal')

        # if self.RlimZlim is not None:	ax.plot(self.RlimZlim[0],self.RlimZlim[1],c='k',lw=3)

        # ------------------------------------
        # --------- X-point evolution
        # ------------------------------------

        colors = GRAPHICStools.listColors()

        ax = ax3
        for i in range(len(self.xpoints_sol)):
            z = self.xpoints_sol[i][0]
            ax.plot(self.t, z, lw=3, label=f"R {i + 1}", c=colors[i])
        for i in range(len(self.xpoints)):
            z = self.xpoints[i][0]
            ax.plot(self.t, z, lw=3, ls="--", c=colors[i])

        ax.set_xlabel("Time (s)")
        ax3.set_ylabel("R (m)")
        ax.legend(loc="upper left")
        # ax.set_ylim([z.min()-0.15,z.max()+0.1])
        ax.set_title("X-point location")
        ax.legend(loc="best")

        ax = ax4
        for i in range(len(self.xpoints_sol)):
            z = self.xpoints_sol[i][1]
            ax.plot(self.t, z, lw=3, label=f"Z {i + 1}", c=colors[i])
        for i in range(len(self.xpoints)):
            z = self.xpoints[i][1]
            ax.plot(self.t, z, lw=3, ls="--", c=colors[i])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Z (m)")
        ax.legend(loc="lower right")
        # ax.set_ylim([z.min()-0.1,z.max()+0.1])
        ax.legend(loc="best")

        ax = ax5
        ax.plot(self.t, self.xpoint_rhoPol, c="b", lw=1)
        ax.axhline(y=1, c="r", ls="-", lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$\\psi_n$ at x-point")
        ax.set_title("Resolution X point in $\\psi_n$")
        ax.set_ylim([1 - 0.01, 1 + 0.01])

        # ax = ax6
        # ax.plot(self.t,self.xpoint_psi,c='b',lw=1,label='interp')
        # ax.plot(self.t,self.psi1,c='r',lw=1,label='output')
        # ax.set_xlabel('Time (s)'); ax.set_ylabel('$\\psi$ at x-point')
        # ax.set_title('Resolution X point in $\\psi$')
        # ax.set_ylim([self.psi1[-1]-0.01,self.psi1[-1]+0.01])
        # ax.legend()

    def plotCoils(self, fig=None, MAturns=True):
        grid = plt.GridSpec(nrows=2, ncols=3, hspace=0.3, wspace=0.4)
        ax00 = fig.add_subplot(grid[0, 0])
        ax10 = fig.add_subplot(grid[1, 0], sharex=ax00)
        ax01 = fig.add_subplot(grid[0, 1], sharex=ax00)
        ax11 = fig.add_subplot(grid[1, 1], sharex=ax00)
        ax1 = fig.add_subplot(grid[:, 2])

        colors = GRAPHICStools.listColors()

        if MAturns:
            unit = " (MA*turn)"
            var = "Ic"
            fact = 1.0
        else:
            unit = " (kA), circuit"
            var = "Ik"
            fact = 1e3

        lw = 2
        fontsize = 8

        cont = -1
        ax = ax00
        for i in self.coils:
            if "PF" in i:
                if "L" not in i:
                    cont += 1
                    ls = "-"
                else:
                    ls = "--"
                ax.plot(
                    self.t,
                    self.coils[i][var] * fact,
                    ls=ls,
                    c=colors[cont],
                    lw=lw,
                    label=i,
                    markersize=1,
                )
                if not MAturns and self.coils[i]["Ipf"] is not None:
                    ax.plot(
                        self.t,
                        self.coils[i]["Ipf"] * fact,
                        ls=":",
                        alpha=0.5,
                        c=colors[cont],
                        lw=lw,
                        markersize=1,
                    )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("I" + unit)
        ax.set_title("PF Coil Currents")
        ax.axhline(y=0, ls="--", lw=0.5, c="k")
        GRAPHICStools.addLegendApart(ax, ratio=0.9, size=fontsize)

        ax = ax10
        for i in self.coils:
            if "CS" in i:
                if "L" not in i:
                    cont += 1
                    ls = "-"
                else:
                    ls = "--"
                ax.plot(
                    self.t,
                    self.coils[i][var] * fact,
                    ls=ls,
                    c=colors[cont],
                    lw=lw,
                    label=i,
                    markersize=1,
                )
                if not MAturns and self.coils[i]["Ipf"] is not None:
                    ax.plot(
                        self.t,
                        self.coils[i]["Ipf"] * fact,
                        ls=":",
                        alpha=0.5,
                        c=colors[cont],
                        lw=lw,
                        markersize=1,
                    )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("I" + unit)
        ax.set_title("CS Coil Currents")
        ax.axhline(y=0, ls="--", lw=0.5, c="k")
        GRAPHICStools.addLegendApart(ax, ratio=0.9, size=fontsize)

        ax = ax01
        for i in self.coils:
            if "DV" in i:
                if "L" not in i:
                    cont += 1
                    ls = "-"
                else:
                    ls = "--"
                ax.plot(
                    self.t,
                    self.coils[i][var] * fact,
                    ls=ls,
                    c=colors[cont],
                    lw=lw,
                    label=i,
                    markersize=1,
                )
                if not MAturns and self.coils[i]["Ipf"] is not None:
                    ax.plot(
                        self.t,
                        self.coils[i]["Ipf"] * fact,
                        ls=":",
                        alpha=0.5,
                        c=colors[cont],
                        lw=lw,
                        markersize=1,
                    )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("I" + unit)
        ax.set_title("DV Coil Currents")
        ax.axhline(y=0, ls="--", lw=0.5, c="k")
        GRAPHICStools.addLegendApart(ax, ratio=0.9, size=fontsize)

        ax = ax11
        for i in self.coils:
            if "VS" in i:
                if "L" not in i:
                    cont += 1
                    ls = "-"
                else:
                    ls = "--"
                ax.plot(
                    self.t,
                    self.coils[i][var] * fact,
                    ls=ls,
                    c=colors[cont],
                    lw=lw,
                    label=i,
                    markersize=1,
                )
                if not MAturns and self.coils[i]["Ipf"] is not None:
                    ax.plot(
                        self.t,
                        self.coils[i]["Ipf"] * fact,
                        ls=":",
                        alpha=0.5,
                        c=colors[cont],
                        lw=lw,
                        markersize=1,
                    )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("I" + unit)
        ax.set_title("VS Coil Currents")
        ax.axhline(y=0, ls="--", lw=0.5, c="k")
        GRAPHICStools.addLegendApart(ax, ratio=0.9, size=fontsize)

        ax = ax1
        time = 100.0
        cs = self.plotSurfaces(ax, time, rhoPol=False)  # ,resol=resol)

        ax.set_aspect("equal")
        ax.set_title("Norm. Poloidal Flux")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")

        if self.inputs is not None:
            self.inputs.plotStructures(ax=ax)

        # ax = ax01
        # for i in self.coils:
        # 	if self.coils[i]['VV'] is not None:
        # 		if 'CS' not in i: ax.plot(self.t,self.coils[i]['VV'],'-^',lw=1,label=i)
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Voltage (V)')
        # ax.set_title('PF Coil terminal voltage')
        # GRAPHICStools.addLegendApart(ax,ratio=0.9)

        # ax = ax11
        # for i in self.coils:
        # 	if self.coils[i]['VV'] is not None:
        # 		if 'CS' in i: ax.plot(self.t,self.coils[i]['VV'],'-^',lw=1,label=i)
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Voltage (V)')
        # ax.set_title('CS Coil terminal voltage')
        # GRAPHICStools.addLegendApart(ax,ratio=0.9)


class ISOLVERinputs:
    def __init__(self, folder, name="sprc"):
        self.folder = IOtools.expandPath(folder)

        self.file_conductor = self.folder / f"{name}_conductor_regions.dat"
        self.file_limiters = self.folder / f"{name}_limiter_surface.dat"
        self.file_connections = self.folder / f"{name}_coil_connections.dat"

        self.readCoils()

        self.readLimiter()

    def readCoils(self):
        with open(self.file_conductor, "r") as f:
            aux = f.readlines()

        for i in range(len(aux)):
            if "# ir.s" in aux[i]:
                coil_line = i
            if "# Inner vacuum vessel" in aux[i]:
                coil_vi = i
            if "# Outer vacuum vessel" in aux[i]:
                coil_vo = i
            if "# Vertical stability passive plates" in aux[i]:
                coil_vs = i
        last = i

        self.coils = grabVals(aux, coil_line + 1, coil_vi - 2)
        self.vv_inner = grabVals(aux, coil_vi + 1, coil_vo - 1)
        self.vv_outer = grabVals(aux, coil_vo + 1, coil_vs - 1)
        self.vs = grabVals(aux, coil_vs + 1, last + 1)

        self.vv_inner_x, self.vv_inner_z = gatherStructureLine(self.vv_inner)
        self.vv_outer_x, self.vv_outer_z = gatherStructureLine(self.vv_outer)
        self.vs_x, self.vs_z = gatherStructureLine(self.vs)

    def readLimiter(self):
        with open(self.file_limiters, "r") as f:
            aux = f.readlines()

        x, z = [], []
        for i in range(len(aux) - 3):
            line = aux[i + 3].split()
            x.append(float(line[0]))
            z.append(float(line[1]))

        self.lim_x = np.array(x)
        self.lim_z = np.array(z)

    def plotStructures(self, ax=None, names=True):
        if ax is None:
            fig, ax = plt.subplots()

        lw = 0.5
        ll = "-"

        self.plotCoils(ax, names=names, lw=lw)

        ax.plot(self.vv_inner_x, self.vv_inner_z, "-", c="k", lw=lw, markersize=2)
        ax.plot(self.vv_outer_x, self.vv_outer_z, "-", c="k", lw=lw, markersize=2)
        for vv in self.vs:
            ax.plot(
                [self.vs[vv]["R"]], [self.vs[vv]["Z"]], "o", markersize=2, c="g", lw=lw
            )
        # ax.plot(self.vs_x, self.vs_z,c='g',lw=1.5)

        ax.plot(self.lim_x, self.lim_z, "-", c="k", lw=lw, markersize=2)

    def plotCoils(self, ax, names=True, lw=0.5, c="k"):
        for coil in self.coils:
            x, y = pointsCoil(
                self.coils[coil]["R"],
                self.coils[coil]["Z"],
                self.coils[coil]["W"],
                self.coils[coil]["H"],
            )

            ax.plot(x, y, "-", c=c, lw=lw)

            if names:
                ax.text(
                    self.coils[coil]["R"],
                    self.coils[coil]["Z"],
                    coil,
                    color="k",
                    fontsize=5,
                    horizontalalignment="center",
                    verticalalignment="center",
                )


def gatherStructureLine(stru):
    vv_inner_x, vv_inner_z = [], []
    for vv in stru:
        vv_inner_x.append(stru[vv]["R"])
        vv_inner_z.append(stru[vv]["Z"])
    vv_inner_x.append(vv_inner_x[0])
    vv_inner_z.append(vv_inner_z[0])

    return np.array(vv_inner_x), np.array(vv_inner_z)


def pointsCoil(R, Z, W, H):
    x, y = [], []
    x.extend(np.linspace(R - W / 2, R + W / 2, 10))
    y.extend(np.linspace(Z - H / 2, Z - H / 2, 10))
    x.extend(np.linspace(R + W / 2, R + W / 2, 10))
    y.extend(np.linspace(Z - H / 2, Z + H / 2, 10))
    x.extend(np.flipud(np.linspace(R - W / 2, R + W / 2, 10)))
    y.extend(np.linspace(Z + H / 2, Z + H / 2, 10))
    x.extend(np.linspace(R - W / 2, R - W / 2, 10))
    y.extend(np.flipud(np.linspace(Z - H / 2, Z + H / 2, 10)))

    return np.array(x), np.array(y)


def grabVals(arr, first, last):
    coils = {}
    for i in np.arange(first, last, 1):
        line = arr[i].split()
        coils[line[-1]] = {
            "R": float(line[1]),
            "Z": float(line[2]),
            "W": float(line[3]),
            "H": float(line[4]),
        }
    return coils
