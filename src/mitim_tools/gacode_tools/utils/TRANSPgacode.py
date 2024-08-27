import matplotlib.pyplot as plt
import numpy as np
from mitim_tools.misc_tools import GRAPHICStools, MATHtools
from mitim_tools.gacode_tools.utils import GACODEdefaults, GACODEplotting, GACODErun

def arrangeTGLF(CDFc, varName="GRATE_TGLF"):
    # (slightly modified from OMFIT: modules/TRANSP/PLOTS/rplot.py)

    grate = CDFc.f[varName][:]
    kys = CDFc.f["KYS_TGLF"][:]

    nx = len(CDFc.x_lw)
    nt = len(CDFc.t)
    nky = kys.shape[1] // (nx + 3)

    # -------------------
    # Arrange vectors
    # -------------------
    """
		X. Yuan said in 04/20/2017:
		'this is boundary value(nx+1), plus two ghost zones (+2), and the number of nky depends on the nky_model (nky+9, or nky+15)'

		I interpret that there is 1 ghost zone on axis and 2 at the edge, so it needs to be reshaped to nx+3
		"""

    kys3 = np.zeros((nt, nky, nx + 3))
    grates3 = np.zeros((nt, nky, nx + 3))

    for i in range(nt):
        kys3[i, :, :] = np.reshape(kys[i, :], (nky, nx + 3))
        grates3[i, :, :] = np.reshape(grate[i, :], (nky, nx + 3))

    kys3 = np.array(np.swapaxes(kys3, 1, 2))
    grates3 = np.array(np.swapaxes(grates3, 1, 2))

    """
		Now I have to remove the ghost regions. If my most inner radial grid point is X=0.005, the axial region has X=0. If my 
		boundary radial grid point is X=0.995, then the edge region has X=1.0 and 1.0+delta
		"""

    kys3 = kys3[:, 1 : nx + 1, :]
    grates3 = grates3[:, 1 : nx + 1, :]

    """
		At this point, these vectors have dimensions (nt,nx,nky)
		However, in those cases that TGLF was not run (at times in simulation with no predictive TGLF or radii
		where TGLF is not used or ghost zones) there are zeros in the ky spectrum. What I'm doing now is to remove those situations.
		For this, instead of 3D arrays, I create objects: kys(nt,nx)(nky), where nky can change size
		"""

    kys, grates = [], []
    for i in range(nt):
        kyst, gratest = [], []
        for j in range(nx):
            mask = kys3[i, j, :] > 0
            kyst.append(kys3[i, j, mask])
            gratest.append(grates3[i, j, mask])
        kys.append(kyst)
        grates.append(gratest)

    kys = np.array(kys, dtype=object)
    grates = np.array(grates, dtype=object)

    return kys, grates


class tglfCDF:
    def __init__(self, transp_output):
        self.t = transp_output.t
        self.x = transp_output.x
        self.x_lw = transp_output.x_lw
        self.tlastsaw = transp_output.tlastsaw
        self.ind_saw = transp_output.ind_saw

        self.nt = transp_output.nt
        self.nx = transp_output.nx

        """
		Convolutions
		"""
        d_perp_dict, dRdx = {}, {}
        for ix in transp_output.x_lw:
            d_perp_dict[ix] = 0.1
            dRdx[ix] = 1.0
        convolution_fun_fluct, factorTot_to_Perp = GACODEdefaults.convolution_CECE(
            d_perp_dict, dRdx=dRdx
        )

        self.getLinearStability(transp_output, convolution_fun_fluct, factorTot_to_Perp)

        self.getMetrics()

    def getLinearStability(self, transp_output, convolution_fun_fluct, factorTot_to_Perp):
        print("\t\t>> Gathering TGLF linear stability")

        self.kys, self.grates = arrangeTGLF(transp_output, varName="GRATE_TGLF")
        _, self.freqs = arrangeTGLF(transp_output, varName="FREQ_TGLF")

        print("\t\t>> Gathering TGLF fluctuations")

        try:
            _, self.neFluct = arrangeTGLF(transp_output, varName="NEKY")
            self.ne_level = self.getFluctuationLevels(
                transp_output, self.neFluct, convolution_fun_fluct, factorTot_to_Perp
            )
        except:
            print("/// Could not get NEKY")

        try:
            _, self.TeFluct = arrangeTGLF(transp_output, varName="TEKY")
            self.Te_level = self.getFluctuationLevels(
                transp_output, self.TeFluct, convolution_fun_fluct, factorTot_to_Perp
            )
        except:
            print("/// Could not get TEKY")

        try:
            _, self.phiFluct = arrangeTGLF(transp_output, varName="POTKY")
        except:
            print("/// Could not get POTKY")

        try:
            _, self.QeSpectrum = arrangeTGLF(transp_output, varName="TEFLXKY")
            _, self.QiSpectrum = arrangeTGLF(transp_output, varName="TIFLXKY")
            _, self.QZSpectrum = arrangeTGLF(transp_output, varName="TZFLXKY")
        except:
            print("/// Could not get *FLXKY")

    def getFluctuationLevels(
        self, transp_output, var, convolution_fun_fluct, factorTot_to_Perp
    ):
        ne_level = []
        cont = 0
        for it in range(len(transp_output.t)):
            ne_aux = []
            for irho in range(len(transp_output.x_lw)):
                if len(self.kys[it, irho]) > 0:
                    yy = GACODErun.obtainFluctuationLevel(
                        self.kys[it, irho],
                        var[it, irho],
                        transp_output.rhos[it, irho],
                        transp_output.TGLF_a[it, irho],
                        convolution_fun_fluct=convolution_fun_fluct,
                        rho_eval=transp_output.x_lw[irho],
                        factorTot_to_Perp=factorTot_to_Perp,
                        printYN=cont == 0,
                    )
                    ne_aux.append(yy)
                    cont += 1
                else:
                    ne_aux.append(0)

            ne_level.append(ne_aux)

        return np.array(ne_level)

    def getMetrics(self):
        pass

        # Note that for the etas I need subdominant, which TRANSP doesn't output
        # it = -1
        # ix = 50
        # etas 	= TGLFtools.processGrowthRates(self.kys[it,ix],self.grates[it,ix],self.freqs[it,ix],None, None)

    def plotComplete_GR(self, fig=None, rhos=[0.8, 0.65, 0.5, 0.35], time=None):
        if time is None:
            time = self.t[self.ind_saw]

        if fig is None:
            fig = plt.figure(figsize=(12, 8))

        grid = plt.GridSpec(2, 2, hspace=0.2, wspace=0.4)

        ax1 = fig.add_subplot(grid[0, 0])
        self.plotLinearStability(ax=ax1, division=0, rhos=rhos, times=[time])

        ax2 = fig.add_subplot(grid[0, 1])
        self.plotLinearStability_x(fig=fig, ax=ax2, division=1, time=time)

        ax3 = fig.add_subplot(grid[1, :])
        self.plotLinearStability_t(fig=fig, ax=ax3, division=1, rho=rhos[0])

        cs = ["m", "g", "r", "b", "k"]
        ax3.axvline(x=time, color=cs[0], ls="--", lw=3)
        i = 0
        for rho in rhos:
            ax2.axvline(x=rho, color=cs[i], ls="--", lw=3)
            i += 1

        return ax1, ax2, ax3

    def plotLinearStability(
        self,
        ax=None,
        plotFreq=False,
        times=[100.0],
        rhos=[0.5],
        division=0,
        markerFreq=True,
        colorstart=0,
        size=15,
        sizemarker=100,
        alpha=1.0,
        leg=True,
        RemoveZeros=False,
    ):
        colors = GRAPHICStools.listColors()

        if ax is None:
            fig, ax = plt.subplots()

        if plotFreq:
            lab = "$\\omega$"
        else:
            lab = "$\\gamma$"
        cont = 0
        for time in times:
            it = np.argmin(np.abs(time - self.t))

            for rho in rhos:
                ix = np.argmin(np.abs(rho - self.x[it, :]))

                freqs = self.freqs[it, ix]
                grates = self.grates[it, ix]
                xx = self.kys[it, ix]

                GACODEplotting.plotTGLFspectrum(
                    ax,
                    xx,
                    grates,
                    freq=freqs,
                    coeff=division,
                    c=colors[cont],
                    ls="-",
                    lw=1,
                    label=f"t={time:.3f}s,$\\rho_N$={rho:.3f}",
                    markersize=20,
                    alpha=1.0,
                    titles=["Growth Rate", "Real Frequency"],
                    removeLow=1e-4,
                    ylabel=True,
                )

                cont += 1

        if leg:
            try:
                ax.legend(loc="best").set_draggable(True)
            except:
                ax.legend(loc="best")

        if ax.get_yscale() != "log":
            ax.set_ylim(bottom=0)

    def plotLinearStability_t(
        self, fig=None, ax=None, rho=0.4, division=0, smoothSawtooth=False, xlims=None
    ):
        if fig is None:
            fig, ax = plt.subplots()

        ix = np.argmin(np.abs(rho - self.x[-1, :]))

        # ~~~~~~ Write vectors
        x = self.t
        y = self.kys[:, ix]
        z = self.grates[:, ix]
        zF = self.freqs[:, ix]
        num = self.kys[-1, ix].shape[0]  # 21
        kysN = np.zeros([self.nt, len(y[-1])])
        gratesN = np.zeros([self.nt, len(y[-1])])
        freqsN = np.zeros([self.nt, len(y[-1])])
        for i in range(self.nt):
            try:
                kysN[i, :] = y[i]
                gratesN[i, :] = z[i]
                freqsN[i, :] = zF[i]
            except:
                kysN[i, :] = np.zeros(num)
                gratesN[i, :] = np.zeros(num)
                freqsN[i, :] = np.zeros(num)
        y = kysN
        z = gratesN
        zF = freqsN

        # ~~~~~~ Smooth Through Sawtooth
        if smoothSawtooth:
            znn = []
            ynn = []
            zFnn = []
            for ik in range(len(y[0])):
                znew, tnew = MATHtools.smoothThroughSawtooth(
                    x, z[:, ik], self.tlastsaw, 1
                )
                ynew, tnew = MATHtools.smoothThroughSawtooth(
                    x, y[:, ik], self.tlastsaw, 1
                )
                zFnew, tnew = MATHtools.smoothThroughSawtooth(
                    x, zF[:, ik], self.tlastsaw, 1
                )
                znn.append(znew)
                ynn.append(ynew)
                zFnn.append(zFnew)
            z = np.transpose(np.array(znn))
            zF = np.transpose(np.array(zFnn))
            y = np.transpose(np.array(ynn))
            x = tnew

        xn = []
        for i in range(len(y[0])):
            xn.append(x)
        x = np.transpose(np.array(xn))

        GRAPHICStools.plotLScontour(
            fig,
            ax,
            x,
            y,
            z,
            zF=zF,
            xlabel="Time (s)",
            zlabel="$\\gamma$ ($c_s/a$)",
            division=division,
            xlims=xlims,
        )

        ax.set_title(f"$\\rho_N$ = {rho:.2f}")

    def plotLinearStability_x(
        self,
        fig=None,
        ax=None,
        time=None,
        division=0,
        smoothSawtooth=False,
        xlims=None,
        size=15,
        zlims=None,
    ):
        if fig is None:
            fig, ax = plt.subplots()

        if time is None:
            ix = self.ind_saw
        else:
            ix = np.argmin(np.abs(time - self.t))

        # ~~~~~~ Write vectors
        x = self.x[ix, :]
        y = self.kys[ix, :]
        z = self.grates[ix, :]
        zF = self.freqs[ix, :]

        xN = []
        for i in range(self.nx):
            if len(y[i]) > 0:
                xN.append(x[i])
                lenk = len(y[i])

        kysN = np.zeros([len(xN), lenk])
        gratesN = np.zeros([len(xN), lenk])
        freqsN = np.zeros([len(xN), lenk])
        cont = 0
        for i in range(self.nx):
            if len(y[i]) > 0:
                kysN[cont, :] = y[i]
                gratesN[cont, :] = z[i]
                freqsN[cont, :] = zF[i]
                cont += 1
        y = kysN
        z = gratesN
        zF = freqsN

        xn = []
        for i in range(lenk):
            xn.append(xN)
        x = np.transpose(np.array(xn))

        GRAPHICStools.plotLScontour(
            fig,
            ax,
            x,
            y,
            z,
            zF=zF,
            xlabel="$\\rho_N$",
            zlabel="$\\mathrm{sgn(\\omega)\\cdot\\gamma}$ ($\\mathrm{c_s/a}$)",
            division=division,
            xlims=xlims,
            size=size,
            zlims=zlims,
        )

        try:
            ax.set_title(f"Time = {time:.3f}s")
        except:
            pass

    def plotComplete_FL(self, fig=None, rhos=[0.8, 0.65, 0.5, 0.35], time=None):
        if time is None:
            time = self.t[self.ind_saw]

        if fig is None:
            fig = plt.figure(figsize=(12, 8))

        grid = plt.GridSpec(2, 2, hspace=0.2, wspace=0.4)

        ax1 = fig.add_subplot(grid[0, 0])
        self.plotFluctSpectrum(var="ne", ax=ax1, times=[time], rhos=rhos)
        ax1.set_title("Density spectrum")

        ax2 = fig.add_subplot(grid[1, 0])
        self.plotFluctSpectrum(var="Te", ax=ax2, times=[time], rhos=rhos)
        ax2.set_title("Temperature spectrum")

        ax3 = fig.add_subplot(grid[0, 1])
        self.plotFluctLevels_t(ax=ax3, rhos=rhos, Te=False, ne=True)
        ax3.set_title("ne Fluctuation Levels")

        ax4 = fig.add_subplot(grid[1, 1])
        self.plotFluctLevels_t(ax=ax4, rhos=rhos, Te=True, ne=False)
        ax4.set_title("Te Fluctuation Levels")

        return ax1, ax2, ax3

    def plotFluctSpectrum(
        self,
        var="ne",
        ax=None,
        times=[1.0, 1.1],
        rhos=[0.4, 0.6],
        colorstart=0,
        leg=True,
    ):
        cs = ["m", "g", "r", "b", "k"]

        if ax is None:
            fig, ax = plt.subplots()

        if var == "ne":
            lab = "$\\delta n_e/n_{e,0}$"
            val = self.neFluct
        elif var == "phi":
            lab = "$\\delta \\phi/\\phi_0$"
            val = self.phiFluct
        elif var == "Te":
            lab = "$\\delta T_e/T_{e,0}$"
            val = self.TeFluct
        elif var == "Qe":
            lab = "$Q_e$"
            val = self.QeSpectrum
        elif var == "Qi":
            lab = "$Q_i$"
            val = self.QiSpectrum
        elif var == "QZ":
            lab = "$Q_Z$"
            val = self.QZSpectrum

        cont = 0
        for time in times:
            it = np.argmin(np.abs(time - self.t))

            for rho in rhos:
                ix = np.argmin(np.abs(rho - self.x[it, :]))

                varplot = val[it, ix]

                ax.plot(
                    self.kys[it, ix],
                    varplot,
                    c=cs[cont + colorstart],
                    lw=2,
                    label=f"$\\rho_N={rho:.1f}$",
                )

                for ik in range(len(varplot)):
                    mark = "o"
                    face = cs[cont + colorstart]

                    ax.scatter(
                        [self.kys[it, ix][ik]],
                        [varplot[ik]],
                        50,
                        facecolor=face,
                        edgecolor=cs[cont + colorstart],
                        marker=mark,
                    )
                cont += 1

        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.set_xlim([0.05, 30.0])
        ax.set_xlabel("$k_\\theta\\rho_s$")
        ax.set_ylabel(lab)

        if leg:
            try:
                ax.legend(loc="best").set_draggable(True)
            except:
                ax.legend(loc="best")

    def plotFluctLevels_t(
        self, ax=None, rhos=[0.4, 0.6], Te=True, ne=True, colorstart=0, leg=True
    ):
        cs = ["m", "g", "r", "b", "k"]

        if ax is None:
            fig, ax = plt.subplots()

        if Te:
            cont = 0
            for rho in rhos:
                ix = np.argmin(np.abs(rho - self.x_lw))
                ax.plot(
                    self.t,
                    self.Te_level[:, ix],
                    c=cs[cont + colorstart],
                    lw=2,
                    label=f"Te ,$\\rho_N={rho:.1f}$",
                )

                cont += 1

        if ne:
            cont = 0
            for rho in rhos:
                ix = np.argmin(np.abs(rho - self.x_lw))
                ax.plot(
                    self.t,
                    self.ne_level[:, ix],
                    c=cs[cont + colorstart],
                    lw=2,
                    ls="--",
                    label=f"ne ,$\\rho_N={rho:.1f}$",
                )

                cont += 1

        ax.set_ylim(bottom=0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$\\delta T_e/T_{e,0}$, $\\delta n_e/n_{e,0}$")

        if leg:
            try:
                ax.legend(loc="best").set_draggable(True)
            except:
                ax.legend(loc="best")
