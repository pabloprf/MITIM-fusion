import netCDF4, fnmatch, pdb, copy, os, sys, pickle
import numpy as np
from mitim_tools.misc_tools.IOtools import *
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mitim_tools.transp_tools.CDFtools import *


def plotSummaryScan(
    axSummary,
    whichOnes,
    mainF,
    colorsC,
    axMachines=None,
    ssh=None,
    keepCDFs=0,
    cdfs_ready=None,
):
    Qtot = []
    Costtot = []
    shotsListFlagsTot = []
    subfolder = ""

    tlim_m = 100
    tlim_M = 0
    cont1 = True

    inter = False

    maxxQ = 0.1

    Tlim = 0
    Nlim = 0

    cdfs = ["none"]
    cdfs_i = ["none"]
    cdfs_p = ["none"]
    for i in range(500):
        cdfs.append("none")
        cdfs_i.append("none")
        cdfs_p.append("none")

    for cont, i in enumerate(whichOnes):
        shotsListFlags = str(i)
        try:
            if cdfs_ready is not None:
                netCDFfile = cdfs_ready[cont]
            else:
                # Grab CDF
                netCDFfile = None
                files = mainF + subfolder + f"/Evaluation.{i}/FolderTRANSP/"
                for file in os.listdir(files):
                    if fnmatch.fnmatch(file, "*CDF"):
                        netCDFfile = files + file
            cdf = CDFreactor(netCDFfile, ssh=ssh)
            cdfs[i] = cdf
        except:
            try:
                # Grab CDF
                netCDFfile = None
                files = mainF + subfolder + f"/Evaluation.{i}/FolderTRANSP/"
                for file in os.listdir(files):
                    if fnmatch.fnmatch(file, "*CDF_prev"):
                        netCDFfile = files + file
                cdf = CDFreactor(netCDFfile, ssh=ssh)
                cdfs[i] = cdf
            except:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(f"Evaluation.{i} does not have TRANSP yet")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                continue

        if cdfs_ready is None:
            try:
                # Grab CDF interpretive
                netCDFfilei = None
                files = (
                    mainF + subfolder + f"/Evaluation.{i}/FolderTRANSP_interpretive/"
                )
                for file in os.listdir(files):
                    if fnmatch.fnmatch(file, "*CDF"):
                        netCDFfilei = files + file
                # print 'Opening {}'.format(netCDFfilei)
                cdf_i = CDFreactor(netCDFfilei, ssh=ssh)
                # cdfs_i.append(cdf_i)
                cdfs_i[i] = cdf_i
                inter = True
            except:
                print(f"\n >>>>>>>>>>> Evaluation.{i} does not have interpretive yet")
                cdf_i = cdf  # CDFreactor(netCDFfile)
                inter = False
                # cdfs_i.append('none')

            try:
                # Grab CDF prediction 1st
                netCDFfilei = None
                files = mainF + subfolder + f"/Evaluation.{i}/FolderTRANSP_predictive1/"
                for file in os.listdir(files):
                    if fnmatch.fnmatch(file, "*CDF"):
                        netCDFfilei = files + file

                # cdf_p = CDFreactor(netCDFfilei,ssh=ssh)
                cdf = CDFreactor(netCDFfilei, ssh=ssh)

                # cdfs_p[i] = cdf_p
                cdfs_p[i] = cdf

                inter2 = True
            except:
                print(f"\n >>>>>>>>>>> Evaluation.{i} does not have 1st predictive")
                cdf_p = cdf  # CDFreactor(netCDFfile)
                inter2 = False
                # cdfs_i.append('none')

        if axMachines is not None:
            cdf.plotGeometry(ax=axMachines, color=colorsC[cont], plotComplete=False)

        # print '\t\t\t{}'.format((cdf.f['PCUR'][-1]-cdf.f['PCUREQ'][-1])/cdf.f['PCUR'][-1] *100.0)
        # print '\t\t\t{}'.format((cdf.f['PCUR'][-1]-cdf.f['PCURC'][-1])/cdf.f['PCUR'][-1] *100.0)

        _, _, index_b, index_a = cdf.findIndeces(time=-1, offsets=[-0.01, 0.01])
        try:
            _, _, index_b_i, index_a_i = cdf_i.findIndeces(
                time=-1, offsets=[-0.01, 0.01]
            )
        except:
            cdf_i = cdf
            _, _, index_b_i, index_a_i = cdf_i.findIndeces(
                time=-1, offsets=[-0.01, 0.01]
            )

        # If no sawtooth, select last one
        if index_b == 0:
            index_b = -1
            index_a = -1
        if index_b_i == 0:
            index_b_i = -1
            index_a_i = -1

        x = cdf.x_lw
        x_i = cdf_i.x_lw

        t = cdf.t
        t_i = cdf_i.t

        Q = cdf.Q
        Q_i = cdf_i.Q

        q95 = cdf.q95
        q95_i = cdf_i.q95

        fGv = cdf.fGv
        fGv_i = cdf_i.fGv

        Te = cdf.Te
        Ne = cdf.ne
        Te_i = cdf_i.Te
        Ne_i = cdf_i.ne
        Ti = cdf.Ti
        Ti_i = cdf_i.Ti

        Teped = cdf.Te_height
        neped = cdf.ne_height
        Teped_i = cdf_i.Te_height
        neped_i = cdf_i.ne_height
        width = cdf.Te_width
        width_i = cdf_i.Te_width

        Ip = cdf.Ip
        Ip_i = cdf_i.Ip

        Pe = cdf.Peich + cdf.Pech + cdf.Plhe + cdf.Pnbie
        Pi = cdf.Piich + cdf.Plhi + cdf.Pnbii

        Bt = cdf.Bt
        Bt_i = cdf_i.Bt

        cptim = cdf.cptim
        cptim_i = cdf_i.cptim

        ne_avol = cdf.ne_avol
        ne_avol_i = cdf_i.ne_avol

        Te_avol = cdf.Te_avol
        Te_avol_i = cdf_i.Te_avol

        Ti_avol = cdf.Ti_avol
        Ti_avol_i = cdf_i.Ti_avol

        saws = cdf.tlastsawU

        # Release memory
        if keepCDFs == 0:
            del cdf_i
            cdfs_i[i] = None
            try:
                del cdf_p
            except:
                pass
            cdfs_p[i] = None
            del cdf
            cdfs[i] = None
            print("### Remove files")
        elif keepCDFs == 1:
            del cdf_i
            cdfs_i[i] = None
            if cdfs_p[i] is not None and cdfs_p[i] is not "none":
                del cdf
                cdfs[i] = None
            print("### Remove cdf_i")
        else:
            print("### Do not remove files")

        # ----

        Tlim = np.max([np.max(Te[:, 0]), np.max(Ti[:, 0]), Tlim])
        Nlim = np.max([Nlim, np.max(Ne[:, 0])])

        linet_interp = "-."
        linet_extra = "--"
        linet_extra2 = ":"

        tlim_m = np.min([tlim_m, t_i[0]])
        tlim_M = np.max([tlim_M, t[-1]])

        tlim = [tlim_m - 0.1, tlim_M + 0.1]

        # ----------------------

        axx = axSummary[0, 0]
        axx.plot(t, Q, lw=3, label=shotsListFlags, c=colorsC[cont])
        axx.set_xlabel("Time (s)")
        axx.set_ylabel("Q")
        axx.plot(t_i, Q_i, ls=linet_interp, lw=1, label=shotsListFlags, c=colorsC[cont])
        maxxQ = np.max([np.max(Q), maxxQ])
        axx.set_ylim([0, np.min([maxxQ, 15])])
        axx.set_xlim(tlim)

        try:
            Qs = []
            varWork = Q  # cdf.Pout#Q
            howmanybef = 1
            zlabb = "$\\Delta P_{fus}$ (%)"  #'$\\Delta Q$ (%)'
            for iss in saws:
                Qs.append(varWork[np.argmin(np.abs(t - iss)) - howmanybef])
            Qs = np.array(Qs)
            # axConvergence[0].scatter(saws,Qs,c=colorsC[cont],s=50)
            # axConvergence[0].set_ylabel('Q')
            # axConvergence[0].set_ylim([0,10])
            tdiff = saws[1:]
            relchange = np.abs(Qs[1:] - Qs[:-1]) / Qs[1:] * 100.0
            axx = axSummary[1, 0]  # [0,4]
            # axx.scatter(tdiff,relchange,s=50)
            axx.plot(tdiff, relchange, c=colorsC[cont])
            axx.scatter(tdiff, relchange, c=[colorsC[cont]], s=25)
            # axx.scatter([tdiff[-1]],[relchange[-1]],s=50,c='r')
            axx.set_ylim([0, 15])
            axx.set_ylabel(zlabb)
            axx.set_xlabel("Time (s)")
            axx.axhline(y=1.0, ls="--", c="k")
            axx.axhline(y=2.0, ls="--", c="k")
            axx.axhline(y=3.0, ls="--", c="k")
            axx.axhline(y=4.0, ls="--", c="k")
            axx.axhline(y=5.0, ls="--", c="k")
            axx.set_xlim(tlim)
        except:
            pass

        # ----------------------
        Tlimextra = 3

        axx = axSummary[0, 1]
        axx.plot(t, Te[:, 0], lw=1, label=shotsListFlags, c=colorsC[cont])
        axx.set_xlabel("Time (s)")
        axx.set_ylabel("$T_e$")
        axx.plot(t_i, Te_i[:, 0], ls=linet_interp, lw=1, c=colorsC[cont])
        axx.plot(t, Te_avol, lw=3, c=colorsC[cont])
        axx.plot(t_i, Te_avol_i, lw=1, ls=linet_interp, c=colorsC[cont])
        axx.set_ylim([0, Tlim + Tlimextra])
        axx.set_xlim(tlim)

        axx = axSummary[1, 1]
        axx.plot(x, Te[index_b, :], lw=3, label=shotsListFlags, c=colorsC[cont])
        axx.plot(x, Te[index_a, :], lw=1, ls=linet_extra, c=colorsC[cont])
        axx.plot(x_i, Te_i[index_b_i, :], lw=1, ls=linet_interp, c=colorsC[cont])
        axx.set_xlabel("$\\rho_N$")
        axx.set_ylabel("Te (keV)")
        axx.set_ylim([0, Tlim + Tlimextra])

        # ----------------------

        axx = axSummary[0, 2]  # 5]
        axx.plot(t, Ti[:, 0], lw=1, label=shotsListFlags, c=colorsC[cont])
        axx.set_xlabel("Time (s)")
        axx.set_ylabel("$T_i$")
        axx.plot(t_i, Ti_i[:, 0], ls=linet_interp, lw=1, c=colorsC[cont])
        axx.plot(t, Ti_avol, lw=3, c=colorsC[cont])
        axx.plot(t_i, Ti_avol_i, lw=1, ls=linet_interp, c=colorsC[cont])

        axx.set_ylim([0, Tlim + Tlimextra])
        axx.set_xlim(tlim)

        axx = axSummary[1, 2]
        axx.plot(x, Ti[index_b, :], lw=3, label=shotsListFlags, c=colorsC[cont])
        axx.plot(x, Ti[index_a, :], lw=1, ls=linet_extra, c=colorsC[cont])
        axx.plot(x_i, Ti_i[index_b_i, :], lw=1, ls=linet_interp, c=colorsC[cont])
        axx.set_xlabel("$\\rho_N$")
        axx.set_ylabel("Ti (keV)")
        axx.set_ylim([0, Tlim + Tlimextra])

        # -------------------------

        axx = axSummary[0, 3]
        axx.plot(t, Ne[:, 0], lw=1, c=colorsC[cont])
        axx.plot(t_i, Ne_i[:, 0], ls=linet_interp, lw=1, c=colorsC[cont])
        axx.plot(t, ne_avol, lw=3, c=colorsC[cont])
        axx.plot(t_i, ne_avol_i, lw=1, ls=linet_interp, c=colorsC[cont])
        axx.set_xlabel("Time (s)")
        axx.set_ylabel("ne (1E20m^-3)")
        axx.set_ylim([0, Nlim + 1])
        axx.set_xlim(tlim)

        axx = axSummary[1, 3]
        axx.plot(x, Ne[index_b, :], lw=3, label=shotsListFlags, c=colorsC[cont])
        axx.plot(x, Ne[index_a, :], lw=1, ls=linet_extra, c=colorsC[cont])
        axx.plot(x_i, Ne_i[index_b_i, :], lw=1, ls=linet_interp, c=colorsC[cont])
        axx.set_xlabel("$\\rho_N$")
        axx.set_ylabel("ne (1E20m^-3)")
        axx.set_ylim([0, Nlim + 1])

        # -------------------------

        if cont1:
            ll = "e-"
            ll2 = "i+"
        else:
            ll = ""
            ll2 = ""
        axx = axSummary[0, 4]
        axx.plot(x, Pe[index_b, :], lw=2, label=ll, c=colorsC[cont])
        axx.plot(x, Pi[index_b, :], lw=2, ls=linet_extra, label=ll2, c=colorsC[cont])
        axx.set_xlabel("$\\rho_N$")
        axx.set_ylabel("Aux power")
        # axx.set_xlim([0,0.8])

        # -------------------------

        if cont1:
            ll = "Ip"
            ll2 = "Bt"
            ll3 = "q95"
            ll4 = "cpu (10h)"
            ll5 = "fG*10"
            cont1 = False
        else:
            ll = ""
            ll2 = ""
            ll3 = ""
            ll4 = ""
            ll5 = ""
        axx = axSummary[0, 5]  # [1,2]
        axx.plot(t, Ip, lw=3, label=ll, c=colorsC[cont])
        axx.plot(t, Bt, lw=2, ls=linet_interp, label=ll2, c=colorsC[cont])
        axx.plot(t_i, Ip_i, ls=linet_extra, lw=1, c=colorsC[cont])
        axx.plot(t_i, Bt_i, ls=linet_interp, lw=1, c=colorsC[cont])
        axx.plot(t, fGv * 10, lw=1, ls="--", label=ll5, c=colorsC[cont])
        axx.plot(t_i, fGv_i * 10, lw=1, ls="--", c=colorsC[cont])
        axx.set_xlabel("Time (s)")
        axx.set_ylabel("Ip (MA)")

        # axx = axSummary[0,1]
        axx.plot(t, q95, ls=linet_extra2, lw=3, label=ll3, c=colorsC[cont])
        axx.plot(t_i, q95_i, ls=linet_interp, lw=1, c=colorsC[cont])
        axx.set_xlabel("Time (s)")
        axx.set_ylabel("q95")
        # axx.set_ylim([2.,4.0])

        suma = 0
        if inter:
            suma = cptim_i[-1]
        try:
            if inter2:
                suma += cptim_p[-1]
        except:
            pass

        axx.plot(t, (cptim + suma) * 1e-1, lw=1, label=ll4, c=colorsC[cont])
        # axx.set_xlabel('Time (s)')
        # axx.set_ylabel('CPU time (10h)')
        axx.plot(t_i, cptim_i * 1e-1, ls=linet_interp, lw=1, c=colorsC[cont])
        axx.scatter([t[-1]], [(cptim[-1] + suma) * 1e-1], s=50, c=[colorsC[cont]])
        axx.set_ylim(bottom=0)

        axx.set_xlim(tlim)

        # -------------------------

        axx = axSummary[1, 4]
        axx.plot(x, Te[index_b, :], lw=2, label=shotsListFlags, c=colorsC[cont])
        axx.plot(x_i, Te_i[index_b_i, :], ls=linet_interp, lw=1, c=colorsC[cont])
        try:
            axx.scatter([1 - width[index_b]], [Teped[index_b]], s=50, c=[colorsC[cont]])
            axx.scatter(
                [1 - width_i[index_b]], [Teped_i[index_b]], s=100, c=[colorsC[cont]]
            )
        except:
            pass
        axx.set_xlim([0.9, 1.0])
        axx.set_ylim([0.0, 7])
        axx.set_xlabel("$\\rho_N$")
        axx.set_ylabel("Te (keV)")

        axx = axSummary[1, 5]
        axx.plot(x, Ne[index_b, :], lw=2, label=shotsListFlags, c=colorsC[cont])
        axx.plot(
            x_i,
            Ne_i[index_b_i, :],
            ls=linet_interp,
            lw=1,
            label=shotsListFlags,
            c=colorsC[cont],
        )
        try:
            axx.scatter([1 - width[index_b]], [neped[index_b]], s=50, c=[colorsC[cont]])
            axx.scatter(
                [1 - width_i[index_b]], [neped_i[index_b]], s=100, c=[colorsC[cont]]
            )
        except:
            pass
        axx.set_xlim([0.9, 1.0])
        axx.set_ylim([0.0, 7])
        axx.set_xlabel("$\\rho_N$")
        axx.set_ylabel("ne (1E20m^-3)")

        try:
            # Get results
            file = mainF + "/Evaluation.{0}/results.out.{0}".format(i)
            with open(file, "r") as f:
                linn = f.readlines()
            vec = []
            for ii in linn:
                vec.append(float(ii.split(" ")[0]))
            Qtot.append(vec[0])
            Costtot.append(vec[1])
        except:
            pass  # print('<<ISSUE>> Evaluation.{}, results not evaluated yet'.format(i))

        shotsListFlagsTot.append(shotsListFlags)

    try:
        axSummary[0, 4].legend(loc="best").set_draggable(True)
        axSummary[0, 5].legend(loc="best").set_draggable(True)
        axSummary[0, 1].legend(loc="best").set_draggable(True)
    except:
        try:
            axSummary[0, 4].legend(loc="best")
            axSummary[0, 5].legend(loc="best")
            axSummary[0, 1].legend(loc="best")
        except:
            print("cannot plot legends")
    axSummary[0, 4].set_ylim(bottom=0)

    return np.array(Qtot), np.array(Costtot), shotsListFlagsTot, cdfs, cdfs_i, cdfs_p


def prettyLabels(lab):
    if lab == "Rmajor":
        lab1 = "$R_{major}$ [m]"
    elif lab == "bt":
        lab1 = "$B_{T}$ [T]"
    elif lab == "delta":
        lab1 = "$\\delta$"
    elif lab == "q95":
        lab1 = "$q_{95}$"
    elif lab == "BetaN":
        lab1 = "$\\beta_N$"
    elif lab == "ip":
        lab1 = "$I_p$"
    elif lab == "Q_plasma":
        lab1 = "$Q_{plasma}$"
    elif lab == "taue":
        lab1 = "$\\tau_E$ [ms]"
    elif lab == "ne_avol":
        lab1 = "$\\langle n_e \\rangle$ [$10^{20}m^{-3}$]"
    elif lab == "Te_avol":
        lab1 = "$\\langle T_e \\rangle$ [keV]"
    elif lab == "Ti_avol":
        lab1 = "$\\langle T_i \\rangle$ [keV]"
    elif lab == "kappa":
        lab1 = "$\\kappa$"
    else:
        lab1 = lab

    return lab1
