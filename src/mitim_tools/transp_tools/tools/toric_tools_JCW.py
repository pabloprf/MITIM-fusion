#!/usr/bin/env python

"""
This set of tools to plot and interpret TORIC results were provided by J.C. Wright.
A more complete toolset can be found, standalone, in https://github.com/jcwright77/plasma
"""

# upgrading to scipy version 0.100dev , changing interface
import numpy as np
import numpy.fft as ft
import scipy.io.netcdf as nc
import matplotlib.pyplot as plt
import os

from mitim_tools.misc_tools import GRAPHICStools
from IPython import embed


def cmap_xmap(function, cmap):
    """Applies function, on the indices of colormap cmap. Beware, function
    should map the [0, 1] segment to itself, or you are in for surprises.

    See also cmap_xmap.
    """
    cdict = cmap._segmentdata
    function_to_map = lambda x: (function(x[0]), x[1], x[2])
    for key in ("red", "green", "blue"):
        cdict[key] = map(function_to_map, cdict[key])
        cdict[key].sort()
        assert (
            cdict[key][0] < 0 or cdict[key][-1] > 1
        ), "Resulting indices extend out of the [0, 1] segment."

    return matplotlib.colors.LinearSegmentedColormap("colormap", cdict, 1024)


class toric_analysis:
    """Class to encapsulate tools used for toric analysis.
    Works with scipy 0.8.0 and scipy 0.10.0 and numpy 1.6

    Typical invocation:
    import toric_tools
    R=toric_tools.toric_analysis('toric.ncdf',mode='ICRF') #'LH' for lower hybrid
    R.plot_2Dfield(component='Re2Ezeta',logl=10) #etc

    Important functions:
    R.info() # netcdf contents and metadata
    R.plotpower(power='PwIF',species=1) #different power profiles
    R.plot_2Dfield(component='Re2Ezeta',logl=10) #Two Dim plots of quantities
    R.plot_1Dfield(component='Re2Ezeta') #One Dim plots of quantities
    R.threeplots() #2D Ez, electron power and poynting flux and poloidal spectrum, works for LH only, presently
    """

    def __init__(
        self,
        toric_name="None",
        mode="LH",
        idebug=False,
        comment="",
        layout="poster",
        path="./",
    ):
        import socket
        from time import gmtime

        __version__ = 1.0

        self.toric_name = toric_name
        self.mode = mode
        self.__version__ = __version__
        self.idebug = idebug

        self.mylw = 1.0
        self.mypt = 18.0
        self.fsc = 2.0
        self.fw = "normal"  # 'bold'
        # self.set_layout(layout)

        self.path = path

        self.prov = {
            "user": "noname",
            "host": "noname",
            "gmtime": "notime",
            "runid": "noid",
            "path": "",
            "comment": "",
        }
        self.label = True
        self.equigs = {}
        self.toricdict = {}

        if self.mode[:2] == "LH":
            self.namemap = {
                "xpsi": "tpsi",
                "poynt": "vpoynt",
                "pelec": "S_eld",
                "e2d_z": "E2d_z_re",
                "xplasma": "x_plasma",
                "zplasma": "z_plasma",
                "xeqpl": "xeqpl",
            }
            if self.toric_name == "None":
                self.toric_name = "TORICLH.cdf"
        else:
            self.namemap = {
                "xpsi": "Pw_abscissa",
                "poynt": "PoyFlx",
                "pelec": "PwE",
                "e2d_z": "Re2Ezeta",
                "xplasma": "Xplasma",
                "zplasma": "Zplasma",
                "xeqpl": "Ef_abscissa",
            }
            if self.toric_name == "None":
                self.toric_name = "toric.ncdf"
        ##Open the toric netcdf file read only
        try:
            self.cdf_hdl = nc.netcdf_file(self.toric_name, mmap=False)  # ,'r')
        except IOError:
            print("\t\t\t* CRITICAL: ", self.toric_name, " not found.")
            self.cdf_hdl = -1

        try:
            self.qlde_hdl = nc.netcdf_file(path + "toric_qlde.cdf", mmap=False)  # ,'r')
        except IOError:
            print("\t\t\t* Non-CRITICAL: ", path + "toric_qlde.cdf", " not found.")
            self.qlde_hdl = -1

        self.prov["host"] = socket.getfqdn()
        self.prov["user"] = os.getenv("USER")
        self.prov["gmtime"] = gmtime()
        self.prov["comment"] = comment
        self.prov["path"] = os.path.abspath("") + "/" + self.toric_name

        # marker sequence
        self.markers = ["o", "1", "2", "s", "*", "+", "H", "x"]
        self.ls = ["-", "--", "-.", ":", "--", "-.", ":"]
        self.bw = False

        return

    def close(self):
        try:
            self.cdf_hdl.close()
        except IOError:
            print("\t\t\t* CRITICAL: ", self.toric_name, " not found.")

        try:
            self.qlde_hdl.close()
        except IOError:
            print("\t\t\t* Non-CRITICAL: ", path + "toric_qlde.cdf", " not found.")

        return

    def info(self):
        "Prints a list of the contents of present TORIC3D output files"

        if self.cdf_hdl != -1:
            for hdl in [self.cdf_hdl]:
                print("The toric file, ", self.toric_name, ", contains:")
                print("----------------------------------------------")
                print("The global attributes: ", hdl.dimensions.keys())
                print("File contains the variables: ", hdl.variables.keys())

        if self.qlde_hdl != -1:
            for hdl in [self.qlde_hdl]:
                print("The toric file, ", self.toric_name, ", contains:")
                print("----------------------------------------------")
                print("The global attributes: ", hdl.dimensions.keys())
                print("File contains the variables: ", hdl.variables.keys())

        print("----------------------------------------------")
        print("Provenance metadata: ", self.prov)

        return

    def toricparam(self):
        "Fill a dictionary of toric scalar values"

        self.toricdict["comment"] = "Dictionary of relevant toric scalars"
        self.toricdict["mode"] = self.mode
        self.toricdict["nelm"] = self.cdf_hdl.dimensions["nelm"]
        self.toricdict["nptpsi"] = self.cdf_hdl.dimensions["nptpsi"]
        self.toricdict["ntt"] = self.cdf_hdl.dimensions["ntt"]

        return

    def plotb0(self, ir=45, db=0, eps=0):
        """Contour plot of bounce averaged Dql coefficient, dB0"""
        if self.qlde_hdl == -1:
            print("qlde File not found")
            return
        if eps > 0:
            db = 2 * eps / (1.0 - eps)

        dqlpsi = self.qlde_hdl.variables["Psi"].data
        dqltemp = self.qlde_hdl.variables["Tem"].data
        dql_LD = self.qlde_hdl.variables["Qldce_LD"].data
        nuperp = self.qlde_hdl.dimensions["VelPrpDim"]
        nupar = self.qlde_hdl.dimensions["VelDim"]

        umax = (self.qlde_hdl.variables["Umax"].data)[0]
        umin = (self.qlde_hdl.variables["Umin"].data)[0]
        upar = np.arange(nupar) / float(nupar - 1) * (umax - umin) + umin
        uperp = np.arange(nuperp) / float(nuperp - 1) * umax
        vx, vz = np.meshgrid(uperp, upar)

        fig = plt.figure(figsize=(2.0 * 8.3, 2.0 * 3.7))
        plt.axes().set_aspect(1, "box")
        # plot passing trapped boundary
        roa = dqlpsi[ir]
        if eps < 0:
            db = 2.0 * np.abs(eps) * roa / (1.0 - np.abs(eps) * roa)

        if db > 0:
            vpar = np.sqrt(db) * umax
            plt.plot([0, vpar], [0, umax], "k", [0, -vpar], [0, umax], "k", linewidth=2)

        dq = np.transpose(np.log((dql_LD[ir, :, :]) + 1.0) / np.log(10))  # np.abs
        mxdq = int(dq.max())
        ll = range(mxdq - 10, mxdq)
        cd = plt.contourf(vz, vx, dq, levels=ll)
        # ,10)
        plt.gca().set_ylim(0, umax)
        cbar = plt.colorbar(cd)

        plt.title(r"log10 $\lambda$<B> at r/a=" + str(roa)[0:4], size=30)
        plt.ylabel(r"$u_{\bot0}/u_{n}$", size=20)
        plt.xlabel(r"$u_{||0}/u_{n}$", size=20)
        plt.draw()  # make sure ylimits are updated in displayed plot
        return

    def plotpower(self, ax, xaxis=None, power=None, species=None, label=""):
        """Plot power profiles versus specified radius for all, or listed species.
        Overplots by default. species is 0 indexed.
        """

        l = -1
        if self.mode[:2] == "LH":
            if xaxis == None:
                xaxis = "tpsi"
            l = self.__plot1D(
                ax, xaxis, "S_eld", "Power absorbed on electrons", label=label
            )
            ax.set_xlabel(r"$\sqrt{\psi_{pol}}$")
        else:
            if xaxis == None:
                xaxis = "Pw_abscissa"
            if power == None:
                power = "PwE"
            if species == None:  # or species==0):
                l = self.__plot1D(ax, xaxis, power, label=label)
            else:
                nspec = self.cdf_hdl.dimensions["SpecDim"]
                if species <= nspec and species > 0:
                    l = self.__plot1D(
                        ax, xaxis, power, idx2=species - 1, label=label
                    )  # zero indexing
                else:
                    print("Invalid species label:" + str(species))
            ax.set_xlabel(r"$\sqrt{\psi_{pol}}$")
        # cf=plt.gcf()
        # cf.subplots_adjust(bottom=0.14)

    def psiplot(self, y):
        "Plot versus rhopsi. Returns handle on line to modify line style if desired using setp."
        psi = self.namemap["xpsi"]

        line = self.__plot1D(psi, y)
        plt.xlabel(r"$\sqrt{\psi_{pol}}$")
        plt.ylabel(y)

        return line

    def plot_1Dfield(self, ax, component, label="", offsetX=0, multX=1.0):
        "Field versus midplane specified."

        line = self.__plot1D(
            ax,
            self.namemap["xeqpl"],
            component,
            "",
            component,
            label=label,
            offsetX=offsetX,
            multX=multX,
        )
        plt.xlabel(r"$X[cm]$")
        return line

    def __plot1D(
        self,
        ax,
        xvar,
        yvar,
        ptitle="",
        plabel="",
        idx2=None,
        label="",
        offsetX=0,
        multX=1.0,
    ):
        "Internal 1D plot"

        x = self.cdf_hdl.variables[xvar]  # .data
        y = self.cdf_hdl.variables[yvar]  # .data
        if self.mode[:2] == "LH":
            xname = ""
            yname = plabel
        else:
            xname = x.long_name
            yname = (y.long_name[0:10]).decode("UTF-8") + y.units.decode("UTF-8")
        x = x.data
        y = y.data
        if idx2 != None:
            if len(y.shape) == 2:
                y = y[:, idx2]
            else:
                print(yvar + " has wrong number of dims in __plot1d.")
        if np.size(y) > np.size(x):
            print("ToricTools.__plot1D resizing", yvar)
            y = np.array(y)[0 : np.size(x)]

        line = ax.plot(x * multX + offsetX, y, label=label)
        ax.set_title(ptitle)
        ax.set_xlabel(str(xname))
        ax.set_ylabel(str(yname[:25]))  # limit length to 25 char

        return line

    def __map_celef(self):
        """Mapping toric field in celef.cdf to re and im parts of the three components
        for LH mode.
        """

        return

    def __getvar__(self, name):
        """Internal function to retrieve variable from data file with checking."""

        try:
            value = self.cdf_hdl.variables[name].data
        except NameError:
            print("CRITICAL: variable not found")
            exit  # raise Exception,'Bad variable name in getvar: %s' % name

        return value

    def fft(self, component="undef", maxr=1.0):
        if self.mode[:2] == "LH":
            radius = "psime"
        if component == "undef":
            if self.mode[:2] == "LH":
                component = "E2d_z_re"
            else:
                component = "Re2Ezeta"

        field = self.__getvar__(component)
        rad = self.__getvar__(radius)
        # field taken to be 2D with shape (ntheta,npsi)
        ntt = field.shape[0]
        nelm = int(field.shape[1] * maxr)
        nlevels = 100
        levels = np.arange(nelm / nlevels, nelm - 1, nelm / nlevels)
        fftfield = np.zeros((ntt, levels.shape[0]), "complex128")
        i = 0
        for ir in levels:
            ffield = ft.fft(field[:, ir])
            fftfield[:, i] = ffield
            i = i + 1

        return fftfield

    def spectrum(self, component="undef", maxr=1.0, cx=0, levels=-1, q=None):
        """Calculate poloidal spectrum of two dimensional field component."""

        if self.mode[:2] == "LH":
            radius = "psime"
        else:
            radius = "Pw_abscissa"

        if component == "undef":
            if self.mode[:2] == "LH":
                component = "E2d_z_re"
                componenti = "E2d_z_im"
            else:
                component = "Re2Eplus"
                componenti = "Im2Eplus"

        f = plt.figure()

        if component == "power":
            field = self.get_power2D()
        else:
            field = self.__getvar__(component)  # [:,:]

        if cx == 1:
            fieldi = self.__getvar__(componenti)  # [:,:]
            field = np.array(field) + np.complex(0.0, 1.0) * np.array(fieldi)

        rad = self.__getvar__(radius)

        # field taken to be 2D with shape (ntheta,npsi)
        field = field + 1.0e-20
        ntt = field.shape[0]
        # nelm=int(field.shape[1]*maxr)
        nelm = int(np.size(rad) * maxr)
        if np.size(levels) == 1:
            nlevels = 7
            #            levels=np.arange(nelm/nlevels,nelm-1,nelm/nlevels)
            levels = np.arange(nlevels) * nelm / nlevels
        else:
            levels = (np.array(levels) * nelm).astype(int)
            nlevels = np.size(levels)

        levels = levels[1:nlevels]
        #        levels=nelm-np.arange(1,20,2)
        rlevels = rad[levels]

        th = np.arange(ntt) - ntt / 2

        ymax = 0.0
        ymin = 0.0

        i = 0
        thq = th
        for indr in range(levels.size):  # levels:  #fft in python isn't normalized to N
            ir = levels[indr]
            print("levels", ir, levels[indr], rlevels[indr], th)
            if q != None:
                thq = (
                    -2.5
                    * (1 + 0.3)
                    / (1 + 0.3 * rlevels[indr])
                    * (1 + th / 191.0 / q(rlevels[indr]))
                )

            print(thq)
            ffield = ft.fftshift(
                np.log10(abs(ft.fft(field[:, ir])) / float(ntt) + 1.0e-20)
            )
            ymax = np.max([ymax, np.max(ffield)])
            ymin = np.min([ymin, np.min(ffield)])
            plabel = f"{rad[ir]:5.2f}"
            if self.bw:
                plt.plot(thq, ffield, label=plabel, linestyle=self.ls[i], color="k")
                i = i + 1
            else:
                plt.plot(thq, ffield, label=plabel)

        ffield = ft.fftshift(
            np.log10(abs(ft.fft(field[:, nelm - 1])) / float(ntt) + 1.0e-20)
        )
        ymax = np.max([ymax, np.max(ffield)])
        ymin = np.min([ymin, np.min(ffield)])
        # plot antenna spectrum
        plabel = "ant"
        print("range, levels", rlevels)
        print("ymax", ymax, ymin)
        plt.plot(thq, ffield, label=plabel, color="grey", linewidth=2)
        cf = plt.gcf()
        cf.subplots_adjust(right=0.76)
        plt.axis("tight")
        if q != None:
            plt.axis(xmin=-8, xmax=8)
        else:
            plt.axis(xmin=-ntt / 4, xmax=ntt / 4)
        plt.axis(ymin=-10)
        plt.legend(loc=(1.05, 0))
        plt.xlabel("m")
        plt.ylabel("log10 scale")
        plt.title("Poloidal spectrum on labeled flux surfaces")
        plt.draw()
        return

    def set_layout(self, layout="poster"):
        if layout == "paper":
            self.mylw = 1.0
            self.mypt = 10.0
            self.fsc = 1.0
            self.fw = "normal"

        params = {
            "axes.linewidth": self.mylw,
            "lines.linewidth": self.mylw,
            "axes.labelsize": self.mypt,
            "font.size": self.mypt,
            "legend.fontsize": self.mypt,
            "axes.titlesize": self.mypt + 2.0,
            "xtick.labelsize": self.mypt,
            "ytick.labelsize": self.mypt,
            "font.weight": self.fw,
        }
        plt.rcParams.update(params)

        return

    # note that if plot commands are in the toplevel, they will not return
    # to the prompt, but wait to be killed.
    def plot_2Dfield(
        self,
        ax,
        component="E2d_z",
        species=None,
        logl=0,
        xunits=1.0,
        axis=(0.0, 0.0),
        im=False,
        scaletop=1.0,
        fig="undef",
        offsetX=0,
        multX=1,
        title="",
    ):
        """

        example of using netcdf python modules to plot toric solutions
        requires numpy and matplotlib and netcdf modules for python.

        Note, under windows you need netcdf.dll installed in SYSTEM32 and the file
        system cannot follow symbolic links.  The DLL needs to have executable
        permissions.

        To overplot with limiter, made from efit plotter:
        R.plot_2Dfield(component='Im2Eplus',logl=20,xunits=0.01,axis=maxis,fig=fig1)

        Easier is to plot solution first, then overplot limiter, scaled appropriately:
        p.plot ( rlim*100.-maxis[0], zlim*100.-maxis[1], 'k', linewidth = 2 )

        """

        R0 = axis[0]
        Z0 = axis[1]
        barfmt = None  #'%5.2e' #'%4.1e' #'%3.1f'
        # what should colorbar with be? format=4.1e means 8 characters
        # the bar and title of the bar add about 4 characters.
        # there are 72.27 pt/in
        # 12 characters * self.mypt /72.27 pt/in = #in
        legend_frac = 12 * self.mypt / 72.27

        xx = self.cdf_hdl.variables[self.namemap["xplasma"]].data
        yy = self.cdf_hdl.variables[self.namemap["zplasma"]].data

        if self.mode[:2] == "LH":
            if im:
                im_e2dname = component + "_im"
                title = "|" + component + "|"
                component = component + "_re"
        else:
            if component == "E2d_z":
                component = "Ezeta"
            if im:
                im_e2dname = "Im2" + component
                title = "|" + component + "|"
                component = "Re2" + component

        # note change to use ().data instead of np.array() in scipy0.8.0
        if component == "power" and self.mode[:2] == "LH":
            e2d = self.get_power2D()
        else:
            e2d = (self.cdf_hdl.variables[component]).data

        if im:
            im_e2d = (self.cdf_hdl.variables[im_e2dname]).data
            e2d = abs(e2d + np.complex(0.0, 1.0) * im_e2d)

        if self.mode[:2] != "LH" and species:
            # print('plot2D, indexing species', species)
            e2d = e2d[:, :, species - 1]

        # print ("2D Matrix shape:", np.shape(xx))

        # contour with 3 args is confused unless arrays are indexed slices
        # need wrapper to close periodicity in theta direction for this
        # tricky, array indexing different from ncdf slicing
        # this step is needed because periodic dimension is not closed.
        # i.e. its [0,pi) not [0,pi]
        dd = np.shape(xx)
        sx = dd[0]
        sy = dd[1]

        xxx = np.zeros((sx + 1, sy), "d")
        xxx[0:sx, :] = xx[:, :]
        xxx[sx, :] = xx[0, :]
        yyy = np.zeros((sx + 1, sy), "d")
        yyy[0:sx, :] = yy[:, :]
        yyy[sx, :] = yy[0, :]

        xxx = (xxx + R0) * xunits
        yyy = (yyy + Z0) * xunits

        ee2d = np.zeros((sx + 1, sy), "d")
        ee2d[0:sx, :] = e2d[:, :]
        ee2d[sx, :] = e2d[0, :]
        emax = ee2d.ravel()[ee2d.argmax()]
        emin = ee2d.ravel()[ee2d.argmin()]

        # contouring levels
        rmax = max([abs(emax), abs(emin)]) * scaletop
        rmin = min([0.0, emax, emin])
        # val=arange(emin,emax,(emax-emin)/25.,'d')
        val = np.arange(-rmax * 1.1, rmax * 1.1, (rmax + rmax) / 25.0, "d")
        if im:
            val = np.arange(0.05 * rmax, rmax * 1.1, (rmax) / 24.0, "d")
            # print ("values",val)

        # reverse redblue map so red is positive
        #            revRBmap=cmap_xmap(lambda x: 1.-x, cm.get_cmap('RdBu'))

        # finally, make the plot
        cwidth = xxx.max() - xxx.min()
        cheight = yyy.max() - yyy.min()
        asp = cheight / cwidth
        # print ("plot aspect ratio:", asp)

        # leave space for bar
        # if (fig=='undef'):
        #     fig=plt.figure(figsize=(self.fsc*3.0+legend_frac,3.0*self.fsc*asp))
        #     fig.subplots_adjust(left=0.02,bottom=0.15,top=0.90)

        maxpsi = xxx.shape[1] - 1
        ax.plot(xxx[:, maxpsi] * multX + offsetX, yyy[:, maxpsi] * multX, "k-")

        # add LCF
        lcfpsi = self.cdf_hdl.dimensions["PsiPwdDim"]
        ax.plot(xxx[:, lcfpsi] * multX + offsetX, yyy[:, lcfpsi] * multX, "grey")

        # read ant length.  Calculate arc length vs theta to this value/2
        # in each direction, this plots the antenna location
        # anthw=max(int(sx*0.01),4)
        # ax.plot(xxx[sx-anthw+1:sx+1,maxpsi]*multX+offsetX,yyy[sx-anthw+1:sx+1,maxpsi]*multX,'g-',linewidth=3)
        # ax.plot(xxx[0:anthw,maxpsi]*multX+offsetX,yyy[0:anthw,maxpsi]*multX,'g-',linewidth=3)

        # if self.label:
        #     sublabel=self.prov['path']
        #     print (sublabel)
        #     ax.text(-0.2,-0.2,sublabel,transform = ax.transAxes)

        # print ("interactive off while plotting")
        #

        if logl > 0:
            title = "log10 " + title
            barfmt = None  #'%3.1f'

        ##labels and titles
        # xlabel(getattr(xx,'long_name')+'('+getattr(xx,'units')+')')
        # ylabel(getattr(yy,'long_name')+'('+getattr(yy,'units')+')')
        # title(getattr(e2d,'long_name')+'('+getattr(e2d,'units')+')')
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        #        plt.title(r'$Re E_{||}$',fontsize=self.mypt+2.0)
        # ax.set_title(title,fontsize=self.mypt+2.0)

        if logl <= 0:
            CS = ax.contourf(
                xxx * multX + offsetX, yyy * multX, ee2d, val * scaletop
            )  # 30APR2009 removed *0.2

        if logl > 0:
            #            lee2d=np.sign(ee2d)*np.log(np.sqrt(np.abs(ee2d)**2+1)+np.abs(ee2d))/np.log(10)
            lee2d = np.log(np.abs(ee2d) + 0.1) / np.log(10)
            CS = ax.contourf(xxx * multX + offsetX, yyy * multX, lee2d, logl)

        # print ("interactive on")
        #        plt.ion()
        ##put the contour scales on the plot
        # tricky, fraction needs to be specified to be part by which horizontal exceed vertical

        sax = ax.set_aspect(1, "box")  # the right way to control aspect ratio
        barfmt = None  #'%3.1f'
        cbar = GRAPHICStools.addColorbarSubplot(
            ax, CS, barfmt=barfmt, title=title
        )  # ,fontsize=6,fontsizeTitle=10)

        # print ("contour values",CS.levels,'xx',rmax,rmin)

        return CS, cbar

    # Handling equigs file
    def __get_varname(self, f):
        "Reads next line from file f and returns it, optionally printing it."
        varname = f.readline()
        if self.idebug:
            print(f.name, varname)
        return varname

    def read_equigs(self, equigsfile="equigs.data"):
        "Read the equilibrium file created by toric in toricmode='equil',isol=0."
        if self.idebug:
            print("Using ", equigsfile)
        equigs_hdl = file(equigsfile, "r")

        varname = self.__get_varname(equigs_hdl)
        self.equigs["rtorm"] = np.fromfile(equigs_hdl, sep=" ", count=1, dtype=float)

        varname = self.__get_varname(equigs_hdl)
        self.equigs["raxis"] = np.fromfile(equigs_hdl, sep=" ", count=1, dtype=float)

        varname = self.__get_varname(equigs_hdl)
        self.equigs["bzero"] = np.fromfile(equigs_hdl, sep=" ", count=1, dtype=float)

        varname = self.__get_varname(equigs_hdl)
        self.equigs["torcur"] = np.fromfile(equigs_hdl, sep=" ", count=1, dtype=float)

        varname = self.__get_varname(equigs_hdl)
        self.equigs["imom"] = np.fromfile(equigs_hdl, sep=" ", count=1, dtype=int)
        imom = self.equigs["imom"]

        varname = self.__get_varname(equigs_hdl)
        self.equigs["nmhd"] = np.fromfile(equigs_hdl, sep=" ", count=1, dtype=int)
        nmhd = self.equigs["nmhd"]

        varname = self.__get_varname(equigs_hdl)
        self.equigs["srad"] = np.fromfile(equigs_hdl, sep=" ", count=nmhd, dtype=float)

        # this needs to be reshaped or remapped into the R,Z sin cos arrays toric uses
        varname = self.__get_varname(equigs_hdl)
        self.equigs["rzmcs2d"] = np.fromfile(
            equigs_hdl, sep=" ", count=2 * nmhd + 4 * nmhd * imom, dtype=float
        )

        varname = self.__get_varname(equigs_hdl)
        self.equigs["qqf"] = np.fromfile(equigs_hdl, sep=" ", count=nmhd, dtype=float)

        # logic checking for "END"
        varname = self.__get_varname(equigs_hdl)
        self.equigs["jcurr"] = np.fromfile(equigs_hdl, sep=" ", count=nmhd, dtype=float)

        varname = self.__get_varname(equigs_hdl)
        self.equigs["gcov"] = np.fromfile(equigs_hdl, sep=" ", count=nmhd, dtype=float)

        varname = self.__get_varname(equigs_hdl)
        self.equigs["rhotor"] = np.fromfile(
            equigs_hdl, sep=" ", count=nmhd, dtype=float
        )

        varname = self.__get_varname(equigs_hdl)
        self.equigs["lastpsi"] = np.fromfile(equigs_hdl, sep=" ", count=1, dtype=float)

        equigs_hdl.close()

    ### user routines using the above, could be in a different module
    def powpoynt(self):
        "Plots electron power and poynting flux"
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        (line1,) = self.psiplot(self.namemap["pelec"])
        # can use setp(lines, ) to change plot properties.
        plt.setp(line1, color="b", marker="+", label="seld")
        ax1.set_ylabel("Power_e", color="b")

        ax2 = ax1.twinx()
        (line2,) = self.psiplot(self.namemap["poynt"])
        # set axis floor at 0
        plt.gca().set_ylim(0)
        # change color and symbol
        plt.setp(line2, color="r", marker=".", label="Poynt")
        ax2.set_ylabel("Poynting", color="r")
        # make  legend too
        plt.legend((line1, line2), (r"$P_{eld}$", "<ExB>"), loc=2)
        fig.subplots_adjust(left=0.12, bottom=0.12, top=0.96, right=0.82, hspace=0.32)
        plt.draw()
        return fig

    def powerion(self):
        "Plots electron power and poynting flux"
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        (line1,) = self.psiplot(self.namemap["pelec"])
        # can use setp(lines, ) to change plot properties.
        plt.setp(line1, color="b", marker="+", label="seld")
        ax1.set_ylabel("Power_e", color="b")

        ax2 = ax1.twinx()
        (line2,) = self.psiplot(self.namemap["poynt"])
        # set axis floor at 0
        plt.gca().set_ylim(0)
        # change color and symbol
        plt.setp(line2, color="r", marker=".", label="Poynt")
        ax2.set_ylabel("Poynting", color="r")
        # make  legend too
        plt.legend((line1, line2), (r"$P_{eld}$", "<ExB>"), loc=2)
        fig.subplots_adjust(left=0.12, bottom=0.12, top=0.96, right=0.82, hspace=0.32)
        plt.draw()
        return fig

    def xpsi_map(self):
        """Return map of x(theta=0)/x(psi=1,theta=0) versus psipol."""
        xmap = 1.0

        return xmap

    def get_power2D(self):
        # figure out a sed way of cutting these lines into the file.
        # also need to replace '-0.' with ' -0.'
        # sed -n -e '/elec/,/,/p' filename | sed -e '/-0\./ -0./g' > torica_2dpower.sol
        try:
            toricsol = file("torica_2dpower.sol", "r")
        except IOError:
            print("CRITICAL: torica_2dpower.sol not found.")
            print("Try to generate:")
            #            cmd="sed -n '/elec/,/,/ s/-0\./ -0./pg'  < torica.sol > torica_2dpower.sol"
            if self.mode[:2] == "LH":
                cmd = "sed -n  '/elec/,$p' torica.sol| sed 's/-0\./ -0./g' > torica_2dpower.sol"
            else:
                cmd = "sed -n  '/elec/,$p' toric.sol| sed 's/-0\./ -0./g' > torica_2dpower.sol"

            os.system(cmd)
            toricsol = file("torica_2dpower.sol", "r")

        # skip title and max value
        toricsol.readline()
        toricsol.readline()

        if self.mode[:2] == "LH":
            nt = self.cdf_hdl.dimensions["ntt"]
            nr = self.cdf_hdl.dimensions["mptpsi"]
        else:
            nt = self.cdf_hdl.dimensions["ThetaDim"]
            nr = self.cdf_hdl.dimensions["PsiPwdDim"]

        power = np.fromfile(toricsol, sep=" ", count=nt * nr, dtype=float)
        toricsol.close()

        power = np.transpose(np.reshape(power, (nr, nt)))
        return power

    def threeplots(self, prefix=""):
        """Makes and saves the three most commonly used plots. Plots are saved in
        the current directory. An optional prefix can be used to label them or change
        the save path.
        * Power and poynting flux on one plot as eps.
        * The polodial power spectrum on six flux surfaces for convergence as eps.
        * And the 2D parallel electric field contour plot as a png."""

        self.spectrum(cx=1)
        plt.draw()
        plt.savefig(prefix + "spectrum.pdf", format="pdf")
        plt.savefig(prefix + "spectrum.png", format="png")

        if self.mode[:2] != "LH":
            self.plot_2Dfield(component="Eplus", im=True, logl=25)
            plt.draw()
            plt.savefig("log10Eplus2d.png", format="png")
            self.plot_2Dfield(component="Eplus")
            plt.draw()
            plt.savefig("Eplus2d.png", format="png")

        self.plot_2Dfield(im=True, logl=25)
        plt.draw()
        plt.savefig(prefix + "log10Ez2d.png", format="png")

        self.powpoynt()
        plt.draw()
        plt.savefig(prefix + "powerpoynt.pdf", format="pdf")
        plt.savefig(prefix + "powerpoynt.png", format="png")

        return


####main block
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import toric_tools
    import sys
    import getopt

    # get file name if provided
    iprefix = ""
    ifile = "TORICLH.cdf"
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hp:f:", ["help", "prefix=", "file="])
    except getopt.GetoptError:
        print("Accepted flags are help and prefix=")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-p", "--prefix"):
            iprefix = arg
        elif opt in ("-f", "--file"):
            ifile = arg
        elif opt in ("-h", "--help"):
            print(toric_tools.toric_analysis.__doc__)
            print('run "help toric_tools.toric_analysis" for help on whole class')

    # Load a run
    LHRun = toric_tools.toric_analysis(toric_name=ifile)
    LHRun.threeplots(prefix=iprefix)


# make sequence of plots, ala the old toric idl driver as an option.
