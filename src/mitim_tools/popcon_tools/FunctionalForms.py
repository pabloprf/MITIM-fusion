import pdb
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from mitim_tools.popcon_tools.utils import PRFfunctionals, FUNCTIONALScalc
from mitim_tools.misc_tools import MATHtools, IOtools, FARMINGtools
from mitim_tools.misc_tools.IOtools import printMsg as print

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PARABOLIC
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def parabolic(Tbar=5, nu=2.5, rho=None, Tedge=0):
    if rho is None:
        rho = np.linspace(0, 1, 100)

    nu_mod = (nu - Tedge / Tbar) / (1 - Tedge / Tbar)

    T = (Tbar - Tedge) * nu_mod * ((1.0 - rho**2.0) ** (nu_mod - 1.0)) + Tedge

    T[-1] += 1e-5  # To avoid zero problems

    return rho, T


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# H-mode
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def PRFfunctionals_Hmode(T_avol, n_avol, nu_T, nu_n, aLT=2.0, width_ped=0.05, rho=None):
    return PRFfunctionals.nTprofiles(
        T_avol, n_avol, nu_T, nu_n, aLT=aLT, width_ped=width_ped, rho=rho
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# L-mode
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def PRFfunctionals_Lmode(T_avol, n_avol, nu_n, rho=None, debug=False):
    """
    Double linear

    Method: Cubic spline to search for value of edge gradient that provides vol avg
    """

    points_search = 50

    if rho is None:
        x = np.linspace(0, 1, 100)
    else:
        x = rho

    x = np.repeat(np.atleast_2d(x), points_search, axis=0)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ Temperature
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    g2_T = 3
    g1Range_T = [1, 50]
    xtransition_T = 0.8
    T_roa1 = 0.25

    gs, Tvol = np.linspace(g1Range_T[0], g1Range_T[1], points_search), []

    T = FUNCTIONALScalc.doubleLinear_aLT(x, gs, g2_T, xtransition_T, T_roa1)
    Tvol = FUNCTIONALScalc.calculate_simplified_volavg(x, T)
    g1 = MATHtools.extrapolateCubicSpline(T_avol, Tvol, gs)
    T = FUNCTIONALScalc.doubleLinear_aLT(
        np.atleast_2d(x[0]), g1, g2_T, xtransition_T, T_roa1
    )[0]

    if g1 < g1Range_T[0] or g1 > g1Range_T[1]:
        print(f">> Gradient aLT outside of search range ({g1})", typeMsg="w")

    if debug:
        fig, ax = plt.subplots()
        ax.plot(gs, Tvol, "-s")
        ax.axhline(y=T_avol)
        ax.axvline(x=g1)
        plt.show()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~ Density
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    n_roa1 = n_avol / 3

    _, n = parabolic(Tbar=n_avol, nu=nu_n, rho=x[0], Tedge=n_roa1)

    # g2_n 		= 1
    # g1Range_n 	= [0.1,10]
    # xtransition_n 		= 0.8

    # gs,nvol = np.linspace(g1Range_n[0],g1Range_n[1],points_search), []

    # n 		= FUNCTIONALScalc.doubleLinear_aLT(x,gs,g2_n,xtransition_n,n_roa1)
    # nvol 	= FUNCTIONALScalc.calculate_simplified_volavg(x,n)
    # g1 		= MATHtools.extrapolateCubicSpline(n_avol,nvol,gs)
    # n 		= FUNCTIONALScalc.doubleLinear_aLT(np.atleast_2d(x[0]),g1,g2_n,xtransition_n,n_roa1)[0]

    # if g1<g1Range_n[0] or g1>g1Range_n[1]: print(f'>> Gradient aLn outside of search range ({g1})',typeMsg='w')

    return x[0], T, n



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Belonging to old PEDmodule.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def create_dummy_plasmastate(file, rho, rhob, psipol, ne, te, ti):
    """
    To "trick" the IDL routines to think this is a plasmstate file to extract the pdestal features

    """

    # Create file
    ncfile = netCDF4.Dataset(file, mode="w", format="NETCDF4_CLASSIC")

    # --------------------------------
    # Store profiles (in center grid)
    # --------------------------------

    # Dimensions
    ncfile.createDimension("xdim", rho.shape[0])
    ncfile.createDimension("sdim", 2)
    ncfile.createDimension("ndim", 1)

    value = ncfile.createVariable("s_name", "S1", ("sdim",))
    value[:] = np.array(["e", "i"], dtype="S1")

    value = ncfile.createVariable(
        "ns",
        "f4",
        (
            "sdim",
            "xdim",
        ),
    )
    value[0, :] = ne

    value = ncfile.createVariable("ns_bdy", "f4", ("sdim",))
    value[:] = np.array([ne[-1], ne[-1]])

    value = ncfile.createVariable(
        "Ts",
        "f4",
        (
            "sdim",
            "xdim",
        ),
    )
    value[0, :] = te

    value = ncfile.createVariable("Te_bdy", "f4", ("ndim",))
    value[:] = te[-1]

    value = ncfile.createVariable("Ti", "f4", ("xdim",))
    value[:] = ti

    value = ncfile.createVariable("Ti_bdy", "f4", ("ndim",))
    value[:] = ti[-1]

    # --------------------------------
    # Store rho and psi (in boundary grid with an extra zero)
    # --------------------------------

    # add one more point to boundary
    rhob = np.append([0], rhob)
    psipol = np.append([0], psipol)

    ncfile.createDimension("xdim_rhob", rhob.shape[0])

    value = ncfile.createVariable("rho", "f4", ("xdim_rhob",))
    value[:] = rhob

    value = ncfile.createVariable("psipol", "f4", ("xdim_rhob",))
    value[:] = psipol

    ncfile.close()



def fit_pedestal_mtanh(
    width_top,
    netop,
    p1,
    ptop,
    plasmastate,
    folderWork="~/scratchFolder/",
    nameRunid=1,
    tetop_previous=None,
    debug=False,
    ):
    """

    Inputs:
            netop in 1E20 m^-3
            width_top in psin
            pressures in Pa

            (Note that temperature is defined from these inputs, in the IDL routine)

            tetop_previous is just for testing how different the value of the IDL calculated tetop from the pressure. Discrepancies
            may be due simply to the linear interpolation

    Outputs:
            x is psin
            ne in 1E20 m^-3
            Te, Ti in keV

    """

    pedestal_job = FARMINGtools.mitim_job(folderWork)

    pedestal_job.define_machine(
        "idl",
        f"mitim_idl_{nameRunid}/",
        launchSlurm=False,
    )

    path = pedestal_job.folderExecution
    plasmastate_path = path + IOtools.reducePathLevel(plasmastate, isItFile=True)[-1]

    with open(folderWork + "/idl_in", "w") as f:
        f.write(".r /home/nthoward/SPARC_mtanh/make_pedestal_profiles_portals.pro\n")
        f.write(
            f"make_profiles,[{width_top},{netop},{p1},{ptop}],'{plasmastate_path}','{path}'\n\n"
        )
        f.write("exit")

    start = "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/labombard/idl_lib/fortran && export IDL_STARTUP=/home/nthoward/idl/idl_startup"

    command = f"cd {path} && {start} && idl < idl_in"

    inputFiles = [folderWork + "/idl_in", plasmastate]
    outputFiles = ["mtanh_fits"]

    print(f"\t\t- Proceeding to run idl pedestal fitter (psi_pol = {width_top:.3f})")

    pedestal_job.prep(
        command,
        output_files=outputFiles,
        input_files=inputFiles,
    )

    pedestal_job.run(timeoutSecs=30)

    x, ne, Te, Ti = read_mtanh(folderWork + "/mtanh_fits")

    tetop = ptop / netop / 3.2e1 * 1e-3

    print(
        "\t- Fitted pedestal, resulting in Tetop = {0:.2f}keV, netop = {1:.1f}E19 m^-3, psipol={2:.3f}".format(
            tetop, netop * 10, 1 - width_top
        )
    )

    if tetop_previous is not None:
        percent_change = np.abs(tetop_previous - tetop) / tetop * 100
        if percent_change > 0.5:
            print(
                "\t\t- Tetop differs from the previous reported value by {0:.1f}% (likely because of linearly interpolating ptop instead of netop and Tetop)".format(
                    percent_change
                ),
                typeMsg="w",
            )

    if debug:
        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
        ax = axs[0]
        ax.plot(x, Te, "-o", markersize=1, lw=1.0, label="fit")
        ax.scatter([1 - width_top], [tetop], label="top")
        ax = axs[1]
        ax.plot(x, Ti, "-o", markersize=1, lw=1.0)
        # ax.scatter([1-width_top],[titop])
        ax = axs[2]
        ax.plot(x, ne, "-o", markersize=1, lw=1.0)
        ax.scatter([1 - width_top], [netop])

        from mitim_tools.transp_tools.utils.PLASMASTATEtools import Plasmastate

        p = Plasmastate(plasmastate)

        p.plot(axs=axs, label=".cdf")

        axs[0].legend()

        plt.show()

        pdb.set_trace()

    return x, ne, Te, Ti

def read_mtanh(file_out):
    with open(file_out, "r") as f:
        aux = f.readlines()

    v = []
    for i in aux:
        if len(i) > 0:
            a = [float(j) for j in i.split()]
        for k in a:
            v.append(k)

    nr = int(len(v) / 4)

    x = np.array(v[:nr])
    ne = np.array(v[nr : nr * 2])  # *1E-20
    Te = np.array(v[nr * 2 : nr * 3]) * 1e-3
    Ti = np.array(v[nr * 3 :]) * 1e-3

    return x, ne, Te, Ti

