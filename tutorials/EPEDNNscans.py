import numpy as np
from mitim_tools.misc_tools import GRAPHICStools, IOtools
from mitim_tools.surrogate_tools import NNtools

import matplotlib.pyplot as plt

def eped_1d_scan(model: NNtools.eped_nn, 
                 Ip, Bt, R, a, kappa995, delta995, neped, betan, zeff, 
                 tesep=75, 
                 nesep_ratio=0.3, 
                 varname=None, 
                 varrange=None,
                 plotYN=True
                 ):
    
    inputs = {
        "Ip": Ip,
        "Bt": Bt,
        "R": R,
        "a": a,
        "kappa995": kappa995,
        "delta995": delta995,
        "neped": neped,
        "betan": betan,
        "zeff": zeff,
        "tesep": tesep,
        "nesep_ratio": nesep_ratio
    }

    if varname is None:
        varname = 'neped'

    if varname in inputs.keys():
        if varrange is None:
            varrange = [inputs[varname] * 0.7, inputs[varname] * 1.3]  # default range is +/- 30%

        varrange = np.linspace(varrange[0], varrange[1], 50)

        ptop = []
        wtop = []

        # define the input parameters for the EPED NN
        for var in varrange:
            inputs[varname] = var
            inputs_array = np.array([inputs[key] for key in inputs.keys()])

            p, w = model(*inputs_array)
            ptop.append(p)
            wtop.append(w)

    else:
        raise ValueError(f"Variable {varname} not an EPED input")

    if plotYN:
        GRAPHICStools.prep_figure_papers(size=20, slower_but_latex=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        GRAPHICStools.addDenseAxis(ax)

        ax.plot(varrange, ptop)
        ax.set_ylabel(r"$P_{top}$ [kPa]")
        ax.set_xlabel(f"{varname}")
        ax.set_title("EPED-NN density scan")

    return varrange, ptop, wtop

def eped_2d_scan(model: NNtools.eped_nn,
                    Ip, Bt, R, a, kappa995, delta995, neped, betan, zeff,
                    tesep=75,
                    nesep_ratio=0.3,
                    varname1=None,
                    varrange1=None,
                    varname2=None,
                    varrange2=None,
                    plotYN=True
                    ):
    
    inputs = {
        "Ip": Ip,
        "Bt": Bt,
        "R": R,
        "a": a,
        "kappa995": kappa995,
        "delta995": delta995,
        "neped": neped,
        "betan": betan,
        "zeff": zeff,
        "tesep": tesep,
        "nesep_ratio": nesep_ratio
    }

    if varname1 is None:
        varname1 = 'neped'
    if varname2 is None:
        varname2 = 'betan'
    
    if varname1 in inputs.keys() and varname2 in inputs.keys():
        if varrange1 is None:
            varrange1 = [inputs[varname1] * 0.7, inputs[varname1] * 1.3]
        if varrange2 is None:
            varrange2 = [inputs[varname2] * 0.7, inputs[varname2] * 1.3]
        
        varrange1 = np.linspace(varrange1[0], varrange1[1], 20)
        varrange2 = np.linspace(varrange2[0], varrange2[1], 20)

        ptop = np.zeros((len(varrange1), len(varrange2)))
        wtop = np.zeros((len(varrange1), len(varrange2)))

        # define the input parameters for the EPED NN

        for i, var1 in enumerate(varrange1):
            for j, var2 in enumerate(varrange2):
                inputs[varname1] = var1
                inputs[varname2] = var2
                inputs_array = np.array([inputs[key] for key in inputs.keys()])

                p, w = model(*inputs_array)
                ptop[i, j] = p
                wtop[i, j] = w

    else:
        raise ValueError(f"Variable {varname1} or {varname2} not an EPED input")
    
    if plotYN:
        GRAPHICStools.prep_figure_papers(size=20, slower_but_latex=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        GRAPHICStools.addDenseAxis(ax)

        ax.contourf(varrange1, varrange2, ptop)
        ax.set_ylabel(f"{varname2}")
        ax.set_xlabel(f"{varname1}")
        ax.set_title("EPED-NN 2D scan")

    return varrange1, varrange2, ptop, wtop

def betan_scan_plot(model: NNtools.eped_nn, Ip, Bt, R, a, kappa995, delta995, zeff, tesep=75, nesep_ratio=0.3):
    greenwald_density = Ip / (np.pi * a ** 2) * 10  # in units of 10^19m^-3
    nped = np.linspace(1, greenwald_density, 50)
    betan_array = np.linspace(1.0, 2.0, 10)
    max_density = []

    # define the input parameters for the EPED NN
    for betan in betan_array:
        ptop = []
        wtop = []
        for neped in nped:
            inputs_array = np.array([Ip, Bt, R, a, kappa995, delta995, neped, betan, zeff, tesep, nesep_ratio])

            p, w = model(*inputs_array)
            ptop.append(p)
            wtop.append(w)

        max_p = np.max(ptop)
        max_p_index = np.argmax(ptop)
        max_n = nped[max_p_index]
        print(f"Maximum pedestal pressure: {max_p} kPa at {max_n} 1e19 m^-3")
        max_density.append(max_n)

    GRAPHICStools.prep_figure_papers(size=20, slower_but_latex=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    GRAPHICStools.addDenseAxis(ax)

    ax.scatter(betan_array, max_density)
    ax.set_ylabel(r"$n_{peak}$ [$10^{19}m^{-3}$]")
    ax.set_xlabel(r"$\beta_N$")
    ax.set_title("EPED-NN betan scan")


if __name__ == '__main__':
    nn = NNtools.eped_nn(type='tf')

    nn_location = IOtools.expandPath('$MFEIM_PATH/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-MODEL-ARC.h5')
    norm_location = IOtools.expandPath('$MFEIM_PATH/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-NORMALIZATION.txt')

    nn.load(nn_location, norm=norm_location)

    _,_,_ = eped_1d_scan(
        model=nn,
        Ip=10.95,
        Bt=10.8,
        R=4.25,
        a=1.17,
        kappa995=1.77,
        delta995=0.534,
        neped=18,
        betan=1.13288488,
        zeff=2.0,
        tesep=75,
        nesep_ratio=1. / 3.,
        varname="neped"
    )
    """
    _,_,_,_ = eped_2d_scan(
        model=nn,
        Ip=10.95,
        Bt=10.8,
        R=4.25,
        a=1.17,
        kappa995=1.77,
        delta995=0.534,
        neped=18,
        betan=1.5,
        zeff=2.0,
        tesep=75,
        nesep_ratio=1. / 3.
    )
    """

    plt.show()
