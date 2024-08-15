import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools, IOtools
from mitim_tools.surrogate_tools import NNtools

def pedestal_density_scan_plot(
        model:NNtools.eped_nn,
        Ip,
        Bt,
        R,
        a,
        kappa995,
        delta995,
        betan,
        zeff,
        tesep=75,    #eV
        nesep_ratio=0.3
    ):

    greenwald_density = Ip / (np.pi * a**2) * 10 # in units of 10^19m^-3
    nped = np.linspace(1, greenwald_density, 50)
    ptop = []
    wtop = []

    # define the input parameters for the EPED NN
    for neped in nped:

        inputs = np.array([Ip, Bt, R, a, kappa995, delta995, neped, betan, zeff, tesep, nesep_ratio])

        p, w = model(*inputs)
        ptop.append(p)
        wtop.append(w)

    GRAPHICStools.prep_figure_papers(size=20,slower_but_latex=True)
    fig, ax = plt.subplots(figsize=(10,8))
    GRAPHICStools.addDenseAxis(ax)

    ax.plot(nped, ptop)
    ax.set_ylabel(r"$P_{top}$ [kPa]")
    ax.set_xlabel(r"$n_{ped}$ [$10^{19}m^{-3}$]")
    ax.set_title("EPED-NN density scan")

if __name__ == '__main__':

    nn = NNtools.eped_nn(type='tf')

    nn_location = IOtools.expandPath('$MFEIM_PATH/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-MODEL-ARC.h5')
    norm_location = IOtools.expandPath('$MFEIM_PATH/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-NORMALIZATION.txt')

    nn.load(nn_location, norm=norm_location)
    
    pedestal_density_scan_plot(
        model=nn,
        Ip=10.95,
        Bt=10.8,
        R=4.25,
        a=1.17,
        kappa995=1.68,
        delta995=0.516,
        betan=1.9,
        zeff=1.5,
        tesep=75,
        nesep_ratio=0.3
    )

    plt.show()