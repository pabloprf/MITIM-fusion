import argparse
import os
from mitim_tools.gs_tools import GEQtools
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.astra_tools import ASTRA_CDFtools
import matplotlib.pyplot as plt
import datetime

#parser = argparse.ArgumentParser()
#parser.add_argument("directories", type=str, nargs="*")
#args = parser.parse_args()

#directories = args.directories

def convert_astra_gacode(astra_root, 
                  nexp=112, # number of grid points for gacode output
                  nion=4, # number of thermal and fast ion species
                  gacode_out=True, # whether to output a file along with gacode object
                  plot_result=False # whether to plot the gacode object
                  ):
    """
    Converts ASTRA run directory to input.gacode file format
    1. Reads the ASTRA output CDF file and extracts transport and profiles info
    2. Reads the astra.geqdsk file and extracts the geometry
    3. Writes the input.gacode file - default is scratch directory
    4. returns a mitim gacode object
    """

    template_path = "/Users/hallj/MITIM-fusion/tests/data/input.gacode"
    p = PROFILEStools.PROFILES_GACODE(template_path)
    params = p.profiles
    #print(params['type'])
    #print([type(item[0]) for key, item in params.items()])

    # Extract CDF file
    cdf_file = None
    astra_results_dir = os.path.join(astra_root, "ncdf_out")
    for file in os.listdir(astra_results_dir):
        if file.endswith(".CDF"):
            cdf_file = os.path.join(astra_results_dir, file)
            break

    if cdf_file is None:
        raise(FileNotFoundError("No CDF file found in {}".format(astra_results_dir)))
    else:
        print("Found CDF file: {}".format(cdf_file))

    c = ASTRA_CDFtools.CDFreactor(cdf_file)
    c.calcProfiles()
    print("Testing ASTRA output:")
    print("Q:",c.Q[-2])
    print("Pfus:",c.Pfus[-2,-1])
    print("H98:",c.H98[-2])
    print("betan:",c.betaN[-2])

    # Extract Geometry info
    geometry_file = None
    for file in os.listdir(astra_root):
        if file.endswith(".geqdsk"):
            geometry_file = os.path.join(astra_root, file)
            break

    if geometry_file is None:
        raise(FileNotFoundError("No geqdsk file found in {}".format(astra_results_dir)))
    else:
        print("Found gfile: {}".format(geometry_file))

    g = GEQtools.MITIMgeqdsk(geometry_file)

    # Aquire MXH Coefficients
    print("Finding flux surface geometry ...")
    shape_cos, shape_sin, bbox = g.get_MXH_coeff(n=1000, n_coeff=6, plot=False)
    print("Done.")

    params["nexp"] = np.array([str(nexp)])
    params["nion"] = np.array([str(nion)])
    params["shot"] = np.array(['12345'])

    nexp_grid = np.linspace(0,1,nexp)
    interp_to_nexp = lambda x: np.interp(nexp_grid,np.linspace(0,1,x.size),x)
    
    print("Getting Profiles from ASTRA...")
    name = np.array(["D", "T", "He", "F"]) ; params['name'] = name
    iontype = np.array(['[therm]', '[therm]', '[therm]', '[therm]']) ; params['type'] = iontype
    masse = np.array([0.00054489]) ; params['masse'] = masse
    mass = np.array([2., 3., c.AIM1[-2], c.AIM3[-2]]) ; params['mass'] = mass # assumes 50/50 D/T !!!!
    z = np.array([1., 1., c.ZIM1[-2,-2], c.ZIM3[-2,-2]]) ; params['z'] = z

    torfluxa = np.array([c.TF[-2]])             ; params['torfluxa(Wb/radian)'] = torfluxa
    rcenter = np.array([c.RTOR[-2]])            ; params['rcentr(m)'] = rcenter
    bcentr = np.array([c.BTOR[-2]])             ; params['bcentr(T)'] = bcentr
    current = np.array([c.IPL[-2]])             ; params['current(MA)'] = current
    rho = interp_to_nexp(c.rho[-2]/c.rho[-2,-1]); params['rho(-)'] = rho
    polflux = interp_to_nexp(c.FP[-2])          ; params['polflux(Wb/radian)'] = polflux
    q = interp_to_nexp(c.q[-2])                 ; params['q(-)'] = q
    rmaj = interp_to_nexp(bbox[0,:])            ; params['rmaj(m)'] = rmaj
    rmin = interp_to_nexp(bbox[1,:])            ; params['rmin(m)'] = rmin
    zmag = interp_to_nexp(bbox[2,:])            ; params['zmag(m)'] = zmag
    kappa = interp_to_nexp(bbox[3,:])           ; params['kappa(-)'] = kappa
    delta = interp_to_nexp(shape_sin[1,:])      ; params['delta(-)'] = delta
    zeta = interp_to_nexp(-shape_sin[2,:])      ; params['zeta(-)'] = zeta
    shape_cos0 = interp_to_nexp(shape_cos[0,:]) ; params['shape_cos0(-)'] = shape_cos0
    shape_cos1 = interp_to_nexp(shape_cos[1,:]) ; params['shape_cos1(-)'] = shape_cos1
    shape_cos2 = interp_to_nexp(shape_cos[2,:]) ; params['shape_cos2(-)'] = shape_cos2
    shape_cos3 = interp_to_nexp(shape_cos[3,:]) ; params['shape_cos3(-)'] = shape_cos3
    shape_cos4 = interp_to_nexp(shape_cos[4,:]) ; params['shape_cos4(-)'] = shape_cos4
    shape_cos5 = interp_to_nexp(shape_cos[5,:]) ; params['shape_cos5(-)'] = shape_cos5
    shape_sin3 = interp_to_nexp(shape_sin[3,:]) ; params['shape_sin3(-)'] = shape_sin3
    shape_sin4 = interp_to_nexp(shape_sin[4,:]) ; params['shape_sin4(-)'] = shape_sin4
    shape_sin5 = interp_to_nexp(shape_sin[5,:]) ; params['shape_sin5(-)'] = shape_sin5

    ne = interp_to_nexp(c.ne[-2,:])             ; params['ne(10^19/m^3)'] = ne
    ni_main = interp_to_nexp(c.NMAIN[-2,:])
    ni_He = interp_to_nexp(c.NIZ1[-2,:])
    ni_I = interp_to_nexp(c.NIZ3[-2,:])
    ni = np.vstack((ni_main/2, ni_main/2, ni_He, ni_I)).T
    params['ni(10^19/m^3)'] = ni

    te = interp_to_nexp(c.Te[-2,:])             ; params['te(keV)'] = te
    ti = interp_to_nexp(c.Ti[-2,:])             ; params['ti(keV)'] = np.tile(ti, (nion, 1)).T # all ions have same temperature
    ptot = interp_to_nexp(c.ptot[-2,:])         ; params['ptot(Pa)'] = ptot
    jbs = interp_to_nexp(c.Cubs[-2,:])          ; params["jbs(MA/m^2)"] = jbs
    z_eff = interp_to_nexp(c.ZEF[-2])           ; params['z_eff(-)'] = z_eff
    vtor = interp_to_nexp(c.VTOR[-2])           ; params['vtor(m/s)'] = vtor
    qohme = interp_to_nexp(c.POH[-2])           ; params['qohme(MW/m^3)'] = qohme
    qbrem = interp_to_nexp(c.PRAD[-2])          ; params['qbrem(MW/m^3)'] = qbrem 
    # setting qbrem equal to total radiation
    # includes line and synchrotron radiation
    qsync = np.zeros(nexp)                      ; params['qsync(MW/m^3)'] = qsync
    qline = np.zeros(nexp)                      ; params['qline(MW/m^3)'] = qline
    qei = interp_to_nexp(c.PEICR[-2])           ; params["qei(MW/m^3)"] = qei
    qrfe = interp_to_nexp(c.PEICR[-2])          ; params['qrfe(MW/m^3)'] = qrfe
    qfuse = interp_to_nexp(c.PEDT[-2])          ; params['qfuse(MW/m^3)'] = qfuse
    qfusi = interp_to_nexp(c.PIDT[-2])          ; params['qfusi(MW/m^3)'] = qfusi
    # remaining parameters, need to derive w0 but set the rest zero for now
    jbstor = np.zeros(nexp) ; params["jbstor(MA/m^2)"] = jbstor
    johm = np.zeros(nexp) ; params["johm(MA/m^2)"] = johm
    qbeame = np.zeros(nexp) ; params['qbeame(MW/m^3)'] = qbeame
    qbeami = np.zeros(nexp) ; params['qbeami(MW/m^3)'] = qbeami
    qione = np.zeros(nexp) ; params['qione(MW/m^3)'] = qione
    qioni = np.zeros(nexp) ; params['qioni(MW/m^3)'] = qioni
    qpar_beam = np.zeros(nexp) ; params['qpar_beam(MW/m^3)'] = qpar_beam
    qmom = np.zeros(nexp) ; params['qmom(MW/m^3)'] = qmom
    qpar_wall = np.zeros(nexp) ; params['qpar_wall(MW/m^3)'] = qpar_wall
    qmom = np.zeros(nexp) ; params['qmom(N/m^2)'] = qmom
    w0 = np.zeros(nexp) ; params['w0(rad/s)'] = w0
    qpar_wall = np.zeros(nexp)
    print("Done.")

    # rederive quantities
    p.deriveQuantities()
    
    if gacode_out:
        gacode_filename = os.path.join(astra_root, "input.gacode")
        p.writeCurrentStatus(file=gacode_filename)
    if plot_result:
        p.plot()

    return p