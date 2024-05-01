import argparse
import os
from mitim_tools.gs_tools import GEQtools
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.astra_tools import ASTRA_CDFtools
import matplotlib.pyplot as plt
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("directories", type=str, nargs="*")
args = parser.parse_args()

directories = args.directories

def convertGACODE(astra_root, 
                  nexp=112, # number of grid points for gacode output
                  nion=4, # number of thermal and fast ion species
                  shot=None,
                  gacode_out=True # whether to output a file along with gacode object
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
    print(params.keys())

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
    print("Pfus:",c.Pfus[-1,-1])
    print("H98:",c.H98[-1])
    print("betan:",c.betaN[-1])

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
    print("Testing Geometry output:")
    print(g.delta95)
    print(g.delta995)

    # Aquire MXH Coefficients
    print("Finding flux surface geometry ...")
    shape_cos, shape_sin, bbox = g.get_MXH_coeff(n=100, n_coeff=6, plot=False)
    print(bbox.shape)
    print("Done.")

    params["nexp"] = nexp
    params["nion"] = nion
    params["shot"] = 12345

    nexp_grid = np.linspace(0,1,nexp)
    interp_to_nexp = lambda x: np.interp(nexp_grid,np.linspace(0,1,x.size),x)
    '''(['nexp', 'nion', 'shot', 'name', 'type', 'masse', 'mass', 
'ze', 'z', 'torfluxa(Wb/radian)', 'rcentr(m)', 'bcentr(T)', 
'current(MA)', 'rho(-)', 'rmin(m)', 'polflux(Wb/radian)', 
'q(-)', 'w0(rad/s)', 'rmaj(m)', 'zmag(m)', 'kappa(-)', 
'delta(-)', 'zeta(-)', 'shape_cos0(-)', 'shape_cos1(-)', 
'shape_cos2(-)', 'shape_cos3(-)', 'shape_cos4(-)', 'shape_cos5(-)', 
'shape_sin3(-)', 'shape_sin4(-)', 'shape_sin5(-)', 'ne(10^19/m^3)', 
'ni(10^19/m^3)', 'te(keV)', 'ti(keV)', 'ptot(Pa)', 'johm(MA/m^2)', 'jbs(MA/m^2)', 
'jbstor(MA/m^2)', 'z_eff(-)', 'vtor(m/s)', 'qohme(MW/m^3)', 'qbeame(MW/m^3)', 
'qbeami(MW/m^3)', 'qrfe(MW/m^3)', 'qbrem(MW/m^3)', 'qsync(MW/m^3)', 'qline(MW/m^3)', 
'qei(MW/m^3)', 'qione(MW/m^3)', 'qioni(MW/m^3)', 'qpar_beam(MW/m^3)', 'qmom(MW/m^3)', 
'qpar_wall(MW/m^3)', 'qmom(N/m^2)'])'''
    print("Getting Profiles from ASTRA...")
    name = "D T C He" ; params['name'] = name
    type = '[therm] [therm] [therm] [fast]' ; params['type'] = type
    masse = 0.00054489 ; params['masse'] = masse
    mass = np.zeros(nion) ; params['mass'] = mass
    print("masse:",params["masse"])
    ze = 0.
    z = np.zeros(nion)
    torfluxa = c.TF[-1] ; params['torfluxa(Wb/radian)'] = torfluxa
    rcenter = c.RTOR[-1] ; params['rcentr(m)'] = rcenter
    bcentr = c.BTOR[-1] ; params['bcentr(T)'] = bcentr
    current = c.IPL[-1] ; params['current(MA)'] = current
    rho = interp_to_nexp(c.rho[-1]/c.rho[-1,-1]) ; params['rho(-)'] = rho
    polflux = interp_to_nexp(c.FP[-1]) ; params['polflux(Wb/radian)'] = polflux
    q = interp_to_nexp(c.q[-1]) ; params['q(-)'] = q
    rmaj = interp_to_nexp(bbox[0,:]) ; params['rmaj(m)'] = rmaj
    rmin = interp_to_nexp(bbox[1,:]) ; params['rmin(m)'] = rmin
    zmag = interp_to_nexp(bbox[2,:]) ; params['zmag(m)'] = zmag
    kappa = interp_to_nexp(bbox[3,:]) ; params['kappa(-)'] = kappa
    delta = interp_to_nexp(shape_sin[1,:]) ; params['delta(-)'] = delta
    zeta = -interp_to_nexp(shape_sin[2,:]) ; params['zeta(-)'] = zeta
    shape_cos0 = interp_to_nexp(shape_cos[0,:]) ; params['shape_cos0(-)'] = shape_cos0
    shape_cos1 = interp_to_nexp(shape_cos[1,:]) ; params['shape_cos1(-)'] = shape_cos1
    shape_cos2 = interp_to_nexp(shape_cos[2,:]) ; params['shape_cos2(-)'] = shape_cos2
    shape_cos3 = interp_to_nexp(shape_cos[3,:]) ; params['shape_cos3(-)'] = shape_cos3
    shape_cos4 = interp_to_nexp(shape_cos[4,:]) ; params['shape_cos4(-)'] = shape_cos4
    shape_cos5 = interp_to_nexp(shape_cos[5,:]) ; params['shape_cos5(-)'] = shape_cos5
    shape_sin3 = interp_to_nexp(shape_sin[3,:]) ; params['shape_sin3(-)'] = shape_sin3
    shape_sin4 = interp_to_nexp(shape_sin[4,:]) ; params['shape_sin4(-)'] = shape_sin4
    shape_sin5 = interp_to_nexp(shape_sin[5,:]) ; params['shape_sin5(-)'] = shape_sin5
    ne = interp_to_nexp(c.ne[-1,:]) ; params['ne(10^19/m^3)'] = ne
    ni = interp_to_nexp(c.ne[-1,:]) ; params['ni(10^19/m^3)'] = np.tile(ni, (nion, 1)).T
    te = interp_to_nexp(c.Te[-1,:]) ; params['te(keV)'] = te
    ti = np.zeros((nexp,nion)) ; params['ti(keV)'] = np.tile(te, (nion, 1)).T # te=ti
    ptot = interp_to_nexp(c.ptot[-1,:]) ; params['ptot(Pa)'] = ptot
    johm = np.zeros(nexp) ; params["johm(MA/m^2)"] = johm
    jbs = np.zeros(nexp) ; params["jbs(MA/m^2)"] = jbs
    jbstor = np.zeros(nexp) ; params["jbstor(MA/m^2)"] = jbstor
    z_eff = interp_to_nexp(c.ZEF[-1]) ; params['z_eff(-)'] = z_eff
    vtor = interp_to_nexp(c.VTOR[-1]) ; params['vtor(m/s)'] = vtor
    qohme = interp_to_nexp(c.QOH[-1]) ; params['qohme(MW/m^3)'] = qohme
    qbeame = np.zeros(nexp) ; params['qbeame(MW/m^3)'] = qbeame
    qbeami = np.zeros(nexp) ; params['qbeami(MW/m^3)'] = qbeami
    qbrem = interp_to_nexp(c.QRAD[-1]) ; params['qbrem(MW/m^3)'] = qbrem
    qsync = np.zeros(nexp) ; params['qsync(MW/m^3)'] = qsync
    qline = np.zeros(nexp) ; params['qline(MW/m^3)'] = qline
    qei = interp_to_nexp(c.PEICR[-1]) ; params["qei(MW/m^3)"] = qei#np.zeros(nexp)
    qrfe = interp_to_nexp(c.PEICR[-1]) ; params['qrfe(MW/m^3)'] = qrfe
    qfuse = interp_to_nexp(c.PEDT[-1]) ; params['qfuse(MW/m^3)'] = qfuse
    qfusi = interp_to_nexp(c.PIDT[-1]) ; params['qfusi(MW/m^3)'] = qfusi
    qione = np.zeros(nexp) ; params['qione(MW/m^3)'] = qione
    qioni = np.zeros(nexp) ; params['qioni(MW/m^3)'] = qioni
    qpar_beam = np.zeros(nexp) ; params['qpar_beam(MW/m^3)'] = qpar_beam
    qmom = np.zeros(nexp) ; params['qmom(MW/m^3)'] = qmom
    qpar_wall = np.zeros(nexp) ; params['qpar_wall(MW/m^3)'] = qpar_wall
    qmom = np.zeros(nexp) ; params['qmom(N/m^2)'] = qmom
    w0 = np.zeros(nexp) ; params['w0(rad/s)'] = w0
    qpar_wall = np.zeros(nexp)
    print("Done.")

    gacode_filename = os.path.join(astra_root, "input.gacode")
    print(params["ni(10^19/m^3)"])
    print(params["ni(10^19/m^3)"].shape)
    print("Writing File ...")
    with open(gacode_filename, 'w') as f:
        f.write(f"#  *original : {datetime.datetime.now().strftime('%a %b %d %H:%M:%S %Z %Y')}\n")
        f.write(f"# *statefile : {os.path.basename(cdf_file)}\n")
        f.write(f"#     *gfile : {os.path.basename(geometry_file)}\n")
        for key, param in params.items():
            print(f"Writing {key} ...")
            f.write(f"# {key}\n")
            try:
                if isinstance(param, np.ndarray) and param.ndim == 1:
                    if param.size == nexp:
                        for i, value in enumerate(param, start=1):
                            f.write(f"{i:3d}  {value:.7e}\n")
                    elif param.size == nion:
                        for i, value in enumerate(param, start=1):
                            f.write(f" {value:.7e}")
                        f.write("\n")
                    elif param.size == 1:
                        f.write(f" {param[0]:.7e}\n")
                elif isinstance(param, np.ndarray) and param.ndim == 2:
                    for i, row in enumerate(param, start=1):
                        f.write(f"{i:3d}")
                        for value in row:
                            f.write(f"  {value:.7e}")
                        f.write("\n")
                elif isinstance(param, str):
                    f.write(f"{param}\n")
                elif isinstance(param, int):
                    f.write(f"{param}\n")
                elif isinstance(param, float):
                    f.write(f"{param:.7e}\n")
                else:
                    raise(ValueError(f"Unrecognized type for {key}"))
            except:
                print(f"Error writing {key}")
    
    return p

    

for directory in directories:
    p = convertGACODE(directory)

p.plot()