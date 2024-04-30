import argparse
import os
from mitim_tools.gs_tools import GEQtools
import numpy as np
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.astra_tools import ASTRA_CDFtools
import matplotlib.pyplot as plt

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
    shape_sin, shape_cos, bbox = g.get_MXH_coeff(n=200, n_coeff=6, plot=False)
    print(shape_cos.shape)
    print("Done.")

    nexp_grid = np.linspace(0,1,nexp)
    interp_to_nexp = lambda x: np.interp(nexp_grid,np.linspace(0,1,x.size),x)

    print("Getting Profiles ...")
    mass = np.zeros(nion)
    z = np.zeros(nion)
    torfluxa = 0
    rcenter = 0
    bcentr = 0 
    current = 0
    rho = np.zeros(nexp) ; rho[:] = interp_to_nexp(c.rho[-1])
    rmin = np.zeros(nexp) ; rmin[:] = interp_to_nexp(bbox[:,1])
    polflux = np.zeros(nexp) ; polflux[:] = interp_to_nexp(c.FP[-1])
    q = np.zeros(nexp) ; q[:] = interp_to_nexp(c.q[-1])
    rmaj = np.zeros(nexp) ; rmaj[:] = interp_to_nexp(bbox[:,0])
    zmag = np.zeros(nexp)
    kappa = np.zeros(nexp)
    delta = np.zeros(nexp)
    zeta = np.zeros(nexp)
    shape_cos0 = np.zeros(nexp)
    shape_cos1 = np.zeros(nexp)
    shape_cos2 = np.zeros(nexp)
    shape_cos3 = np.zeros(nexp)
    shape_cos4 = np.zeros(nexp)
    shape_cos5 = np.zeros(nexp)
    shape_sin3 = np.zeros(nexp)
    shape_sin4 = np.zeros(nexp)
    shape_sin5 = np.zeros(nexp)
    ne = np.zeros(nexp)
    ni = np.zeros((nexp,nion))
    te = np.zeros(nexp)
    ti = np.zeros((nexp,nion))
    ptot = np.zeros(nexp)
    johm = np.zeros(nexp)
    jbs = np.zeros(nexp)
    jbstor = np.zeros(nexp)
    z_eff = np.zeros(nexp)
    qohme = np.zeros(nexp)
    qrfe = np.zeros(nexp)
    qrfi = np.zeros(nion)
    qfuse = np.zeros(nexp)
    qfusi = np.zeros(nexp)
    qbrem = np.zeros(nexp)
    qsync = np.zeros(nexp)
    qline = np.zeros(nexp)
    qei = np.zeros(nexp)
    qione = np.zeros(nexp)
    qioni = np.zeros(nion)
    qpar_beam = np.zeros(nexp)
    w0 = np.zeros(nexp)
    qpar_wall = np.zeros(nexp)
    print("Done.")

    params = {'nexp':nexp, 'nion':nion, 'shot':12345, 'name':'D T F He',
              'mass':mass,'z':z,'torfluxa':torfluxa,'rcenter':rcenter,
              'bcentr':bcentr,'current':current,
              'rho':rho,'rmin':rmin,'polflux':polflux,'q':q,
              'rmaj':rmaj,'zmag':zmag,'kappa':kappa,'delta':delta,'zeta':zeta,
              'shape_cos0':shape_cos0,'shape_cos1':shape_cos1,
              'shape_cos2':shape_cos2,'shape_cos3':shape_cos3,
              'shape_cos4':shape_cos4,'shape_cos5':shape_cos5,
              'shape_sin3':shape_sin3,'shape_sin4':shape_sin4,
              'shape_sin5':shape_sin5,'ne':ne,'ni':ni,'te':te,
              'ti':ti,'ptot':ptot,'johm':johm,'jbs':jbs,'jbstor':jbstor,
              'z_eff':z_eff,'qohme':qohme,'qrfe':qrfe,'qrfi':qrfi,
              'qfuse':qfuse,'qfusi':qfusi,'qbrem':qbrem,'qsync':qsync,
              'qline':qline,'qei':qei,'qione':qione,'qioni':qioni,
              'qpar_beam':qpar_beam,'w0':w0,'qpar_wall':qpar_wall,
              }

    filename = os.path.join(astra_root, "input.gacode")
    print("Writing File ...")
    with open(filename, 'w') as f:
        for key, param in params.items():
            print(f"Writing {key} ...")
            f.write(f"# {key} | -\n")
            try:
                if isinstance(param, np.ndarray) and param.ndim == 1:
                    for i, value in enumerate(param, start=1):
                        f.write(f"{i:3d}  {value:.7e}\n")
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
    print("Done.")

    

for directory in directories:
    convertGACODE(directory)