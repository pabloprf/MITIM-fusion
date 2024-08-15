import os
import tarfile
import numpy as np
from mitim_tools.misc_tools import IOtools,FARMINGtools
from mitim_tools.astra_tools import ASTRA_CDFtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gs_tools import GEQtools
from mitim_tools import __mitimroot__
from IPython import embed

class ASTRA():

    def __init__(self):

        pass

    def prep(self,folder,file_repo = __mitimroot__ + '/templates/ASTRA8_REPO.tar.gz'): 

        # Folder is the local folder where ASTRA things are, e.g. ~/scratch/testAstra/

        self.folder = IOtools.expandPath(folder)
        self.file_repo = file_repo

        # Create folder
        IOtools.askNewFolder(self.folder)

        # Move files
        os.system(f'cp {self.file_repo} {self.folder}/ASTRA8_REPO.tar.gz')

        # untar
        with tarfile.open(
            os.path.join(self.folder, "ASTRA8_REPO.tar.gz"), "r"
        ) as tar:
            tar.extractall(path=self.folder)

        #os.system(f'cp -r {self.folder}/ASTRA8_REPO/* {self.folder_as}/')
        os.remove(os.path.join(self.folder, "ASTRA8_REPO.tar.gz"))

        # Define basic controls
        self.equfile = 'fluxes'
        self.expfile = 'aug34954'

    def run(self,
            t_ini,
            t_end,
            name='run1',
            slurm_options = {
                'time': 10,
                'cpus': 16}):

        self.t_ini = t_ini
        self.t_end = t_end

        self.folder_astra = f'{self.folder}/{name}/'
        IOtools.askNewFolder(self.folder_astra)
        os.system(f'cp -r {self.folder}/ASTRA8_REPO/* {self.folder_astra}/')

        astra_name = f'mitim_astra_{name}'

        self.astra_job = FARMINGtools.mitim_job(self.folder)

        self.astra_job.define_machine(
            "astra",
            f"{astra_name}/",
            launchSlurm=False,
        )

        # What to run 
        self.command_to_run_astra = f'''
cd {self.astra_job.folderExecution}/{name} 
scripts/as_exe -m {self.equfile} -v {self.expfile} -s {self.t_ini} -e {self.t_end} -dev aug -batch
'''

        self.shellPreCommand = f'cd {self.astra_job.folderExecution}/{name} &&  ./install.sh'

        # ---------------------------------------------
        # Execute
        # ---------------------------------------------

        self.output_folder = f'{name}/.res/ncdf/'

        self.astra_job.prep(
            self.command_to_run_astra,
            shellPreCommands=[self.shellPreCommand],
            input_folders=[self.folder_astra],
            output_folders=[self.output_folder],
        )

        self.astra_job.run(waitYN=False)


    def read(self):

        self.cdf_file = f'{self.output_folder}/'

        self.cdf = ASTRA_CDFtools.transp_output(self.cdf_file)

    def plot(self):

        pass

def convert_ASTRA_to_gacode(astra_root, 
                  nexp=112, # number of grid points for gacode output
                  nion=5, # number of thermal and fast ion species
                  ai=-3, # array index of astra timestep to convert
                  gacode_out=None, # whether to output a file along with gacode object
                  plot_result=False # whether to plot the gacode object
                  ):
    """
    Converts ASTRA run directory to input.gacode file format
    1. Reads the ASTRA output CDF file and extracts transport and profiles info
    2. Reads the astra.geqdsk file and extracts the geometry
    3. Writes the input.gacode file - default is scratch directory
    4. returns a mitim gacode object
    """

    template_path = __mitimroot__ + "/tests/data/input.gacode"
    p = PROFILEStools.PROFILES_GACODE(template_path)
    params = p.profiles

    # Extract CDF file
    cdf_file = None
    astra_results_dir = os.path.join(astra_root, "ncdf_out")
    for file in os.listdir(astra_results_dir):
        if file.endswith(".CDF"):
            cdf_file = os.path.join(astra_results_dir, file)
            break

    if cdf_file is None:
        raise(FileNotFoundError(f"No CDF file found in {astra_results_dir}"))
    else:
        print(f"Found CDF file: {cdf_file}")

    c = ASTRA_CDFtools.transp_output(cdf_file)
    c.calcProfiles()

    # Extract Geometry info
    geometry_file = None
    for file in os.listdir(astra_root):
        if file.endswith(".geqdsk") or file.endswith(".eqdsk"):
            geometry_file = os.path.join(astra_root, file)
            break

    if geometry_file is None:
        raise(FileNotFoundError(f"No geqdsk file found in {astra_results_dir}"))
    else:
        print(f"Found gfile: {geometry_file}")

    try:
        g = GEQtools.MITIMgeqdsk(geometry_file, removeCoils=False)
    except:
        g = GEQtools.MITIMgeqdsk(geometry_file)

    # Aquire MXH Coefficients
    print("Finding flux surface geometry ...")
    shape_cos, shape_sin, bbox, psin_grid  = g.get_MXH_coeff(n=1000, n_coeff=6, plot=False)
    print("Done.")

    params["nexp"] = np.array([str(nexp)])
    params["nion"] = np.array([str(nion)])
    params["shot"] = np.array(['12345'])

    nexp_grid = np.linspace(0,1,nexp)

    # interpolate ASTRA's rho grid to gacode rho grid
    interp_to_nexp = lambda x: np.interp(nexp_grid,np.linspace(0,1,x.size),x)
    
    print("Getting Profiles from ASTRA...")
    name = np.array(["D", "T", "He", "W", "F"]) ; params['name'] = name
    iontype = np.array(['[therm]', '[therm]', '[therm]', '[therm]', '[therm]']) ; params['type'] = iontype
    masse = np.array([0.00054489]) ; params['masse'] = masse
    mass = np.array([2., 3., c.AIM1[ai],c.AIM2[ai], c.AIM3[ai]]) ; params['mass'] = mass # assumes 50/50 D/T !!!!
    z = np.array([1., 1., c.ZIM1[ai,ai], 74., c.ZIM3[ai,ai]]) ; params['z'] = z

    torfluxa = np.array([c.TF[ai]])             ; params['torfluxa(Wb/radian)'] = torfluxa
    rcenter = np.array([c.RTOR[ai]])            ; params['rcentr(m)'] = rcenter
    bcentr = np.array([c.BTOR[ai]])             ; params['bcentr(T)'] = bcentr
    current = np.array([c.IPL[ai]])             ; params['current(MA)'] = current

    rho_normalized = (c.rho[-1,]-c.rho[-1,0])/(c.rho[-1,-1]-c.rho[-1,0])

    rho = interp_to_nexp(rho_normalized)        ; params['rho(-)'] = rho
    polflux = interp_to_nexp(c.FP[ai])          ; params['polflux(Wb/radian)'] = polflux

    polflux_norm = (polflux-polflux[0])/(polflux[-1]-polflux[0])
                                         
    # interpolate geqdsk quantities from psin grid to rho grid using polflux_norm
    interp_to_rho = lambda x: np.interp(polflux_norm, psin_grid, x)    

    q = interp_to_nexp(c.q[ai])                ; params['q(-)'] = q
    rmaj = interp_to_rho(bbox[0,:])            ; params['rmaj(m)'] = rmaj
    rmin = interp_to_rho(bbox[1,:])            ; params['rmin(m)'] = rmin
    zmag = interp_to_rho(bbox[2,:])            ; params['zmag(m)'] = zmag
    kappa = interp_to_rho(bbox[3,:])           ; params['kappa(-)'] = kappa
    delta = interp_to_rho(shape_sin[1,:])      ; params['delta(-)'] = delta
    zeta = interp_to_rho(-shape_sin[2,:])      ; params['zeta(-)'] = zeta
    shape_cos0 = interp_to_rho(shape_cos[0,:]) ; params['shape_cos0(-)'] = shape_cos0
    shape_cos1 = interp_to_rho(shape_cos[1,:]) ; params['shape_cos1(-)'] = shape_cos1
    shape_cos2 = interp_to_rho(shape_cos[2,:]) ; params['shape_cos2(-)'] = shape_cos2
    shape_cos3 = interp_to_rho(shape_cos[3,:]) ; params['shape_cos3(-)'] = shape_cos3
    shape_cos4 = interp_to_rho(shape_cos[4,:]) ; params['shape_cos4(-)'] = shape_cos4
    shape_cos5 = interp_to_rho(shape_cos[5,:]) ; params['shape_cos5(-)'] = shape_cos5
    shape_cos6 = np.zeros(nexp)                ; params['shape_cos6(-)'] = shape_cos6
    shape_sin3 = interp_to_rho(shape_sin[3,:]) ; params['shape_sin3(-)'] = shape_sin3
    shape_sin4 = interp_to_rho(shape_sin[4,:]) ; params['shape_sin4(-)'] = shape_sin4
    shape_sin5 = interp_to_rho(shape_sin[5,:]) ; params['shape_sin5(-)'] = shape_sin5
    shape_sin6 = np.zeros(nexp)                ; params['shape_sin6(-)'] = shape_sin6

    ne = interp_to_nexp(c.ne[ai,:])            ; params['ne(10^19/m^3)'] = ne
    ni_main = interp_to_nexp(c.NMAIN[ai,:])
    ni_He = interp_to_nexp(c.NIZ1[ai,:])
    ni_W = interp_to_nexp(c.NIZ2[ai,:])
    ni_F = interp_to_nexp(c.NIZ3[ai,:])
    ni = np.vstack((ni_main/2, ni_main/2, ni_He, ni_W, ni_F)).T
    
    params['ni(10^19/m^3)'] = ni

    te = interp_to_nexp(c.Te[ai,:])             ; params['te(keV)'] = te
    # all ions have same temperature
    ti = interp_to_nexp(c.Ti[ai,:])             ; params['ti(keV)'] = np.tile(ti, (nion, 1)).T 
    ptot = interp_to_nexp(c.ptot[ai,:])         ; params['ptot(Pa)'] = ptot * 1602 # convert to Pa
    jbs = interp_to_nexp(c.Cubs[ai,:])          ; params["jbs(MA/m^2)"] = jbs
    z_eff = interp_to_nexp(c.ZEF[ai])           ; params['z_eff(-)'] = z_eff
    vtor = interp_to_nexp(c.VTOR[ai])           ; params['vtor(m/s)'] = vtor
    qohme = interp_to_nexp(c.POH[ai])           ; params['qohme(MW/m^3)'] = qohme
    qbrem = interp_to_nexp(c.PRAD[ai])          ; params['qbrem(MW/m^3)'] = qbrem 
    # setting qbrem equal to total radiation
    # includes line and synchrotron radiation
    qsync = np.zeros(nexp)                      ; params['qsync(MW/m^3)'] = qsync
    qline = np.zeros(nexp)                      ; params['qline(MW/m^3)'] = qline
    qei = np.zeros(nexp)                            ; params["qei(MW/m^3)"] = qei
    qrfe = interp_to_nexp(c.PEICR[ai])          ; params['qrfe(MW/m^3)'] = qrfe
    qrfi = interp_to_nexp(c.PIICR[ai])          ; params['qrfi(MW/m^3)'] = qrfi
    qfuse = interp_to_nexp(c.PEDT[ai])          ; params['qfuse(MW/m^3)'] = qfuse
    qfusi = interp_to_nexp(c.PIDT[ai])          ; params['qfusi(MW/m^3)'] = qfusi

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

    # Print output to check Q, Pfus, etc.
    p.printInfo()
    
    if gacode_out is not None:
        p.writeCurrentStatus(file=gacode_out)
    if plot_result:
        p.plot()

    return p