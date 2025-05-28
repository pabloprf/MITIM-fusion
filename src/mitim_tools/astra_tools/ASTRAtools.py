import shutil
import tarfile
import numpy as np
from mitim_tools.misc_tools import IOtools,FARMINGtools, GUItools, GRAPHICStools
from mitim_tools.astra_tools import ASTRA_CDFtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.popcon_tools import FunctionalForms
from mitim_tools import __mitimroot__
from IPython import embed
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class ASTRA():

    def __init__(self):

        pass

    def prep(self,folder,file_repo = __mitimroot__ / 'templates' / 'ASTRA8_REPO.tar.gz'): 

        # Folder is the local folder where ASTRA things are, e.g. ~/scratch/testAstra/

        self.folder = IOtools.expandPath(folder)
        self.file_repo = IOtools.expandPath(file_repo)

        # Create folder
        IOtools.askNewFolder(self.folder)

        # Move files
        shutil.copy2(self.file_repo, self.folder / 'ASTRA8_REPO.tar.gz')

        # untar
        with tarfile.open(
            self.folder / "ASTRA8_REPO.tar.gz", "r"
        ) as tar:
            tar.extractall(path=self.folder)

        (self.folder / "ASTRA8_REPO.tar.gz").unlink(missing_ok=True)

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

        self.folder_astra = self.folder /  name
        IOtools.askNewFolder(self.folder_astra)
        for item in self.folder.glob('ASTRA8_REPO*'):
            shutil.copy2(item, self.folder_astra)

        astra_name = f'mitim_astra_{name}'

        self.astra_job = FARMINGtools.mitim_job(self.folder)

        self.astra_job.define_machine(
            "astra",
            f"{astra_name}",
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

        self.output_folder = name / '.res' / 'ncdf'

        self.astra_job.prep(
            self.command_to_run_astra,
            shellPreCommands=[self.shellPreCommand],
            input_folders=[self.folder_astra],
            output_folders=[self.output_folder],
        )

        self.astra_job.run(waitYN=False)


    def read(self):

        self.cdf = ASTRA_CDFtools.transp_output(self.output_folder)

    def plot(self):

        pass

def convert_ASTRA_folder_to_gacode(astra_root, 
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

    astra_root = IOtools.expandPath(astra_root)

    # Extract CDF file
    cdf_file = None
    astra_results_dir = astra_root / "ncdf_out"
    for file in astra_results_dir.glob("*"):
        if file.suffix in [".CDF"]:
            cdf_file = file.resolve()
            break

    if cdf_file is None:
        raise(FileNotFoundError(f"No CDF file found in {astra_results_dir}"))
    else:
        print(f"Found CDF file: {cdf_file}")

    c = ASTRA_CDFtools.transp_output(cdf_file)

    c.calcProfiles()

    convert_ASTRA_to_gacode_from_transp_output(
        c,
        nexp=nexp,
        nion=nion,
        ai=ai,
        gacode_out=gacode_out,
        plot_result=plot_result
    )

def convert_ASTRA_to_gacode_fromCDF(astra_cdf, 
                  nexp=112, # number of grid points for gacode output
                  nion=5, # number of thermal and fast ion species
                  ai=-3, # array index of astra timestep to convert
                  gacode_out=None, # whether to output a file along with gacode object
                  plot_result=False # whether to plot the gacode object
                  ):
    """
    Converts ASTRA run directory to input.gacode file format
    1. Reads the ASTRA output CDF file and extracts transport, profiles and the geometry
    3. Writes the input.gacode file - default is scratch directory
    4. returns a mitim gacode object
    """

    template_path = __mitimroot__ / "tests" / "data"/ "input.gacode"
    p = PROFILEStools.PROFILES_GACODE(template_path)
    params = p.profiles

    # Extract CDF file
    cdf_file = None
    #astra_results = os.path.join(astra_cdf, "ncdf_out")
    astra_results = astra_cdf
    #for file in os.listdir(astra_results):
    #    if file.endswith(".CDF"):
    #        cdf_file = os.path.join(astra_results, file)
    #        break
    cdf_file = astra_results

    if cdf_file is None:
        raise(FileNotFoundError(f"No CDF file found in {astra_results}"))
    else:
        print(f"Found CDF file: {cdf_file}")

    c = ASTRA_CDFtools.transp_output(cdf_file)

    c.calcProfiles()

    convert_ASTRA_to_gacode_from_transp_output(
        c,
        nexp=nexp,
        nion=nion,
        ai=ai,
        gacode_out=gacode_out,
        plot_result=plot_result
    )

def convert_ASTRA_to_gacode_from_transp_output(c, 
                  nexp=112, # number of grid points for gacode output
                  nion=5, # number of thermal and fast ion species
                  ai=-3, # array index of astra timestep to convert
                  gacode_out=None, # whether to output a file along with gacode object
                  plot_result=False # whether to plot the gacode object
                  ):
    """
    Converts ASTRA run directory to input.gacode file format
    1. Reads the ASTRA output CDF file and extracts transport, profiles and the geometry
    3. Writes the input.gacode file - default is scratch directory
    4. returns a mitim gacode object
    """

    template_path = __mitimroot__ / "tests" / "data"/ "input.gacode"
    p = PROFILEStools.PROFILES_GACODE(template_path)
    params = p.profiles

    #c.calcProfiles()

    # Aquire MXH Coefficients
    print("Finding flux surface geometry ...")
    #print("time indicies",c.t.shape)
    #print("R shape",c.R.shape)
    # need this try block to deal with both old and new astra output
    r=c.R[ai,:,:]
    z=c.Z[ai,:,:]
    r=np.atleast_2d(r)
    z=np.atleast_2d(z)
    psi=c.FP_norm[ai,:]
    
    surfaces = GEQtools.mitim_flux_surfaces()
    surfaces.reconstruct_from_RZ(r, z)
    coeffs_MXH=6
    surfaces._to_mxh(n_coeff=coeffs_MXH)
    #print('r is '+str(r))
    #print('z is '+str(z))
    #print(f"Shape of r: {r.shape}")
    #print(f"Shape of z: {z.shape}")
    for i in range(coeffs_MXH):
        shape_cos = surfaces.cn[:,i]
        if i > 2:
            shape_sin = surfaces.sn[:,i]
    kappa = surfaces.kappa
    delta = np.sin(surfaces.sn[:,1])
    zeta= -surfaces.sn[:,2]
    rmin= surfaces.a
    rmaj= surfaces.R0
    zmag= surfaces.Z0

    #shape_cos, shape_sin, bbox, psin_grid  = GEQtools.get_MXH_coeff_fromRZ(r,z,psi)
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
    z = np.array([1., 1., c.ZIM1[ai,-1], 74., c.ZIM3[ai,-1]]) ; params['z'] = z

    torfluxa = np.array([c.TF[ai]])             ; params['torfluxa(Wb/radian)'] = torfluxa
    rcenter = np.array([c.RTOR[ai]])            ; params['rcentr(m)'] = rcenter
    bcentr = np.array([c.BTOR[ai]])             ; params['bcentr(T)'] = bcentr
    current = np.array([c.IPL[ai]])             ; params['current(MA)'] = current
    rho = interp_to_nexp(c.rho[ai]/c.rho[ai,-1]); params['rho(-)'] = rho
    polflux = interp_to_nexp(c.FP[ai])          ; params['polflux(Wb/radian)'] = polflux/2/np.pi

    polflux_norm = (polflux-polflux[0])/(polflux[-1]-polflux[0])
    
    f_interp=interp1d(np.linspace(0,1,len(psi)),psi)
    psis=f_interp(np.linspace(0,1,len(r)))

    # interpolate geqdsk quantities from psin grid to rho grid using polflux_norm
    interp_to_rho = lambda x: np.interp(polflux_norm, psis, x)    

    q = interp_to_nexp(c.q[ai])                ; params['q(-)'] = q
    rmaj = interp_to_rho(rmaj)            ; params['rmaj(m)'] = rmaj
    rmin = interp_to_rho(rmin)            ; params['rmin(m)'] = rmin
    zmag = interp_to_rho(zmag)            ; params['zmag(m)'] = zmag
    kappa = interp_to_rho(kappa)           ; params['kappa(-)'] = kappa
    delta = interp_to_rho(surfaces.sn[:,1])      ; params['delta(-)'] = delta
    zeta = interp_to_rho(-surfaces.sn[:,2])      ; params['zeta(-)'] = zeta
    shape_cos0 = interp_to_rho(surfaces.cn[:,0]) ; params['shape_cos0(-)'] = shape_cos0
    shape_cos1 = interp_to_rho(surfaces.cn[:,1]) ; params['shape_cos1(-)'] = shape_cos1
    shape_cos2 = interp_to_rho(surfaces.cn[:,2]) ; params['shape_cos2(-)'] = shape_cos2
    shape_cos3 = interp_to_rho(surfaces.cn[:,3]) ; params['shape_cos3(-)'] = shape_cos3
    shape_cos4 = interp_to_rho(surfaces.cn[:,4]) ; params['shape_cos4(-)'] = shape_cos4
    shape_cos5 = interp_to_rho(surfaces.cn[:,5]) ; params['shape_cos5(-)'] = shape_cos5
    shape_cos6 = np.zeros(nexp)                ; params['shape_cos6(-)'] = shape_cos6
    shape_sin3 = interp_to_rho(surfaces.sn[:,3]) ; params['shape_sin3(-)'] = shape_sin3
    shape_sin4 = interp_to_rho(surfaces.sn[:,4]) ; params['shape_sin4(-)'] = shape_sin4
    shape_sin5 = interp_to_rho(surfaces.sn[:,5]) ; params['shape_sin5(-)'] = shape_sin5
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

    # Some checks (to work with Joe when he's back)
    for i in range(nexp-1):
        if np.isnan(p.profiles['kappa(-)'][-1-(i+1)]):
            p.profiles['kappa(-)'][-1-(i+1)] = p.profiles['kappa(-)'][-1-i]
    p.profiles['rho(-)'][0] = 0.0

    # rederive quantities
    p.deriveQuantities()

    # Print output to check Q, Pfus, etc.
    p.printInfo()
    
    if gacode_out is not None:
        p.writeCurrentStatus(file=gacode_out)
    if plot_result:
        p.plot()

    return p

def create_initial_conditions(te_avg,
                              ne_avg,
                              q_profile=None,
                              use_eped_pedestal=True, # add this later
                              eped_nn=None,
                              eped_params=None,
                              file_output_location=None,
                              width_top=0.05,
                              n_rho=104,
                              rho=None,
                              Te=None,
                              ne=None,
                              Ti=None, 
                              geometry=None,
                              plotYN=False,
                              ):
    
    """Returns a PRF functional form of the kinetic profiles for the initial conditions
    as a U-file that can be fed directly into ASTRA. Makes an initial guess for the pedestal and uses a tanh functional form
    outside of psi_n = 0.95"""

    # Define the radial grid
    if rho is not None:
        n_rho = len(rho)
    else:
        rho = np.linspace(0,1,n_rho)

    file_output_location = IOtools.expandPath(file_output_location)
    file_output_location.mkdir(parents=True, exist_ok=True)

    psi_n = None

    if geometry is not None:
        # load in geqdsk file and extract geometry
        g = GEQtools.MITIMgeqdsk(geometry)
        RZ = np.array([g.Rb,g.Yb]).T

        R = g.Rb
        Z = g.Yb

        # order profiles correctly
        # ASTRA expects them to be nonoverlapping, 
        # starting from outboard midplane and going counter-clockwise


        from scipy.signal import savgol_filter

        # apply minor smoothing to the profiles to avoid x point 
        R = savgol_filter(R, 8, 3, mode='wrap')
        Z = savgol_filter(Z, 8, 3, mode='wrap')

        rmax_index = np.argmax(R)
        R = np.roll(R, -(rmax_index+1))
        Z = np.roll(Z, -(rmax_index+1))
        if Z[1] < Z[-1]:
            Z = np.flip(Z)
            R = np.flip(R)
        Z[0] = Z[-1] = 0


        r_preamble = f"""  34954AUGD 2 0 6              ;-SHOT #- F(X) DATA -UF2DWR- 27Nov2019
                               ;-SHOT DATE-  UFILES ASCII FILE SYSTEM
   0                           ;-NUMBER OF ASSOCIATED SCALAR QUANTITIES-
 Time                Seconds   ;-INDEPENDENT VARIABLE LABEL: X-
 POSITION                      ;-INDEPENDENT VARIABLE LABEL: Y-
 R BOUNDARY    [M]             ;-DEPENDENT VARIABLE LABEL-
 0                             ;-PROC CODE- 0:RAW 1:AVG 2:SM. 3:AVG+SM
          1                    ;-# OF  X PTS-
        {len(R)}                    ;-# OF  Y PTS-
"""

        z_preamble = f"""  34954AUGD 2 0 6              ;-SHOT #- F(X) DATA -UF2DWR- 27Nov2019
                               ;-SHOT DATE-  UFILES ASCII FILE SYSTEM
   0                           ;-NUMBER OF ASSOCIATED SCALAR QUANTITIES-
 Time                Seconds   ;-INDEPENDENT VARIABLE LABEL: X-
 POSITION                      ;-INDEPENDENT VARIABLE LABEL: Y-
 R BOUNDARY    [M]             ;-DEPENDENT VARIABLE LABEL-
 0                             ;-PROC CODE- 0:RAW 1:AVG 2:SM. 3:AVG+SM
          1                    ;-# OF  X PTS-
        {len(Z)}                    ;-# OF  Y PTS-
"""

        with open(file_output_location / "R_BOUNDARY", 'w') as f:
            x = range(1, len(g.Rb) + 1)
            f.write(r_preamble)
            f.write(f"  1.000000e-01\n ")
            f.write("\n ".join("".join(f" {num:.6e}" if num >= 0 else f"{num:.6e}" for num in x[i:i + 6]) for i in
                               range(0, len(x), 6)))
            f.write("\n ")
            f.write("\n ".join("".join(f" {num:.6e}" if num >= 0 else f"{num:.6e}" for num in R[i:i + 6]) for i in
                               range(0, len(x), 6)))
            f.write("\n ")
            f.write(";----END-OF-DATA-----------------COMMENTS:-----------;")

        with open(file_output_location / "Z_BOUNDARY", 'w') as f:
            f.write(";----END-OF-DATA-----------------COMMENTS:-----------;")

        psi_n = g.g["AuxQuantities"]["PSI_NORM"]

    # replace this two-step process with one functional form: Pablo said he would do this

    if use_eped_pedestal:

        if eped_nn is not None:

            # parameters are hardcoded for now
            
            p, w = eped_nn(*eped_params)

            print(f"Pedestal values: p = {p}, w = {w}")

            rho_n = np.linspace(0,1,len(psi_n)) if psi_n is not None else rho
            rhotop = np.interp(1-w, psi_n, rho_n) if psi_n is not None else 0.5

            print(f"Pedestal location: psin = {w}, rho = {rhotop}")

            netop_19 = 1.08*eped_params[6]
            print(netop_19, "TOP DENS")
            Ttop_keV = (p*1E3) / (1.602176634E-19 * 2*netop_19 * 1e19) * 1E-3

            print(Ttop_keV, "TOP TEMP")

            #basically want to wrap this in a function to produce the correct betan

            betan_desired = eped_params[7]
            
            from scipy.optimize import minimize

            def _betan_initial_conditions(x, geometry_object, rhotop, Ttop_keV, netop_19, eped_params, x_a=0.3, n_rho=104):

                # Calculate the pressure profile from the initial conditions:

                aLT, aLn = x
                

                x, T = FunctionalForms.MITIMfunctional_aLyTanh(rhotop, Ttop_keV, eped_params[9]*1e-2, aLT, x_a=x_a, nx=n_rho)
                x, n = FunctionalForms.MITIMfunctional_aLyTanh(rhotop, netop_19, eped_params[10]*netop_19, aLn, x_a=x_a, nx=n_rho)


                #calculate the flux surface averaged pressure and approximate betan
                profiles = geometry_object.to_profiles()
                profiles.profiles['te(keV)'] = np.interp(profiles.profiles['rho(-)'], x, T)
                profiles.profiles['ne(10^19/m^3)'] = np.interp(profiles.profiles['rho(-)'], x, n)
                profiles.profiles['ti(keV)'][:,0] = np.interp(profiles.profiles['rho(-)'], x, T)
                profiles.makeAllThermalIonsHaveSameTemp()
                profiles.profiles['ni(10^19/m^3)'][:,0] = profiles.profiles['ne(10^19/m^3)']
                profiles.enforceQuasineutrality()
                profiles.deriveQuantities()

                print("residual:", ((profiles.derived['BetaN_engineering']-betan_desired) / betan_desired)**2)

                return ((profiles.derived['BetaN_engineering']-betan_desired) / betan_desired)**2

            aLT = 2.0
            aLn = 0.2
            x_a = 0.3

            x0 = [aLT, aLn]
            bounds = [(0, 3), (0, 1)]

            res = minimize(_betan_initial_conditions, 
                           x0, 
                           args=(g,rhotop, Ttop_keV, netop_19, eped_params, x_a, n_rho), 
                           method='Nelder-Mead',
                           bounds=bounds,
                           tol=1e-1,
                           options={'disp': True})
            
            if res.success:
                print(res.x)
                aLT, aLn = res.x
            
            x, T = FunctionalForms.MITIMfunctional_aLyTanh(rhotop, Ttop_keV, eped_params[9]*1e-2, aLT, x_a=x_a, nx=n_rho)
            x, n = FunctionalForms.MITIMfunctional_aLyTanh(rhotop, netop_19, eped_params[10]*netop_19, aLn, x_a=x_a, nx=n_rho)

        else:

            BC_index = np.argmin(np.abs(rho-0.95))
            width_top = width_top
            ne_ped = n[BC_index]
            Te_ped = T[BC_index]
            ne_sep = 0.3*ne_ped
            T_sep = 1

            n_ped = FunctionalForms.pedestal_tanh(ne_ped, ne_sep, width_top, x=rho)[1]
            T_ped = FunctionalForms.pedestal_tanh(Te_ped, T_sep, width_top, x=rho)[1]
            n[BC_index:] = n_ped[BC_index:]
            T[BC_index:] = T_ped[BC_index:]

            print(f"Pedestal values: ne_ped = {ne_ped}, Te_ped = {Te_ped}")

    preamble_Temp = f""" 900052D3D  2 0 6              ;-SHOT #- F(X) DATA WRITEUF OMFIT
                               ;-SHOT DATE-  UFILES ASCII FILE SYSTEM
   0                           ;-NUMBER OF ASSOCIATED SCALAR QUANTITIES-
 Time                Seconds   ;-INDEPENDENT VARIABLE LABEL: X1-
 rho_tor                       ;-INDEPENDENT VARIABLE LABEL: X0-
 Electron Temp       eV        ;-DEPENDENT VARIABLE LABEL-
 0                             ;-PROC CODE- 0:RAW 1:AVG 2:SM 3:AVG+SM
          1                    ;-# OF  X1 PTS-
        {n_rho}                    ;-# OF  X0 PTS-
"""
    preamble_dens = f""" 900052D3D  2 0 6              ;-SHOT #- F(X) DATA WRITEUF OMFIT
                               ;-SHOT DATE-  UFILES ASCII FILE SYSTEM
   0                           ;-NUMBER OF ASSOCIATED SCALAR QUANTITIES-
 Time                Seconds   ;-INDEPENDENT VARIABLE LABEL: X1-
 rho_tor                       ;-INDEPENDENT VARIABLE LABEL: X0-
 Electron Density    cm**-3    ;-DEPENDENT VARIABLE LABEL-
 0                             ;-PROC CODE- 0:RAW 1:AVG 2:SM 3:AVG+SM
          1                    ;-# OF  X1 PTS-
        {n_rho}                    ;-# OF  X0 PTS-
"""
    
    preamble_q = f""" 900052D3D  2 0 6              ;-SHOT #- F(X) DATA WRITEUF OMFIT
                               ;-SHOT DATE-  UFILES ASCII FILE SYSTEM
   0                           ;-NUMBER OF ASSOCIATED SCALAR QUANTITIES-
 Time                Seconds   ;-INDEPENDENT VARIABLE LABEL: X0-
 rho_tor                       ;-INDEPENDENT VARIABLE LABEL: X1-
 EFIT q profile                ;-DEPENDENT VARIABLE LABEL-
 0                             ;-PROC CODE- 0:RAW 1:AVG 2:SM 3:AVG+SM
          1                    ;-# OF  X0 PTS-
        {n_rho}                    ;-# OF  X1 PTS-
"""
    
    if Te is None:
        Te = T

    with open(file_output_location / "TE_ASTRA", 'w')  as f:
        f.write(preamble_Temp)
        f.write(f" 1.000000e-01\n ")
        f.write("\n ".join(" ".join(f"{num:.6e}" for num in x[i:i + 6]) for i in range(0, len(x), 6)))
        f.write("\n ")
        f.write("\n ".join(" ".join(f"{num:.6e}" for num in Te[i:i + 6]) for i in range(0, len(x), 6)))
        f.write("\n ")
        f.write(";----END-OF-DATA-----------------COMMENTS:-----------;")

    if Ti is None:
        Ti = T

    with open(file_output_location / "TI_ASTRA", 'w')  as f:
        f.write(preamble_Temp)
        f.write(f" 1.000000e-01\n ")
        f.write("\n ".join(" ".join(f"{num:.6e}" for num in x[i:i + 6]) for i in range(0, len(x), 6)))
        f.write("\n ")
        f.write("\n ".join(" ".join(f"{num:.6e}" for num in Ti[i:i + 6]) for i in range(0, len(x), 6)))
        f.write("\n ")
        f.write(";----END-OF-DATA-----------------COMMENTS:-----------;")

    if ne is None:
        ne = n

    with open(file_output_location / "NE_ASTRA", 'w')  as f:
        f.write(preamble_dens)
        f.write(f" 1.000000e-01\n ")
        f.write("\n ".join(" ".join(f"{num:.6e}" for num in x[i:i + 6]) for i in range(0, len(x), 6)))
        f.write("\n ")
        f.write("\n ".join(" ".join(f"{num:.6e}" for num in ne[i:i + 6]) for i in range(0, len(x), 6)))
        f.write("\n ")

    if q_profile is not None:

        q = q_profile
        
        with open(file_output_location / "q_ASTRA", 'w')  as f:
            f.write(preamble_q)
            f.write(f" 1.000000e-01\n ")
            f.write("\n ".join(" ".join(f"{num:.6e}" for num in x[i:i + 6]) for i in range(0, len(x), 6)))
            f.write("\n ")
            f.write("\n ".join(" ".join(f"{num:.6e}" for num in q[i:i + 6]) for i in range(0, len(x), 6)))
            f.write("\n ")
            f.write(";----END-OF-DATA-----------------COMMENTS:-----------;")

    if plotYN==True:
        fn = GUItools.FigureNotebook("ASTRA Initial Conditions")
        fig = fn.add_figure(label='Kinetic Profiles', tab_color=1)
        ax = fig.add_subplot(121)
        ax1 = ax.twinx()
        GRAPHICStools.addDenseAxis(ax)
        ax.plot(rho, T, label='T')
        ax.plot(rho, n, label='n',color='tab:orange')  
        ax.set_ylabel(r"$T_e$ [eV]")
        ax1.set_ylabel(r"$n_e$ [$10^{19}m^{-3}$]")
        ax.set_xlabel(r"$\rho$")
        ax.set_ylim(0,np.max(T)+5)
        ax1.set_ylim(0,np.max(T)+5)
        ax.set_title("Initial temperature profile")
        ax.legend()

        ax_geo = fig.add_subplot(122)
        ax_geo.plot(R,Z)
        ax_geo.set_aspect('equal')

        fn.show()
