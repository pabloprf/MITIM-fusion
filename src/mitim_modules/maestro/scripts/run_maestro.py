import argparse
import shutil
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.maestro.MAESTROmain import maestro
from mitim_modules.maestro.utils import TRANSPbeat, PORTALSbeat
from mitim_tools.misc_tools.IOtools import mitim_timer
from mitim_tools.misc_tools import PLASMAtools
from IPython import embed

def parse_maestro_nml(file_path):
    # Extract engineering parameters, initializations, and desired beats to run
    maestro_namelist = IOtools.read_mitim_nml(file_path)

    if "seed" in maestro_namelist:
        seed = maestro_namelist["seed"]
    else:
        seed = 0

    # ---------------------------------------------------------------------------------------
    # Engineering parameters
    # ---------------------------------------------------------------------------------------

    Ip = maestro_namelist["machine"]["Ip"]
    Bt = maestro_namelist["machine"]["Bt"]
    
    if maestro_namelist["assumptions"]["initialization"]["assume_neped"]:
        neped = maestro_namelist["assumptions"]["initialization"]["neped_20"]
        nesepratio = maestro_namelist["assumptions"]["initialization"]["nesep_ratio"]
        nesep = neped * nesepratio
    else:
        raise ValueError("[MITIM] Only assume_neped is supported for now")
    
    if maestro_namelist["machine"]["heating"]["type"] == "ICRH":
        Pich = maestro_namelist["machine"]["heating"]["parameters"]["P_icrh"]
        Zmini = maestro_namelist["machine"]["heating"]["parameters"]["minority"][0]
        Amini = maestro_namelist["machine"]["heating"]["parameters"]["minority"][1]
        fmini = maestro_namelist["machine"]["heating"]["parameters"]["fmini"]
    else:
        raise ValueError("[MITIM] Only ICRH heating is supported for now")
    
    Zeff = maestro_namelist["assumptions"]["Zeff"]
    Tsep = maestro_namelist["assumptions"]["Tesep_eV"]*1E-3

    parameters_engineering = {'Ip_MA': Ip, 'B_T': Bt, 'Zeff': Zeff, 'PichT_MW': Pich, 'neped_20' : neped , 'Tesep_keV': Tsep, 'nesep_20': nesep}
    
    # ---------------------------------------------------------------------------------------
    # Plasma mix and impurity parameters
    # ---------------------------------------------------------------------------------------

    fmain = maestro_namelist["assumptions"]["mix"]["fmain"]
    fW = maestro_namelist["assumptions"]["mix"]["fW"]
    ZW = maestro_namelist["assumptions"]["mix"]["ZW"]

    LowZ, Wratio = PLASMAtools.estimateLowZ(fmain,Zeff,Zmini,fmini,ZW,fW)

    parameters_mix = {'DTplasma': True, 'lowZ_impurity': LowZ, 'impurity_ratio_WtoZ': Wratio, 'minority': [Zmini,Amini,fmini]}

    # ---------------------------------------------------------------------------------------
    # Initialization parameters
    # ---------------------------------------------------------------------------------------

    separatrix_type = maestro_namelist["machine"]["separatrix"]["type"]
    parameters_initialize = {
        'BetaN_initialization': maestro_namelist["assumptions"]["initialization"]["BetaN"],
        'peaking_initialization': maestro_namelist["assumptions"]["initialization"]["density_peaking"],
        "initializer":separatrix_type}

    # ---------------------------------------------------------------------------------------
    # Geometry parameters
    # ---------------------------------------------------------------------------------------

    if separatrix_type == "freegs":
        # Initialize geometry from first 4 MXH moments
        R = maestro_namelist["machine"]["separatrix"]["parameters"]["R"]
        a = maestro_namelist["machine"]["separatrix"]["parameters"]["a"]
        kappa_sep = maestro_namelist["machine"]["separatrix"]["parameters"]["kappa_sep"]
        delta_sep = maestro_namelist["machine"]["separatrix"]["parameters"]["delta_sep"]
        n_mxh = maestro_namelist["machine"]["separatrix"]["parameters"]["n_mxh"]
        geometry = {'R': R, 'a': a, 'kappa_sep': kappa_sep, 'delta_sep': delta_sep, 'zeta_sep': 0.0, 'z0': 0.0, 'coeffs_MXH' : n_mxh}
    elif separatrix_type == "geqdsk":
        # Initialize geometry from geqdsk file
        geqdsk_file = maestro_namelist["machine"]["separatrix"]["parameters"]["geqdsk_file"]
        n_mxh = maestro_namelist["machine"]["separatrix"]["parameters"]["n_mxh"]
        geometry = {'geqdsk_file':geqdsk_file,'coeffs_MXH' : n_mxh}
    else:
        raise ValueError('[MITIM] Only "freegs" (mxh) or "geqdsk" are supported')

    # ---------------------------------------------------------------------------------------
    # Read user settings and default namelists for individual Beats
    # ---------------------------------------------------------------------------------------

    beat_namelists = {}

    for beat_type in ["eped", "transp", "transp_soft", "portals", "portals_soft"]:

        if f"{beat_type}_beat" in maestro_namelist["maestro"]:

            # ***************************************************************************
            # Do I want a default namelist?
            # ***************************************************************************
            if maestro_namelist["maestro"][f"{beat_type}_beat"]["use_default"]:

                if beat_type == "transp":
                    beat_namelist = TRANSPbeat.transp_beat_default_nml(parameters_engineering,parameters_mix)
                elif beat_type == "transp_soft":
                    beat_namelist = TRANSPbeat.transp_beat_default_nml(parameters_engineering,parameters_mix,only_current_diffusion=True)
                elif beat_type == "portals_soft":
                    # PORTALS soft requires right now that portals namelist is defined
                    if "portals_beat" in maestro_namelist["maestro"]:
                        beat_namelist = PORTALSbeat.portals_beat_soft_criteria(maestro_namelist["maestro"]["portals_beat"]["portals_namelist"])
                    else:
                        raise ValueError("[MITIM] For PORTALS soft default I need PORTALS namelist")
                else:
                    raise ValueError(f"[MITIM] {beat_type} beat does not have a default namelist yet")

            # ***************************************************************************
            # Read user namelist
            # ***************************************************************************
            elif f"{beat_type}_namelist" in maestro_namelist["maestro"][f"{beat_type}_beat"]:

                beat_namelist = maestro_namelist["maestro"][f"{beat_type}_beat"][f"{beat_type}_namelist"]

            # ***************************************************************************
            # Nothin yet
            # ***************************************************************************
            else:
                raise ValueError(f"[MITIM] {beat_type} beat not found in the MAESTRO namelist nor you wanted default")

            # ***************************************************************************
            # Additional modifications that are required but not in JSON
            # ***************************************************************************

            # soft portals namelist
            if beat_type in ["portals","portals_soft"]:

                lumpImpurities = maestro_namelist["maestro"]["portals_beat"]["transport_preprocessing"]["lumpImpurities"]
                enforce_same_density_gradients = maestro_namelist["maestro"]["portals_beat"]["transport_preprocessing"]["enforce_same_density_gradients"]

                # add postprocessing function
                def profiles_postprocessing_fun(file_profs):
                    p = PROFILEStools.gacode_state(file_profs)
                    if lumpImpurities:
                        p.lumpImpurities()
                    if enforce_same_density_gradients:
                        p.enforce_same_density_gradients()
                    p.writeCurrentStatus(file=file_profs)
                beat_namelist['PORTALSparameters']['profiles_postprocessing_fun'] = profiles_postprocessing_fun

        else:
            raise ValueError(f"[MITIM] {beat_type} beat not found in the MAESTRO namelist")

        beat_namelists[beat_type] = beat_namelist

    maestro_beats = maestro_namelist["maestro"]

    return parameters_engineering, parameters_initialize, geometry, beat_namelists, maestro_beats, seed

@mitim_timer('MAESTRO')
def run_maestro_local(    
        parameters_engineering, 
        parameters_initialize, 
        geometry, 
        beat_namelists, 
        maestro_beats,
        seed,
        folder=None,
        terminal_outputs = False,
        force_cold_start = False,
        cpus = 8,
        keep_all_files = True,
        ):
    
    # -------------------------------------------------------------------------
    # Initialize object
    # -------------------------------------------------------------------------

    if folder is None:
        folder = IOtools.expandPath('./')

    m = maestro(
        folder, 
        master_seed = seed, 
        terminal_outputs = terminal_outputs, 
        overall_log_file = True,
        master_cold_start = force_cold_start, 
        keep_all_files = keep_all_files)

    # -------------------------------------------------------------------------
    # Loop through beats
    # -------------------------------------------------------------------------

    creator_added = False

    while maestro_beats["beats"]:

        # ****************************************************************************
        # Define beat
        # ****************************************************************************
        if maestro_beats["beats"][0] in ["transp", "transp_soft"]:
            label_beat = "transp"
        elif maestro_beats["beats"][0] in ["eped"]:
            label_beat = "eped"
        elif maestro_beats["beats"][0] in ["portals", "portals_soft"]:
            label_beat = "portals"

        m.define_beat(label_beat, initializer=None if creator_added else parameters_initialize["initializer"])

        # ****************************************************************************
        # Define creator
        # ****************************************************************************
        if not creator_added:
            m.define_creator(
                'eped', 
                BetaN = parameters_initialize["BetaN_initialization"], 
                nu_ne = parameters_initialize["peaking_initialization"], 
                **beat_namelists["eped"],
                **parameters_engineering
                )
            m.initialize(BetaN = parameters_initialize["BetaN_initialization"], **geometry, **parameters_engineering)
            creator_added = True

        # ****************************************************************************
        # Define preparation and run
        # ****************************************************************************

        run_namelist = {}
        if maestro_beats["beats"][0] in ["transp", "transp_soft"]:
            run_namelist = {'mpisettings' : {"trmpi": cpus, "toricmpi": cpus, "ptrmpi": 1}}
        elif maestro_beats["beats"][0] in ["eped"]:
            run_namelist = {'cold_start': force_cold_start, 'cpus': cpus}

        m.prepare(**beat_namelists[maestro_beats["beats"][0]])
        m.run(**run_namelist)

        maestro_beats["beats"].pop(0)

    m.finalize()

    return m

def main():
    parser = argparse.ArgumentParser(description='Parse MAESTRO namelist')
    parser.add_argument('folder', type=str, help='Folder to run MAESTRO')
    parser.add_argument('file_path', type=str, help='Path to MAESTRO namelist file')
    parser.add_argument('cpus', type=int, help='Number of CPUs to use')
    parser.add_argument('--terminal', action='store_true', help='Print terminal outputs')
    args = parser.parse_args()
    folder = IOtools.expandPath(args.folder)
    file_path = args.file_path
    cpus = args.cpus
    terminal_outputs = args.terminal

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    
    IOtools.recursive_backup(folder / 'maestro_namelist.json')
    
    run_maestro_local(*parse_maestro_nml(file_path),folder=folder,cpus = cpus, terminal_outputs = terminal_outputs)


if __name__ == "__main__":
    main()