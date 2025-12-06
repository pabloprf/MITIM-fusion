import argparse
from pathlib import Path
from mitim_tools.misc_tools import IOtools
from mitim_modules.maestro.MAESTROmain import maestro
from mitim_tools.misc_tools.IOtools import mitim_timer
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

@mitim_timer('MAESTRO')
def run_maestro_local(    
        file_path,
        folder              = None,
        terminal_outputs    = False,
        force_cold_start    = False,
        cpus                = 8,
        keep_all_files      = True,
        ):
    
    
    maestro_namelist = IOtools.read_mitim_yaml(file_path)
    
    # ****************************************************************************************************************
    # ****************************************************************************************************************
    # Parse namelist
    # ****************************************************************************************************************
    # ****************************************************************************************************************
    
    seed = maestro_namelist["seed"] if "seed" in maestro_namelist else 0

    # ---------------------------------------------------------------------------------------
    # Read problem parameters
    # ---------------------------------------------------------------------------------------

    parameters_engineering = {
        'Ip_MA':        maestro_namelist["machine"]["Ip"],
        'B_T':          maestro_namelist["machine"]["Bt"],
        'Zeff':         maestro_namelist["assumptions"]["Zeff"],
        'PichT_MW':     maestro_namelist["machine"]["heating"]["parameters"]["P_icrh"],
        'neped_20' :    maestro_namelist["assumptions"]["initialization"]["neped_20"] ,
        'Tesep_keV':    maestro_namelist["assumptions"]["Tesep_eV"]*1E-3,
        'nesep_20':     maestro_namelist["assumptions"]["initialization"]["neped_20"] * maestro_namelist["assumptions"]["initialization"]["nesep_ratio"]
        }
    
    separatrix_type = maestro_namelist["machine"]["separatrix"]["type"]
    
    parameters_initialize = {
        'BetaN_initialization':     maestro_namelist["assumptions"]["initialization"]["BetaN"],
        'peaking_initialization':   maestro_namelist["assumptions"]["initialization"]["density_peaking"],
        "initializer":              separatrix_type
        }

    # Initialize geometry from first 4 MXH moments
    if separatrix_type == "freegs":
        
        R           = maestro_namelist["machine"]["separatrix"]["parameters"]["R"]
        a           = maestro_namelist["machine"]["separatrix"]["parameters"]["a"]
        kappa_sep   = maestro_namelist["machine"]["separatrix"]["parameters"]["kappa_sep"]
        delta_sep   = maestro_namelist["machine"]["separatrix"]["parameters"]["delta_sep"]
        n_mxh       = maestro_namelist["machine"]["separatrix"]["parameters"]["n_mxh"]
        geometry    = {'R': R, 'a': a, 'kappa_sep': kappa_sep, 'delta_sep': delta_sep, 'zeta_sep': 0.0, 'z0': 0.0, 'coeffs_MXH' : n_mxh}
    
    elif separatrix_type == 'fibe': 
        R           = maestro_namelist["machine"]["separatrix"]["parameters"]["R"]
        a           = maestro_namelist["machine"]["separatrix"]["parameters"]["a"]
        kappa_sep   = maestro_namelist["machine"]["separatrix"]["parameters"]["kappa_sep"]
        delta_sep   = maestro_namelist["machine"]["separatrix"]["parameters"]["delta_sep"]
        zeta_sep    = maestro_namelist["machine"]["separatrix"]["parameters"]["zeta_sep"]
        n_mxh       = maestro_namelist["machine"]["separatrix"]["parameters"]["n_mxh"]
        geometry    = {'R': R, 'a': a, 'kappa_sep': kappa_sep, 'delta_sep': delta_sep, 'zeta_sep': 0.0, 'z0': 0.0, 'coeffs_MXH' : n_mxh}
    
    # Initialize geometry from geqdsk file
    elif separatrix_type == "geqdsk":
        
        geqdsk_file = maestro_namelist["machine"]["separatrix"]["parameters"]["geqdsk_file"]
        n_mxh       = maestro_namelist["machine"]["separatrix"]["parameters"]["n_mxh"]
        geometry    = {'geqdsk_file':geqdsk_file,'coeffs_MXH' : n_mxh}
    
    else:
        raise ValueError('[MITIM] Only "freegs" (mxh) or "geqdsk" are supported')

    # ---------------------------------------------------------------------------------------
    # Read user settings and default namelists for individual Beats
    # ---------------------------------------------------------------------------------------

    potential_beats = maestro_namelist["maestro"]["beats"] + ["eped_initializer"] # The ones that I want to use plus the special one


    beat_prepare_namelists, beat_run_namelists = {}, {}
    for beat in potential_beats:
        
        # Read beat parameters
        beat_parameters = maestro_namelist["maestro"][f"{beat}_beat"]

        # ********
        # ******** Prepare the prepare() parameters
        # ********
        
        beat_prepare_namelist_mod = beat_parameters["parameters_prepare"]
        
        # I can also provide a "base namelist" from another beat so that I don't repeat all the inputs and I have just indicated the changes
        beat_base = beat_parameters["base_beat"]
        if beat_base is not None:
            beat_base_namelist = maestro_namelist["maestro"][beat_base]["parameters_prepare"]
            beat_prepare_namelist = IOtools.deep_dict_update(beat_base_namelist, beat_prepare_namelist_mod)
        else:
            beat_prepare_namelist = beat_prepare_namelist_mod
        
        # Potentially modify namelist based on rest of the namelist
        preprocess_prepare_function = None
        
        if "preprocess_prepare" in beat_parameters:
            preprocess_prepare_function = beat_parameters["preprocess_prepare"]
            preprocess_prepare_parameters = beat_parameters["preprocess_prepare_parameters"]
        elif beat_base is not None:
            preprocess_prepare_function = maestro_namelist["maestro"][beat_base]["preprocess_prepare"]
            preprocess_prepare_parameters = maestro_namelist["maestro"][beat_base]["preprocess_prepare_parameters"]
            
        if preprocess_prepare_function is not None:
            beat_prepare_namelist = preprocess_prepare_function(
                beat_prepare_namelist,
                maestro_namelist,
                preprocess_prepare_parameters
                )

        # ********
        # ******** Prepare the run() parameters
        # ********

        # Run 
        if "preprocess_run" in beat_parameters and beat_parameters["preprocess_run"] is not None: 
            beat_run_namelist = beat_parameters["preprocess_run"]({}, maestro_namelist, cpus, force_cold_start)
        elif beat_base is not None:
            beat_run_namelist = maestro_namelist["maestro"][beat_base]["preprocess_run"]({}, maestro_namelist, cpus, force_cold_start)
        else:
            beat_run_namelist = {}

        beat_prepare_namelists[beat] = beat_prepare_namelist
        beat_run_namelists[beat] = beat_run_namelist

    # ****************************************************************************************************************
    # ****************************************************************************************************************
    # Execute MAESTRO
    # ****************************************************************************************************************
    # ****************************************************************************************************************    

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
    
    for beat in maestro_namelist["maestro"]["beats"]:
        
        beat_parameters = maestro_namelist["maestro"][f"{beat}_beat"]
        
        # ****************************************************************************
        # Define beat
        # ****************************************************************************
        
        m.define_beat(
            beat_parameters["beat_type"],
            initializer = None if creator_added else parameters_initialize["initializer"]
            )

        # ****************************************************************************
        # Define creator
        # ****************************************************************************
        
        if not creator_added:
            
            m.define_creator(
                'eped_initializer', 
                BetaN = parameters_initialize["BetaN_initialization"], 
                nu_ne = parameters_initialize["peaking_initialization"], 
                **beat_prepare_namelists["eped_initializer"],
                **parameters_engineering
                )
            
            m.initialize(
                BetaN = parameters_initialize["BetaN_initialization"],
                **geometry,
                **parameters_engineering
                )
            
            creator_added = True

        # ****************************************************************************
        # Prepare beat
        # ****************************************************************************
        
        m.prepare(**beat_prepare_namelists[beat])
        
        # ****************************************************************************
        # Run beat
        # ****************************************************************************
        
        m.run(**beat_run_namelists[beat])
        
    # ****************************************************************************
    # Finalize MAESTRO run
    # ****************************************************************************

    m.finalize()

    return m

def main():
    parser = argparse.ArgumentParser(description='Parse MAESTRO namelist')
    parser.add_argument('folder', type=str, help='Folder to run MAESTRO')
    parser.add_argument("--namelist", type=str, required=False, default=None) # namelist.maestro.yaml file, otherwise what's in the current folder
    parser.add_argument('--cpus', type=int, required=False, default=8, help='Number of CPUs to use')
    parser.add_argument('--terminal', action='store_true', help='Print terminal outputs')
    args = parser.parse_args()
    
    folder = IOtools.expandPath(args.folder)
    maestro_namelist = args.namelist
    cpus = args.cpus
    terminal_outputs = args.terminal

    maestro_namelist = Path(maestro_namelist) if  maestro_namelist is not None else IOtools.expandPath('.') / "namelist.maestro.yaml"

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    
    run_maestro_local(maestro_namelist,folder=folder,cpus = cpus, terminal_outputs = terminal_outputs)

if __name__ == "__main__":
    main()