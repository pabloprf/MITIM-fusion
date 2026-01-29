import argparse
import copy
import json
import numpy as np
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
    
    # In case a beat requests this information (e.g. EPED initializer)
    maestro_namelist['maestro']['master_cpus'] = cpus
    
    # ****************************************************************************************************************
    # ****************************************************************************************************************
    # Parse namelist
    # ****************************************************************************************************************
    # ****************************************************************************************************************
    
    seed = maestro_namelist["seed"] if "seed" in maestro_namelist else 0

    # ---------------------------------------------------------------------------------------
    # Read problem parameters
    # ---------------------------------------------------------------------------------------

    # Define total power
    if maestro_namelist["plasma"]["heating"]["type"] == "ICRH":
        Ptotal = maestro_namelist["plasma"]["heating"]["parameters"]["P_icrh"]
    elif maestro_namelist["plasma"]["heating"]["type"] == "gaussian_sources":
        Ptotal = maestro_namelist["plasma"]["heating"]["parameters"]["Pe"] + maestro_namelist["plasma"]["heating"]["parameters"]["Pi"]

    parameters_engineering = {
        'Ip_MA':        maestro_namelist["plasma"]["parameters"]["Ip"],
        'B_T':          maestro_namelist["plasma"]["parameters"]["Bt"],
        'Zeff':         maestro_namelist["plasma"]["species"]["Zeff"],
        'PichT_MW':     Ptotal,
        'neped_20' :    maestro_namelist["plasma"]["parameters"]["neped_20"] ,
        'Tesep_keV':    maestro_namelist["plasma"]["parameters"]["Tesep_eV"]*1E-3,
        'nesep_20':     maestro_namelist["plasma"]["parameters"]["neped_20"] * maestro_namelist["plasma"]["parameters"]["ne_ratio_sep_ped"]
        }
    
    initialization_type =  maestro_namelist["plasma"]["profiles_initialization"]["initialization_type"]
    
    initialization_creator_type = maestro_namelist["plasma"]["profiles_initialization"]["creator_type"]
    parameters_initialize =  maestro_namelist["plasma"]["profiles_initialization"]["parameters"]

    if "freeze_995_from" in maestro_namelist["plasma"]["parameters"]["separatrix"]:
        if maestro_namelist["plasma"]["parameters"]["separatrix"]["freeze_995_from"] == "geo":
            print('[MAESTRO] Warning: "geo" option for freeze_995_from is deprecated, use "analytic_interpolation" instead', typeMsg='w')
            maestro_namelist["plasma"]["parameters"]["separatrix"]["freeze_995_from"] = "analytic_interpolation"
            
    # Initialize geometry from first 4 MXH moments
    if initialization_type in ['fibe','separatrix',"freegs"]:
        
        R           = maestro_namelist["plasma"]["parameters"]["separatrix"]["R"]
        a           = maestro_namelist["plasma"]["parameters"]["separatrix"]["a"]
        kappa_sep   = maestro_namelist["plasma"]["parameters"]["separatrix"]["kappa_sep"]
        delta_sep   = maestro_namelist["plasma"]["parameters"]["separatrix"]["delta_sep"]
        zeta_sep    = maestro_namelist["plasma"]["parameters"]["separatrix"]["zeta_sep"]
        n_mxh       = maestro_namelist["plasma"]["parameters"]["separatrix"]["n_mxh"]
        extract_995_from = maestro_namelist["plasma"]["parameters"]["separatrix"]["freeze_995_from"]
        rz_boundary_file = maestro_namelist["plasma"]["parameters"]["separatrix"]["rz_boundary_file"]
        internal_flux_file = maestro_namelist["plasma"]["parameters"]["separatrix"]["internal_flux_file"]
        geometry    = {'R': R, 'a': a, 'kappa_sep': kappa_sep, 'delta_sep': delta_sep, 'zeta_sep': zeta_sep, 'z0': 0.0, 'coeffs_MXH' : n_mxh, 'rz_boundary_file': rz_boundary_file, 'extract_995_from': extract_995_from, 'internal_flux_file': internal_flux_file}

    # Initialize geometry from geqdsk file
    elif initialization_type == "geqdsk":
        
        geqdsk_file = maestro_namelist["plasma"]["parameters"]["separatrix"]["geqdsk_file"]
        n_mxh       = maestro_namelist["plasma"]["parameters"]["separatrix"]["n_mxh"]
        extract_995_from = maestro_namelist["plasma"]["parameters"]["separatrix"]["freeze_995_from"]
        geometry    = {'geqdsk_file':geqdsk_file,'coeffs_MXH' : n_mxh, 'extract_995_from': extract_995_from}
    
    else:
        geometry = {}
        
    # ---------------------------------------------------------------------------------------
    # Read user settings and default namelists for individual Beats
    # ---------------------------------------------------------------------------------------

    potential_beats = maestro_namelist["maestro"]["beats"] + ["eped_initializer"] # The ones that I want to use plus the special one


    beat_prepare_namelists, beat_run_namelists = {}, {}
    for beat in potential_beats:
        
        # Read beat parameters
        beat_parameters = copy.deepcopy(maestro_namelist["maestro"][beat])

        # ********
        # ******** Prepare the prepare() parameters
        # ********
        
        beat_prepare_namelist_mod = beat_parameters["parameters_prepare"]
        
        # I can also provide a "base namelist" from another beat so that I don't repeat all the inputs and I have just indicated the changes
        beat_base = beat_parameters["base_module"]
        if beat_base is not None:
            beat_base_namelist = copy.deepcopy(maestro_namelist["maestro"][beat_base]["parameters_prepare"])
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
        elif (beat_base is not None) and (maestro_namelist["maestro"][beat_base]["preprocess_run"] is not None):
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

    # Copy namelist to folder
    IOtools.write_mitim_yaml(maestro_namelist, Path(folder) / "maestro.namelist.actual.yaml")

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
        keep_all_files = keep_all_files,
        maestro_namelist = maestro_namelist
        )

    # -------------------------------------------------------------------------
    # Loop through beats
    # -------------------------------------------------------------------------

    creator_added = False
    
    for beat in maestro_namelist["maestro"]["beats"]:
        
        beat_parameters = maestro_namelist["maestro"][beat]
        
        # ****************************************************************************
        # Define beat
        # ****************************************************************************
        
        # Initialization chosen (profiles, freegs, geqdsk, fibe) for the first beat
        initialize_this_beat_with = initialization_type if (not creator_added) else None
        
        m.define_beat(
            beat_parameters["beat_type"],
            initializer = initialize_this_beat_with
            )

        # ****************************************************************************
        # Define creator
        # ****************************************************************************
        
        if not creator_added:
            
            if initialization_creator_type is not None:
                
                # Special case #TODO: Improve in future
                if initialization_creator_type == 'fixed_profiles':
                    profiles_insert = read_fixed_profiles(parameters_initialize['profiles_file'])
                    parameters_initialize['profiles_insert'] = profiles_insert
                else:
                    # If normal creator, append the **beat_prepare_namelists[initialization_creator_type],
                    parameters_initialize = IOtools.deep_dict_update(
                        beat_prepare_namelists[initialization_creator_type],
                        parameters_initialize
                        )
                
                m.define_creator(
                    initialization_creator_type, # e.g. 'eped_initializer', 
                    **parameters_initialize,
                    **parameters_engineering
                    )
            
            m.initialize(
                **parameters_initialize,
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
        # Post-process beat
        # ****************************************************************************
        
        m.interpret()
        
    # ****************************************************************************
    # Finalize MAESTRO run
    # ****************************************************************************

    m.finalize()

    return m

def read_fixed_profiles(file):
    
    with open(file, 'r') as f:
        profiles_insert_tmp = json.load(f)
    
    profiles_insert = {}
    
    variables_to_extract = {
        'rho': 'rho',
        'roa': 'roa',
        'Te_keV': 'Te',
        'Ti_keV': 'Ti',
        'ne_1e20m3': 'ne',
        'w0_rads': 'w0',
    }
    
    for key, new_key in variables_to_extract.items():
        if key in profiles_insert_tmp:
            profiles_insert[new_key] = np.array(profiles_insert_tmp[key])
    
    return profiles_insert


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