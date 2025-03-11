import argparse
import os
from mitim_tools.misc_tools import IOtools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_modules.maestro.MAESTROmain import maestro
from mitim_modules.maestro.utils import TRANSPbeat, PORTALSbeat
from mitim_tools.misc_tools.IOtools import mitim_timer
from mitim_tools.misc_tools import PLASMAtools

def parse_maestro_nml(file_path):
    # Extract engineering parameters, initializations, and desired beats to run
    maestro_namelist = IOtools.read_mitim_nml(file_path)
    print(maestro_namelist)

    
    Ip = maestro_namelist["machine"]["Ip"]
    Bt = maestro_namelist["machine"]["Bt"]
    
    if maestro_namelist["assumptions"]["initialization"]["assume_neped"]:
        neped = maestro_namelist["assumptions"]["initialization"]["neped_20"]
        nesepratio = maestro_namelist["assumptions"]["initialization"]["nesep_ratio"]
        nesep = neped * nesepratio
    else:
        raise ValueError("Only assume_neped is supported for now")
    
    if maestro_namelist["machine"]["heating"]["type"] == "ICRH":
        Pich = maestro_namelist["machine"]["heating"]["parameters"]["P_icrh"]
        Zmini = maestro_namelist["machine"]["heating"]["parameters"]["Zmini"] 
        fmini = maestro_namelist["machine"]["heating"]["parameters"]["fmini"]
    else:
        raise ValueError("Only ICRH heating is supported for now")
    
    Zeff = maestro_namelist["assumptions"]["Zeff"]

    parameters_engineering = {'Ip_MA': Ip, 'B_T': Bt, 'Zeff': Zeff, 'PichT_MW': Pich, 'neped_20' : neped , 'Tesep_keV': 0.075, 'nesep_20': neped*0.3}
    
    fmain = maestro_namelist["assumptions"]["mix"]["fmain"]
    fW = maestro_namelist["assumptions"]["mix"]["fW"]
    ZW = maestro_namelist["assumptions"]["mix"]["ZW"]

    LowZ, Wratio = PLASMAtools.estimateLowZ(fmain,Zeff,Zmini,fmini,ZW,fW)

    parameters_mix = {'DTplasma': True, 'lowZ_impurity': LowZ, 'impurity_ratio_WtoZ': Wratio, 'minority': [Zmini,1,fmini]}

    parameters_initialize = {'BetaN_initialization': 2.0, 'peaking_initialization': 1.5, "initializer":"freegs"}

    if maestro_namelist["machine"]["separatrix"]["type"] == "mxh":
        R = maestro_namelist["machine"]["separatrix"]["parameters"]["R"]
        a = maestro_namelist["machine"]["separatrix"]["parameters"]["a"]
        kappa_sep = maestro_namelist["machine"]["separatrix"]["parameters"]["kappa_sep"]
        delta_sep = maestro_namelist["machine"]["separatrix"]["parameters"]["delta_sep"]
        n_mxh = maestro_namelist["machine"]["separatrix"]["parameters"]["n_mxh"]
    else:
        raise ValueError("Only mxh separatrix is supported for now, use a separate script for geqdsk")

    geometry = {'R': R, 'a': a, 'kappa_sep': kappa_sep, 'delta_sep': delta_sep, 'zeta_sep': 0.0, 'z0': 0.0, 'coeffs_MXH' : n_mxh}

    beat_namelists = {}

    for beat_type in ["eped", "transp", "transp_soft", "portals"]: # Add more beats as we grow maestro
        if f"{beat_type}_beat" in maestro_namelist["maestro"]:
            if f"{beat_type}_namelist" in maestro_namelist["maestro"][f"{beat_type}_beat"]:
                beat_namelist = maestro_namelist["maestro"][f"{beat_type}_beat"][f"{beat_type}_namelist"]

                # soft portals namelist
                if beat_type == "portals":
                    portals_namelist_soft = PORTALSbeat.portals_beat_soft_criteria(beat_namelist)
                    # add postprocessing function
                    def profiles_postprocessing_fun(file_profs):
                        p = PROFILEStools.PROFILES_GACODE(file_profs)
                        p.lumpImpurities()
                        p.enforce_same_density_gradients()
                        p.writeCurrentStatus(file=file_profs)
                    beat_namelist['PORTALSparameters']['profiles_postprocessing_fun'] = profiles_postprocessing_fun
        else:
            if beat_type == "transp":
                beat_namelist = TRANSPbeat.transp_beat_default_nml(parameters_engineering,parameters_mix)
            elif beat_type == "transp_soft":
                beat_namelist = TRANSPbeat.transp_beat_default_nml(parameters_engineering,parameters_mix,only_current_diffusion=True)
    
        beat_namelists[beat_type] = beat_namelist

    maestro_beats = maestro_namelist["maestro"]

    return parameters_engineering, parameters_mix, parameters_initialize, geometry, beat_namelists, maestro_beats

def build_maestro_run_local(
        parameters_engineering, 
        parameters_mix, 
        parameters_initialize, 
        geometry, 
        beat_namelists, 
        maestro_beats,
        folder=None,
        terminal_outputs = False
        ):

    if folder is None:
        folder = os.getcwd()

    m = maestro(folder, terminal_outputs = terminal_outputs)

    return m

@mitim_timer('\t\t* MAESTRO')
def run_maestro_local(    
        parameters_engineering, 
        parameters_mix, 
        parameters_initialize, 
        geometry, 
        beat_namelists, 
        maestro_beats,
        folder=None,
        terminal_outputs = False
        ):
    
    m = build_maestro_run_local(folder=folder,terminal_outputs = terminal_outputs)
    
    while maestro_beats["beats"]: # iterates through list of beats fron the json file until it's empty
        if maestro_beats["beats"][0] == "transp":
            m.define_beat('transp')
            m.prepare(**beat_namelists['transp'])
            m.run()
        elif maestro_beats["beats"][0] == "transp_soft":
            m.define_beat('transp', initializer=parameters_initialize["initializer"])
            m.define_creator(
                'eped', 
                BetaN = parameters_initialize["BetaN_initialization"], 
                nu_ne = parameters_initialize["peaking_initialization"], 
                **beat_namelists["eped"],
                **parameters_engineering
                )
            m.initialize(BetaN = parameters_initialize["BetaN_initialization"], **geometry, **parameters_engineering)
            m.prepare(**beat_namelists['transp_soft'])
        elif maestro_beats["beats"][0] == "eped":
            m.define_beat('eped')
            m.prepare(**beat_namelists['eped'])
            m.run()
        elif maestro_beats["beats"][0] == "portals":
            m.define_beat('portals')
            m.prepare(**beat_namelists['portals'], 
                      change_last_radial_call = maestro_beats["beats"]["portals_beat"]["change_last_radial_call"], 
                      use_previous_surrogate_data=maestro_beats["beats"]["portals_beat"]["use_previous_surrogate_data"], 
                      try_flux_match_only_for_first_point=maestro_beats["beats"]["portals_beat"]["try_flux_match_only_for_first_point"])
            m.run()

        maestro_beats["beats"].pop(0)

    m.finalize()

    return m

def main():
    parser = argparse.ArgumentParser(description='Parse MAESTRO namelist')
    parser.add_argument('file_path', type=str, help='Path to MAESTRO namelist file')
    args = parser.parse_args()
    file_path = args.file_path
    parse_maestro_nml(file_path)
    print("success")
    m = build_maestro_run_local(**parse_maestro_nml(file_path))
    run_maestro_local(m, *parse_maestro_nml(file_path))
