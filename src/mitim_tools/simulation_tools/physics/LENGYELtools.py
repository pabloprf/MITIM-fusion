import os
from pathlib import Path
import matplotlib.pyplot as plt
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools import __mitimroot__
from IPython import embed

from extended_lengyel.cli import run_extended_lengyel

class Lengyel():
    def __init__(
        self
        ):
        
        self.nml_default = Path(__mitimroot__ / 'templates' / 'input.lengyel.controls.yaml')
  
    # Optional preparation step
    def prep(
        self,
        radas_dir,
        input_gacode = None
        ):
        
        # Read default namelist
        self.nml = IOtools.read_mitim_yaml(self.nml_default)
        
        # Change RADAS directory
        radas_dir = Path(radas_dir)
        if radas_dir.exists():
            self.nml['input']['radas_dir'] = f"PATH:{radas_dir.resolve()}"
        else:
            raise FileNotFoundError(f"[MITIM] The provided RADAS_DIR '{radas_dir}' does not exist; please do 'radas -c radas_config.yml -s tungsten' with the proper config and impurities")

        # Potentially change some parameters from the input.gacode
        
        params = {
            'major_radius': ['profiles', 'rcentr(m)', 'm', 0],
            'minor_radius': ['derived', 'a', 'm', None],
            'elongation_psi95': ['derived', 'kappa95', ' ', None],
            'triangularity_psi95': ['derived', 'delta95', ' ', None],
            'magnetic_field_on_axis': ['profiles', 'bcentr(T)', 'T', 0],
            'plasma_current': ['profiles', 'current(MA)', 'MA', 0],
            'ion_mass': ['derived', 'mbg_main', 'amu', None],
            'power_crossing_separatrix': ['derived', 'Psol', 'MW', None],
            'separatrix_electron_density': ['profiles', 'ne(10^19/m^3)', 'e19/m^3', -1],
        }
        
        if input_gacode is not None:
            print(f"\t- Populating Lengyel input from provided GACODE profile:")
            if isinstance(input_gacode, PROFILEStools.gacode_state):
                p = input_gacode
            else:
                p = PROFILEStools.gacode_state(input_gacode)
                
            for par in params:
                val = p.__dict__[params[par][0]][params[par][1]]
                if params[par][3] is not None:
                    val = val[params[par][3]]
                print(f"\t\t* Setting '{par}' to MITIMstate value '{params[par][1]} = {val}'")
                self.nml['input'][par] = f'{val}{params[par][2]}' 

    def run(
        self,
        folder,
        cold_start = False,
        **input_dict
        ):
        
        folder = Path(folder)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        elif cold_start:
            print(f"\t- Cold starting Lengyel run; cleaning folder '{folder}'")
            os.system(f'rm -rf {folder}/*')
        
        # Potentially modify namelist with input_dict
        for key in input_dict:
            print(f"\t- Setting Lengyel input parameter '{key}' to '{input_dict[key]}'")
            self.nml['input'][key] = input_dict[key]
        
        # Write modified namelist to folder
        nml_file = folder / 'input.lengyel.controls.yml'
        IOtools.write_mitim_yaml( self.nml, nml_file )
        
        # Run
        output_file = folder / 'output.lengyel.results.yml'
        run_extended_lengyel(
            config_file = nml_file,
            output_file = output_file,
        )
        
        # Read output
        self.results = IOtools.read_mitim_yaml( output_file )

    def run_scan(
        self,
        folder,
        scan_name,
        scan_values,
        cold_start = False,
        plotYN = True,
        **input_dict
    ):
        
        folder = Path(folder)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        
        self.results_scan = {}
        for val in scan_values:
            print(f"\t- Running Lengyel scan '{scan_name}' with value '{val}'")
            scan_folder = folder / f"{scan_name}_{val}"
            scan_input = input_dict.copy()
            scan_input[scan_name] = val
            self.run(
                folder = scan_folder,
                cold_start = cold_start,
                **scan_input
            )
            self.results_scan[val] = self.results
    
        # Plot
        if plotYN:
            plt.ion(); fig, ax = plt.subplots()
            for val in scan_values:
                res = self.results_scan[val]
                ax.plot(
                    val,
                    float(res['separatrix_electron_temp'].split()[0]),
                    'o', markersize=15
                )
                
            ax.set_xlabel(f"{scan_name}")
            ax.set_ylabel("Separatrix electron temperature [eV]")
            GRAPHICStools.addDenseAxis(ax)
            
        
        
        