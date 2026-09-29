import os
from pathlib import Path
import matplotlib.pyplot as plt
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools.misc_tools import IOtools, GRAPHICStools
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools import __mitimroot__
from IPython import embed

class Lengyel():
    def __init__(
        self,
        namelist_location = None,  # Custom controls yaml; if None, use the default template namelist
        ):

        self.nml_default = Path(namelist_location) if namelist_location is not None else Path(__mitimroot__ / 'templates' / 'input.lengyel.controls.yaml')
  
    # Optional preparation step
    def prep(
        self,
        radas_dir = None,   # required for the seeded (detachment) workflow; unused by run_forward()
        input_gacode = None
        ):

        # Read default namelist
        self.nml = IOtools.read_mitim_yaml(self.nml_default)

        # Change RADAS directory (only if this namelist uses atomic data, i.e. seeded mode)
        if 'radas_dir' in self.nml['input']:
            if radas_dir is None:
                raise ValueError("[MITIM] This Lengyel namelist requires atomic data; please provide radas_dir (or RADAS_DIR env)")
            radas_dir = Path(radas_dir)
            if radas_dir.exists():
                self.nml['input']['radas_dir'] = f"PATH:{radas_dir.resolve()}"
            else:
                raise FileNotFoundError(f"[MITIM] The provided RADAS_DIR '{radas_dir}' does not exist; please do 'radas -c radas_config.yml -s tungsten' with the proper config and impurities")

        # Potentially change some parameters from the input.gacode
        # (only keys present in the loaded namelist template are populated, so the
        #  seeded and forward/clean templates each receive exactly their input set)

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
            'z_effective': ['derived', 'Zeff_vol', ' ', None],
            'average_total_pressure': ['derived', 'ptot_manual_vol', 'MPa', None],
        }

        if input_gacode is not None:
            print(f"\t- Populating Lengyel input from provided GACODE profile:")
            if isinstance(input_gacode, PROFILEStools.gacode_state):
                p = input_gacode
            else:
                p = PROFILEStools.gacode_state(input_gacode)

            for par in params:
                if par not in self.nml['input']:
                    continue
                val = p.__dict__[params[par][0]][params[par][1]]
                if params[par][3] is not None:
                    val = val[params[par][3]]
                # GACODE carries signed Bt/Ip (COCOS conventions); the SOL model wants magnitudes
                if par in ('magnetic_field_on_axis', 'plasma_current'):
                    val = abs(val)
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
            for item in folder.iterdir():
                IOtools.shutil_rmtree(item) if item.is_dir() else item.unlink()
        
        # Potentially modify namelist with input_dict
        for key in input_dict:
            print(f"\t- Setting Lengyel input parameter '{key}' to '{input_dict[key]}'")
            self.nml['input'][key] = input_dict[key]
        
        # Write modified namelist to folder
        nml_file = folder / 'input.lengyel.controls.yml'
        IOtools.write_mitim_yaml( self.nml, nml_file )
        
        # Run
        output_file = folder / 'output.lengyel.results.yml'
        from extended_lengyel.cli import run_extended_lengyel
        run_extended_lengyel(
            config_file = nml_file,
            output_file = output_file,
        )
        
        # Read output
        self.results = IOtools.read_mitim_yaml( output_file )

    def run_forward(
        self,
        folder,
        cold_start = False,
        **input_dict
        ):
        '''
        Non-detached FORWARD mode: upstream separatrix Te from the package's own
        registered conduction algorithms (see templates/input.lengyel_clean.controls.yaml
        for the algorithm list and the modeling notes). No impurity seeding, no
        detachment root-find, no atomic data.

        The package CLI driver (run_extended_lengyel) is bypassed on purpose: its
        output writer unconditionally reads the seeded solve's impurity_fraction and
        fails on a forward algorithm list. The same package config machinery is used
        to parse the namelist and resolve the algorithms, so all physics and unit
        handling stays package-side.
        '''
        import numpy as np
        import xarray as xr
        import yaml
        import cfspopcon
        from extended_lengyel import config as el_config

        folder = Path(folder)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        elif cold_start:
            print(f"\t- Cold starting Lengyel forward run; cleaning folder '{folder}'")
            for item in folder.iterdir():
                IOtools.shutil_rmtree(item) if item.is_dir() else item.unlink()

        # Potentially modify namelist with input_dict
        for key in input_dict:
            print(f"\t- Setting Lengyel input parameter '{key}' to '{input_dict[key]}'")
            self.nml['input'][key] = input_dict[key]

        # Write modified namelist to folder (traceability, and the package config
        # reader consumes it from file)
        nml_file = folder / 'input.lengyel.controls.yml'
        IOtools.write_mitim_yaml( self.nml, nml_file )

        algorithm = cfspopcon.CompositeAlgorithm.from_list(
            el_config.read_config_from_yaml(nml_file)["algorithms"]
        )
        data_vars = el_config.read_config(
            elements          = ["input"],
            filepath          = nml_file,
            keys              = algorithm.input_keys,
            allowed_missing   = algorithm.default_keys,
            overrides         = {},
            warn_if_unused    = True,
            convert_overrides = True,
        )

        # q_parallel convention: the seeded driver internally multiplies the divertor
        # power fraction by (1 - 1/e) before calc_parallel_heat_flux_density
        # (Lengyel_model_extended_S_Zeff_alphat.py); apply the same factor so forward
        # and seeded modes share the q_parallel definition.
        data_vars['fraction_of_P_SOL_to_divertor'] = data_vars['fraction_of_P_SOL_to_divertor'] * (1.0 - 1.0 / np.e)

        ds = xr.Dataset(data_vars=data_vars)
        algorithm.validate_inputs(ds)
        ds = algorithm.update_dataset(ds)

        self.results_forward_ds = ds

        # Results dict in the same string-with-units format as the seeded output yaml
        # (sanitize_variable strips the pint wrapper into value + units, as the CLI does)
        from cfspopcon.file_io import sanitize_variable
        self.results = {}
        for key in list(ds.keys()) + list(ds.coords):
            val = sanitize_variable(ds[key], str(key))
            units = getattr(val, "units", None)
            v = val.values.tolist()
            self.results[str(key)] = f"{v} {units}" if units is not None else v

        output_file = folder / 'output.lengyel.results.yml'
        with open(output_file, "w") as f:
            f.write(yaml.dump(self.results))

        print("Extended lengyel forward (clean) model ran successfully.")

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
            fig, ax = plt.subplots()
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
            
        
        
        