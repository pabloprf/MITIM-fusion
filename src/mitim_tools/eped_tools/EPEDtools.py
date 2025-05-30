import os
import re
import copy
import subprocess
import matplotlib.pyplot as plt
import f90nml
from pathlib import Path
from mitim_tools.misc_tools import FARMINGtools, GRAPHICStools, IOtools
import numpy as np
import pandas as pd
import xarray as xr
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class EPED:
    def __init__(
            self,
            folder
            ):
        
        self.folder = Path(folder)

        self.folder.mkdir(parents=True, exist_ok=True)

        self.results = {}

    def run(
            self,
            subfolder = 'run1',
            input_params = None,    # {'ip': 12.0, 'bt': 12.16, 'r': 1.85, 'a': 0.57, 'kappa': 1.9, 'delta': 0.5, 'neped': 30.0, 'betan': 1.0, 'zeffped': 1.5, 'nesep': 10.0, 'tesep': 100.0},
            scan_param = None,      # {'variable': 'neped', 'values': [10.0, 20.0, 30.0]}
            keep_nsep_ratio = None, # Ratio of neped to nesep
            nproc_per_run = 64,
            minutes_slurm = 30,
            cold_start = False,
            ):

        # ------------------------------------
        # Prepare job
        # ------------------------------------

        # Prepare folder structure
        self.folder_run = self.folder / subfolder

        # Prepare scan parameters
        scan_param_variable = scan_param['variable'] if scan_param is not None else None
        scan_param_values = scan_param['values'] if scan_param is not None else [None]

        # Prepare job array setup
        job_array = ''
        for i in range(len(scan_param_values)):
            job_array += f'{i+1}' if i == 0 else f',{i+1}'

        # Initialize Job
        self.eped_job = FARMINGtools.mitim_job(self.folder_run)

        self.eped_job.define_machine(
            "eped",
            "mitim_eped",
            slurm_settings={
                'name': 'mitim_eped',
                'minutes': minutes_slurm,
                'ntasks': nproc_per_run,
                'job_array': job_array
            }
        )

        # ------------------------------------
        # Prepare each individual case
        # ------------------------------------

        folder_cases, output_files, shellPreCommands = [], [], []
        for i,value in enumerate(scan_param_values):

            # Folder structure
            subfolder = f'run{i+1}'
            folder_case = self.folder_run / subfolder
            
            # Prepare input parameters
            if scan_param_variable is not None:
                input_params_new = input_params.copy() if input_params is not None else {}
                input_params_new[scan_param_variable] = value
            else:
                input_params_new = input_params

            if keep_nsep_ratio is not None:
                print(f'\t> Setting nesep to {keep_nsep_ratio} * neped')
                input_params_new['nesep'] = keep_nsep_ratio * input_params_new['neped']

            # *******************************
            # Check if the case should be run
            run_case = True
            force_res = False
            if (self.folder_run / f'output_{subfolder}.nc').exists():
                if cold_start:
                    res = print(f'\t> Run {subfolder} already exists but cold_start is set to True. Running from scratch.', typeMsg='i' if force_res else 'q')
                    if res:
                        IOtools.shutil_rmtree(folder_case)
                        (self.folder_run / f'output_{subfolder}.nc').unlink(missing_ok=True)
                        force_res = True
                    else:
                        run_case = False
                else:
                    print(f'\t> Run {subfolder} already exists and cold_start is set to False. Skipping run.', typeMsg='i')
                    run_case = False

            if not run_case:
                continue
            # *******************************

            # Set up folder
            folder_case.mkdir(parents=True, exist_ok=True)

            # Preparation of the run folder by copying the template files
            eped_input_file = 'eped.input.1'
            required_files_folder = '$EPED_SOURCE_PATH/template/engaging/eped_run_template'
            shellPreCommands.append(f'cp {required_files_folder}/* {self.eped_job.folderExecution}/{subfolder}/. && mv {self.eped_job.folderExecution}/{subfolder}/{eped_input_file} {self.eped_job.folderExecution}/{subfolder}/eped.input')

            # Write input file to EPED, determining the expected output file
            output_file = self._prep_input_file(folder_case,input_params=input_params_new,eped_input_file=eped_input_file)

            output_files.append(output_file.as_posix())
            folder_cases.append(folder_case)

        # If no cases to run, exit
        if len(folder_cases) == 0:
            return

        # -------------------------------------
        # Execute
        # -------------------------------------

        # Command to execute by each job in the array
        EPEDcommand  = f'cd {self.eped_job.folderExecution}/run"$SLURM_ARRAY_TASK_ID" && export NPROC_EPED={nproc_per_run} && ips.py --config=eped.config --platform=psfc_cluster.conf'

        # Prepare the job script
        self.eped_job.prep(EPEDcommand,input_folders=folder_cases,output_files=copy.deepcopy(output_files),shellPreCommands=shellPreCommands)

        # Run the job
        self.eped_job.run() #removeScratchFolders=False)

        # -------------------------------------
        # Postprocessing
        # -------------------------------------

        # Remove potential output files from previous runs
        output_files_old = sorted(list(self.folder_run.glob("*.nc")))
        for output_file in output_files_old:
            output_file.unlink()

        # Rename output files
        for i in range(len(output_files)):
            os.system(f'mv {self.folder_run / output_files[i]} {self.folder_run / f"output_run{i+1}.nc"}')

    def _prep_input_file(
            self,
            folder_case,
            input_params = None,
            eped_input_file = 'eped.input.1', # Do not call it directly 'eped.input' as it may be overwritten by the job script template copying commands
            ):
        
        shot = 0
        timeid = 0

        # Update with fixed parameters
        input_params.update(
            {'num_scan': 1,
             'shot': shot,
             'timeid': timeid,
             'runid': 0,
             'm': 2,
             'z': 1,
             'mi': 20,
             'zi': 10,
             'tewid': 0.03,
             'ptotwid': 0.03,
             'teped': -1,
             'ptotped': -1,
            }
        )

        eped_input = {'eped_input': input_params}
        nml = f90nml.Namelist(eped_input)
        
        # Write the input file
        f90nml.write(nml, folder_case / eped_input_file, force=True)

        # What's the expected output file?
        output_file = folder_case.relative_to(self.folder_run) / 'eped' / 'SUMMARY' / f'e{shot:06d}.{timeid:05d}'

        return output_file

    def read(
            self,
            subfolder = 'run1',
            label = None,
            ):

        self.results[label if label is not None else subfolder] = {}
        
        output_files = sorted(list((self.folder / subfolder).glob("*.nc")))

        for output_file in output_files:


            data = xr.open_dataset(f'{output_file.resolve()}', engine='netcdf4')
            data = postprocess_eped(data, 'G', 0.03)

            sublabel = output_file.name.split('_')[-1].split('.')[0]

            self.results[label if label is not None else subfolder][sublabel] = data

    def plot(
            self,
            labels = ['run1'],
            ):

        plt.ion(); fig, axs = plt.subplots(2, 1, figsize=(10, 6))

        colors = GRAPHICStools.listColors()

        for i,name in enumerate(labels):

            data = self.results[name]

            neped, ptop, wtop = [], [], []
            for sublabel in data:
                neped.append(float(data[sublabel]['neped']))
                if 'ptop' in data[sublabel].data_vars:
                    ptop.append(float(data[sublabel]['ptop']))
                    wtop.append(float(data[sublabel]['wptop']))
                else:
                    ptop.append(0.0)
                    wtop.append(0.0)
            
            axs[0].plot(neped,ptop,'-s', c = colors[i], ms = 10)
            axs[1].plot(neped,wtop,'-s', c = colors[i], ms = 10)

        ax = axs[0]
        ax.set_xlabel('neped ($10^{19}m^{-3}$)')
        ax.set_ylabel('ptop (kPa)')
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[1]
        ax.set_xlabel('neped ($10^{19}m^{-3}$)')
        ax.set_ylabel('wptop (psi_pol)')
        ax.set_ylim(bottom=0)
        GRAPHICStools.addDenseAxis(ax)

        plt.tight_layout()

# ************************************************************************************************************
# ************************************************************************************************************

def convert_to_dimensional(df):
    #ee = 1.60217663e-19
    mu0 = 1.25663706127e-6
    df['a'] = df['r'] / df['epsilon']
    df['ip'] = 1.0e-6 * (2.0 * np.pi * np.square(df['a']) * df['kappa'] * df['bt']) / (df['qstar'] * df['r'] * mu0)
    df['neped'] = 10.0 * df['fgped'] * df['ip'] / (np.pi * np.square(df['a']))
    df['nesep'] = 0.25 * df['neped']
    #df['teped'] = 2500 * df['bt'] * df['ip'] * df['betan'] / (3 * df['a'] * 1.5 * df['neped'])
    #df['teped'] = df['teped'].clip(upper=8000)
    df['teped'] = df['r'] * 0.0 - 1.0
    return df


def convert_to_dimensionless(df):
    mu0 = 1.25663706127e-6
    df['epsilon'] = df['r'] / df['a']
    df['fgped'] = df['neped'] * np.pi * np.square(df['a']) / (10.0 * df['ip'])
    df['qstar'] = (2.0 * np.pi * np.square(df['a']) * df['kappa'] * df['bt']) / (1.0e6 * df['ip'] * df['r'] * mu0)
    df['nesep'] = 0.25 * df['neped']
    df['teped'] = df['r'] * 0.0 - 1.0
    return df


def setup_eped(output_path, inputs_list, template_path):

    output_path = Path(output_path).resolve()  # Ensure absolute path
    output_path.mkdir(parents=True, exist_ok=True)

    subprocess.run(['cp', str(template_path.resolve() / 'exec_eped.sh'), str(output_path)])
    subprocess.run(['cp', str(template_path.resolve() / 'submit_eped_array_psfc.batch'), str(output_path)])
    subprocess.run(['cp', str(template_path.resolve() / 'postprocessing.py'), str(output_path)])
    rpaths = []

    for run_num, inputs in enumerate(inputs_list):
        run_id = f'run{run_num + 1:03d}'
        rpath = output_path / run_id  # Construct the absolute path for the run directory
        subprocess.run(['cp', '-r', str(template_path.resolve() / 'eped_run_template'), str(rpath)])

        # Edit input file
        input_file = rpath / 'eped.input'
        contents = f90nml.read(str(input_file))
        for param, value in inputs.items():
            contents['eped_input'][param] = value
        contents.write(str(input_file), force=True)

        rpaths.append(rpath)

    #logger.info(f'{len(inputs_list)} Runs created at {output_path}')

    return rpaths


def setup_array_batch(launch_path, rpaths, maxqueue=5):

    # Convert to Path object and ensure absolute path
    launch_path = Path(launch_path).resolve()
    
    s = ''
    for path in rpaths:
        if s:
            s += '\n'
        s += f'"./exec_eped.sh {path.resolve()}"'
    
    # Use proper Path object for file operations
    batch_file = launch_path / 'submit_eped_array_psfc.batch'
    with batch_file.open('r') as f:
        content = f.read()
        new_content = re.sub('<numruns>', str(len(rpaths) - 1), content)
        new_content = re.sub('<maxqueue>', str(maxqueue), new_content)
        new_content = re.sub('<rundir>', str(launch_path), new_content)  # Convert to string for substitution
        new_content = re.sub('#<launchdirs>', s, new_content)
    with batch_file.open('w') as f:
        f.write(new_content)

    #logger.info('Batch array created')

    return batch_file


def postprocess_eped(data, diamagnetic_stab_rule, stability_threshold):

    coords = {k: data[k].values for k in ['dim_height', 'dim_widths', 'dim_nmodes', 'dim_rho', 'dim_three', 'dim_one']}
    data = data.assign_coords(coords)

    x = data['eq_betanped'].data
    index = np.where(x < 0)[0]
    if diamagnetic_stab_rule == 'G':
        y = data['gamma'].data.copy()
    elif diamagnetic_stab_rule in ['GH', 'HG']:
        y = data['gamma_PB'].data.copy()
        y *= data['gamma'].data.copy()
    elif diamagnetic_stab_rule == 'H':
        y = data['gamma_PB'].data.copy()
    else:
        y = data['gamma'].data.copy()
    y[index, :] = np.nan

    data['stability'] = (('dim_height', 'dim_nmodes'), y)
    y0 = np.nanmax(y, 1)
    y0 = np.where(y0 == None, 0, y0)
    indices = np.where(y0 > stability_threshold)[0]
    if len(indices):
        step = indices[0]
    else:
        step = -1

    dims = ('dim_one')
    data['stability_rule'] = (dims, [diamagnetic_stab_rule])
    data['stability_threshold'] = (dims, np.array([stability_threshold]))
    if step > 0:
        data['stability_index'] = (dims, np.array([step]))
        data['pped'] = (dims, np.array([data['eq_pped'].data[step] * 1.0e3]))
        data['ptop'] = (dims, np.array([data['eq_ptop'].data[step] * 1.0e3]))
        data['tped'] = (dims, np.array([data['eq_tped'].data[step]]))
        data['ttop'] = (dims, np.array([data['eq_ttop'].data[step]]))
        data['wpped'] = (dims, np.array([data['eq_wped_psi'].data[step]]))
        data['wptop'] = (dims, np.array([data['eq_wped_psi'].data[step] * 1.5]))
        data['wrped'] = (dims, np.array([data['eq_wped_rho'].data[step]]))
        if np.any(data['tesep'].data < 0):
            data['tesep'] = (dims, np.array([75.0]))
            data['nesep'] = 0.25 * data['neped']

    return data

def read_eped_file(ipaths):
    invars = ['ip', 'bt', 'r' , 'a', 'kappa', 'delta', 'neped', 'betan', 'zeffped', 'nesep', 'tesep']
    data_arrays = []
    for ipath in ipaths:
        dummy_coords = {
            'dim_height': np.empty((0, ), dtype=int),
            'dim_nmodes': np.empty((0, ), dtype=int),
            'dim_widths': np.empty((0, ), dtype=int),
            'dim_rho': np.empty((0, ), dtype=int),
            'dim_three': np.empty((0, ), dtype=int),
            'dim_one': np.arange(1),
        }
        set_inputs = f90nml.read(str(ipath.parent.parent.parent / 'eped.input'))
        dummy_vars = {k: (['dim_one'], [v]) for k, v in set_inputs['eped_input'].items() if k in invars}
        data = xr.Dataset(coords=dummy_coords, data_vars=dummy_vars)
        if ipath.is_file():
            data = xr.open_dataset(f'{ipath.resolve()}', engine='netcdf4')
            data = postprocess_eped(data, 'G', 0.03)
        data_arrays.append(data.expand_dims({'filename': [ipath.parent.parent.parent.name]}))

    dataset = xr.merge(data_arrays, join='outer', fill_value=np.nan).sortby('filename')
    return dataset

def launch_eped_slurm(input_params, scan_params, nscan, output_path, template_path, run_tag, wait=False): 
    ivars = ['ip', 'bt', 'r', 'a', 'kappa', 'delta', 'neped', 'betan', 'zeffped', 'nesep', 'tesep', 'teped']
    input_params.update(scan_params)
    data = {}
    for var, val in input_params.items():
        if isinstance(val, (tuple, list, np.ndarray)) and len(val) > 1:
            data[var] = np.linspace(val[0], val[1], nscan)
        else:
            data[var] = np.zeros((nscan, )) + val
    #if scan_var == 'qstar': # Use for ip scan
    #    data['fgped'] = (0.5 / 3.5) * data['qstar']
    inp = pd.DataFrame(data=data, index=pd.RangeIndex(nscan))
    #inp = convert_to_dimensional(inp)
    inputs  = [{ivar: inp[ivar].iloc[i] for ivar in ivars} for i in range(len(inp))]
    run_paths = setup_eped(output_path, inputs, template_path)
    spath = setup_array_batch(output_path, run_paths)
    inp.to_hdf(output_path / f'{output_path.name}.h5', key='/data')
    command = ['sbatch']
    if wait:
        command.append('--wait')
    command.append(f'{spath.resolve()}')
    subprocess.run(command)

    return run_paths

def main():

    rootdir = Path(os.environ.get('PIXI_PROJECT_ROOT', './'))
    run_tag = 'mitim_eped_test'
    base_input_path = Path('./') / 'eped.input'
    scan_params = {
    #    'tesep': [50.0, 300.0],
    }
    nscan = 1
    output_path = Path('./') / f'eped_{run_tag}'
    template_path = rootdir / 'ips-eped-master' / 'template' / 'engaging'
    wait = False

    input_params = f90nml.read(str(base_input_path)).todict().get('eped_input', {})

    launch_eped_slurm(input_params, scan_params, nscan, output_path, template_path, run_tag, wait=wait)


if __name__ == '__main__':
    main()
