import os
import re
import subprocess
import f90nml
from pathlib import Path
from mitim_tools.misc_tools import FARMINGtools
import numpy as np
import pandas as pd
import xarray as xr
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
            input_params = None,
            nproc = 64,
            ):

        # Set up folder
        self.folder_run = self.folder / subfolder
        self.folder_run.mkdir(parents=True, exist_ok=True)

        # ------------------------------------
        # Write input file to EPED
        # ------------------------------------

        eped_input_file = self.folder_run / 'eped.input'

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
        f90nml.write(nml, eped_input_file, force=True)

        # Initialize Job
        self.eped_job = FARMINGtools.mitim_job(self.folder_run)

        self.eped_job.define_machine(
            "eped",
            "mitim_eped",
            launchSlurm=False,
        )

        # -------------------------------------
        # Executable commands
        # -------------------------------------

        EPEDcommand = f'cp $EPED_SOURCE_PATH/template/engaging/eped_run_template/* {self.eped_job.folderExecution}/. && export NPROC_EPED={nproc} && ips.py --config=eped.config --platform=psfc_cluster.conf'

        # -------------------------------------
        # Execute
        # -------------------------------------

        output_file = f'e{shot:06d}.{timeid:05d}'

        self.eped_job.prep(
            EPEDcommand,
            input_files=[eped_input_file],
            output_files=[f'eped/SUMMARY/{output_file}'],
        )

        self.eped_job.run(removeScratchFolders=False)


        # Rename output file
        os.system(f'mv {self.folder_run / output_file} {self.folder_run / "output.nc"}')

    def read(
            self,
            subfolder = 'run1',
            name = 'run1',
            ):

        output_file = self.folder / subfolder / 'output.nc'

        data = xr.open_dataset(f'{output_file.resolve()}', engine='netcdf4')
        data = postprocess_eped(data, 'G', 0.03)

        self.results[name] = data

    def plot(
            self,
            name = 'run1',
            ):

        data = self.results[name]

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
