import os
import tarfile
from mitim_tools.misc_tools import IOtools,FARMINGtools
from mitim_tools.astra_tools import ASTRA_CDFtools
from IPython import embed

class ASTRA():

    def __init__(self):

        pass

    def prep(self,folder,file_repo = '$MITIM_PATH/templates/ASTRA8_REPO.tar.gz'): 

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
exe/as_exe -m {self.equfile} -v {self.expfile} -s {self.t_ini} -e {self.t_end} -dev aug -batch
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

        self.cdf = ASTRA_CDFtools.CDFreactor(self.cdf_file)

    def plot(self):

        pass
