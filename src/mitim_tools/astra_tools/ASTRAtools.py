import os
from mitim_tools.misc_tools import IOtools,CONFIGread,FARMINGtools
from mitim_tools.astra_tools import ASTRA_CDFtools
from IPython import embed

class ASTRA():

    def __init__(self):

        pass

    def prep(self,folder,folder_repo = '$MITIM_PATH/templates/ASTRA8_REPO/'): 

        self.folder = IOtools.expandPath(folder)
        self.folder_repo = IOtools.expandPath(folder_repo)

        self.folder_astra_reduced = 'execution/'

        self.folder_astra = f'{self.folder}/{self.folder_astra_reduced}/'

        # Create folder
        IOtools.askNewFolder(self.folder)
        IOtools.askNewFolder(self.folder_astra)

        # Move files
        os.system(f'cp -r {self.folder_repo}/* {self.folder_astra}/')

        # Define basic controls
        self.equfile = 'fluxes'
        self.expfile = 'aug34954'

    def run(self,
            t_ini,
            t_end,name='run1',
            slurm_options = {
                'time': 10,
                'cpus': 16}):

        self.t_ini = t_ini
        self.t_end = t_end

        # Where to run 
        machineSettings = CONFIGread.machineSettings(
                code="astra",
                nameScratch=f"astra_tmp_{name}/",
            )
        
        self.folderExecution = machineSettings["folderWork"]

        # What to run 

        self.command_to_run_astra = f'''
cd {self.folderExecution}
./install.sh
exe/as_exe -m {self.equfile} -v {self.expfile} -s {self.t_ini} -e {self.t_end} -dev aug
'''

        # ---------------------------------------------
        # Execute
        # ---------------------------------------------

        self.output_folder_reduced = f'{self.folder_astra_reduced}/.res/ncdf/'
        self.output_folder = f'{self.folder_astra}/{self.output_folder_reduced}/'

        FARMINGtools.SLURMcomplete(
            self.command_to_run_astra,
            self.folder_astra,
            [],
            [],
            slurm_options['time'],
            1,
            name,
            machineSettings,
            cpuspertask=slurm_options['cpus'],
            inputFolders=[self.folder_astra],
            outputFolders=[self.output_folder_reduced],
        )

    def read(self):

        self.cdf_file = f'{self.output_folder}/'

        self.cdf = ASTRA_CDFtools.CDFreactor(self.cdf_file)

    def plot(self):

        pass
