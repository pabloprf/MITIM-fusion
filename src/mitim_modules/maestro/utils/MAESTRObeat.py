import os
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.IOtools import printMsg as print

from IPython import embed

# --------------------------------------------------------------------------------------------
# Generic beat class with required methods
# --------------------------------------------------------------------------------------------

class beat:

    def __init__(self, maestro_instance, beat_name = 'generic'):

        self.maestro_instance = maestro_instance
        self.folder_beat = f'{self.maestro_instance.folder_beats}/Beat_{self.maestro_instance.counter}/'

        # Where to run it
        self.name = beat_name
        self.folder = f'{self.folder_beat}/run_{self.name}/'
        os.makedirs(self.folder, exist_ok=True)

        # Where to save the results
        self.folder_output = f'{self.folder_beat}/beat_results/'
        os.makedirs(self.folder_output, exist_ok=True)

    def check(self, restart = False, folder_search = None, suffix = ''):
        '''
        Check if output file already exists so that I don't need to run this beat again
        '''

        if folder_search is None:
            folder_search = self.folder_output

        output_file = None
        if not restart:
            output_file = IOtools.findFileByExtension(folder_search, suffix, agnostic_to_case=True, provide_full_path=True)

            if output_file is not None:
                print('\t\t- Output file already exists, not running beat', typeMsg = 'i')

        else:
            print('\t\t- Forced restarting of beat', typeMsg = 'i')

        return output_file is not None

    def define_initializer(self, *args, **kwargs):
        pass

    def initialize(self, *args, **kwargs):
        pass

    def prepare(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        pass

    def finalize(self, *args, **kwargs):
        pass

    def finalize_maestro(self, *args, **kwargs):
        pass

    def plot(self, *args, **kwargs):
        return ''

    def inform_save(self, *args, **kwargs):
        pass

    def _inform(self, *args, **kwargs):
        pass

# --------------------------------------------------------------------------------------------
# Generic initializer class with required methods
# --------------------------------------------------------------------------------------------

class beat_initializer:
    
    def __init__(self, beat_instance, label = ''):
            
        self.beat_instance = beat_instance
        self.folder = f'{self.beat_instance.folder_beat}/initializer_{label}/'

        if len(label) > 0:
            os.makedirs(self.folder, exist_ok=True)

    def __call__(self, *args, **kwargs):
        pass

