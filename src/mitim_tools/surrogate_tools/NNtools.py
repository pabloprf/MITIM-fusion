import numpy as np
from mitim_tools.misc_tools import IOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

# ---------------------------------------------------------------------------------------------
# Main class for Neural Network tools
# ---------------------------------------------------------------------------------------------

class mitim_nn:

    def __init__(self, type = 'tf', force_within_range = False):
        
        if type == 'tf':
            print('Initializing Tensorflow Neural Network')

            self.load = self._load_tf
            self.__call__ = self._evaluate_tf
        
        self.force_within_range = force_within_range

    def _load_tf(self, model_path, norm=None):

        from tensorflow.keras.models import load_model

        self.model = load_model(model_path)
        self.normalization = None

        if norm is not None:
            with open(norm, 'r') as f:
                self.normalization = np.array([float(x) for x in f.readline().split()])
                try:
                    self.inputs = np.array([x for x in f.readline().split()])
                except:
                    self.inputs = None

                try:
                    self.ranges = {}
                    for inp in self.inputs:
                        self.ranges[inp] = [float(x) for x in f.readline().split()]
                except:
                    self.ranges = None

            print(f'\t- Normalization file from {IOtools.clipstr(norm,30)} loaded')
            print("Norm:", self.normalization)
            if self.inputs is not None:
                print("Expected inputs to NN:")
                print(self.inputs)
            else:
                print("No input information found in normalization file")
            if self.ranges is not None:
                print("Trained ranges:")
                print(self.ranges)
            else:
                print("No ranges found in normalization file")
        
        print(f'\t- Weights file from {IOtools.clipstr(model_path,30)} loaded')
        
    def _evaluate_tf(self, inputs):
        
        print('Evaluating NN with inputs: ', inputs)

        if self.normalization is not None:
            return self.model.predict(np.expand_dims(inputs, axis=0))[0]*self.normalization
        else:
            return self.model.predict(np.expand_dims(inputs, axis=0))[0]

'''
---------------------------------------------------------------------------------------------
Class for the EPED NN
---------------------------------------------------------------------------------------------
Example usage:

    nn = eped_nn(type='tf')

    nn_location = IOtools.expandPath('$MFEIM_PATH/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-MODEL-ARC.h5')
    norm_location = IOtools.expandPath('$MFEIM_PATH/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-NORMALIZATION.txt')

    nn.load(nn_location, norm=norm_location)
    
    ptop, wtop = nn(10.95, 10.8, 4.25, 1.17, 1.68, 0.516, 19.27, 1.9, 1.5)

    print(ptop, wtop)

'''

# ---------------------------------------------------------------------------------------------
# Class for the EPED NN
# ---------------------------------------------------------------------------------------------

class eped_nn(mitim_nn):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, 
                Ip,            # MA
                Bt,            # T
                R,             # m
                a,             # m
                kappa995, 
                delta995, 
                neped,         # 10e19m^-3
                betan, 
                zeff, 
                tesep=75,    #eV
                nesep_ratio=0.3
                 ):


        # 1) Calculate any extra derived quantities
        nesep = neped * nesep_ratio

        # 2) Collect all possible arguments in a dictionary
        all_args = {
            'Ip':         Ip,
            'Bt':         Bt,
            'R':          R,
            'a':          a,
            'kappa995':   kappa995,
            'delta995':   delta995,
            'neped':      neped,
            'betan':      betan,
            'zeff':       zeff,
            'tesep':      tesep,
            'nesep':      nesep
        }

        # 3) Construct the input list/array in the exact order the parent expects (as listed in self.inputs).
        if self.inputs is not None:
            inputs = [all_args[key] for key in self.inputs]
        else:
            inputs = list(all_args.values())

        # 4) Potentially check the ranges
        if self.ranges is not None:
            for i, inp in enumerate(self.inputs):
                if inputs[i] < self.ranges[inp][0] or inputs[i] > self.ranges[inp][1]:
                    print(f'\t- Input {inp} out of range: {inputs[i]} not in {self.ranges[inp]}', typeMsg='w' if not self.force_within_range else 'q')

        # 5) Call the parent method to run the actual inference 
        return self.__call__(inputs)
