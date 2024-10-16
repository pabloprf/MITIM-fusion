from mitim_tools.misc_tools import IOtools
import numpy as np

# ---------------------------------------------------------------------------------------------
# Main class for Neural Network tools
# ---------------------------------------------------------------------------------------------

class mitim_nn:

    def __init__(self, type = 'tf'):
        
        if type == 'tf':
            print('Initializing Tensorflow Neural Network')

            self.load = self._load_tf
            self.__call__ = self._evaluate_tf

    def _load_tf(self, model_path, norm=None):

        from tensorflow.keras.models import load_model

        self.model = load_model(model_path)
        self.normalization = None

        if norm is not None:
            self.normalization = np.loadtxt(norm)
            print(f'\t- Normalization file from {IOtools.clipstr(norm,30)} loaded')
            print("Norm:", self.normalization)
        
        print(f'\t- Weights file from {IOtools.clipstr(model_path,30)} loaded')
        
    def _evaluate_tf(self, inputs):
        
        print('Evaluating NN with inputs: ', inputs)

        if self.normalization is not None:
            return self.model.predict(inputs)[0]*self.normalization
        else:
            return self.model.predict(inputs)[0]

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
                R,             #m
                a,             #m
                kappa995, 
                delta995, 
                neped,         #10e19m^-3
                betan, 
                zeff, 
                tesep=75,    #eV
                nesep_ratio=0.3
                 ):

        nesep = neped*nesep_ratio
        inputs = np.array([Ip, Bt, R, a, kappa995, delta995, neped, betan, zeff, tesep, nesep])

        return self.__call__(inputs)
