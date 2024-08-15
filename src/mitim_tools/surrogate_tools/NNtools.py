from mitim_tools.misc_tools import IOtools
import numpy as np
import os

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
        self.norm = None

        if norm is not None:
            self.norm = np.loadtxt('/Users/hallj/MITIM-fusion/src/mitim_tools/surrogate_tools/EPED-NN-NORMALIZATION.txt')
            print(f'\t- Normalization file from {IOtools.clipstr(norm,30)} loaded')
            print("Norm:", self.norm)
        
        print(f'\t- Weights file from {IOtools.clipstr(model_path,30)} loaded')
        
    def _evaluate_tf(self, inputs):
        
        print('testing evaluation with ', inputs)

        if self.norm is not None:
            return self.model.predict(inputs)[0]*self.norm
        else:
            return self.model.predict(inputs)[0]


# ---------------------------------------------------------------------------------------------
# Class for the EPED NN
# ---------------------------------------------------------------------------------------------

class eped_nn(mitim_nn):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, 
                 R,             #m
                 a,             #m
                 Ip,            # MA
                 Bt,            # T
                 kappa995, 
                 delta995, 
                 neped,         #10e19m^-3
                 betan, 
                 zeff, 
                 nesep_ratio=0.3, 
                 tesep=75    #eV
                 ):

        nesep = neped*nesep_ratio
        inputs = np.array([Ip, Bt, R, a, kappa995, delta995, neped, betan, zeff, tesep, nesep])

        self._evaluate_tf(inputs)

        return self.__call__(inputs)




if __name__ == '__main__':

    nn = eped_nn(type='tf')
    nn.load('/Users/hallj/MITIM-fusion/src/mitim_tools/surrogate_tools/EPED-NN-MODEL-ARC.h5', 
            norm="/Users/hallj/MITIM-fusion/src/mitim_tools/surrogate_tools/EPED-NN-NORMALIZATION.txt"
            )
    ptop, wtop = nn(4.25, 1.17, 10.95, 10.8, 1.68, 0.516, 19.27, 1.9, 1.5)
    print(ptop, wtop)
