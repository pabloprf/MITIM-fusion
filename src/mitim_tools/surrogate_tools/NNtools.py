import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian
from mitim_tools.misc_tools import IOtools, GRAPHICStools
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

        if type == 'pt':
            print('Initializing Pytorch Neural Network')

            self.load = self._load_pt
            self.__call__ = self._evaluate_pt

        self.force_within_range = force_within_range

    def _load_tf(self, model_path, norm=None):

        from tensorflow.keras.models import load_model

        self.model = load_model(model_path)
        self.normalization = None

        if norm is not None:
            self._read_norm_file(norm)
        
        print(f'\t- Model file from {IOtools.clipstr(model_path,30)} loaded')

    def _read_norm_file(self, norm):

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
        
    def _evaluate_tf(self, inputs, print_msg=True):
        
        if print_msg:
            print('Evaluating NN with inputs: ', inputs)

        if self.normalization is not None:
            return self.model.predict(np.expand_dims(inputs, axis=0))[0]*self.normalization
        else:
            return self.model.predict(np.expand_dims(inputs, axis=0))[0]

    def _load_pt(self, model_path, norm=None):

        import torch
        self.model = torch.load(model_path, weights_only=False).eval()

        self.normalization = None

        if norm is not None:
            self._read_norm_file(norm)
            self.normalization = 1.0

        print(f'\t- Model file from {IOtools.clipstr(model_path,30)} loaded')

    def _evaluate_pt(self, inputs, print_msg=True):

        if print_msg:
            print('Evaluating NN with inputs: ', inputs)

        output = self.model(inputs.unsqueeze(0)).squeeze()

        if output.dim() == 0:
            output = output.unsqueeze(0)

        if self.normalization is not None:
            return output*self.normalization
        else:
            return output

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


        # 0) Calculate any extra derived quantities
        nesep = neped * nesep_ratio

        # 1) Collect all possible arguments in a dictionary
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

        # 2) Construct the input list/array in the exact order the parent expects (as listed in self.inputs).
        if self.inputs is not None:
            inputs = [all_args[key] for key in self.inputs]
        else:
            inputs = list(all_args.values())

        # 3) Potentially check the ranges
        if self.ranges is not None and self.force_within_range is not None:
            for i, inp in enumerate(self.inputs):
                if inputs[i] < self.ranges[inp][0] or inputs[i] > self.ranges[inp][1]:
                    print(f'\t- Input {inp} out of range: {inputs[i]} not in {self.ranges[inp]}', typeMsg='w' if not self.force_within_range else 'q')

        # 5) Call the parent method to run the actual inference 
        return self.__call__(inputs)


'''
---------------------------------------------------------------------------------------------
Class for the TGLF NN
---------------------------------------------------------------------------------------------
'''

class tglf_nn(mitim_nn):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, inputs):
        '''
        inputs could be [batch, dim]
        '''

        # Potentially check the ranges
        if self.ranges is not None and self.force_within_range is not None:
            for k in range(inputs.shape[0]):
                input0 = inputs[k,:]
                for i, inp in enumerate(self.inputs):
                    if input0[i] < self.ranges[inp][0] or input0[i] > self.ranges[inp][1]:
                        print(f'\t- Input {inp} out of range for batch position {k}: {inputs[k,i]} not in {self.ranges[inp]}', typeMsg='w' if not self.force_within_range else 'q')

        # 5) Call the parent method to run the actual inference 
        return self.__call__(inputs, print_msg=False)


    def scan_parameter(self, inputs, param, values=np.linspace(0.5, 2.0, 100), values_are_relative=True):

        if self.inputs is not None:
            print(f'- Scanning parameter {param}')
        else:
            raise ValueError('No input information found in normalization file')
    
        param_i = np.where(self.inputs == param)[0][0]

        inputs_c = inputs.clone()

        x = []
        results, resultsJ = [], []
        for val in values:
            if values_are_relative:
                inputs[:,param_i] = val*inputs_c[:,param_i]
            else:
                inputs[:,param_i] = val
            x.append(inputs[:,param_i].clone())
        
            q = self.__call__(inputs, print_msg=False)

            results.append(q)
            qJ = []
            for j in range(inputs.shape[0]):
                qJ0 = jacobian(self,inputs[j,:].unsqueeze(0)).squeeze()[param_i].item()
                qJ.append(qJ0)
            resultsJ.append(qJ)
        
        x = torch.stack(x, dim=1)    
        results = torch.stack(results, dim=1)
        resultsJ = torch.Tensor(resultsJ).transpose(0,1)

        plt.ion(); fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(12,6))
        for candidate in range(results.shape[0]):
            axs[0].plot(x[candidate,:], results[candidate,:].detach().numpy(), 'o-', lw=0.5, ms=1, label=f'Candidate {candidate}' if results.shape[0] > 1 else '')
            axs[1].plot(x[candidate,:], resultsJ[candidate,:].detach().numpy(), 'o-', lw=0.5, ms=1, label=f'Candidate {candidate}' if results.shape[0] > 1 else '')

        ax = axs[0]
        ax.set_ylabel('Q')
        GRAPHICStools.addLegendApart(ax, ratio=0.9, size=8, withleg=results.shape[1] > 1)
        GRAPHICStools.addDenseAxis(ax)

        ax = axs[1]
        ax.set_xlabel(param)
        ax.set_ylabel(f'dQ/d{param}')
        GRAPHICStools.addDenseAxis(ax)
        GRAPHICStools.addLegendApart(ax, ratio=0.9, size=8, withleg=False)
