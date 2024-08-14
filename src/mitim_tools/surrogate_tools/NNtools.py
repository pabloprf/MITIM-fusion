from mitim_tools.misc_tools import IOtools
# ---------------------------------------------------------------------------------------------
# Main class for Neural Network tools
# ---------------------------------------------------------------------------------------------

class mitim_nn:

    def __init__(self, type = 'tf'):
        
        if type == 'tf':
            print('Initializing Tensorflow Neural Network')

            self.load = self._load_tf
            self.__call__ = self._evaluate_tf

    def _load_tf(self, model_path):

        #from tensorflow.keras.models import load_model
        #return load_model(model_path)
        print(f'\t- Weights file from {IOtools.clipstr(model_path,30)} loaded')
        
    def _evaluate_tf(self, inputs):

        print('testing evaluation with ', inputs)


# ---------------------------------------------------------------------------------------------
# Class for the EPED NN
# ---------------------------------------------------------------------------------------------

class eped_nn(mitim_nn):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, R, a, kappa):

        inputs = [R, a, kappa]

        return self.__call__(inputs)




if __name__ == '__main__':

    nn = eped_nn(type='tf')
    nn.load('model.h5')
    p_top = nn(1.67, 0.6, 1.75)

