import os
import numpy as np
import tarfile
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import IOtools
from mitim_tools.surrogate_tools import NNtools
from mitim_tools.astra_tools import ASTRAtools

if __name__=="__main__":
    mfe_im_path = '/Users/hallj/MFE-IM'

    nn = NNtools.eped_nn(type='tf')

    nn_location = IOtools.expandPath('$MFEIM_PATH/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-MODEL-ARC.h5')
    norm_location = IOtools.expandPath('$MFEIM_PATH/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-NORMALIZATION.txt')

    nn.load(nn_location, norm=norm_location)
    ip, bt, r, a, kappa995, delta995, neped, betan, zeff, tesep, nesep_ratio = 10.95, 10.8, 4.25, 1.17, 1.68, 0.516, 18, 1.9, 1.5, 100, 1./3.
    eped_inputs = np.array([ip, bt, r, a, kappa995, delta995, neped, betan, zeff, tesep, nesep_ratio])

    ASTRAtools.create_initial_conditions(10,
                                         20, 
                                         file_output_location='/Users/hallj/MITIM-fusion/src/mitim_tools/astra_tools/tmp',
                                         geometry=f'{mfe_im_path}/private_data/ARCV2B.geqdsk', 
                                         eped_nn=nn,
                                         eped_params=eped_inputs, 
                                         q_profile=None,
                                         plotYN=True)