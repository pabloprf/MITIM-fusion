if __name__=="__main__":
    mfe_im_path = '/Users/hallj/MFE-IM'

    nn = NNtools.eped_nn(type='tf')

    nn_location = IOtools.expandPath('$MFEIM_PATH/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-MODEL-ARC.h5')
    norm_location = IOtools.expandPath('$MFEIM_PATH/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-NORMALIZATION.txt')

    nn.load(nn_location, norm=norm_location)

    create_initial_conditions(10,20, file_output_location='/Users/hallj/MITIM-fusion/src/mitim_tools/astra_tools/tmp',
                              geometry=f'{mfe_im_path}/private_data/ARCV2B.geqdsk', eped_nn=nn, q_profile=None, plotYN=True)