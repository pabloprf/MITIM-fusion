import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools, IOtools
from mitim_tools.surrogate_tools import NNtools
from scipy.optimize import minimize

# Uses EPED-NN-ARC model to determine the optimum pedestal pressure and width for a given set of inputs.

nn = NNtools.eped_nn(type='tf')

nn_location = IOtools.expandPath('$MFEIM_PATH/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-MODEL-ARC.h5')
norm_location = IOtools.expandPath('$MFEIM_PATH/private_code_mitim/NN_DATA/EPED-NN-ARC/EPED-NN-NORMALIZATION.txt')

nn.load(nn_location, norm=norm_location)

p, w = nn(10.95, 10.8, 4.25, 1.17, 1.68, 0.516, 19.27, 1.9, 1.5)
print(p, w)

params = {
    'ip' : [10.0,10.8],
    'bt' : [10, 11.5],
    'r' : [4.0, 4.5],
    'a' : [1.0, 1.3],
    'kappa995' : [1.68,1.8],
    'delta995' : [0.516, 0.534],
    'neped' : [5,35],
    'betan' : [1.9],
    'zeff' : [1.5, 2.5],
    'tesep' : [75],
    'nesep_ratio' : [0.3,0.6]
}

# constraints on problem:
# - qstar is constant: as we raise kappa/delta, ip has to drop
# - constraints on the absolute values of kappa and delta due to engineering
# for now, I will take this as 5% higher than v2b values
# - divertor upstream parameter alpha_t [Eich 2021] is constant at 0.7
# - density from 0 to 1 times greenwald density
# proposed flow:
# - R, a are fixed
# B is fixed from B*R=const
# Ip is fixed from qstar=const
# ne range is calculated from ne/nGW=0.1 to 1

x0 = np.array([value[0] for key, value in params.items()])

def apply_constraint(x):

    R, a = 4.25, 1.17
    ip = 10.95
    bt = 10.8
    kappa_ratio = 1.69/1.490
    delta_ratio = 0.534/0.395
    kappa95_0 = 1.69 / kappa_ratio # assume this ratio is essentially constant
    delta95_0 = 0.534 / delta_ratio
    
    kappa95 = x[4] / kappa_ratio # assume this ratio is essentially constant
    delta95 = x[5] / delta_ratio

    qstar = ((5 * bt * R * a**2)/( ip * R**2)) * 0.5 * (1+kappa95_0**2 * (1+2*delta95_0**2-1.2*delta95_0**3))

    # constrain the variables
    qstar_new = ((5*x[1]*x[2]*x[3]**2)/(x[0]*x[2]**2))*0.5*(1+kappa95**2 * (1+2*delta95**2-1.2*delta95**3))

    total = qstar_new - qstar

    return total

my_constraints = ({'type': 'eq', "fun": apply_constraint })

def objective(x):
    p, w = nn(*x)
    return -p

bounds = [(value[0], value[0]) if len(value) == 1 else (value[0], value[1]) for key, value in params.items()]

print(bounds)

result = minimize(objective, x0, bounds=bounds,method='COBYQA',constraints=my_constraints)

if result.success:
    print('\nOptimum: ', result.x)
    print('Pressure: ', nn(*result.x)[0])
    print('Width: ', nn(*result.x)[1])

