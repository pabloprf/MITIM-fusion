import torch
import numpy as np
from mitim_tools.gacode_tools.utils import GACODEdefaults
from IPython import embed

def profiles_to_tglf(self, rho, TGLFsettings=5):

    # <> Function to interpolate a curve <> 
    from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function

    def interpolator(y):
        return interpolation_function(rho, self.profiles['rho(-)'],y).item()

    TGLFinput, TGLFoptions, label = GACODEdefaults.addTGLFcontrol(TGLFsettings)

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Controls come from options
    # ---------------------------------------------------------------------------------------------------------------------------------------
    controls = TGLFoptions

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Species come from profiles
    # ---------------------------------------------------------------------------------------------------------------------------------------

    mass_e = 0.000272445 * self.derived["mi_ref"]

    species = {
        1: {
            'ZS': -1.0,
            'MASS': mass_e/self.derived["mi_ref"],
            'RLNS': interpolator(self.derived['aLne']),
            'RLTS': interpolator(self.derived['aLTe']),
            'TAUS': 1.0,
            'AS': 1.0,
            'VPAR': interpolator(self.derived['vpar']),
            'VPAR_SHEAR': interpolator(self.derived['vpar_shear']),
            'VNS_SHEAR': 0.0,
            'VTS_SHEAR': 0.0},
    }

    for i in range(len(self.Species)):
        species[i+2] = {
            'ZS': self.Species[i]['Z'],
            'MASS': self.Species[i]['A']/self.derived["mi_ref"],
            'RLNS': interpolator(self.derived['aLni'][:,i]),
            'RLTS': interpolator(self.derived['aLTi'][:,0] if self.Species[i]['S'] == 'therm' else self.derived["aLTi"][:,i]),
            'TAUS': interpolator(self.derived['tite'] if self.Species[i]['S'] == 'therm' else self.derived["tite_all"][:,i]),
            'AS': interpolator(self.derived['fi'][:,i]),
            'VPAR': interpolator(self.derived['vpar']),
            'VPAR_SHEAR': interpolator(self.derived['vpar_shear']),
            'VNS_SHEAR': 0.0,
            'VTS_SHEAR': 0.0
            }

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Plasma comes from profiles
    # ---------------------------------------------------------------------------------------------------------------------------------------

    plasma = {
        'NS': len(species)+1,
        'SIGN_BT': -1.0,
        'SIGN_IT': -1.0,
        'VEXB': 0.0,
        'VEXB_SHEAR': interpolator(self.derived['vexb_shear']),
        'BETAE': interpolator(self.derived['betae']),
        'XNUE': interpolator(self.derived['xnue']),
        'ZEFF': interpolator(self.derived['Zeff']),
        'DEBYE': interpolator(self.derived['debye']),
        }

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Geometry comes from profiles
    # ---------------------------------------------------------------------------------------------------------------------------------------

    parameters = {
        'RMIN_LOC':     self.derived['roa'],
        'RMAJ_LOC':     self.derived['Rmajoa'],
        'ZMAJ_LOC':     self.derived["Zmagoa"],
        'DRMINDX_LOC':  self.derived['drmin/dr'],
        'DRMAJDX_LOC':  self.derived['dRmaj/dr'],
        'DZMAJDX_LOC':  self.derived['dZmaj/dr'],
        'Q_LOC':        self.profiles["q(-)"],
        'KAPPA_LOC':    self.profiles["kappa(-)"],
        'S_KAPPA_LOC':  self.derived['s_kappa'],
        'DELTA_LOC':    self.profiles["delta(-)"],
        'S_DELTA_LOC':  self.derived['s_delta'],
        'ZETA_LOC':     self.profiles["zeta(-)"],
        'S_ZETA_LOC':   self.derived['s_zeta'],
        'P_PRIME_LOC':  self.derived['pprime'],
        'Q_PRIME_LOC':  self.derived['s_q'],
    }

    geom = {}
    for k in parameters:
        par = torch.nan_to_num(torch.from_numpy(parameters[k]) if type(parameters[k]) is np.ndarray else parameters[k], nan=0.0, posinf=1E10, neginf=-1E10)
        geom[k] = interpolator(par)

    geom['BETA_LOC'] = 0.0
    geom['KX0_LOC'] = 0.0

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Merging
    # ---------------------------------------------------------------------------------------------------------------------------------------

    input_dict = {**controls, **plasma, **geom}

    for i in range(len(species)):
        for k in species[i+1]:
            input_dict[f'{k}_{i+1}'] = species[i+1][k]

    return input_dict
