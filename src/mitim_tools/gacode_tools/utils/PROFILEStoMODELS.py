import torch
import numpy as np
from mitim_tools.gacode_tools.utils import GACODEdefaults
from IPython import embed

def profiles_to_tglf(self, rho, TGLFsettings=5):

    # <> Function to interpolate a curve <> 
    from mitim_tools.misc_tools.MATHtools import extrapolateCubicSpline as interpolation_function
    from mitim_tools.gacode_tools.PROFILEStools import grad as deriv_function

    TGLFinput, TGLFoptions, label = GACODEdefaults.addTGLFcontrol(TGLFsettings)

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Controls come from options
    # ---------------------------------------------------------------------------------------------------------------------------------------
    controls = TGLFoptions

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Species come from profiles
    # ---------------------------------------------------------------------------------------------------------------------------------------

    mass_ref = 2.01355 #self.Species[0]['A']
    mass_e = 0.000272445

    species = {
        1: {
            'ZS': -1.0,
            'MASS': mass_e,
            'RLNS': interpolation_function(rho, self.profiles['rho(-)'],self.derived['aLne']).item(),
            'RLTS': interpolation_function(rho, self.profiles['rho(-)'],self.derived['aLTe']).item(),
            'TAUS': 1.0,
            'AS': 1.0,
            'VPAR': 0.0,
            'VPAR_SHEAR': 0.0,
            'VNS_SHEAR': 0.0,
            'VTS_SHEAR': 0.0},
    }

    for i in range(len(self.Species)):
         species[i+2] = {
            'ZS': self.Species[i]['Z'],
            'MASS': self.Species[i]['A']/mass_ref,
            'RLNS': interpolation_function(rho, self.profiles['rho(-)'],self.derived['aLni'][:,i]).item(),
            'RLTS': interpolation_function(rho, self.profiles['rho(-)'],self.derived['aLTi'][:,i]).item(),
            'TAUS': interpolation_function(rho, self.profiles['rho(-)'],self.profiles['ti(keV)'][:,i]/self.profiles['te(keV)']).item(),
            'AS': interpolation_function(rho, self.profiles['rho(-)'],self.derived['fi'][:,i]).item(),
            'VPAR': 0.0,
            'VPAR_SHEAR': 0.0,
            'VNS_SHEAR': 0.0,
            'VTS_SHEAR': 0.0}

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Plasma comes from profiles
    # ---------------------------------------------------------------------------------------------------------------------------------------

    plasma = {
        'NS': len(species)+1,
        'SIGN_BT': -1.0,
        'SIGN_IT': -1.0,
        'VEXB': 0.0,
        'VEXB_SHEAR': 0.0,
        'BETAE': interpolation_function(rho, self.profiles['rho(-)'],self.derived['betae']).item(),
        'XNUE': interpolation_function(rho, self.profiles['rho(-)'],self.derived['xnue']).item(),
        'ZEFF': interpolation_function(rho, self.profiles['rho(-)'],self.derived['Zeff']).item(),
        'DEBYE': interpolation_function(rho, self.profiles['rho(-)'],self.derived['debye']).item(),
        }

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # Geometry comes from profiles
    # ---------------------------------------------------------------------------------------------------------------------------------------

    s = torch.from_numpy(self.profiles["rmin(m)"]/ self.profiles["q(-)"])*deriv_function(self.profiles["rmin(m)"], self.profiles["q(-)"])

    parameters = {
        'RMIN_LOC':     self.derived['roa'],
        'RMAJ_LOC':     self.derived['Rmajoa'],
        'ZMAJ_LOC':     self.derived["Zmagoa"],
        'DRMINDX_LOC': deriv_function(self.profiles["rmin(m)"], self.profiles["rmin(m)"]), 
        'DRMAJDX_LOC': deriv_function(self.profiles["rmin(m)"], self.profiles["rmaj(m)"]),
        'DZMAJDX_LOC': deriv_function(self.profiles["rmin(m)"], self.profiles["zmag(m)"]),
        'Q_LOC':        self.profiles["q(-)"],
        'KAPPA_LOC':    self.profiles["kappa(-)"],
        'S_KAPPA_LOC':  torch.from_numpy(self.profiles["rmin(m)"]/self.profiles["kappa(-)"]) * deriv_function(self.profiles["rmin(m)"], self.profiles["kappa(-)"]),
        'DELTA_LOC':    self.profiles["delta(-)"],
        'S_DELTA_LOC':  torch.from_numpy(self.profiles["rmin(m)"]) * deriv_function(self.profiles["rmin(m)"], self.profiles["delta(-)"]),
        'ZETA_LOC':     self.profiles["zeta(-)"],
        'S_ZETA_LOC':   torch.from_numpy(self.profiles["rmin(m)"]) * deriv_function(self.profiles["rmin(m)"], self.profiles["zeta(-)"]),
        'P_PRIME_LOC':  self.derived['pprime'] * 1E-7,
        'Q_PRIME_LOC':  torch.from_numpy(self.profiles["q(-)"]/self.derived['roa'])**2*s,
    }

    geom = {}
    for k in parameters:
        par = torch.nan_to_num(torch.from_numpy(parameters[k]) if type(parameters[k]) is np.ndarray else parameters[k], nan=0.0, posinf=1E10, neginf=-1E10)
        geom[k] = interpolation_function(rho, self.profiles['rho(-)'],par).item()

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
