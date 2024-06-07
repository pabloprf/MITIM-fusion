import numpy as np
import xarray as xr
from pathlib import Path
import cfspopcon
from cfspopcon.unit_handling import ureg
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.astra_tools import ASTRA_CDFtools
from mitim_tools.gs_tools import GEQtools

import matplotlib.pyplot as plt

class MITIMpopcon:
    def __init__(self, filename):
        """
        Read input.yaml file
        """
        self.input_parameters, self.algorithm, self.points, self.plots = cfspopcon.read_case(filename)
        self.algorithm.validate_inputs(self.input_parameters)
        self.dataset = xr.Dataset(self.input_parameters)

    def evaluate(self):
        self.results = self.algorithm.update_dataset(self.dataset)

    def update_from_gacode(self, 
                           profiles_gacode: PROFILEStools.PROFILES_GACODE,
                           confinement_type="H98"
                           ):
        
        self.dataset['major_radius'].data = profiles_gacode.profiles['rcentr(m)'][-1] * ureg.meter

        self.dataset["magnetic_field_on_axis"].data = np.abs(profiles_gacode.derived["B0"]) * ureg.tesla

        rmin = profiles_gacode.profiles["rmin(m)"][-1]
        rmaj = profiles_gacode.profiles["rmaj(m)"][-1]
        self.dataset["inverse_aspect_ratio"].data =  (rmin / rmaj) * ureg.dimensionless

        kappa_a = profiles_gacode.derived["kappa_a"]
        kappa_sep = profiles_gacode.profiles["kappa(-)"][-1]
        self.dataset["areal_elongation"].data = kappa_a * ureg.dimensionless
        self.dataset["elongation_ratio_sep_to_areal"].data = (kappa_sep / kappa_a) * ureg.dimensionless

        delta_95 = np.interp(0.95, profiles_gacode.derived["rho_pol"], profiles_gacode.profiles["delta(-)"])
        delta_sep = profiles_gacode.profiles["delta(-)"][-1]
        self.dataset["triangularity_psi95"].data = delta_95 * ureg.dimensionless
        self.dataset["triangularity_ratio_sep_to_psi95"].data = (delta_sep / delta_95) * ureg.dimensionless

        ip = np.abs(profiles_gacode.profiles["current(MA)"][-1]) * 1e6
        self.dataset["plasma_current"].data = ip * ureg.ampere

        ne_vol_19 = profiles_gacode.derived["ne_vol20"] * 10
        self.dataset = self.dataset.assign_coords(dim_average_electron_density=np.array([ne_vol_19]))
        self.dataset["average_electron_density"].data = np.array([ne_vol_19]) * ureg._1e19_per_cubic_metre
        self.dataset["nesep_over_nebar"].data = profiles_gacode.profiles["ne(10^19/m^3)"][-1] / ne_vol_19

        te_vol_keV = profiles_gacode.derived["Te_vol"]
        self.dataset = self.dataset.assign_coords(dim_average_electron_temp=np.array([te_vol_keV]))
        self.dataset["average_electron_temp"].data = np.array([te_vol_keV]) * ureg.kiloelectron_volt

        impurity_zs = profiles_gacode.profiles["z"][np.where(profiles_gacode.profiles["z"] > 1)]
        imputity_fs = profiles_gacode.derived["fi_vol"][np.where(profiles_gacode.profiles["z"] > 1)]
        impurities = []
        concentrations = []

        for i in range(impurity_zs.size):
            try:
                impurities.append(cfspopcon.named_options.AtomicSpecies(int(impurity_zs[i])))
                concentrations.append(imputity_fs[i])
            except:
                print(f"Could not find atomic number {impurity_zs[i]} in list of named quantities.")
                print(f"Z={impurity_zs[i]}")
                print("Attempting to lump impurity content using oxygen (Z=8)")
                impurities.append(cfspopcon.named_options.AtomicSpecies(8))
                concentrations.append((imputity_fs[i] * impurity_zs[i]) / 8)

        self.dataset = self.dataset.assign_coords(dim_species=np.array(impurities))
        self.dataset['impurities'] = cfspopcon.helpers.make_impurities_array(impurities, concentrations)

        arg_min_rho = np.argmin(np.abs(profiles_gacode.profiles["rho(-)"] - 0.4))
        arg_max_rho = np.argmin(np.abs(profiles_gacode.profiles["rho(-)"] - 0.8))

        # calculate the predicted density peaking using the Angioni 2007 scaling
        betan = profiles_gacode.derived["BetaN"]
        beta_percent = betan * ip  / rmin / profiles_gacode.profiles['bcentr(T)'][-1]
        nu_eff = profiles_gacode.derived["nu_eff"]
        nu_n_scaling = cfspopcon.formulas.calc_density_peaking(nu_eff, beta_percent * 1e-2, nu_noffset=0.0)

        aLTe = profiles_gacode.derived["aLTe"][arg_min_rho:arg_max_rho].mean()
        self.dataset["normalized_inverse_temp_scale_length"].data = aLTe * ureg.dimensionless

        nu_ne_offset = (nu_n_scaling - profiles_gacode.derived["ne_peaking"])
        self.dataset["electron_density_peaking_offset"].data = nu_ne_offset * ureg.dimensionless
        self.dataset["ion_density_peaking_offset"].data = nu_ne_offset * ureg.dimensionless

        nu_te = profiles_gacode.derived["Te_peaking"]
        self.dataset["temperature_peaking"].data = nu_te * ureg.dimensionless

        confinement_scalar = profiles_gacode.derived[confinement_type]
        self.dataset["confinement_time_scalar"].data = confinement_scalar * ureg.dimensionless

        ti_over_te = profiles_gacode.derived["tite_vol"]
        self.dataset["ion_to_electron_temp_ratio"].data = ti_over_te * ureg.dimensionless

    def update_transport(self, 
                         aLTe, 
                         confinement_scalar, 
                         nu_te, 
                         nu_ne_offset, 
                         ti_over_te
                         ):

        self.dataset["normalized_inverse_temp_scale_length"].data = aLTe * ureg.dimensionless
        self.dataset["electron_density_peaking_offset"].data = nu_ne_offset * ureg.dimensionless
        self.dataset["ion_density_peaking_offset"].data = nu_ne_offset * ureg.dimensionless
        self.dataset["temperature_peaking"].data = nu_te * ureg.dimensionless
        self.dataset["confinement_time_scalar"].data = confinement_scalar * ureg.dimensionless
        self.dataset["ion_to_electron_temp_ratio"].data = ti_over_te * ureg.dimensionless
