import numpy as np
import xarray as xr
from pathlib import Path
import cfspopcon
from cfspopcon.unit_handling import ureg
from mitim_tools.gacode_tools import PROFILEStools
from mitim_tools.misc_tools import GRAPHICStools
from mitim_tools.astra_tools import ASTRA_CDFtools
from mitim_tools.gs_tools import GEQtools
from mitim_tools.misc_tools import IOtools

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

        delta_95 = np.interp(0.95, profiles_gacode.derived["psi_pol_n"], profiles_gacode.profiles["delta(-)"])
        delta_sep = profiles_gacode.profiles["delta(-)"][-1]
        self.dataset["triangularity_psi95"].data = delta_95 * ureg.dimensionless
        self.dataset["triangularity_ratio_sep_to_psi95"].data = (delta_sep / delta_95) * ureg.dimensionless

        ip = np.abs(profiles_gacode.profiles["current(MA)"][-1]) * 1e6
        self.dataset["plasma_current"].data = ip * ureg.ampere

        ne_vol_19 = profiles_gacode.derived["ne_vol20"] * 10
        self.dataset = self.dataset.assign_coords(dim_average_electron_density=np.array([ne_vol_19]))
        self.dataset["average_electron_density"].data = np.array([ne_vol_19]) * ureg._1e19_per_cubic_metre
        #self.dataset["nesep_over_nebar"].data = profiles_gacode.profiles["ne(10^19/m^3)"][-1] / ne_vol_19

        te_vol_keV = profiles_gacode.derived["Te_vol"]
        self.dataset = self.dataset.assign_coords(dim_average_electron_temp=np.array([te_vol_keV]))
        self.dataset["average_electron_temp"].data = np.array([te_vol_keV]) * ureg.kiloelectron_volt

        impurity_zs = profiles_gacode.profiles["z"][np.where(profiles_gacode.profiles["z"] > 1)]
        imputity_fs = profiles_gacode.derived["fi_vol"][np.where(profiles_gacode.profiles["z"] > 1)]
        impurities = []
        concentrations = []
        named_options_array = [1,2,3,4,6,7,8,10,18,36,54,74] # atomicspecies built into cfspopcon

        for i in range(impurity_zs.size):
            try:
                impurities.append(cfspopcon.named_options.AtomicSpecies(int(impurity_zs[i])))
                concentrations.append(imputity_fs[i])
            except:
                print(f"Could not find atomic number {impurity_zs[i]} in list of named quantities.")
                print(f"Z={impurity_zs[i]}")
            
                closest_element = min(named_options_array, key=lambda x: abs(x - impurity_zs[i]))
                print(f"Attempting to lump impurity content using Z={closest_element} while preserving quasineutrality.")
                print(f"May produce inaccurate results from radiation model.")
                impurities.append(cfspopcon.named_options.AtomicSpecies(int(closest_element)))
                concentrations.append((imputity_fs[i] * impurity_zs[i]) / closest_element)

        self.dataset = self.dataset.assign_coords(dim_species=np.array(impurities))
        self.dataset['impurities'] = cfspopcon.helpers.make_impurities_array(impurities, concentrations)

        arg_min_rho = np.argmin(np.abs(profiles_gacode.profiles["rho(-)"] - 0.4))
        arg_max_rho = np.argmin(np.abs(profiles_gacode.profiles["rho(-)"] - 0.8))

        # calculate the predicted density peaking using the Angioni 2007 scaling
        beta_percent = (profiles_gacode.derived["BetaN"]
                        *profiles_gacode.profiles["current(MA)"][-1] 
                        / profiles_gacode.profiles["rmin(m)"][-1] 
                        / profiles_gacode.profiles['bcentr(T)'][-1])
        
        nu_n_scaling = cfspopcon.formulas.calc_density_peaking(profiles_gacode.derived["nu_eff"],
                                                                beta_percent*1e-2,
                                                                  nu_noffset=0.0)

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

    def match_to_gacode(self,
                        profiles_gacode: PROFILEStools.PROFILES_GACODE,
                        confinement_type="H98",
                        print_progress=False,
                        plot_convergence=True,
                        bounds=None
                        ):
        
        from scipy.optimize import minimize

        self.parameter_history = []
        self.popcon_history = []

        print("Starting optimization...")
        print("... Initializing POPCON with GACODE parameters")

        self.update_from_gacode(profiles_gacode=profiles_gacode,
                                confinement_type=confinement_type
                                )
        x0 = [
            self.dataset["normalized_inverse_temp_scale_length"].data.magnitude,
            self.dataset["confinement_time_scalar"].data.magnitude,
            self.dataset["temperature_peaking"].data.magnitude,
            self.dataset["electron_density_peaking_offset"].data.magnitude,
            self.dataset["ion_to_electron_temp_ratio"].data.magnitude
        ]

        if bounds is None:
            flag=True
            bounds = [(None,None),
                        (0.99*x0[1],1.01*x0[1]),
                        (None,None),
                        (None,None),
                        (0,1)]
            
        print(bounds)

        # first optomization step: make profiles match
        print("... Optimizing profiles")

        res = minimize(self.match_profiles,
                        x0,
                        args=(profiles_gacode, print_progress), 
                        method='Nelder-Mead',
                        bounds=bounds, 
                        options={'disp': True},
                        tol=1e-1)
        x1 = [
            res.x[0], 
            res.x[1], 
            res.x[2], 
            res.x[3], 
            res.x[4]
        ]

        if x1[3] >=0:
            nu_ne_offset_bounds = (0.99*x1[3],1.01*x1[3])
        else:
            nu_ne_offset_bounds = (1.01*x1[3],0.99*x1[3])

        if flag:
            bounds = [(0.99*x1[0],1.01*x1[0]),
                            (None,None),
                            (0.99*x1[2],1.01*x1[2]),
                            nu_ne_offset_bounds,
                            (0.99*x1[4],1.01*x1[4])]
            
        print(bounds)

        # second optimization step: make power balance match
        print("... Optimizing confinement time")

        res = minimize(self.match_pfus,
                       x1,
                       args=(profiles_gacode, print_progress), 
                       method='Nelder-Mead',
                       bounds=bounds, 
                       options={'disp': True},
                       tol=1e-1)
        
        if res.success:
            self.update_transport(res.x[0], res.x[1], res.x[2], res.x[3], res.x[4])
            self.evaluate()

        if plot_convergence:
            self.plot_convergence()
                        
    def match_pfus(self, 
                   x, 
                   profiles_gacode,
                   print_progress=False
                   ):
        
        aLTe, H_98, nu_te, nu_ne_offset, ti_over_te = x[0], x[1], x[2], x[3], x[4]

        # returns the residual between the POPCON and GACODE power balance

        self.update_transport(aLTe, H_98, nu_te, nu_ne_offset, ti_over_te)
        self.evaluate()

        point = self.results.isel(dim_average_electron_temp=0, dim_average_electron_density=0)

        Pfus_residual = point['P_fusion'].data.magnitude - profiles_gacode.derived['Pfus'] 

        Psol_residual = ((point['P_LH_thresh'].data.magnitude 
                         * point['ratio_of_P_SOL_to_P_LH'].data.magnitude) 
                         - profiles_gacode.derived['Psol'])

        Pin_derived = (profiles_gacode.derived['qi_aux_MWmiller'][-1]
                       +profiles_gacode.derived['qe_aux_MWmiller'][-1]
                       )
        
        Pin_residual = point['P_auxillary'].data.magnitude - Pin_derived

        self.parameter_history.append(x)

        self.popcon_history.append({"Pfus": point['P_fusion'].data.magnitude,
                            "Q": point['Q'].data.magnitude, 
                            "Pin": point['P_in'].data.magnitude, 
                            "Psol": point['P_LH_thresh'].data.magnitude *point['ratio_of_P_SOL_to_P_LH'].data.magnitude, 
                            "taue": point['energy_confinement_time'].data.magnitude, 
                            "Paux": point['P_auxillary'].data.magnitude}
                            )
        
        if print_progress:
            print("Parameters:", x)
            print("Residual:", Pfus_residual**2 + Psol_residual**2 + Pin_residual**2)
            print("P_fusion:", point['P_fusion'].data.magnitude, "MW")
        
        return Pfus_residual**2 + Pin_residual**2 + Psol_residual**2
    
    def match_profiles(self,
                       x,
                       profiles_gacode,
                       print_progress=False
                   ):
        
        aLTe, H_98, nu_te, nu_ne_offset, ti_over_te = x[0], x[1], x[2], x[3], x[4]
        # returns the difference between the POPCON and GACODE power balance
        # optimizes over Pfus, Q, and P_in

        self.update_transport(aLTe, H_98, nu_te, nu_ne_offset, ti_over_te)
        self.evaluate()

        point = self.results.isel(dim_average_electron_temp=0, dim_average_electron_density=0)

        te_profiles = np.interp(point["dim_rho"], point["dim_rho"].size*profiles_gacode.profiles["rho(-)"] ,profiles_gacode.profiles["te(keV)"])
        te_popcon = point["electron_temp_profile"].data.magnitude
        te_L2 = np.sum((te_profiles-te_popcon)**2)

        ti_profiles = np.interp(point["dim_rho"], point["dim_rho"].size*profiles_gacode.profiles["rho(-)"] ,profiles_gacode.profiles["ti(keV)"][:, 0])
        ti_popcon = point["ion_temp_profile"].data.magnitude
        ti_L2 = np.sum((ti_profiles-ti_popcon)**2)

        ne19_profiles = np.interp(point["dim_rho"], point["dim_rho"].size*profiles_gacode.profiles["rho(-)"] ,profiles_gacode.profiles["ne(10^19/m^3)"])
        ne19_popcon = point["electron_density_profile"].data.magnitude
        ne19_L2 = np.sum((ne19_profiles-ne19_popcon)**2)

        self.parameter_history.append(x)

        self.popcon_history.append({"Pfus": point['P_fusion'].data.magnitude,
                            "Q": point['Q'].data.magnitude, 
                            "Pin": point['P_in'].data.magnitude, 
                            "Psol": point['P_LH_thresh'].data.magnitude *point['ratio_of_P_SOL_to_P_LH'].data.magnitude, 
                            "taue": point['energy_confinement_time'].data.magnitude, 
                            "Paux": point['P_auxillary'].data.magnitude}
                            )
        
        if print_progress:
            print("Parameters:", x)
            print("Residual:", te_L2 + ti_L2 + ne19_L2)
            print("P_fusion:", point['P_fusion'].data.magnitude, "MW")
        
        return te_L2 + ti_L2 + ne19_L2
    
    def plot_convergence(self):

        parameter_history = np.array(self.parameter_history)
        iteration = np.arange(parameter_history.shape[0])

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # Plot aLTe
        axs[0, 0].plot(iteration, parameter_history[:, 0])
        axs[0, 0].set_xlabel('Iteration')
        axs[0, 0].set_ylabel('$a/LT_e$')
        axs[0, 0].set_title('$a/LT_e$ History')

        # Plot H_98
        axs[0, 1].plot(iteration, parameter_history[:, 1],label='H_98')
        axs[0, 1].set_xlabel('Iteration')
        axs[0, 1].set_title('H_98 and Ti/Te History')
        axs[0, 1].plot(iteration, parameter_history[:, 4],label='ti/te')
        axs[0, 1].legend()

        # Plot nu_te
        axs[1, 0].plot(iteration, parameter_history[:, 2])
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel(r'$\nu_{Te}$')
        axs[1, 0].set_title(r'$\nu_{Te}$ History')

        # Plot nu_ne_offset and nu_ni_offset on the same axis
        axs[1, 1].plot(iteration, parameter_history[:, 3], label=r'$\nu_{n}$ offset')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel(r'$\nu_n$ offset')
        axs[1, 1].set_title(r'$\nu_n$ offset History')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def plot_profile_comparison(self, profiles_gacode: PROFILEStools.PROFILES_GACODE):

        point = self.results.isel(dim_average_electron_temp=0, dim_average_electron_density=0)

        fig, ax = plt.subplots(figsize=(12,8))
        ax2 = ax.twinx()
        GRAPHICStools.addDenseAxis(ax)
        x = np.linspace(0,1, point["dim_rho"].size) # normalized toroidal flux

        ax.plot(x, 
                np.interp(point["dim_rho"], point["dim_rho"].size*profiles_gacode.profiles["rho(-)"] ,profiles_gacode.profiles["te(keV)"]),
                label="Te (GACODE)",color='tab:red')

        ax.plot(x,point["electron_temp_profile"], label="Te (POPCON)",ls="--",color='tab:red')

        ax.plot(x, 
                np.interp(point["dim_rho"], point["dim_rho"].size*profiles_gacode.profiles["rho(-)"] ,profiles_gacode.profiles["ti(keV)"][:, 0]),
                label="Ti (GACODE)",color='tab:purple')
        ax.plot(x,point["ion_temp_profile"], label="Ti (POPCON)",ls="--",color='tab:purple')

        ax2.plot(x, 
                np.interp(point["dim_rho"], point["dim_rho"].size*profiles_gacode.profiles["rho(-)"] ,profiles_gacode.profiles["ne(10^19/m^3)"]),
                label="ne (GACODE)",color='tab:blue')
        ax2.plot(x,point["electron_density_profile"], label="ne (POPCON)",ls="--",color='tab:blue')

        ax.text(0,11, f"Pfus = {point['P_fusion'].data.magnitude:.2f}")
        ax.text(0,13, f"Q = {point['Q'].data.magnitude:.2f}")

        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel("Temperature [keV]")
        ax2.set_ylabel("Electron Density [$10^{19}m^{-3}$]")
        ax.legend(loc='lower left')
        ax2.legend(loc='upper right')
        plt.title("ASTRA-GACODE-POPCON Matching",fontsize=24)
        plt.tight_layout()

    def evaluate_on_grid(self,
             Te_range=np.linspace(5, 15, 10), # temperature range to evaluate, keV
             ne_range=np.linspace(8, 30, 10), # density range to evaluate, in 10^19 m^-3
             use_result=False
            ):
        
        # Establish new <Te>, <ne> ranges

        popcon_2D = self.input_parameters

        popcon_2D["average_electron_density"] = xr.DataArray(ne_range, coords={"dim_average_electron_density": ne_range})
        popcon_2D["average_electron_density"].data *= ureg._1e19_per_cubic_metre

        popcon_2D["average_electron_temp"] = xr.DataArray(Te_range, coords={"dim_average_electron_temp": Te_range})
        popcon_2D["average_electron_temp"].data *= ureg.kiloelectron_volt

        popcon_2D = xr.Dataset(popcon_2D)

        if use_result:
            # updates 0D parameters and transport with the results of the optimization
            for key in popcon_2D.keys():
                if key not in ["average_electron_density", "average_electron_temp"]:
                    popcon_2D[key] = self.results[key]

        popcon_2D = self.algorithm.update_dataset(popcon_2D)

        return popcon_2D
    
    def plot(self,
             dataset_2D=None,
             plot_template=None,
             plot_options={}, 
             use_result=True,
             title="POPCON Results"
            ):
        
        if dataset_2D is None:
            print("No 2D popcon dataset passed. Evaluating based on default ranges...")
            dataset_2D = self.evaluate_on_grid(use_result=use_result)

        # Read template plot options
        if plot_template is None:
            plot_template = IOtools.expandPath("$MITIM_PATH/templates/plot_popcon.yaml")
            plot_style = cfspopcon.read_plot_style(plot_template)

        # Update plot options
        for key, value in plot_options.items():
            plot_style[key] = value
        
        cfspopcon.plotting.make_plot(
            dataset_2D,
            plot_style,
            points=self.points,
            title=title,
            save_name=None
        )

    def print_data(self,
                   compare_to_gacode=False,
                   profiles_gacode=None
                   ):
        
        try:
            point = self.results.isel(dim_average_electron_temp=0, dim_average_electron_density=0)
        except:
            raise ValueError("No results have been calculated yet. Run evaluate() first.")

        print("POPCON Results:")

        print(f"Operational point: <ne>={point['average_electron_density'].data.magnitude}, <Te>={point['average_electron_temp'].data.magnitude}")

        if compare_to_gacode:
            if profiles_gacode is None:
                raise ValueError("No GACODE profiles passed to compare to.")
            
            print(f"Pfus:  ", f"POPCON: {point['P_fusion'].data.magnitude:.2f}", f"GACODE: {profiles_gacode.derived['Pfus']:.2f}", "(MW)")
            print(f"Q:     ", f"POPCON: {point['Q'].data.magnitude:.2f}", f"GACODE: {profiles_gacode.derived['Q']:.2f}")
            print(f"TauE:  ", f"POPCON: {point['energy_confinement_time'].data.magnitude:.2f}", f"GACODE: {profiles_gacode.derived['tauE']:.2f}", "(s)")
            print(f"Beta_N:", f"POPCON: {point['normalized_beta'].data.magnitude:.2f}", f"GACODE: {profiles_gacode.derived['BetaN']:.2f}")
            print(f"P_sol: ", f"POPCON: {(point['P_LH_thresh'].data.magnitude *point['ratio_of_P_SOL_to_P_LH'].data.magnitude):.2f}",
                f"GACODE: {profiles_gacode.derived['Psol']:.2f}","(MW)", f"({point['P_LH_thresh'].data.magnitude:.2f} of LH threshold)")
            print(f"P_aux: ", f"POPCON: {point['P_auxillary'].data.magnitude:.2f}",
                f"GACODE: {(profiles_gacode.derived['qi_aux_MWmiller'][-1]+profiles_gacode.derived['qe_aux_MWmiller'][-1]):.2f}",
                "(MW)")
            print(f"P_rad: ", f"POPCON: {point['P_radiation'].data.magnitude:.2f}",f"GACODE: {profiles_gacode.derived['Prad']:.2f}","(MW)")
            print(f"P_ext: ", f"POPCON: {point['P_external'].data.magnitude:.2f}","(MW)")
            print(f"P_ohm: ", f"POPCON: {point['P_ohmic'].data.magnitude:.2f}", f"GACODE: {profiles_gacode.derived['qOhm_MWmiller'][-1]:.2f}","(MW)")
            print(f"P_in:  ", f"POPCON: {point['P_in'].data.magnitude:.2f}", 
                f"GACODE: {(profiles_gacode.derived['qOhm_MWmiller'][-1]+profiles_gacode.derived['qi_aux_MWmiller'][-1]+profiles_gacode.derived['qe_aux_MWmiller'][-1]+profiles_gacode.derived['Pfus']*0.2):.2f}",
                "(MW)")
            print(f"q95:   ", f"POPCON: {point['q_star'].data.magnitude:.2f}", f"GACODE: {profiles_gacode.derived['q95']:.2f}")
            print(f"Wtot:  ", f"POPCON: {point['plasma_stored_energy'].data.magnitude:.2f}", f"GACODE: {profiles_gacode.derived['Wthr']:.2f}", "(MJ)")
            print(" ")

            print("Transport Parameters:")
            print("aLTe:", f"{point['normalized_inverse_temp_scale_length'].data.magnitude:.2f}")
            print("H98:", f"{point['confinement_time_scalar'].data.magnitude:.2f}")
            print("nu_Te:", f"{point['temperature_peaking'].data.magnitude:.2f}")
            print(" ")

        else:
            print(f"Pfus:  ", f"POPCON: {point['P_fusion'].data.magnitude:.2f}", "(MW)")
            print(f"Q:     ", f"POPCON: {point['Q'].data.magnitude:.2f}")
            print(f"TauE:  ", f"POPCON: {point['energy_confinement_time'].data.magnitude:.2f}", "(s)")
            print(f"Beta_N:", f"POPCON: {point['normalized_beta'].data.magnitude:.2f}",)

            print(f"P_sol: ", 
                f"POPCON: {(point['P_LH_thresh'].data.magnitude *point['ratio_of_P_SOL_to_P_LH'].data.magnitude):.2f}",
                    "(MW)")
            
            print(f"P_aux: ", f"POPCON: {point['P_auxillary'].data.magnitude:.2f}","(MW)")
            print(f"P_rad: ", f"POPCON: {point['P_radiation'].data.magnitude:.2f}","(MW)")
            print(f"P_ext: ", f"POPCON: {point['P_external'].data.magnitude:.2f}","(MW)")
            print(f"P_ohm: ", f"POPCON: {point['P_ohmic'].data.magnitude:.2f}","(MW)")
            print(f"P_in:  ", f"POPCON: {point['P_in'].data.magnitude:.2f}", "(MW)")
            print(f"q95:   ", f"POPCON: {point['q_star'].data.magnitude:.2f}")
            print(f"Wtot:  ", f"POPCON: {point['plasma_stored_energy'].data.magnitude:.2f}", "(MJ)")
            print(" ")

            print("Transport Parameters:")
            print("aLTe:", f"{point['normalized_inverse_temp_scale_length'].data.magnitude:.2f}")
            print("H98:", f"{point['confinement_time_scalar'].data.magnitude:.2f}")
            print("nu_Te:", f"{point['temperature_peaking'].data.magnitude:.2f}")
            print(" ")