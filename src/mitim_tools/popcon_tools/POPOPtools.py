import numpy as np
import xarray as xr
from IPython import embed
from popop.xarrays_with_units import ureg, Quantity
from popop.drivers import run_original_POPCON
from IPython import embed
from mitim_tools.misc_tools.LOGtools import printMsg as print


class CFSpopop:
    def run(
        self,
        engineering_parameters,
        assumption_parameters,
        resolution=100,
        constraints_set=None,
        ranges_neTe=[[2, 16], [0.1, 100]],
        extraInfo={"maxPaux": 25, "maxfG": 1, "maxLH": 1},
    ):
        """
        Compatibility updates:
                - remove dilution and zeff
                - radiation_in_tau --> tau_e_scaling_uses_P_in
                - profile form split into electron_density, fuel_density, electron_temp, and fuel_temp
        """

        self.constraints_set = constraints_set
        self.extraInfo = extraInfo

        nebar19 = Quantity(
            np.linspace(ranges_neTe[1][0], ranges_neTe[1][1], num=resolution), ureg.n19
        )
        TebarkeV = Quantity(
            np.linspace(ranges_neTe[0][0], ranges_neTe[0][1], num=resolution), ureg.keV
        )

        self.nebar19 = xr.DataArray(
            nebar19,
            dims="average_density",
            coords=dict(average_density=nebar19.magnitude),
        )
        self.TebarkeV = xr.DataArray(
            TebarkeV,
            dims="average_temperature",
            coords=dict(average_temperature=TebarkeV.magnitude),
        )

        R0 = Quantity(engineering_parameters["R0"], ureg.m)
        BR = R0 * Quantity(engineering_parameters["Bt"], ureg.T)
        epsilon = engineering_parameters["epsilon"]
        kappaA = engineering_parameters["kappaA"]
        delta95 = engineering_parameters["delta95"]
        q_star = engineering_parameters["q_star"]
        reaction_type = engineering_parameters["reaction_type"]

        aLTe = assumption_parameters["aLTe"]
        nu_T = assumption_parameters["nu_T"]
        # 		dilution          				= assumption_parameters['dilution']
        frac_heavier_fuel = assumption_parameters["frac_heavier_fuel"]
        # 		zeff              				= assumption_parameters['zeff']
        impurities = assumption_parameters["impurities"]
        H = assumption_parameters["H"]
        factor_P_rad = assumption_parameters["factor_P_rad"]
        nu_nEoffset = assumption_parameters["nu_nEoffset"]
        nu_nIoffset = assumption_parameters["nu_nIoffset"]
        tau_e_scaling_uses_P_in = assumption_parameters["tau_e_scaling_uses_P_in"]
        tauE_scaling = assumption_parameters["tauE_scaling"]
        electron_density_profile_form = assumption_parameters[
            "electron_density_profile_form"
        ]
        fuel_density_profile_form = assumption_parameters["fuel_density_profile_form"]
        electron_temp_profile_form = assumption_parameters["electron_temp_profile_form"]
        fuel_temp_profile_form = assumption_parameters["fuel_density_profile_form"]
        radiation_method = assumption_parameters["radiation_method"]
        temp_ratio = assumption_parameters["temp_ratio"]
        zeff = assumption_parameters["zeff"]
        dilution = assumption_parameters["dilution"]

        self.results = run_original_POPCON(
            nebar19=self.nebar19,
            TebarkeV=self.TebarkeV,
            R0=R0,
            BR=BR,
            epsilon=epsilon,
            kappaA=kappaA,
            delta95=delta95,
            q_star=q_star,
            aLTe=aLTe,
            nu_T=nu_T,
            frac_heavier_fuel=frac_heavier_fuel,
            impurities=impurities,
            H=H,
            factor_P_rad=factor_P_rad,
            nu_nEoffset=nu_nEoffset,
            nu_nIoffset=nu_nIoffset,
            tau_e_scaling_uses_P_in=tau_e_scaling_uses_P_in,
            tauE_scaling=tauE_scaling,
            electron_density_profile_form=electron_density_profile_form,
            fuel_density_profile_form=fuel_density_profile_form,
            electron_temp_profile_form=electron_temp_profile_form,
            fuel_temp_profile_form=fuel_temp_profile_form,
            radiation_method=radiation_method,
            reaction_type=reaction_type,
            temp_ratio=temp_ratio,
            zeff=zeff,
            dilution=dilution,
        )


# 	def mask_var(self,var):

# 		(Ip, Ip_inductive, fG, Q, Wp, tauE, P_in, P_sol, P_fusion, P_radiation,
# 		P_external, P_alpha, P_neutron, P_ohmic, P_aux, neutron_rate, P_LH_thresh,P_LI_thresh,
# 		ratio_P_LH, ratio_P_LI, ne19_peak, ni_peak, TekeV_peak, TikeV_peak, betaN, beta_pol, V_loop, fBS,
# 		nu_spitz, nu_trap, nu_neo, surface_area, volume) = self.results

# 		if self.constraints_set == 'H':		im = var.where(ratio_P_LH > self.extraInfo['maxLH']).where(P_aux < Quantity(self.extraInfo['maxPaux'], "MW")).where(fG < self.extraInfo['maxfG'])
# 		elif self.constraints_set == 'L':	im = var.where(ratio_P_LH < self.extraInfo['maxLH']).where(P_aux < Quantity(self.extraInfo['maxPaux'], "MW")).where(fG < self.extraInfo['maxfG'])

# 		return im

# 	def optimize(self,ax=None):

# 		# I didn't update this tuple assignment from self.results yet

# 		(Ip, volume, surface_area, ne19_peak, TekeV_peak, fG, Q, Wp, tauE, nTtauE,
# 		P_transport, P_fusion, P_radiation, P_external, P_alpha, P_neutron, P_ohmic, P_aux, P_sol,
# 		P_LH_thresh, ratio_P_LH, betaN, beta, beta_pol, fBS, neutron_rate, q_neutron,V_loop
# 		) = self.results

# 		Q_mask = self.mask_var(Q)

# 		Qmax = Q_mask.max().data.magnitude.round(2)

# 		da = Q_mask.argmax(...)
# 		Tmax_i = da['average_temperature'].item()
# 		Tmax   = Q_mask.coords['average_temperature'][Tmax_i].item()

# 		nmax_i = da['average_density'].item()
# 		nmax   = Q_mask.coords['average_density'][nmax_i].item()

# 		print(f'> Max Q value found to be Q={Qmax} at <T>={Tmax:.1f}keV, <n>={nmax:.1f}E19m3',typeMsg='i')

# 		if ax is not None:
# 			ax.plot([Tmax],[nmax],'-o',markersize=10,c='r')

# 			Q_contours = Q.plot.contour(levels= [Qmax], colors="r",linewidths=1)
# 			Q_artist = Q_contours.legend_elements()[0][0]
# 			label_contour(ax, Q_contours)

# 		return Qmax

# 	def plot(self,
# 			windowSet 			= 'Q',
# 			countours_optional 	= {
# 									'Q': 		[1.0, 2.0, 5.0, 10.0, 20.0],
# 									'Paux': 	[0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
# 									'Pfusion': 	[0.0, 50.0, 100, 250],
# 									'fG': 		[1.0],
# 									'betaN': 	[3.0],
# 									'Plh': 		[1.0] } ):

# 		# (Ip, Ip_inductive, fG, Q, Wp, tauE, P_in, P_sol, P_fusion, P_radiation,
# 		# P_external, P_alpha, P_neutron, P_ohmic, P_aux, neutron_rate, P_LH_thresh,P_LI_thresh,
# 		# ratio_P_LH, ratio_P_LI, ne19_peak, ni_peak, TekeV_peak, TikeV_peak, betaN, beta_pol, V_loop, fBS,
# 		# nu_spitz, nu_trap, nu_neo, surface_area, volume) = self.results

# 		plt.ion()

# 		Q = self.results['Q']
# 		P_aux = self.results['P_aux']
# 		# fG = self.results['greenwald_fraction']
# 		ratio_P_LH = self.results['ratio_P_LH']
# 		P_fusion = self.results['P_fusion']


# 		fig, ax = plt.subplots(figsize=(8,6))

# 		legs,legslab = [], []

# 		if 'Q' in countours_optional:
# 			Q_contours = Q.plot.contour(levels= countours_optional['Q'], colors="r")
# 			Q_artist = Q_contours.legend_elements()[0][0]
# 			label_contour(ax, Q_contours)

# 			legs.append(Q_contours)
# 			legslab.append("$Q$")

# 			if 1.0 in countours_optional['Q']:
# 				Q_contours = Q.plot.contour(levels= [1.0], colors="r",linewidths=2)
# 				Q_artist = Q_contours.legend_elements()[0][0]
# 				label_contour(ax, Q_contours)

# 		if 'Paux' in countours_optional:
# 			P_aux_contours = P_aux.plot.contour(levels=countours_optional['Paux'], colors=["k"])
# 			label_contour(ax, P_aux_contours)

# 			legs.append(P_aux_contours)
# 			legslab.append("$P_{aux}$")

# 			if 0.0 in countours_optional['Paux']:
# 				P_aux_contours = P_aux.plot.contour(levels=[0.0], colors=["k"],linewidths=2)
# 				label_contour(ax, P_aux_contours)

# 		if 'Pfusion' in countours_optional:
# 			P_fusion_contours = P_fusion.plot.contour(levels=countours_optional['Pfusion'], colors=["purple"])
# 			label_contour(ax, P_fusion_contours)

# 			legs.append(P_fusion_contours)
# 			legslab.append("$P_{fusion}$")

# 		if 'fG' in countours_optional:
# 			greenwald_fraction_contours = fG.plot.contour(levels=countours_optional['fG'], colors=["green"])
# 			label_contour(ax, greenwald_fraction_contours)

# 			legs.append(greenwald_fraction_contours)
# 			legslab.append("$f_G$")

# 		if 'betaN' in countours_optional:
# 			betaN_contours = betaN.plot.contour(levels=countours_optional['betaN'], colors=["gold"])
# 			label_contour(ax, betaN_contours)

# 			legs.append(betaN_contours)
# 			legslab.append("$\\beta_N$")

# 		if 'Plh' in countours_optional:
# 			LH_contours = ratio_P_LH.plot.contour(levels=countours_optional['Plh'], colors=["blue"])
# 			label_contour(ax, LH_contours)

# 			legs.append(LH_contours)
# 			legslab.append("$f_{LH}$")

# 		if 'ne_peaking' in countours_optional:
# 			var = ne19_peak/self.nebar19
# 			var_contours = var.plot.contour(levels=countours_optional['ne_peaking'], colors=["cyan"])
# 			label_contour(ax, var_contours)

# 			legs.append(var_contours)
# 			legslab.append("$\\nu_{ne}$")

# 		if 'Wp' in countours_optional:
# 			var = Wp
# 			var_contours = var.plot.contour(levels=countours_optional['Wp'], colors=["orange"])
# 			label_contour(ax, var_contours)

# 			legs.append(var_contours)
# 			legslab.append("$W_p$")

# 		if 'tauE' in countours_optional:
# 			var = tauE
# 			var_contours = var.plot.contour(levels=countours_optional['tauE'], colors=["orange"])
# 			label_contour(ax, var_contours)

# 			legs.append(var_contours)
# 			legslab.append("$\\tau_E$")

# 		if self.constraints_set is not None:

# 			if windowSet == 'Q': 			var, lab = Q, 'Feasible Q'
# 			elif windowSet == 'Pfusion': 	var, lab = P_fusion, 'Feasible $P_{fusion}$ (MW)'
# 			elif windowSet == 'ne_peaking': 	var, lab = ne19_peak/self.nebar19, 'Feasible $\\nu_{ne}$'

# 			var_mask = self.mask_var(var)
# 			im = var_mask.plot(ax=ax, add_colorbar=False)

# 			cbar = fig.colorbar(im, ax=ax)
# 			cbar.ax.get_yaxis().labelpad = 15
# 			cbar.ax.set_ylabel(lab)

# 		ax.legend([C.legend_elements()[0][0] for C in legs],legslab)

# 		GRAPHICStools.addDenseAxis(ax)

# 		ax.set_xlabel('$\\langle T_e \\rangle$ (keV)')
# 		ax.set_ylabel('$\\langle n_e \\rangle$ ($10^{19}m^{-3}$)')

# 		GRAPHICStools.adjust_figure_layout(fig)

# 		return ax

# def label_contour(ax, contour_set):
#     def fmt(x):
#         return f"{x:g}"
#     ax.clabel(contour_set, contour_set.levels, inline=True, fmt=fmt, fontsize=10)
