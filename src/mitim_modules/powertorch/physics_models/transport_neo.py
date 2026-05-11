import numpy as np
from mitim_tools.gacode_tools import NEOtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from IPython import embed

class neo_model:

    def evaluate_neoclassical(self):
        
        # ------------------------------------------------------------------------------------------------------------------------
        # Grab options
        # ------------------------------------------------------------------------------------------------------------------------

        simulation_options = self.transport_evaluator_options["neo"]
        cold_start = self.cold_start
        
        percent_error = simulation_options["percent_error"]
        impurityPosition = self.powerstate.impurityPosition_transport
                
        # ------------------------------------------------------------------------------------------------------------------------        
        # Run
        # ------------------------------------------------------------------------------------------------------------------------
        
        rho_locations = [self.powerstate.plasma["rho"][0, 1:][i].item() for i in range(len(self.powerstate.plasma["rho"][0, 1:]))]
        
        neo = NEOtools.NEO(rhos=rho_locations)

        _ = neo.prep(
            self.powerstate.profiles_transport,
            self.folder,
            cold_start = cold_start,
            )

        # ---- Edge extension: inject per-rho DPHI0DR from powerstate_edge.plasma["E_rad"]
        # Used when ROTATION_MODEL=1 ("edge" NEO model) so that each local NEO run
        # receives the radial electric field derived from the current ExB rotation state.
        # plasma["E_rad"] is computed in powerstate_edge.calculateProfileFunctions():
        #   E_rad = roa * omega_ExB[rad/s] * R_major[m] / v_thi[m/s]   (DPHI0DR normalization)
        try:
            from mitim_modules.powertorch.STATEedge import powerstate_edge as _powerstate_edge
            _is_edge = isinstance(self.powerstate, _powerstate_edge)
        except ImportError:
            _is_edge = False

        if _is_edge and "E_rad" in self.powerstate.plasma:
            E_rad_profile = self.powerstate.plasma["E_rad"][0, 1:]   # (n_rho,) skip rho=0
            for i, rho_val in enumerate(rho_locations):
                dphi0dr = float(E_rad_profile[i].item())
                neo.inputs_files[rho_val].controls["DPHI0DR"] = dphi0dr
            print(
                f"\t[edge NEO] Per-rho DPHI0DR injected from E_rad "
                f"({len(rho_locations)} radii, range "
                f"[{float(E_rad_profile.min()):.3f}, {float(E_rad_profile.max()):.3f}])",
                typeMsg="i",
            )

        neo.run(
            'base_neo',
            cold_start=cold_start,
            forceIfcold_start=True,
            **simulation_options["run"]
        )
    
        neo.read(
            label='base',
            **simulation_options["read"])
        
        Qe = np.array([neo.results['base']['output'][i].Qe for i in range(len(rho_locations))])
        Qi = np.array([neo.results['base']['output'][i].Qi for i in range(len(rho_locations))])
        Ge = np.array([neo.results['base']['output'][i].Ge for i in range(len(rho_locations))])
        GZ = np.array([neo.results['base']['output'][i].GiAll[impurityPosition-1] for i in range(len(rho_locations))])
        Mt = np.array([neo.results['base']['output'][i].Mt for i in range(len(rho_locations))])
        
        # ------------------------------------------------------------------------------------------------------------------------
        # Pass the information to what power_transport expects
        # ------------------------------------------------------------------------------------------------------------------------
        
        self.QeGB_neoc = Qe
        self.QiGB_neoc = Qi
        self.GeGB_neoc = Ge
        self.GZGB_neoc = GZ
        self.MtGB_neoc = Mt
        
        # Uncertainties is just a percent of the value
        self.QeGB_neoc_stds = abs(Qe) * percent_error/100.0
        self.QiGB_neoc_stds = abs(Qi) * percent_error/100.0
        self.GeGB_neoc_stds = abs(Ge) * percent_error/100.0
        self.GZGB_neoc_stds = abs(GZ) * percent_error/100.0
        self.MtGB_neoc_stds = abs(Mt) * percent_error/100.0

        return neo
