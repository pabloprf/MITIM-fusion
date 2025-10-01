import numpy as np
import matplotlib.pyplot as plt
from mitim_tools.misc_tools import GRAPHICStools, IOtools, GUItools
from mitim_tools.gacode_tools.utils import GACODErun, GACODEdefaults
from mitim_tools.simulation_tools import SIMtools
from mitim_tools.misc_tools.LOGtools import printMsg as print
from mitim_tools import __mitimroot__
from IPython import embed

class NEO(SIMtools.mitim_simulation):
    def __init__(
        self,
        rhos=[0.4, 0.6],  # rho locations of interest
    ):
        
        super().__init__(rhos=rhos)

        def code_call(folder, n, p, additional_command="", **kwargs):
            return f"neo -e {folder} -n {n} -p {p} {additional_command}"

        def code_slurm_settings(name, minutes, total_cores_required, cores_per_code_call, type_of_submission, array_list=None, **kwargs_slurm):

            slurm_settings = {
                "name": name,
                "minutes": minutes,
                'job_array_limit': None,    # Limit to this number at most running jobs at the same time?
            }

            if type_of_submission == "slurm_standard":
                
                slurm_settings['ntasks'] = total_cores_required
                slurm_settings['cpuspertask'] = cores_per_code_call

            elif type_of_submission == "slurm_array":

                slurm_settings['ntasks'] = 1
                slurm_settings['cpuspertask'] = cores_per_code_call
                slurm_settings['job_array'] = ",".join(array_list)

            return slurm_settings

        self.run_specifications = {
            'code': 'neo',
            'input_file': 'input.neo',
            'code_call': code_call,
            'code_slurm_settings': code_slurm_settings,
            'control_function': GACODEdefaults.addNEOcontrol,
            'controls_file': 'input.neo.controls',
            'state_converter': 'to_neo',
            'input_class': NEOinput,
            'complete_variation': None,
            'default_cores': 1,  # Default cores to use in the simulation
            'output_class': NEOoutput,
        }
        
        print("\n-----------------------------------------------------------------------------------------")
        print("\t\t\t NEO class module")
        print("-----------------------------------------------------------------------------------------\n")

        self.ResultsFiles = self.ResultsFiles_minimal = ['out.neo.transport_flux']
        
    def plot(
        self,
        fn=None,
        labels=["neo1"],
        extratitle="",
        fn_color=None,
        colors=None,
        ):
        
        if fn is None:
            self.fn = GUItools.FigureNotebook("NEO MITIM Notebook", geometry="1700x900", vertical=True)
        else:
            self.fn = fn
            
        fig1 = self.fn.add_figure(label=f"{extratitle}Summary", tab_color=fn_color)
        
        grid = plt.GridSpec(1, 3, hspace=0.7, wspace=0.2)

        if colors is None:
            colors = GRAPHICStools.listColors()

        axQe = fig1.add_subplot(grid[0, 0])
        axQi = fig1.add_subplot(grid[0, 1])
        axGe = fig1.add_subplot(grid[0, 2])

        for i,label in enumerate(labels):
            roa, QeGB, QiGB, GeGB = [], [], [], []
            for irho in range(len(self.rhos)):
                roa.append(self.results[label]['output'][irho].roa)
                QeGB.append(self.results[label]['output'][irho].Qe)
                QiGB.append(self.results[label]['output'][irho].Qi)
                GeGB.append(self.results[label]['output'][irho].Ge)
                
            axQe.plot(roa, QeGB, label=label, color=colors[i], marker='o', linestyle='-')
            axQi.plot(roa, QiGB, label=label, color=colors[i], marker='o', linestyle='-')
            axGe.plot(roa, GeGB, label=label, color=colors[i], marker='o', linestyle='-')

        for ax in [axQe, axQi, axGe]:
            ax.set_xlabel("$r/a$"); ax.set_xlim([0,1])
            GRAPHICStools.addDenseAxis(ax)
            ax.legend(loc="best")

        axQe.set_ylabel("$Q_e$ ($MW/m^2$)"); axQe.set_yscale('log')
        axQi.set_ylabel("$Q_i$ ($MW/m^2$)"); axQi.set_yscale('log')
        axGe.set_ylabel("$\\Gamma_e$ ($1E20/s/m^2$)"); #axGe.set_yscale('log')


    def read_scan(
        self,
        label="scan1",
        subfolder=None,
        variable="RLTS_1",
        positionIon=2
    ):

        output_object = "output"

        variable_mapping = {
            'scanned_variable': ["parsed", variable, None],
            'Qe_gb': [output_object, 'Qe', None],
            'Qi_gb': [output_object, 'Qi', None],
            'Ge_gb': [output_object, 'Ge', None],
            'Gi_gb': [output_object, 'GiAll', positionIon - 2],
            'Mt_gb': [output_object, 'Mt', None],
        }
        
        variable_mapping_unn = {
            'Qe': [output_object, 'Qe_unn', None],
            'Qi': [output_object, 'Qi_unn', None],
            'Ge': [output_object, 'Ge_unn', None],
            'Gi': [output_object, 'GiAll_unn', positionIon - 2],
            'Mt': [output_object, 'Mt_unn', None],
        }
        
        super().read_scan(
            label=label,
            subfolder=subfolder,
            variable=variable,
            positionIon=positionIon,
            variable_mapping=variable_mapping,
            variable_mapping_unn=variable_mapping_unn
        )

    def plot_scan(
        self,
        fn=None,
        labels=["neo1"],
        extratitle="",
        fn_color=None,
        colors=None,
        ):
        
        if fn is None:
            self.fn = GUItools.FigureNotebook("NEO Scan Notebook", geometry="1700x900", vertical=True)
        else:
            self.fn = fn
            
        fig1 = self.fn.add_figure(label=f"{extratitle}Summary", tab_color=fn_color)
        
        grid = plt.GridSpec(1, 3, hspace=0.7, wspace=0.2)

        if colors is None:
            colors = GRAPHICStools.listColors()

        axQe = fig1.add_subplot(grid[0, 0])
        axQi = fig1.add_subplot(grid[0, 1])
        axGe = fig1.add_subplot(grid[0, 2])

        cont = 0
        for label in labels:
            for irho in range(len(self.rhos)):
                
                x = self.scans[label]['scanned_variable'][irho]
                
                axQe.plot(x, self.scans[label]['Qe'][irho], label=f'{label}, {self.rhos[irho]}', color=colors[cont], marker='o', linestyle='-')
                axQi.plot(x, self.scans[label]['Qi'][irho], label=f'{label}, {self.rhos[irho]}', color=colors[cont], marker='o', linestyle='-')
                axGe.plot(x, self.scans[label]['Ge'][irho], label=f'{label}, {self.rhos[irho]}', color=colors[cont], marker='o', linestyle='-')

                cont += 1

        for ax in [axQe, axQi, axGe]:
            ax.set_xlabel("Scanned variable")
            GRAPHICStools.addDenseAxis(ax)
            ax.legend(loc="best")

        axQe.set_ylabel("$Q_e$ ($MW/m^2$)"); 
        axQi.set_ylabel("$Q_i$ ($MW/m^2$)"); 
        axGe.set_ylabel("$\\Gamma_e$ ($1E20/s/m^2$)")
        
        plt.tight_layout()



    # def prep(self, inputgacode, folder):
    #     self.inputgacode = inputgacode
    #     self.folder = IOtools.expandPath(folder)

    #     self.folder.mkdir(parents=True, exist_ok=True)



    def run_vgen(self, subfolder="vgen1", vgenOptions={}, cold_start=False):

        self.folder_vgen = self.folder / f"{subfolder}"

        # ---- Default options

        vgenOptions.setdefault("er", 2)
        vgenOptions.setdefault("vel", 1)
        vgenOptions.setdefault("numspecies", len(self.inputgacode.Species))
        vgenOptions.setdefault("matched_ion", 1)
        vgenOptions.setdefault("nth", "17,39")

        # ---- Prepare

        runThisCase = check_if_files_exist(
            self.folder_vgen,
            [
                ["vgen", "input.gacode"],
                ["vgen", "input.neo.gen"],
                ["out.vgen.neoequil00"],
                ["out.vgen.neoexpnorm00"],
                ["out.vgen.neontheta00"],
                ["vgen.dat"],
            ],
        )

        if (not runThisCase) and cold_start:
            runThisCase = print("\t- Files found in folder, but cold_start requested. Are you sure?",typeMsg="q",)

            if runThisCase:
                IOtools.askNewFolder(self.folder_vgen, force=True)

        self.inputgacode.write_state(file=(self.folder_vgen / f"input.gacode"))

        # ---- Run

        if runThisCase:
            file_new = GACODErun.runVGEN(
                self.folder_vgen, vgenOptions=vgenOptions, name_run=subfolder
            )
        else:
            print(f"\t- Required files found in {subfolder}, not running VGEN",typeMsg="i",)
            file_new = self.folder_vgen / f"vgen" / f"input.gacode"

        # ---- Postprocess

        from mitim_tools.gacode_tools import PROFILEStools
        self.inputgacode_vgen = PROFILEStools.gacode_state(file_new, derive_quantities=True, mi_ref=self.inputgacode.mi_ref)


def check_if_files_exist(folder, list_files):
    folder = IOtools.expandPath(folder)

    for file_parts in list_files:
        checkfile = folder
        for ii in range(len(file_parts)):
            checkfile = checkfile / f"{file_parts[ii]}"
        if not checkfile.exists():
            return False

    return True

class NEOinput(SIMtools.GACODEinput):
    def __init__(self, file=None):
        super().__init__(
            file=file,
            controls_file= __mitimroot__ / "templates" / "input.neo.controls",
            code='NEO',
            n_species='N_SPECIES'
            )
                
class NEOoutput(SIMtools.GACODEoutput):
    def __init__(self, FolderGACODE, suffix="", **kwargs):
        super().__init__()
        
        self.FolderGACODE, self.suffix = FolderGACODE, suffix

        if suffix == "":
            print(f"\t- Reading results from folder {IOtools.clipstr(FolderGACODE)} without suffix")
        else:
            print(f"\t- Reading results from folder {IOtools.clipstr(FolderGACODE)} with suffix {suffix}")

        self.inputclass = NEOinput(file=self.FolderGACODE / f"input.neo{self.suffix}")

        self.read()

    def read(self):
                
        with open(self.FolderGACODE / ("out.neo.transport_flux" + self.suffix), "r") as f:
            lines = f.readlines()
            
        for i in range(len(lines)):
            if '# Z       pflux_tgyro   eflux_tgyro   mflux_tgyro' in lines[i]:
                # Found the header line, now process the data
                break
        
        line = lines[i+2]
        self.Ge, self.Qe, self.Me = [float(x) for x in line.split()[1:]]
        
        self.GiAll, self.QiAll, self.MiAll = [], [], []
        for i in range(i+3, len(lines)):
            line = lines[i]
            self.GiAll.append(float(line.split()[1]))
            self.QiAll.append(float(line.split()[2]))
            self.MiAll.append(float(line.split()[3]))

        self.GiAll = np.array(self.GiAll)
        self.QiAll = np.array(self.QiAll)
        self.MiAll = np.array(self.MiAll)

        self.Qi = self.QiAll.sum()
        self.Mt = self.Me + self.MiAll.sum()
        
        self.roa = float(lines[0].split()[-1])


        # ------------------------------------------------------------------------
        # Input file
        # ------------------------------------------------------------------------

        with open(self.FolderGACODE / ("input.neo" + self.suffix), "r") as fi:
            lines = fi.readlines()
        self.inputFile = "".join(lines)
        
        
    def unnormalize(self, normalization, rho=None):
        
        if normalization is not None:
            rho_x = normalization["rho"]
            roa_x = normalization["roa"]
            q_gb = normalization["q_gb"]
            g_gb = normalization["g_gb"]
            pi_gb = normalization["pi_gb"]
            s_gb = normalization["s_gb"]
            rho_s = normalization["rho_s"]
            a = normalization["rmin"][-1]

            # ------------------------------------
            # Usage of normalization quantities
            # ------------------------------------

            if rho is None:
                ir = np.argmin(np.abs(roa_x - self.roa))
                rho_eval = rho_x[ir]
            else:
                ir = np.argmin(np.abs(rho_x - rho))
                rho_eval = rho

            self.Qe_unn = self.Qe * q_gb[ir]
            self.Qi_unn = self.Qi * q_gb[ir]
            self.QiAll_unn = self.QiAll * q_gb[ir]
            self.Ge_unn = self.Ge * g_gb[ir]
            self.GiAll_unn = self.GiAll * g_gb[ir]
            self.MiAll_unn = self.MiAll * g_gb[ir]
            self.Mt_unn = self.Mt * s_gb[ir]

            self.unnormalization_successful = True

        else:
            self.unnormalization_successful = False
