# ---------------------------------------------------------------------------------------------------------------------
# Useful aliases for MITIM repository
# ---------------------------------------------------------------------------------------------------------------------

# mitim_tools interfaces: read, run, plot

alias mitim_plot_gacode="ipython3 -i -- $MITIM_PATH/src/mitim_tools/gacode_tools/scripts/read_gacode.py $1"
alias mitim_plot_tgyro="ipython3 -i -- $MITIM_PATH/src/mitim_tools/gacode_tools/scripts/read_tgyro.py $1"
alias mitim_plot_tglf="ipython3 -i -- $MITIM_PATH/src/mitim_tools/gacode_tools/scripts/read_tglf.py $1" 		 # [--suffix _0.55] [--gacode input.gacode]
alias mitim_plot_cgyro="ipython3 -i -- $MITIM_PATH/src/mitim_tools/gacode_tools/scripts/read_cgyro.py $1"
alias mitim_plot_eq="ipython3 -i -- $MITIM_PATH/src/mitim_tools/gs_tools/scripts/read_eq.py $1"
alias mitim_plot_transp="ipython3 -i -- $MITIM_PATH/src/mitim_tools/transp_tools/scripts/read_transp.py $1"

alias mitim_run_tglf="ipython3 -i -- $MITIM_PATH/src/mitim_tools/gacode_tools/scripts/run_tglf.py $1 $2" # (folder input.tglf)  [--gacode input.gacode] [--scan RLTS_2] [--drives True]

# Optimizations
alias mitim_plot_opt="ipython3 -i -- $MITIM_PATH/src/mitim_tools/opt_tools/scripts/read.py --type 4 --resolution 20 $1"
alias mitim_plot_portals="ipython3 -i -- $MITIM_PATH/src/mitim_modules/portals/scripts/read_portals.py $1"
alias mitim_slurm="python3 $MITIM_PATH/src/mitim_tools/opt_tools/scripts/slurm.py $1"

# TRANSP
alias mitim_trcheck="python3 $MITIM_PATH/src/mitim_tools/transp_tools/scripts/run_check.py $1"		        # mitim_trcheck pablorf
alias mitim_trcheck_p="python3 $MITIM_PATH/src/mitim_tools/transp_tools/scripts/run_check_periodic.py $1"	# mitim_trcheck_p pablorf
alias mitim_trclean="python3 $MITIM_PATH/src/mitim_tools/transp_tools/scripts/run_clean.py $1" 		        # mitim_trclean 88664P CMOD --numbers 1,2,3
alias mitim_trlook="ipython3 -i -- $MITIM_PATH/src/mitim_tools/transp_tools/scripts/run_look.py $1"         # mitim_trlook 152895P01 CMOD --nofull --plot --remove

# To run TRANSP (in folder with required files): transp 88664 P01 CMOD --version tshare --trmpi 32 --toricmpi 32 --ptrmpi 32
alias transp="python3 $MITIM_PATH/src/mitim_tools/transp_tools/scripts/run_transp.py"

# IM Aliases
alias runim="python3 $MITIM_PATH/src/mitim_tools/im_tools/scripts/run_im.py ./"             # To run complete IM evaluation:   runim 7 [DebugOption: --debug 0]
alias runmitim="python3 $MITIM_PATH/mitim_opt/scenarios_tools/scripts/runMITIM_BO.py"  # To peform scenario optimization

# Others
alias compare_nml="python3 $MITIM_PATH/src/mitim_tools/transp_tools/scripts/compareNML.py $1"
alias eff_job="python3 $MITIM_PATH/src/mitim_tools/misc_tools/PARALLELtools.py $1" # Give mitim.out or slurm_output.dat
