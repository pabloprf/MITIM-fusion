# ---------------------------------------------------------------------------------------------------------------------
# Useful aliases for MITIM repository
# ---------------------------------------------------------------------------------------------------------------------

# mitim_tools interfaces: read, run, plot

alias mitim_plot_gacode="ipython3 -i -- $MITIM_PATH/src/mitim_tools/gacode_tools/exe/read_gacodes.py --files $1"
alias mitim_plot_tgyro="ipython3 -i -- $MITIM_PATH/src/mitim_tools/gacode_tools/exe/read_tgyros.py --folders $1"
alias mitim_plot_tglf="ipython3 -i -- $MITIM_PATH/src/mitim_tools/gacode_tools/exe/read_tglf.py --folder $1" 		 # [--suffix _0.55] [--gacode input.gacode]
alias mitim_plot_eq="ipython3 -i -- $MITIM_PATH/src/mitim_tools/gs_tools/exe/read_eqs.py --files $1"
alias mitim_plot_cgyro="ipython3 -i -- $MITIM_PATH/src/mitim_tools/gacode_tools/exe/plot_cgyro.py $1"

alias mitim_read_transp="ipython3 -i -- $MITIM_PATH/src/mitim_tools/transp_tools/exe/read_transp.py --files $1"
alias mitim_run_tglf="ipython3 -i -- $MITIM_PATH/src/mitim_tools/gacode_tools/exe/run_tglf.py --folder $1 --tglf $2" #[--gacode input.gacode] [--scan RLTS_2] [--drives True]

# Optimizations
alias mitim_plot_opt="ipython3 -i -- $MITIM_PATH/src/mitim_tools/opt_tools/exe/read.py --type 4 --resolution 20 --folders $1"
alias mitim_plot_portals="ipython3 -i -- $MITIM_PATH/src/mitim_modules/portals/exe/readMetrics.py $1"
alias mitim_slurm="python3 $MITIM_PATH/src/mitim_tools/opt_tools/exe/slurm.py $1"

# TRANSP
alias mitim_trcheck="python3 $MITIM_PATH/src/mitim_tools/transp_tools/exe/run_check.py $1"		# mitim_trcheck pablorf
alias mitim_trclean="python3 $MITIM_PATH/src/mitim_tools/transp_tools/exe/run_clean.py $1" 		# mitim_trclean 88664P CMOD --numbers 1,2,3
alias mitim_trlook="ipython3 -i -- $MITIM_PATH/src/mitim_tools/transp_tools/exe/run_look.py $1" # mitim_trlook 152895P01 CMOD --nofull --plot --remove

# Others
alias compare_nml="python3 $MITIM_PATH/src/mitim_tools/transp_tools/routines/compareNML.py $1"
alias eff_job="python3 $MITIM_PATH/src/mitim_tools/misc_tools/PARALLELtools.py $1" # Give sbatch.out or slurm_output.dat
