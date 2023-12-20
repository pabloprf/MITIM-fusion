# Note that for this to work, $MITIM_PATH must be defined in previously (e.g. .bashrc)
# Note that before the "echo" calls I added those lines because if not interactive (i.e. scp) then it would fail

# Still allow aliases in non-interactive shells (but only do this when shopt exists)
type shopt &>/dev/null && shopt -s expand_aliases

# Grab host name for both Mac ($HOST) and unix ($HOSTNAME)
if [ -z "$HOSTNAME" ]; then export HOSTNAME=$HOST;fi

! [ -z "$PS1" ] && echo -e "\033[32m>>>>>>>>>>>>>>>>>>>>>>>> MITIM <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\033[0m"
if [ -n "$SLURM_SUBMIT_HOST" ];
then 
	! [ -z "$PS1" ] && echo "You are in machine $HOSTNAME (host: $SLURM_SUBMIT_HOST), user $USER"
else
	! [ -z "$PS1" ] && echo "You are in machine $HOSTNAME, user $USER"
fi

# -------------------------------------------------------------------------------------------------------
# Machine specific
# -------------------------------------------------------------------------------------------------------

! [ -z "$PS1" ] && echo "  1. Loading specific machine environment"

# mfews (mfewsXX.psfc.mit.edu, mferwsXX.psfc.mit.edu)
if [ "${HOSTNAME:0:5}" = mfews ] || [ "${HOSTNAME:0:6}" = mferws ]
then
	! [ -z "$PS1" ] && echo "     - Detected MIT MFE workstation, loading mfews.bashrc"
	source $MITIM_PATH/config/machines_sources/mfews.bashrc
# engaging (eofe7.mit.edu, eofe8.mit.edu)
elif [ "${HOSTNAME:0:4}" = eofe ] || [ "${SLURM_SUBMIT_HOST:0:4}" = eofe ] 
then
	! [ -z "$PS1" ] && echo "     - Detected MIT EOFE cluster, loading eofe.bashrc"
	source $MITIM_PATH/config/machines_sources/eofe.bashrc
# iris GA (irisa.gat.com)
elif [ "${HOSTNAME:0:4}" = iris ]
then
	! [ -z "$PS1" ] && echo "     - Detected GA IRIS machine, loading iris.bashrc"
	source $MITIM_PATH/config/machines_sources/iris.bashrc
# toki IPP
elif [ "${HOSTNAME:0:3}" = toki ]
then
	source $MITIM_PATH/config/machines_sources/toki.bashrc
# None of the above
else
	! [ -z "$PS1" ] && echo "     - No specific environment file loaded"
fi



# -------------------------------------------------------------------------------------------------------
# Aliases for quick tools (plotting, reading)
# -------------------------------------------------------------------------------------------------------

! [ -z "$PS1" ] && echo "  2. Defining useful aliases for MITIM utilities"

source $MITIM_PATH/config/machines_sources/aliases.bashrc

# -------------------------------------------------------------------------------------------------------
# TRANSP Aliases
# -------------------------------------------------------------------------------------------------------

# To run TRANSP (in folder with required files): transp 88664 P01 CMOD --version tshare --trmpi 32 --toricmpi 32 --ptrmpi 32
alias transp="python3 $MITIM_PATH/src/mitim_tools/transp_tools/exe/run_transp.py"

# To send and get LOOK or do a FETCH: look 88664P01 CMOD --nofull --plot --remove
alias look="python3 $MITIM_PATH/src/mitim_tools/transp_tools/exe/run_look.py"

# To check the grid: trcheck pablorf
alias trcheck="python3 $MITIM_PATH/src/mitim_tools/transp_tools/exe/run_check.py"

# To check the grid periodically: trcheckp pablorf
alias trcheckp="python3 $MITIM_PATH/src/mitim_tools/transp_tools/exe/run_check_periodic.py"

# To delete runs from TRANSP grid: cleanruns 88664P CMOD --numbers 1 --to 5   OR 88664P CMOD --numbers 1,2,3 
alias cleanruns="python3 $MITIM_PATH/src/mitim_tools/transp_tools/exe/run_clean.py"

# Plot transp: trplot transp.cdf
alias trplot="ipython3 -i $MITIM_PATH/src/mitim_tools/transp_tools/outputs/exploreCDFs.py"

# -------------------------------------------------------------------------------------------------------
# IM Aliases
# -------------------------------------------------------------------------------------------------------

# To run complete IM evaluation:   runim 7 [DebugOption: --debug 0]
alias runim="python3 $MITIM_PATH/src/mitim_tools/im_tools/exe/run_im.py ./"

# To peform scenario optimization
alias runmitim="python3 $MITIM_PATH/mitim_opt/scenarios_tools/routines/runMITIM_BO.py"

# -------------------------------------------------------------------------------------------------------
# Other Useful Aliases
# -------------------------------------------------------------------------------------------------------

# e.g.: grepy hola
function grepy { clear; grep -rno "$1" . --color --include \*.py; }
function grepf90 { clear; grep -rno "$1" . --color --include \*.{f90,F90}; }

! [ -z "$PS1" ] && echo -e "\033[32m>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\033[0m"
