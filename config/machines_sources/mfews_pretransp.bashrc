
# ------------------------------------------------------------------------------------------------
#       Old TRANSP stuff (pretransp era)
# ------------------------------------------------------------------------------------------------

# ***** IDL

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/labombard/idl_lib/fortran
export IDL_STARTUP=/home/nthoward/idl/idl_startup
export LD_LIBRARY_PATH=/usr/local/mdsplus/lib:$LD_LIBRARY_PATH

# ***** MDS+

export MDS_TRANSP_SERVER=alcdata-transp.psfc.mit.edu
export MDS_TRANSP_TREE=TRANSP
source /usr/local/cmod/codes/transp/pppl_tools/transp_setup.bash

# ***** PRETRANSP

export GLOBUS_LOCATION="/usr/local/fusiongrid"
alias w_pretransp="/usr/local/cmod/codes/transp/pretransp/pretransp-mit"
alias w_posttransp="sh /usr/local/cmod/codes/transp/new_posttransp.sh"
alias mg=’/usr/local/cmod/codes/transp/mg’
export MDS_PATH="$MDS_PATH;/usr/local/cmod/codes/transp/tdi/"
LD_LIBRARY_PATH=/usr/local/cmod/codes/transp/lib:$LD_LIBRARY_PATH


