# Stuff only for MFEWS

# ------------------------------------------------------------------------------------------------
#               TRANSP
# ------------------------------------------------------------------------------------------------

# NTCC ******************************************************************************************************
export NTCC_HOME=/home/sachdev/ntcc/ntcc-23.1.0/

export NTCC_ROOT=$NTCC_HOME
export CONFIGDIR=$NTCC_HOME/etc
export TRANSP_LOCATION=$NTCC_HOME/etc
export PATH=$NTCC_HOME/bin:$NTCC_HOME/etc:$PATH
export INCLUDE_PATH=$NTCC_HOME/include:$NTCC_HOME/inclshare:$NTCC_HOME/preact:$NTCC_HOME/mod:$INCLUDE_PATH
export LD_LIBRARY_PATH=$NTCC_HOME/lib:$LD_LIBRARY_PATH

source /usr/local/cmod/codes/transp/pppl_tools/transp_setup.bash # I still keep this to know what tolower is
unset GLOBUS_LOCATION

# ~~~
# Content of what used to be "source /home/sachdev/ntcc/transp_setup.bash"
# ~~~
export NTCCHOME=$NTCC_HOME
export TRANSP_LOCATION=$NTCC_HOME/etc
export PATH=${NTCCHOME}/bin:${TRANSP_LOCATION}:$PATH

if [ -z "$PREACTDIR" ]; then
   export PREACTDIR=${NTCCHOME}/PREACT
fi
if [ -z "$ADASDIR" ]; then
   export ADASDIR=${NTCCHOME}/ADAS
fi
if [ -z "$UIDPATH" ]; then
   export UIDPATH=${NTCCHOME}/uid/%U
else
   export UIDPATH=${NTCCHOME}/uid/%U:$UIDPATH
fi

export XTRANSPIN=${TRANSP_LOCATION}/xtranspin

if [ -z "$IDL_PATH" ]; then
   export IDL_PATH="+${NTCCHOME}/idl"
else
   export IDL_PATH="+${NTCCHOME}/idl:$IDL_PATH"
fi
if [ -z "$LD_LIBRARY_PATH" ]; then
   export LD_LIBRARY_PATH=${NTCCHOME}/lib
else
   export LD_LIBRARY_PATH=${NTCCHOME}/lib:$LD_LIBRARY_PATH
fi

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ***** Command line
export MDS_TRANSP_SERVER='NONE'
export MDS_TRANSP_TREE='NONE'
export DATADIR=''
export EDITOR=vim

# ***** Alias to load pretransp stuff
alias load_pretransp="source $MITIM_PATH/config/machines_sources/mfews_pretransp.bashrc"

