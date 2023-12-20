# Stuff only for IRIS machine

# ------------------------------------------------------------------------------------------------
#               GACODE
# ------------------------------------------------------------------------------------------------

# module load atom/pygacode

! [ -z "$PS1" ] && echo "              * Proceeding to load GACODE modules"

export GACODE_PLATFORM=IRIS
export GACODE_ROOT=/home/$USER/gacode

. ${GACODE_ROOT}/shared/bin/gacode_setup
#. ${GACODE_ROOT}/platform/env/env.${GACODE_PLATFORM}   # IMPORTANT to not do this, otherwise it will then point to atom

if [ $? -eq 124 ]
then
    ! [ -z "$PS1" ] && echo -e "\033[31m                   * GACODE modules could not be loaded\033[0m"
else
	! [ -z "$PS1" ] && echo "                   * GACODE modules loaded"
fi

# ------------------------------------------------------------------------------------------------
#               TRANSP/NTCC
# ------------------------------------------------------------------------------------------------

module load omfit/unstable
module load ntcc
