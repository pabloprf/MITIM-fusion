# This is a setup file for the IRIS cluster. Remember (see Installation guide) that this is not
# stricly necessary. However, it is convenient to have it here, so that you can load the modules
# with a single command. If you do not want to use this file, you can load the modules manually
# if you have these lines in your .bashrc (or equivalent)

# ------------------------------------------------------------------------------------------------
#               GACODE
# ------------------------------------------------------------------------------------------------

! [ -z "$PS1" ] && echo "              * Proceeding to load GACODE modules"

export GACODE_PLATFORM=IRIS
export GACODE_ROOT=/home/$USER/gacode
. ${GACODE_ROOT}/shared/bin/gacode_setup
module load ntcc

#. ${GACODE_ROOT}/platform/env/env.${GACODE_PLATFORM}   # IMPORTANT to not do this, otherwise it will then point to atom

if [ $? -eq 124 ]
then
    ! [ -z "$PS1" ] && echo -e "\033[31m                   * GACODE modules could not be loaded\033[0m"
else
	! [ -z "$PS1" ] && echo "                   * GACODE modules loaded"
fi
