# This is a setup file for NERSC Perlmutter. Remember (see Installation guide) that this is not
# stricly necessary. However, it is convenient to have it here, so that you can load the modules
# with a single command. If you do not want to use this file, you can load the modules manually
# if you have these lines in your .bashrc (or equivalent)

if [ "$NERSC_HOST" = perlmutter ]
then
	! [ -z "$PS1" ] && echo "              * PERLMUTTER"

    # ------------------------------------------------------------------------------------------------
    #               GACODE
    # ------------------------------------------------------------------------------------------------

    export GACODE_PLATFORM=PERLMUTTER_GPU
    export GACODE_ROOT=$HOME/gacode
    . $GACODE_ROOT/shared/bin/gacode_setup
    . ${GACODE_ROOT}/platform/env/env.$GACODE_PLATFORM
fi

source $MITIM_PATH/config/machines_sources/slurm_aliases.bashrc