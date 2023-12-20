
module use /home/software/psfc/modulefiles/

# ************************************************************************************
# Machines: eofe7.mit.edu, eofe8.mit.edu
# ************************************************************************************

if [ "${HOSTNAME:0:5}" = eofe7 ] || [ "${HOSTNAME:0:5}" = eofe8 ] || [ "${SLURM_SUBMIT_HOST:0:5}" = eofe7 ] || [ "${SLURM_SUBMIT_HOST:0:5}" = eofe8 ]
then
	! [ -z "$PS1" ] && echo "              * EOFE7/8"

	# --------------------------------------------------------------------------------
	#               SLURM
	# --------------------------------------------------------------------------------

	module load slurm

	export MITIM_PARTITION=sched_mit_psfc

	# --------------------------------------------------------------------------------
	#               GACODE
	# --------------------------------------------------------------------------------
	# folder: 	/nfs/psfclab001/software/gacode
	# module: 	/home/software/psfc/modulefiles/psfc/gacode/b0bd3767

	module unload python
	# module load python/3.9.4 # Let's have the GACODE env file load its own python

	# module load psfc/gacode/b0bd3767

	module load intel/2017-01 impi psfc/mkl/17
	export GACODE_PLATFORM=PSFCLUSTER
	export GACODE_ROOT=/home/$USER/gacode

	# --------------------------------------------------------------------------------
	#               TRANSP
	# --------------------------------------------------------------------------------
	# folder: 	/nfs/psfclab001/software/transp/
	# module: 	/home/software/psfc/modulefiles/psfc/transp/23.1

	module load psfc/transp/23.1

# ************************************************************************************
# Machines: eofe10.mit.edu
# ************************************************************************************

elif [ "${HOSTNAME:0:6}" = eofe10 ] || [ "${SLURM_SUBMIT_HOST:0:6}" = eofe10 ] || [ "${HOSTNAME:0:5}" = eofe4 ] || [ "${SLURM_SUBMIT_HOST:0:5}" = eofe4 ] 
then
	! [ -z "$PS1" ] && echo "              * EOFE4/10"

	# --------------------------------------------------------------------------------
	#               SLURM
	# --------------------------------------------------------------------------------

	export MITIM_PARTITION=sched_mit_psfc_r8

	# GPU
	alias nvi='nvidia-smi pmon -i 0'
	alias slgpu='python3 $MITIM_PATH/mitim_opt/opt_tools/exe/slurm.py $1 --partition sched_mit_psfc_gpu_r8 $2'
	alias sagpu='salloc --partition sched_mit_psfc_gpu_r8 --nodes=1 --exclusive --time=8:00:00'

	# --------------------------------------------------------------------------------
	#               GACODE
	# --------------------------------------------------------------------------------

	export GACODE_PLATFORM=PSFC_EOFE_RPP
	export GACODE_ROOT=/home/$USER/gacode_sparc

	# --------------------------------------------------------------------------------
	#               ASTRA
	# --------------------------------------------------------------------------------

	! [ -z "$PS1" ] && echo "              * Proceeding to load ASTRA modules"
	#timeout 5s bash -c '{
	module use /orcd/nese/psfc/001/software/spack/2023-07-01-physics-rpp/spack/share/spack/modules-test/linux-rocky8-x86_64
	module load intel-oneapi-compilers/2023.1.0-gcc-12.2.0-module-3vfzgf
	module load intel-oneapi-mkl/2023.1.0-intel-oneapi-mpi-2021.9.0-gcc-12.2.0-module-seow5n
	module load anaconda3/2022.05-x86_64
	module load netcdf-fortran/4.6.0-intel-2021.9.0-module-2v44tym
	module use /home/gtardini/modulefiles
	module load astra
	#}'
	if [ $? -eq 124 ]
	then
	    ! [ -z "$PS1" ] && echo -e "\033[31m                   * ASTRA modules could not be loaded\033[0m"
	else
		! [ -z "$PS1" ] && echo "                   * ASTRA modules loaded"
	fi

fi

# --------------------------------------------------------------------------------
#               GACODE
# --------------------------------------------------------------------------------

! [ -z "$PS1" ] && echo "              * Proceeding to load GACODE modules"
# timeout 5s bash -c '{
. ${GACODE_ROOT}/shared/bin/gacode_setup
. ${GACODE_ROOT}/platform/env/env.${GACODE_PLATFORM}
# }'
if [ $? -eq 124 ]
then
    ! [ -z "$PS1" ] && echo -e "\033[31m                   * GACODE modules could not be loaded\033[0m"
else
	! [ -z "$PS1" ] && echo "                   * GACODE modules loaded"
fi

# --------------------------------------------------------------------------------
#               SLURM
# --------------------------------------------------------------------------------

export MITIM_PARTITIONS_ALL="sched_mit_psfc,sched_mit_nse,sched_mit_psfc_r8,sched_mit_psfc_gpu_r8"

source $MITIM_PATH/config/machines_sources/slurm_aliases.bashrc
