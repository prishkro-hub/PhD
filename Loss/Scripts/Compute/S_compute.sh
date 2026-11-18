#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N S_Comp
#$ -q smoshe.q
#$ -pe shared 20
#$ -l h_vmem=16G
#$ -o /dev/null
#$ -e /dev/null

# 1. Initialize Conda
source ~/.bashrc
conda activate freq_shift_env

# 2. Fetch & setup Logs/Outputs from config
eval "$(python Loss/Scripts/project_config.py Compute)"

mkdir -p "${LOGS}" "${OUTPUTS}"

# 3. Force prioritizing environment's libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 4. Thread Clamping (CRUCIAL)
export OPENBLAS_NUM_THREADS=2
export NUMBA_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS=2
export OPENBLAS_CORETYPE=Haswell

# 5. Bypass MPI Hardware
export FI_PROVIDER=tcp
export I_MPI_OFI_PROVIDER=tcp
export OMPI_MCA_btl="^openib,ofi"
export OMPI_MCA_mtl="^ofi"

# 6. Define Specific Output Logs
LOG_FILE="${LOGS}/S_Compute.log"

echo "--- JOB STARTING AT $(date) ON HOST $HOSTNAME ---" > $LOG_FILE

# 6. Execute
python -u Loss/Scripts/Compute/S_compute.py >> $LOG_FILE 2>&1

echo "--- JOB FINISHED AT $(date) ---" >> $LOG_FILE
