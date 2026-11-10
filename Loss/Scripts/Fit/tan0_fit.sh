#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N Tan0_Fit
#$ -q smoshe.q
#$ -pe shared 20
#$ -l h_vmem=16G
#$ -o /dev/null
#$ -e /dev/null

# 1. Setup Logs
LOGS="Loss/Logs/Al-Sapp/6789_MHz/5V/Fit"
OUTPUTS="Loss/Outputs/Al-Sapp/6789_MHz/5V/Fit"
mkdir -p ${LOGS}
mkdir -p ${OUTPUTS}

# 2. Initialize Conda
source ~/.bashrc
conda activate freq_shift_env

# 3. Force prioritizing environment's libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 4. Thread Clamping (CRUCIAL)
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS=1
export OPENBLAS_CORETYPE=Haswell

# 5. Bypass MPI Hardware
export FI_PROVIDER=tcp
export I_MPI_OFI_PROVIDER=tcp
export OMPI_MCA_btl="^openib,ofi"
export OMPI_MCA_mtl="^ofi"

# 6. Define Specific Output Logs
LOG_FILE="${LOGS}/Tan0_Fit.log"

echo "--- JOB STARTING AT $(date) ON HOST $HOSTNAME ---" > $LOG_FILE

# 6. Execute with MPI
mpirun -np $NSLOTS python -u Loss/Scripts/Fit/tan0_fit.py >> $LOG_FILE 2>&1

echo "--- JOB FINISHED AT $(date) ---" >> $LOG_FILE
