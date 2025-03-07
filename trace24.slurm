#!/bin/bash -l

#####################

# SBATCH Directives
#SBATCH --job-name=trace24

# n jobs will run in this array at the same time
# #SBATCH --array=1-8

# run for n minutes
#              d-hh:mm:ss
#SBATCH --time=1-00:00:00

# 500MB memory per core
# this is a hard limit
# SBATCH --mem-per-cpu=500MB

#SBATCH -o results/trace_job-%j.out # output reporting (%j=jobnumber %N=node)
#SBATCH -e results/trace_job-%j.err # error reporting

#SBATCH --mail-user=alejandro.palomino@utah.edu
#SBATCH --mail-type=FlAIL,BEGIN,END

#SBATCH --account=parvania
#SBATCH --partition=lonepeak

#load necessary modules
module purge
module use $HOME/MyModules
module load miniconda3/latest
# module load python/3.6.3
# export PYTHONPATH=/uufs/chpc.utah.edu/common/home/u1141243/VENV3.6.3/lib/python3.6/site-packages:$PYTHONPATH
# export PYTHONPATH=/uufs/chpc.utah.edu/common/home/u1141243/software/pkg/miniconda3/lib/python3.7/site-packages/:$PYTHONPATH
# module load intel impi

# define and create a unique scratch directory
SCRATCH_DIRECTORY=/scratch/general/lustre/${USER}/${SLURM_JOBID}
# rm -rf /scratch/general/lustre/${USER}
mkdir -p ${SLURM_SUBMIT_DIR}/results/${SLURM_JOBID}/
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# SLURM_SUBMIT_DIR=~/Documents/firstSlurm
cp ${SLURM_SUBMIT_DIR}/trace24.py ${SCRATCH_DIRECTORY}
cp ${SLURM_SUBMIT_DIR}/hdc_wkdy20.csv ${SCRATCH_DIRECTORY}

# each job will see a different ${SLURM_ARRAY_TASK_ID}
echo "Evaluating $SLURM_JOB_ID at `date`"
echo "now processing ..... " ${SLURM_ARRAY_TASK_ID}

python trace24.py > output.txt

echo "Finished $SLURM_JOB_ID at `date`"

# after the job is done we copy our output back to $SLURM_SUBMIT_DIR
cp -r ${SCRATCH_DIRECTORY}/out* ${SLURM_SUBMIT_DIR}/results/${SLURM_JOBID}/

# we step out of the scratch directory and remove it
cd ${SLURM_SUBMIT_DIR}
# rm -rf ${SCRATCH_DIRECTORY}

# happy end
exit 0
