#!/bin/bash

# # SBATCH Directives
#SBATCH --job-name=trace24
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH -o slurmjob-%j.out-%N # output reporting (%j=jobnumber %N=node)
#SBATCH -e slurmjob-%j.err-%N # error reporting

# #SBATCH --mail-user=alejandro.palomino@utah.edu
# #SBATCH --mail-type=FAIL,BEGIN,END

#SBATCH --account=parvania
#SBATCH --partition=lonepeak

#load necessary modules
module purge
module load python/3.6.3
module load intel impi

#set up the temporary directory
SCRDIR=/scratch/general/lustre/$USER/$SLURM_JOBID
mkdir -p $SCRDIR
cd $SCRDIR

#copy over inputs
SUBSCR=~/Documents/firstSlurm
cp $SUBSCR/hr_day_cnctd.csv $SCRDIR
cp $SUBSCR/trace24.py $SCRDIR

#run the program with input
echo "Evaluating $SLURM_JOB_ID at `date`"

mpirun trace24.py


cp $SCDIR/out_yPred.csv $SUBSCR

echo "Finished $SLURM_JOB_ID at `date`"

cd $SUBSCR
rm -rf $SCRDIR
exit 0
