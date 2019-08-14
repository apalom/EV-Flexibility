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

# # Envionment Setup
INPUTFILE=hr_day_cnctd.xlsx
DATADIR=/uufs/chpc.utah.edu/common/home/u1141243/Documents/firstSlurm/data
OUTDIR=/uufs/chpc.utah.edu/common/home/u1141243/Documents/firstSlurm/results

# # Clean Up Environment
module purge
module spider python/3.6.3
module load intel impi

# # Scratch Directory Setup
SCRATCH=/scratch/general/lustre/u1141243/WORKING
mkdir -p $SCRATCH

# # Run (run module, number processes, prgm name, input file, output file)
mpirun -np $SLURM_NTASKS trace24.py -inp $DATADIR/$INPUTFILE.in > $OUTDIR/output.txt
# #mpirun -np $SLURM_NTASKS python simple.py > $OUTDIR/output.txt

# # Copy from Scratch and Clean Up
#rm -rf $SCRATCH # clean up scratch
