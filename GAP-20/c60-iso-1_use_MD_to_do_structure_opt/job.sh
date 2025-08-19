#!/bin/bash
#SBATCH --job-name=c60-iso-1_opt.job
#SBATCH --mail-user=mail@ufl.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --error=c60-iso-1_opt.err
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1gb
#SBATCH --account=mingjieliu
#SBATCH --qos=mingjieliu-b
#SBATCH --distribution=cyclic:cyclic

module purge
module load gcc/12.2.0 openmpi/4.1.5 gsl/2.7 mkl/2020.0.166
ml lammps/02Aug23

LAMMPS=lmp_mpi
INPUT=c60-iso-1_opt.lmp

mpiexec $LAMMPS -in $INPUT > c60-iso-1_opt.out 2>&1
