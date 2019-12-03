#!/bin/bash

#SBATCH -J SI_1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1600mb 
#SBATCH --nodes=1

module purge
module load mpi/mpich-x86_64

srun --mpi=pmi2 python main.py 2> err
