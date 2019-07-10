#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=mem
#SBATCH --partition=gpu
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=mem.%j.out

cd /scratch/`whoami`/GPUClass18/HOL1/

set -o xtrace
cuda-memcheck ./vAdd 1000000 256
