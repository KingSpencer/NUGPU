#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=nvprof
#SBATCH --partition=gpu-class
#SBATCH --mem=8Gb
#SBATCH --gres=gpu:k20:1
#SBATCH --output=nvprof.%j.out

#cd /scratch/`whoami`/GPUClass18/HOL1/

set -o xtrace
echo "NORMAL RUN"
nvprof ./vAdd 100000000 512

echo "GPU TRACE"
nvprof --print-gpu-trace ./vAdd 100000000 512

echo "METRICS"
nvprof -m all ./vAdd 100000000 512
