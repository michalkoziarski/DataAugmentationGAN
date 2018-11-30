#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu
#SBATCH --mem=20000

module add plgrid/tools/python/3.6.0
module add plgrid/apps/cuda/9.0

python3 ${1} ${@:2}
