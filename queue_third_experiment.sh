#!/bin/bash

for dataset in cifar10 mnist stl10
do
    sbatch --output=${dataset}_generation.out slurm.sh run_generation.py -dataset ${dataset}
done
