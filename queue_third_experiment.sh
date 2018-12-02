#!/bin/bash

for dataset in cifar10 mnist stl10
do
    for iteration in 1 2 3 4 5 6 7 8 9 10
    do
        sbatch --output=${dataset}_generation_${iteration}.out slurm.sh run_generation.py -name_suffix ${iteration} -dataset ${dataset}
    done
done
