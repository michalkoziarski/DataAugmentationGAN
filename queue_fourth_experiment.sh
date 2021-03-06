#!/bin/bash

for dataset in cifar10 mnist stl10
do
    for iteration in 1 2 3 4 5 6 7 8 9 10
    do
        for n in 500 1000 2000 4000 8000 16000 32000 64000
        do
            sbatch --output=${dataset}_using_generated_${n}_${iteration}.out slurm.sh run_classification.py -name_suffix ${iteration} -dataset ${dataset} -n_generated_images ${n}
        done
    done
done
