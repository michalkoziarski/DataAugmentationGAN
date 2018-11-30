#!/bin/bash

for dataset in cifar10 mnist stl10
do
    for iteration in 1 2 3 4 5 6 7 8 9 10
    do
        sbatch --output=${dataset}_scale+translation_${iteration}.out slurm.sh run_training.py -augmentations scale translation -name_suffix ${iteration} -dataset ${dataset}
        sbatch --output=${dataset}_rotation+scale_${iteration}.out slurm.sh run_training.py -augmentations rotation scale -name_suffix ${iteration} -dataset ${dataset}
        sbatch --output=${dataset}_rotation+translation_${iteration}.out slurm.sh run_training.py -augmentations rotation translation -name_suffix ${iteration} -dataset ${dataset}
        sbatch --output=${dataset}_rotation+scale+translation_${iteration}.out slurm.sh run_training.py -augmentations rotation scale translation -name_suffix ${iteration} -dataset ${dataset}
        sbatch --output=${dataset}_flip+rotation+scale+translation_${iteration}.out slurm.sh run_training.py -augmentations flip rotation scale translation -name_suffix ${iteration} -dataset ${dataset}
    done
done
