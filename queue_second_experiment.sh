#!/bin/bash

for iteration in 1 2 3 4 5 6 7 8 9 10
do
    sbatch --output=$0_scale+translation_${iteration}.out slurm.sh run_training.py -augmentations scale translation -name_suffix ${iteration} -dataset $0
    sbatch --output=$0_rotation+scale_${iteration}.out slurm.sh run_training.py -augmentations rotation scale -name_suffix ${iteration} -dataset $0
    sbatch --output=$0_rotation+translation_${iteration}.out slurm.sh run_training.py -augmentations rotation translation -name_suffix ${iteration} -dataset $0
    sbatch --output=$0_rotation+scale+translation_${iteration}.out slurm.sh run_training.py -augmentations rotation scale translation -name_suffix ${iteration} -dataset $0
    sbatch --output=$0_flip+rotation+scale+translation_${iteration}.out slurm.sh run_training.py -augmentations flip rotation scale translation -name_suffix ${iteration} -dataset $0
done
