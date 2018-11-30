#!/bin/bash

for iteration in 1 2 3 4 5 6 7 8 9 10
do
    sbatch --output=$0_none_${iteration}.out slurm.sh run_classification.py -name_suffix ${iteration} -dataset $0

    sbatch --output=$0_flip_${iteration}.out slurm.sh run_classification.py -augmentations flip -name_suffix ${iteration} -dataset $0

    for param in 10 20 30 40 50
    do
        sbatch --output=$0_rotation_${param}_${iteration}.out slurm.sh run_classification.py -augmentations rotation -rotation_range ${param} -name_suffix ${param}_${iteration} -dataset $0
    done

    for param in 1.2 1.4 1.6 1.8 2.0
    do
        sbatch --output=$0_scale_${param}_${iteration}.out slurm.sh run_classification.py -augmentations scale -scale_range ${param} -name_suffix ${param}_${iteration} -dataset $0
    done

    for param in 0.1 0.2 0.3 0.4 0.5
    do
        sbatch --output=$0_translation_${param}_${iteration}.out slurm.sh run_classification.py -augmentations translation -translation_range ${param} -name_suffix ${param}_${iteration} -dataset $0
    done

    sbatch --output=$0_color_${iteration}.out slurm.sh run_classification.py -augmentations color -name_suffix ${iteration} -dataset $0

    for param in 1 2 4 8 16
    do
        sbatch --output=$0_gaussian_noise_${param}_${iteration}.out slurm.sh run_classification.py -augmentations gaussian_noise -gaussian_noise_std ${param} -name_suffix ${param}_${iteration} -dataset $0
    done

    for param in 0.1 0.01 0.001 0.0001 0.00001
    do
        sbatch --output=$0_snp_noise_${param}_${iteration}.out slurm.sh run_classification.py -augmentations snp_noise -snp_noise_probability ${param} -name_suffix ${param}_${iteration} -dataset $0
    done
done
