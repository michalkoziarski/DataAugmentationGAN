#!/bin/bash

for iteration in 1 2 3 4 5 6 7 8 9 10
do
    sbatch --output=slurm_none_${iteration}.out slurm.sh run_training.py -name_suffix ${iteration}

    sbatch --output=slurm_flip_${iteration}.out slurm.sh run_training.py -augmentations flip -name_suffix ${iteration}

    for param in 5 10 15 20 25
    do
        sbatch --output=slurm_rotation_${param}_${iteration}.out slurm.sh run_training.py -augmentations rotation -name_suffix ${param}_${iteration}
    done

    for param in 1.1 1.2 1.3 1.4 1.5
    do
        sbatch --output=slurm_scale_${param}_${iteration}.out slurm.sh run_training.py -augmentations scale -name_suffix ${param}_${iteration}
    done

    for param in 0.1 0.2 0.3 0.4 0.5
    do
        sbatch --output=slurm_translation_${param}_${iteration}.out slurm.sh run_training.py -augmentations translation -name_suffix ${param}_${iteration}
    done

    sbatch --output=slurm_color_${iteration}.out slurm.sh run_training.py -augmentations color -name_suffix ${iteration}

    for param in 1 2 4 8 16
    do
        sbatch --output=slurm_gaussian_noise_${param}_${iteration}.out slurm.sh run_training.py -augmentations gaussian_noise -name_suffix ${param}_${iteration}
    done

    for param in 0.1 0.01 0.001 0.0001 0.00001
    do
        sbatch --output=slurm_snp_noise_${param}_${iteration}.out slurm.sh run_training.py -augmentations snp_noise -name_suffix ${param}_${iteration}
    done
done