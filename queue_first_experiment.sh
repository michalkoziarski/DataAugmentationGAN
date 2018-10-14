#!/bin/bash

for iteration in 1 2 3 4 5 6 7 8 9 10
do
    sbatch --output=slurm_none_${iteration}.out slurm.sh run_training.py -name_suffix ${iteration}

    sbatch --output=slurm_flip_${iteration}.out slurm.sh run_training.py -augmentations flip -name_suffix ${iteration}

    for param in 10 20 30 40 50
    do
        sbatch --output=slurm_rotation_${param}_${iteration}.out slurm.sh run_training.py -augmentations rotation -rotation_range ${param} -name_suffix ${param}_${iteration}
    done

    for param in 1.2 1.4 1.6 1.8 2.0
    do
        sbatch --output=slurm_scale_${param}_${iteration}.out slurm.sh run_training.py -augmentations scale -scale_range ${param} -name_suffix ${param}_${iteration}
    done

    for param in 0.1 0.2 0.3 0.4 0.5
    do
        sbatch --output=slurm_translation_${param}_${iteration}.out slurm.sh run_training.py -augmentations translation -translation_range ${param} -name_suffix ${param}_${iteration}
    done

    sbatch --output=slurm_color_${iteration}.out slurm.sh run_training.py -augmentations color -name_suffix ${iteration}

    for param in 1 2 4 8 16
    do
        sbatch --output=slurm_gaussian_noise_${param}_${iteration}.out slurm.sh run_training.py -augmentations gaussian_noise -gaussian_noise_std ${param} -name_suffix ${param}_${iteration}
    done

    for param in 0.1 0.01 0.001 0.0001 0.00001
    do
        sbatch --output=slurm_snp_noise_${param}_${iteration}.out slurm.sh run_training.py -augmentations snp_noise -snp_noise_probability ${param} -name_suffix ${param}_${iteration}
    done
done