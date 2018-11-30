#!/bin/bash

for dataset in cifar10 mnist stl10
do
    for iteration in 1 2 3 4 5 6 7 8 9 10
    do
        sbatch --output=${dataset}_none_${iteration}.out slurm.sh run_classification.py -name_suffix ${iteration} -dataset ${dataset}
    
        sbatch --output=${dataset}_flip_${iteration}.out slurm.sh run_classification.py -augmentations flip -name_suffix ${iteration} -dataset ${dataset}
    
        for param in 10 20 30 40 50
        do
            sbatch --output=${dataset}_rotation_${param}_${iteration}.out slurm.sh run_classification.py -augmentations rotation -rotation_range ${param} -name_suffix ${param}_${iteration} -dataset ${dataset}
        done
    
        for param in 1.2 1.4 1.6 1.8 2.0
        do
            sbatch --output=${dataset}_scale_${param}_${iteration}.out slurm.sh run_classification.py -augmentations scale -scale_range ${param} -name_suffix ${param}_${iteration} -dataset ${dataset}
        done
    
        for param in 0.1 0.2 0.3 0.4 0.5
        do
            sbatch --output=${dataset}_translation_${param}_${iteration}.out slurm.sh run_classification.py -augmentations translation -translation_range ${param} -name_suffix ${param}_${iteration} -dataset ${dataset}
        done
    
        sbatch --output=${dataset}_color_${iteration}.out slurm.sh run_classification.py -augmentations color -name_suffix ${iteration} -dataset ${dataset}
    
        for param in 1 2 4 8 16
        do
            sbatch --output=${dataset}_gaussian_noise_${param}_${iteration}.out slurm.sh run_classification.py -augmentations gaussian_noise -gaussian_noise_std ${param} -name_suffix ${param}_${iteration} -dataset ${dataset}
        done
    
        for param in 0.1 0.01 0.001 0.0001 0.00001
        do
            sbatch --output=${dataset}_snp_noise_${param}_${iteration}.out slurm.sh run_classification.py -augmentations snp_noise -snp_noise_probability ${param} -name_suffix ${param}_${iteration} -dataset ${dataset}
        done
    done
done
