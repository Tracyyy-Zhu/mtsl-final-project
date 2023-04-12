#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=/scratch/bc3088/mstats/log/%j_%x.out
#SBATCH --error=/scratch/bc3088/mstats/log/%j_%x.err
#SBATCH --account=ds_ga_3001_011-2023sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --requeue

#SBATCH --mail-type=ALL
#SBATCH --mail-user=bale.chen@nyu.edu


# Problem 2a: Write a script to be executed by your HPC job in order to fine-tune a BERT-tiny model.

singularity exec --nv --bind /scratch/bc3088/ --overlay /scratch/bc3088/overlay-25GB-500K.ext3:rw /scratch/bc3088/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "source /ext3/env.sh; conda activate; python /scratch/bc3088/mstats/project/src/train_model.py"