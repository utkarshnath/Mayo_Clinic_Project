#!/bin/bash

#SBATCH --get-user-env
#SBATCH -J super
#SBATCH -n 36
#SBATCH -p htcgpu
#SBATCH -q normal

#SBATCH --gres=gpu:4

#SBATCH -t 0-4
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL         # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=unath@asu.edu   # send-to address

conda env list
source activate DAS-copy
module purge
module load cuda/10.2.89
module load cudnn/8.1.0

cd /home/unath/medical_imaging_projects/ModelsGenesis/pytorch-copy

python krasClassification.py
#python siameseClassification.py
#python classification.py
#python mytest.py
#python segmentation.py
