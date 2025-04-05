#!/bin/bash
#SBATCH --job-name=ENEL645_FINAL        # Job name
#SBATCH --output=ENEL645_final_outputs-%j.out              # Output log file
#SBATCH --error=ENEL645_final_errors-%j.log
#SBATCH --partition=gpu                    # Request the GPU partition
#SBATCH --gres=gpu:1                       # Request 1 GPU (Tesla T4, for example)
#SBATCH --time=8:00:00                    # Max run time (hh:mm:ss)
#SBATCH --mem=32G                          # Memory requirement
#SBATCH --ntasks=1                         # Number of tasks (processes)
#SBATCH --cpus-per-task=2              # Number of CPUs per task
# Activate conda environment
source ~/miniforge3/bin
conda activate final_project
cd ~/software/src/enel645_project/
python project.py
