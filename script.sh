#!/bin/bash
#SBATCH --job-name=multiGNN
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --time=47:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=gpupart_p100
#SBATCH --gpus=1

python main.py --data Small_HI --model gin --focal_loss --seed 1 --alpha 0.75 --gamma 1.3