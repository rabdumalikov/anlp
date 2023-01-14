#!/bin/bash

#SBATCH --partition=gpu
#SBATCH -J t5_scan
#SBATCH -o ./metalog.out # STDOUT

#SBATCH -t 48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:a100-80g:1
##SBATCH --gres=gpu:tesla:1

echo "Started..."

python -u ml_t5_exp1b.py

echo "Ended."
