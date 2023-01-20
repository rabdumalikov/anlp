#!/bin/bash

#SBATCH --partition=gpu
#SBATCH -J t5_k2_1e5
#SBATCH -o ./_1e5_k2_5000_log.out # STDOUT

#SBATCH -t 72:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:a100-40g:1
##SBATCH --gres=gpu:tesla:1

echo "Started..."

python -u t5_exp1b.py
#python -u t5_exp1b.py

echo "Ended."
