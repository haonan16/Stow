#!/usr/bin/env bash

#SBATCH --job-name=robocook
#SBATCH --account=viscam
#SBATCH --partition=svl
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/sailhome/hshi74/output/robocook/%A.out

python dynamics/eval.py \
	--stage dy \
	--skill_name $1 \
    --dy_model_path $2 