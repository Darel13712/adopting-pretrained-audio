#!/bin/bash
#SBATCH --job-name="mule-emb"
#SBATCH --time=120:00:00
#SBATCH --mem=30G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

module load libsndfile
module load ffmpeg
source ~/.e/bin/activate
python ./mule_emb.py
