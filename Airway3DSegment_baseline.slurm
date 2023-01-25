#!/bin/bash

#SBATCH --job-name=airway3dseg_baseline
#SBATCH --partition=dgx2
#SBATCH -n 8
#SBATCH --gres=gpu:2
#SBATCH --output=airway3dseg_baseline_%j.out
#SBATCH --error=airway3dseg_baseline_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=frederique.hsu@outlook.com

module load miniconda3

source activate ~/.conda/envs/jupyter-spyder-pytorch-env

which python

python --version

# Training the baseline UNet3D model
python Airway3DSegmentation.py --model=baseline --batch-size=1 --num-workers=0 --save-dir=baseline \
--train-cube-size 80 192 304 --train-stride 64 96 152 --val-cube-size 80 192 304 --val-stride 64 72 72 \
--start-epoch=1 --epochs=60 --sgd=0 --randsel=0 --multi-gpu-parallel=True \
--encoder-path-ad=False --decoder-path-ad=False