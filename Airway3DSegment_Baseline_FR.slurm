#!/bin/bash

#SBATCH --job-name=FR_UNet3D_Airway3DSegment
#SBATCH --partition=dgx2
#SBATCH -n 8
#SBATCH --gres=gpu:2
#SBATCH --output=Airway3DSegment_Baseline_FR_%j.out
#SBATCH --error=Airway3DSegment_Baseline_FR_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=frederique.hsu@outlook.com

module load miniconda3

source activate ~/.conda/envs/jupyter-spyder-pytorch-env

which python

python --version

# Training the baseline UNet3D + Feature Recalibration model
python Airway3DSegmentation.py --model=baseline_fr --batch-size=4 --num-workers=3 --save-dir=baseline_fr \
--train-cube-size 80 192 304 --train-stride 64 96 152 --val-cube-size 80 192 304 --val-stride 64 72 72 \
--start-epoch=1 --epochs=60 --sgd=0 --randsel=0 --multi-gpu-parallel=1 \
--encoder-path-ad=0 --decoder-path-ad=0 \
--enable-training=1 --enable-validating=0 --enable-testing=0


# Validating
python Airway3DSegmentation.py --model=baseline_fr --batch-size=4 --num-workers=3 --save-dir=baseline_fr \
--train-cube-size 80 192 304 --train-stride 64 96 152 --val-cube-size 80 192 304 --val-stride 64 72 72 \
--start-epoch=1 --epochs=60 --sgd=0 --randsel=0 --multi-gpu-parallel=1 \
--encoder-path-ad=0 --decoder-path-ad=0 \
--resume="./results/baseline_fr/model_latest.ckpt" --save-feature=1 \
--enable-training=0 --enable-validating=1 --enable-testing=0


# Testing
python Airway3DSegmentation.py --model=baseline_fr --batch-size=4 --num-workers=3 --save-dir=baseline_fr \
--train-cube-size 80 192 304 --train-stride 64 96 152 --val-cube-size 80 192 304 --val-stride 64 72 72 \
--start-epoch=1 --epochs=60 --sgd=0 --randsel=0 --multi-gpu-parallel=1 \
--encoder-path-ad=0 --decoder-path-ad=0 \
--resume="./results/baseline_fr/model_latest.ckpt" --save-feature=1 \
--enable-training=0 --enable-validating=0 --enable-testing=1
