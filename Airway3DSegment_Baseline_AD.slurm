#!/bin/bash

#SBATCH --job-name=AD_UNet3D_Airway3DSegment
#SBATCH --partition=dgx2
#SBATCH -n 8
#SBATCH --gres=gpu:2
#SBATCH --output=Airway3DSegment_Baseline_AD_%j.out
#SBATCH --error=Airway3DSegment_Baseline_AD_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=frederique.hsu@outlook.com

module load miniconda3

source activate ~/.conda/envs/my-dissertation-env

which python3

python3 --version

# Training the baseline UNet3D + Attention Distillation model
python3 Airway3DSegmentation.py --model=baseline --batch-size=4 --num-workers=3 --save-dir=baseline_ad \
--train-cube-size 80 192 304 --train-stride 64 96 152 --val-cube-size 80 192 304 --val-stride 64 72 72 \
--start-epoch=1 --epochs=60 --sgd=0 --randsel=0 --multi-gpu-parallel=1 \
--encoder-path-ad=0 --decoder-path-ad=1 \
--enable-training=1 --enable-validating=0 --enable-testing=0


# Validating
python3 Airway3DSegmentation.py --model=baseline --batch-size=4 --num-workers=3 --save-dir=baseline_ad \
--train-cube-size 80 192 304 --train-stride 64 96 152 --val-cube-size 80 192 304 --val-stride 64 72 72 \
--start-epoch=1 --epochs=60 --sgd=0 --randsel=0 --multi-gpu-parallel=1 \
--encoder-path-ad=0 --decoder-path-ad=1 \
--resume="./results/baseline_ad/model_latest.ckpt" --save-feature=1 \
--enable-training=0 --enable-validating=1 --enable-testing=0


# Testing
python3 Airway3DSegmentation.py --model=baseline --batch-size=4 --num-workers=3 --save-dir=baseline_ad \
--train-cube-size 80 192 304 --train-stride 64 96 152 --val-cube-size 80 192 304 --val-stride 64 72 72 \
--start-epoch=1 --epochs=60 --sgd=0 --randsel=0 --multi-gpu-parallel=1 \
--encoder-path-ad=0 --decoder-path-ad=1 \
--resume="./results/baseline_ad/model_latest.ckpt" --save-feature=1 \
--enable-training=0 --enable-validating=0 --enable-testing=1
