#!/bin/bash
#SBATCH -J model_test
#SBATCH --gres=gpu:1
#SBATCH -p PV100q
#SBATCH -w node09

margin=(1.0 2.0)
dataset_name=WN18
bern=(False True)
epochs=120
batch_size=(512 1024 2048)
learning_rate=0.01
dim=64
lr_decay=1.8
norm=1

#python3 ../main/pretrain.py --gpu 7 --model TransE--dim 32 --bs 4196 --init_lr 0.01 --lr_decay 1.2 --bern False --margin 6.0  --epochs 1000 --reg 0.0 --neg_ratio 1


























rm "slurm-${SLURM_JOB_ID}.out"
