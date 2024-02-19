#!/bin/sh
#SBATCH -J eval
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n02
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -o log/eval.out
#SBATCH -e log/eval.err
#SBATCH --time 24:00:00

PYTHONPATH=../ python3 eval.py \
 --dataset Atipico1/nq-test-valid-adversary-replace-processed \
 --split train \
 --test True