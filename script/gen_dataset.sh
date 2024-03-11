#!/bin/sh
#SBATCH -J gen-data
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:0
#SBATCH --mem=40G
#SBATCH -o log/gen_dataset.out
#SBATCH -e log/gen_dataset.err
#SBATCH --time 24:00:00

PYTHONPATH=../ python3 gen_dataset.py \
 --origin_path Atipico1/nq-test-valid-adversary-replace \
 --output_path Atipico1/nq-test-replace-format 


PYTHONPATH=../ python3 gen_dataset.py \
 --origin_path Atipico1/nq-test-valid_adv_passage \
 --output_path Atipico1/nq-test-format \
 --split train
