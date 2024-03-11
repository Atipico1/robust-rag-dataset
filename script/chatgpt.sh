#!/bin/sh
#SBATCH -J prompt-test
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:0
#SBATCH --mem=100G
#SBATCH -o log/chatgpt.out
#SBATCH -e log/chatgpt.err
#SBATCH --time 24:00:00

PYTHONPATH=../ python3 chatgpt.py \
 --test False \
 --test_size 20 \
 --split test \
 --dataset Atipico1/nq_test \
 --run_name nq_test \
 --tasks answer_sent adv_sent adv_passage \
 --output_dir Atipico1/nq_test_adversary
