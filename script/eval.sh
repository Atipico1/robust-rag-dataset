#!/bin/sh
#SBATCH -J eval
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH -o log/eval.out
#SBATCH -e log/eval.err
#SBATCH --time 24:00:00

PYTHONPATH=../ python3 eval.py \
 --dataset Atipico1/nq_test_adversary \
 --model meta-llama/Llama-2-13b-chat-hf \
 --split train \
 --test True \
 --test_size 500 \
 --insts best \
 --unans True \
 --unans_string "unanswerable" \
 --adv True \
 --adv_method random_insert

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-replace-processed \
#  --model meta-llama/Llama-2-13b-chat-hf \
#  --split train \
#  --unans True

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-random-replace \
#  --model Qwen/Qwen1.5-72B-Chat \
#  --split test

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-random-replace \
#  --model Qwen/Qwen1.5-7B-Chat \
#  --split test

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-random-replace \
#  --model Qwen/Qwen1.5-14B-Chat \
#  --split test

PYTHONPATH=../ python3 eval.py \
 --dataset Atipico1/nq-test-random-replace \
 --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
 --split test

PYTHONPATH=../ python3 eval.py \
 --dataset Atipico1/nq-test-random-replace \
 --model microsoft/Orca-2-13b \
 --split test

PYTHONPATH=../ python3 eval.py \
 --dataset Atipico1/nq-test-random-replace \
 --model microsoft/Orca-2-7b \
 --split test
 
PYTHONPATH=../ python3 eval.py \
 --dataset Atipico1/nq-test-random-replace \
 --model microsoft/phi-2 \
 --split test

PYTHONPATH=../ python3 eval.py \
 --dataset Atipico1/nq-test-random-replace \
 --model google/gemma-7b-it \
 --split test

PYTHONPATH=../ python3 eval.py \
 --dataset Atipico1/nq-test-random-replace \
 --model google/gemma-2b-it \
 --split test

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-random-replace \
#  --model meta-llama/Llama-2-13b-chat-hf \
#  --split test \
#  --unans True

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model meta-llama/Llama-2-13b-chat-hf \
#  --split train \
#  --unans True

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-format \
#  --model meta-llama/Llama-2-7b-chat-hf \
#  --split train


# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-replace-processed \
#  --model mistralai/Mistral-7B-Instruct-v0.2 \
#  --split train

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model mistralai/Mistral-7B-Instruct-v0.2 \
#  --split train

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-replace-processed \
#  --model Qwen/Qwen1.5-7B-Chat \
#  --split train

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model Qwen/Qwen1.5-7B-Chat \
#  --split train

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-replace-processed \
#  --model Qwen/Qwen1.5-14B-Chat \
#  --split train

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model Qwen/Qwen1.5-14B-Chat \
#  --split train

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-replace-processed \
#  --model Qwen/Qwen1.5-72B-Chat \
#  --split train

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model Qwen/Qwen1.5-72B-Chat \
#  --split train