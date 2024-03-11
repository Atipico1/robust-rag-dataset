#!/bin/sh
#SBATCH -J prompt-test
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -o log/eval_test.out
#SBATCH -e log/eval_test.err
#SBATCH --time 24:00:00

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model google/gemma-2b-it \
#  --split train \
#  --test True \
#  --unans True \
#  --wandb_project prompt-test \
#  --insts default answer_parse mistral_rag openai_rag

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model google/gemma-2b-it \
#  --split train \
#  --test True \
#  --unans False \
#  --wandb_project prompt-test \
#  --insts default answer_parse mistral_rag openai_rag

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model google/gemma-7b-it \
#  --split train \
#  --test True \
#  --unans False \
#  --wandb_project prompt-test \
#  --insts default answer_parse mistral_rag openai_rag

# PYTHONPATH=../ python3 -u eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model mistralai/Mistral-7B-Instruct-v0.2 \
#  --split train \
#  --test True \
#  --mode hf \
#  --parser format_enforcer \
#  --unans False \
#  --wandb_project prompt-test \
#  --insts openai_rag_json OnlyAnsInst StrictOpenAIInst StrictMistralInst

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model mistralai/Mistral-7B-Instruct-v0.2 \
#  --split train \
#  --test True \
#  --unans True \
#  --wandb_project prompt-test \
#  --insts strict_openai_rag strict_mistral_rag only_ans

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
#  --split train \
#  --test False \
#  --unans True \
#  --wandb_project prompt-test \
#  --insts strict_openai_rag strict_mistral_rag only_ans

PYTHONPATH=../ python3 eval.py \
 --dataset Atipico1/nq-test-valid-adversary-processed \
 --model microsoft/Orca-2-13b \
 --split train \
 --test True \
 --unans False \
 --wandb_project prompt-test \
 --insts default answer_parse mistral_rag openai_rag

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model microsoft/Orca-2-7b \
#  --split train \
#  --test True \
#  --unans False \
#  --wandb_project prompt-test \
#  --insts default answer_parse mistral_rag openai_rag

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model microsoft/phi-2 \
#  --split train \
#  --test True \
#  --unans False \
#  --wandb_project prompt-test \
#  --insts default answer_parse mistral_rag openai_rag

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model meta-llama/Llama-2-70b-chat-hf \
#  --split train \
#  --test True \
#  --unans True \
#  --wandb_project prompt-test \
#  --insts default answer_parse mistral_rag openai_rag

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model meta-llama/Llama-2-7b-chat-hf \
#  --split train \
#  --test True \
#  --unans True \
#  --wandb_project prompt-test \
#  --insts default answer_parse mistral_rag openai_rag


# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model Qwen/Qwen1.5-7B-Chat \
#  --split train \
#  --test True \
#  --unans True \
#  --wandb_project prompt-test \
#  --insts default answer_parse mistral_rag openai_rag

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model Qwen/Qwen1.5-7B-Chat \
#  --split train \
#  --test True \
#  --unans False \
#  --wandb_project prompt-test \
#  --insts default answer_parse mistral_rag openai_rag

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model Qwen/Qwen1.5-14B-Chat \
#  --split train \
#  --test True \
#  --unans True \
#  --wandb_project prompt-test \
#  --insts default answer_parse mistral_rag openai_rag

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model Qwen/Qwen1.5-14B-Chat \
#  --split train \
#  --test True \
#  --unans False \
#  --wandb_project prompt-test \
#  --insts default answer_parse mistral_rag openai_rag

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model Qwen/Qwen1.5-72B-Chat \
#  --split train \
#  --test True \
#  --unans True \
#  --wandb_project prompt-test \
#  --insts default answer_parse mistral_rag openai_rag

# PYTHONPATH=../ python3 eval.py \
#  --dataset Atipico1/nq-test-valid-adversary-processed \
#  --model Qwen/Qwen1.5-72B-Chat \
#  --split train \
#  --test True \
#  --unans False \
#  --wandb_project prompt-test \
#  --insts default answer_parse mistral_rag openai_rag