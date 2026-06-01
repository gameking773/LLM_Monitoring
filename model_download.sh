#!/usr/bin/env bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LLM_DIR}/logs/download/model-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --constraint=armgpu

romeo_load_armgpu_env
spack load python@3.13.0/rom

# Get the model from huggingface_hub
pip install huggingface_hub
hf download ${MODEL_REPO} \
  --local-dir ${MODELS_DIRECTORY}