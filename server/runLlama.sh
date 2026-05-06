#!/usr/bin/env bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LLM_DIR}/setup_venv-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --constraint=armgpu

# Load environnement
romeo_load_armgpu_env
spack load cuda@13.0.2
export LD_LIBRARY_PATH=$(spack location -i cuda@13.0.2)/lib64:$LD_LIBRARY_PATH

# Node display
echo "Starting server on node : $HOSTNAME"

# Run model
cd ${LLM_DIR}/llama.cpp
#!/bin/bash
./build/bin/llama-server \
  --model ${LLM_DIR}/models/${MODEL_FILE} \
  --host 0.0.0.0 \
  --port 8080 \
  --metrics