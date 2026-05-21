#!/usr/bin/env bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LLM_DIR}/logs/server/runServer-%j.out
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --constraint=armgpu

# Load environnement
romeo_load_armgpu_env

if [ $# -lt 1 ]; then 
    echo "You must choose an inference engine between vllm and llamacpp"
    exit 1
fi

export ENGINE=$1

if [ "$ENGINE" == "llamacpp" ]; then
    spack load cuda@13.0.2
    source "${LLM_DIR}/server/runLlama.sh"
elif [ "$ENGINE" == "vllm" ]; then
    spack load python@3.13.0/rom
    spack load cuda@12.6.3
    source "${LLM_DIR}/server/runvLLM.sh"
else
    echo "Invalid engine '$ENGINE'. You must choose between vllm and llamacpp"
    exit 1
fi
