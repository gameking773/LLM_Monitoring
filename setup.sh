#!/usr/bin/env bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LLM_DIR}/logs/vLLM/setup-%j.out
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --constraint=armgpu

# Load environnement
romeo_load_armgpu_env

if [ $# -lt 1 ]; then 
    echo "You must choose an inference engine between vllm and llamacpp"
    exit 1
fi

export ENGINE=$1

if [ ! -d "${LLM_DIR}/lighteval_env" ] || [ ! -d "${LLM_DIR}/monitor_env" ]; then
    spack load python@3.12.5
    
    # python monitoring env
    if [ ! -d "${LLM_DIR}/monitor_env" ]; then
        python -m venv ${LLM_DIR}/monitor_env
        source ${LLM_DIR}/monitor_env/bin/activate
        pip install fastapi uvicorn requests dotenv
        deactivate
    fi

    # lighteval env
    if [ ! -d "${LLM_DIR}/lighteval_env" ]; then
        python -m venv ${LLM_DIR}/lighteval_env
        source ${LLM_DIR}/lighteval_env/bin/activate
        pip install --upgrade pip
        pip install lighteval
        deactivate
    fi
    spack unload python@3.12.5
fi

if [ "$ENGINE" == "llamacpp" ]; then
    spack load python@3.13.0/rom
    spack load cuda@13.0.2
    source "${LLM_DIR}/setup/setupLlama.sh"
elif [ "$ENGINE" == "vllm" ]; then
    spack load python@3.12.5
    spack load cuda@12.6.2 
    source "${LLM_DIR}/setup/setupvLLM.sh"
else
    echo "Invalid engine '$ENGINE'. You must choose between vllm and llamacpp"
    exit 1
fi