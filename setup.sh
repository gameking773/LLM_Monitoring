#!/usr/bin/env bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LLM_DIR}/logs/setup/setup-%j.out
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

# if the monitoring and benchmarking env aren't installed, install them
if [ ! -d "${LLM_DIR}/promptfoo_env" ] || [ ! -d "${LLM_DIR}/monitor_env" ]; then
    spack load python@3.12.5
    
    # python monitoring env
    if [ ! -d "${LLM_DIR}/monitor_env" ]; then
        python -m venv ${LLM_DIR}/monitor_env
        source ${LLM_DIR}/monitor_env/bin/activate
        pip install fastapi uvicorn requests dotenv psutil
        deactivate
    fi

    # lm-eval env
    if [ ! -d "${LLM_DIR}/lm-eval_env" ]; then
        python -m venv ${LLM_DIR}/lm-eval_env
        source ${LLM_DIR}/lm-eval_env/bin/activate
        pip install --upgrade pip
        pip install lm-eval lm-eval[api]
        pip install datasets
        deactivate 
    fi
    spack unload python@3.12.5
fi

# Installation of the selected engine
if [ "$ENGINE" == "llamacpp" ]; then
    spack load python@3.13.0/rom
    spack load cuda@13.0.2
    source "${LLM_DIR}/setup/setupLlama.sh"
elif [ "$ENGINE" == "vllm" ]; then
    spack load python@3.12.5
    spack load cuda@13.0.2
    source "${LLM_DIR}/setup/setupvLLM.sh"
else
    echo "Invalid engine '$ENGINE'. You must choose between vllm and llamacpp"
    exit 1
fi