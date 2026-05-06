#!/usr/bin/env bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LLM_DIR}/logs/vLLM/run_venv-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --constraint=armgpu

# Load environnement
romeo_load_armgpu_env
spack load python@3.13.0/rom
spack load cuda@12.6.3

# Activating vLLM env from its venv path 
source ${LLM_DIR}/vllm_env/bin/activate

# Downloading gradio to get a web chat
pip install python-dotenv

# Cache redirection
mkdir -p "${LLM_DIR}/vllm_cache"
mkdir -p "${LLM_DIR}/vllm_cache/tmp" "${LLM_DIR}/vllm_cache/triton" "${LLM_DIR}/vllm_cache/hf" "${LLM_DIR}/vllm_cache/config"

export TMPDIR="${LLM_DIR}/vllm_cache/tmp"
export TRITON_CACHE_DIR="${LLM_DIR}/vllm_cache/triton"
export HF_HOME="${LLM_DIR}/vllm_cache/hf"
export VLLM_CONFIG_ROOT="${LLM_DIR}/vllm_cache/config"
export MULTIPROCESSING_TMPDIR="${LLM_DIR}/vllm_cache/tmp"


# Server launch
/usr/bin/env \
    TMPDIR="${LLM_DIR}/vllm_cache/tmp" \
    HF_HOME="${LLM_DIR}/vllm_cache/hf" \
    TRITON_CACHE_DIR="${LLM_DIR}/vllm_cache/triton" \
    VLLM_CONFIG_ROOT="${LLM_DIR}/vllm_cache/config" \
    PYTHONPATH=$PYTHONPATH \
    python -m vllm.entrypoints.openai.api_server \
        --model ${LLM_DIR}/models/${MODEL_FILE} \
        --served-model-name ${MODEL_FILE} \
        --host 0.0.0.0 \
        --port 8080 \
        --dtype float16 \
        --enforce-eager \
        --gpu-memory-utilization 0.7 \
        --enable-auto-tool-choice \
        --tool-call-parser auto