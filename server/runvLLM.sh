#!/usr/bin/env bash
export LD_LIBRARY_PATH=$(spack location -i cuda@12.6.2)/lib64:$LD_LIBRARY_PATH 
# Activating vLLM env from its venv path 
source ${LLM_DIR}/vllm_env/bin/activate

# Get the model to use
MODEL_TO_USE=${1:-$MODEL_NAME}

# Cache redirection
mkdir -p "${LLM_DIR}/vllm_cache"
mkdir -p "${LLM_DIR}/vllm_cache/tmp" "${LLM_DIR}/vllm_cache/triton" "${LLM_DIR}/vllm_cache/hf" "${LLM_DIR}/vllm_cache/config"

export TMPDIR="${LLM_DIR}/vllm_cache/tmp"
export TRITON_CACHE_DIR="${LLM_DIR}/vllm_cache/triton"
export HF_HOME="${LLM_DIR}/vllm_cache/hf"
export VLLM_CONFIG_ROOT="${LLM_DIR}/vllm_cache/config"
export MULTIPROCESSING_TMPDIR="${LLM_DIR}/vllm_cache/tmp"


# API Launch
${LLM_DIR}/monitor_env/bin/python ${LLM_DIR}/metrics/metricsAPI.py &

# Server launch
/usr/bin/env \
    TMPDIR="${LLM_DIR}/vllm_cache/tmp" \
    HF_HOME="${LLM_DIR}/vllm_cache/hf" \
    TRITON_CACHE_DIR="${LLM_DIR}/vllm_cache/triton" \
    VLLM_CONFIG_ROOT="${LLM_DIR}/vllm_cache/config" \
    PYTHONPATH=$PYTHONPATH \
    ${LLM_DIR}/vllm_env/bin/python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_DIRECTORY}/${MODEL_TO_USE} \
        --served-model-name ${MODEL_TO_USE} \
        --host 0.0.0.0 \
        --port 8080 \
        --dtype float16 \
        --enforce-eager \
        --gpu-memory-utilization 0.7 \
        --enable-auto-tool-choice \
        --tool-call-parser ${VLLM_TOOL_CALL_PARSER} 