#!/usr/bin/env bash
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
        --model ${LLM_DIR}/models/${MODEL_FILE} \
        --served-model-name ${MODEL_NAME} \
        --host 0.0.0.0 \
        --port 8080 \
        --dtype float16 \
        --enforce-eager \
        --gpu-memory-utilization 0.7 \
        --enable-auto-tool-choice \
        --tool-call-parser ${VLLM_TOOL_CALL_PARSER}