#!/usr/bin/env bash
# Load environnement
export LD_LIBRARY_PATH=$(spack location -i cuda@13.0.2)/lib64:$LD_LIBRARY_PATH

# Node display
echo "Starting server on node : $HOSTNAME"

# API Launch
${LLM_DIR}/monitor_env/bin/python ${LLM_DIR}/metrics/metricsAPI.py &

# Run model
./llama.cpp/build/bin/llama-server \
  --model ${LLM_DIR}/models/${MODEL_FILE} \
  --host 0.0.0.0 \
  --port 8080 \
  --metrics