#!/usr/bin/env bash

# .env loading
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo ".env file not found"
    exit 1
fi

if [ "$1" == "sbatch" ]; then
    ENV_VAR='$LLM_DIR:$SBATCH_ACCOUNT:$SBATCH_JOB_NAME:$VLLM_TOOL_CALL_PARSER:$MODEL_REPO:$MODEL_NAME:$MODEL_FILE:$MODEL_DIRECTORY:$MODELS_TO_BENCHMARK:$DATASET_TASKS'
    ENGINE_ARG="$3"

    TMP_SCRIPT=".run_submit_tmp.sh"
    envsubst "$ENV_VAR" < "$2" > "$TMP_SCRIPT"

    sbatch "$TMP_SCRIPT" "$ENGINE_ARG"
    rm -f "$TMP_SCRIPT"

else
    exec "$@"
fi