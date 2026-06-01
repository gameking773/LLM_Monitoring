#!/usr/bin/env bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LLM_DIR}/logs/server/campaign-%j.out
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --constraint=armgpu

romeo_load_armgpu_env
spack load python@3.12.5

# get the model, the engine and craft the model name
export BENCH_MODEL_FILE="${MODELS_TO_BENCHMARK%%:*}"
export ENGINE="${MODELS_TO_BENCHMARK#*:}"
export BENCH_MODEL_NAME="${BENCH_MODEL_FILE%.*}"

# start the right engine and save the PID for later use
if [ "$ENGINE" == "llamacpp" ]; then
    bash "${LLM_DIR}/server/runLlama.sh" "${BENCH_MODEL_FILE}" &
    SERVER_PID=$!
elif [ "$ENGINE" == "vllm" ]; then
    spack load cuda@12.6.2 
    bash "${LLM_DIR}/server/runvLLM.sh" "${BENCH_MODEL_FILE}" &
    SERVER_PID=$!
else
    echo "Unknown engine '$ENGINE' for $BENCH_MODEL_FILE. Skipping."
    continue 
fi

# wait for the server to start
COUNT=0
while ! curl -s http://localhost:8080/v1/models > /dev/null; do
    sleep 10
    ((COUNT++))
    if [ $COUNT -ge 30 ]; then
        echo "Serveur non démarré, abandon."
        exit 1
    fi
done

# activation of the hardware surveillance
source "${LLM_DIR}/monitor_env/bin/activate"
python3 "${LLM_DIR}/benchmark/modelBenchmark.py" "${BENCH_MODEL_FILE}" "${ENGINE}" &
LOGGER_PID=$!
deactivate

# datasets execution with Eleuther lm-eval
source "${LLM_DIR}/lm-eval_env/bin/activate"

export OPENAI_API_KEY="none"
lm_eval --model openai-chat-completions \
    --model_args="base_url=http://localhost:8080/v1/chat/completions,model=Qwen3-8B,num_concurrent=4" \
    --tasks ${DATASET_TASK} \
    --apply_chat_template \
    --num_fewshot 0 \
    --output_path "${LLM_DIR}/logs/results_${BENCH_MODEL_NAME}_${ENGINE}.json" \
    --gen_kwargs="do_sample=false,max_tokens=1024,temperature=0"

deactivate