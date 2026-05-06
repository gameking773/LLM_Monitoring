#!/usr/bin/env bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LLM_DIR}/logs/vLLM/setup_venv-%j.out
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --constraint=armgpu


romeo_load_armgpu_env
spack load python@3.12.5
spack load cuda@12.6.2 

# cache config
mkdir -p "${LLM_DIR}/cache/tmp"
mkdir -p "${LLM_DIR}/cache/pip_cache"
mkdir -p "${LLM_DIR}/cache/hf_cache"
mkdir -p "${LLM_DIR}/cache/nvcc_cache"
mkdir -p "${LLM_DIR}/cache/xdg"

export TMPDIR="${LLM_DIR}/cache/tmp"
export TEMP="${LLM_DIR}/cache/tmp"
export TMP="${LLM_DIR}/cache/tmp"
export PIP_CACHE_DIR="${LLM_DIR}/cache/pip_cache"
export HF_HOME="${LLM_DIR}/cache/hf_cache"

export XDG_CACHE_HOME="${LLM_DIR}/cache/xdg"
export CUDA_CACHE_PATH="${LLM_DIR}/cache/nvcc_cache"
export CUDA_CACHE_DISABLE=1
export TORCH_CUDA_ARCH_LIST="9.0"

export NVCC_THREADS=1
export MAX_JOBS=16

# .venv creation and launching
mkdir -p ${LLM_DIR}/vllm_env
cd ${LLM_DIR}

python -m venv vllm_env
source vllm_env/bin/activate 

# dependencies installation
pip install --upgrade pip
pip install wheel setuptools
pip install numpy

# vLLM installation
pip install vllm \
 --extra-index-url https://download.pytorch.org/whl/cu126 \
 --no-cache-dir
