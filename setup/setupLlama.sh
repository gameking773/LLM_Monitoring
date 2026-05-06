#!/usr/bin/env bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LLM_DIR}/setup_venv-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --constraint=armgpu

romeo_load_armgpu_env
spack load python@3.13.0/rom
spack load cuda@13.0.2
export LD_LIBRARY_PATH=$(spack location -i cuda@13.0.2)/lib64:$LD_LIBRARY_PATH # help the system to find CUDA libs

# Diag and Versions
echo "=== Architecture : $(uname -m) ==="
echo "=== CUDA version ==="
nvcc --version || echo "nvcc absent"
nvidia-smi | head -5

# Pull llama.cpp from github if it isn't here. Else reuse it
cd ${LLM_DIR}

if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp
fi

cd llama.cpp

# Cleaning of an eventual failed build
rm -rf build

# Llama.cpp compilation
cmake -B build \
-DGGML_CUDA=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_CUDA_COMPILER=nvcc

cmake --build build --config Release -j8

# Verification
echo "Version installée"
./build/bin/llama-cli --version