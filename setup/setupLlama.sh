#!/usr/bin/env bash

export LD_LIBRARY_PATH=$(spack location -i cuda@13.0.2)/lib64:$LD_LIBRARY_PATH # help the system to find CUDA libs

# Pull llama.cpp from github if it isn't here. Else reuse it
cd ${LLM_DIR}

# Get llama.cpp repo
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp
fi

cd llama.cpp

# Cleaning of an eventual failed build
rm -rf build

# Llama.cpp compilation
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_CUDA_COMPILER=nvcc

cmake --build build --config Release -j8

# Verification
echo "Version installée"
./build/bin/llama-cli --version