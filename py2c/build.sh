#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

g++ -O3 -shared -fPIC cpu_add.cpp -o libcpu_add.so
nvcc -O3 -shared -Xcompiler -fPIC cuda_add.cu -o libcuda_add.so

echo "built libcpu_add.so and libcuda_add.so"
