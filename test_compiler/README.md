# NVRTC compiler playground

Build and run these samples from the repository root on Linux with a CUDA Toolkit
and an NVIDIA GPU. Each sample compiles a CUDA kernel with NVRTC, loads it with the
CUDA driver, and launches it with the value `13.37`.

The second sample maps standard-library names to libcu++ through its include path.
The flags files are generated locally because toolkit paths vary across machines.

```bash
export CUDA_PATH=/usr/local/cuda-13.0
mkdir -p .cache/test_compiler
"$CUDA_PATH/bin/nvcc" test_compiler/test_compiler.cpp \
    -o .cache/test_compiler/playground \
    -lnvrtc_static -lnvrtc-builtins_static -lnvptxcompiler_static

cat > .cache/test_compiler/cflags1.txt <<FLAGS
--std=c++17
--gpu-architecture=compute_75
--device-as-default-execution-space
--include-path=$CUDA_PATH/include
--include-path=$CUDA_PATH/include/cccl
FLAGS
cp .cache/test_compiler/cflags1.txt .cache/test_compiler/cflags2.txt
printf '%s\n' "--include-path=$CUDA_PATH/include/cccl/cuda/std" \
    >> .cache/test_compiler/cflags2.txt

.cache/test_compiler/playground test_compiler/src/source1.cu .cache/test_compiler/cflags1.txt
.cache/test_compiler/playground test_compiler/src/source2.cu .cache/test_compiler/cflags2.txt
```

Expected kernel output is `13.370000` for the first sample and
`13.37, 3.6565, 178.757` for the second. Both should finish with `Goodbye` and no
CUDA or NVRTC errors. The `compute_75` PTX target requires a Turing or newer GPU.
