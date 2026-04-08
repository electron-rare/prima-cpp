# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Distributed llama.cpp fork — ring-topology pipeline parallelism for 70B+ LLM inference across heterogeneous home clusters. 15x faster than vanilla llama.cpp on large models. Uses ZeroMQ for comms, HiGHS for workload optimization, BitSqueeze for compression.

## Build

```bash
# Makefile (preferred)
make -j$(nproc)                              # basic
make USE_HIGHS=1 -j$(nproc)                 # rank 0 (head device, required for scheduler)
make GGML_CUDA=1 -j$(nproc)                 # with GPU
make GGML_CUDA=1 USE_HIGHS=1 -j$(nproc)     # rank 0 + GPU
make LLAMA_DEBUG=1 -j$(nproc)               # debug build

# CMake alternative
cmake -B build && cmake --build build -j$(nproc)
cd build && ctest                            # tests
```

## Run

```bash
# Single device (degrades to llama.cpp)
./llama-cli -m model.gguf -c 1024 -p "prompt" -n 256 -ngl 30

# Distributed ring
./llama-cli -m model.gguf --world N --rank R --master IP --next NEXT_IP --prefetch [--gpu-mem GB]

# Server mode (rank 0 only, OpenAI-compatible)
./llama-server -m model.gguf --world 2 --rank 0 --master IP --next IP --prefetch --host 0.0.0.0 --port 8080

# Profiling
./profile-tool -m model.gguf
```

## Where to Look

| Task | Location |
|------|----------|
| Core inference + networking | `src/` — llama.cpp, network-utils.cpp |
| Public headers | `include/` — llama.h, zmq.hpp, Highs.h, bitsqueeze.h, profiler.h |
| GGML tensor library | `ggml/` |
| Example binaries | `examples/` — main, server, speculative, perplexity, batched |
| Tests | `tests/` — CTest |
| Python model conversion | `gguf-py/`, convert scripts at root |
| Models | `models/` (~19GB) |

## Fork-Specific Flags

`--world`, `--rank`, `--master`, `--next`, `--prefetch`, `--force`, `--gpu-mem`, `--data-port` (9000), `--signal-port` (10000), `--keep-out-in-cuda`, `-lw`/`--n-layer-window`, `--splits`

## Conventions

- Focus: distributed inference optimization, especially networking (send/recv compression)
- Fork-specific changes must be documented in `CHANGES.md`
- Capture new flags and compatibility notes
- Python scripts: Poetry-managed (`pyproject.toml`), Python >=3.9
