# Introduction

This markdown file is used to demonstrate the change i made compare to original prima.cpp

## Configurable Network Port Options

Enable separate configuration of local listening ports and remote node ports for distributed inference, supporting complex networking scenarios including port forwarding.

- commits (1e7ae71)

### New CLI Arguments

**Local Listening Ports:**
- `--data_port PORT`: Local port where this node listens for data communications
- `--signal_port PORT`: Local port where this node listens for signal communications

**Remote Node Ports (ports that other nodes are listening on):**
- `--master_data_port PORT`: Port that master node is listening on for data
- `--next_node_data_port PORT`: Port that next node is listening on for data  
- `--next_node_signal_port PORT`: Port that next node is listening on for signals

### Port Configuration Logic

Prima.cpp separates local binding from remote addressing:
- **Local ports** (`--data_port`, `--signal_port`): Where this node listens for incoming connections
- **Remote ports** (`--master_*_port`, `--next_node_*_port`): Which ports other nodes are listening on

This design enables port forwarding scenarios where nodes may listen on different ports locally but are accessed through forwarded ports.

### Usage Examples

```shell
# On rank 0 (master server), run:
./llama-server -m gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --host 0.0.0.0 --port 8080 --world 2 --rank 0 --data_port 9000 --signal_port 9001 --master 127.0.0.1 --master_data_port 9000 --next 192.168.4.10 --next_node_data_port 9000 --next_node_signal_port 9001

# On rank 1 (worker node), run:
./llama-cli -m gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --world 2 --rank 1 --data_port 9000 --signal_port 9001 --master 192.168.4.9 --master_data_port 9000 --next 192.168.4.9 --next_node_data_port 9000 --next_node_signal_port 9001
```

This provides complete control over both local binding and remote connectivity for distributed model inference.

## Rank-Specific GGUF Split Loading

Enable selective loading of GGUF split files based on layer assignment, reducing memory usage and download requirements for distributed inference.

- commits (f1f7e37)

### New CLI Argument

- `--splits SPLIT_LIST`: Comma-separated list of split indices to load (e.g., "0,2,3")

### Split Loading Logic

When models are split using `gguf-split`:

```shell
./gguf-split --split-max-tensors 128 llama.gguf llama.gguf
```

You get multiple files:
```
llama.gguf-00001-of-00004.gguf
llama.gguf-00002-of-00004.gguf  
llama.gguf-00003-of-00004.gguf
llama.gguf-00004-of-00004.gguf
```

Unlike `llama.cpp` which loads all splits, `prima.cpp` enables selective loading based on layer assignment:

- Each rank only loads splits containing its assigned layers
- Reduces memory footprint and network transfer
- Eliminates need to download unused model segments

### Layer Assignment Policy

Prima.cpp enforces a specific layer distribution:

- **Rank 0 (master)**: Always owns embedding layers + final transformer blocks
- **Rank 1+ (workers)**: Own contiguous middle transformer blocks
- **Rank 1**: Always starts from layer 0

### Split Selection Strategy

For `--n-layer-window "8,8"` with `--world 2`:

- **Rank 0**: Owns embedding + layers 8-15 → loads splits 0,2,3  
- **Rank 1**: Owns layers 0-7 → loads splits 0,1

### Usage Examples

```bash
# Rank 0 (master) loads splits containing embedding + final layers
./llama-cli -m llama.gguf-00001-of-00004.gguf --splits 0,2,3 --world 2 --rank 0 --n-layer-window "8,8"

# Rank 1 (worker) loads splits containing initial layers  
./llama-cli -m llama.gguf-00001-of-00004.gguf --splits 0,1 --world 2 --rank 1 --n-layer-window "8,8"
```

### Error Handling

- Missing required tensors trigger immediate error with split suggestions
- Explicitly set `--n-layer-window` to ensure predictable layer-to-split mapping
- Validate split coverage before distributed execution

This optimization significantly reduces resource requirements for large model inference across multiple nodes.

## Communication and Compute Logging

Enable detailed timestamped logging of inter-node communication and computation phases for performance analysis and debugging distributed inference. This feature can be controlled with a CLI flag to reduce overhead when not needed.

- commits (eb0cac1, adad23d, a399e49)

### CLI Control

- `--enable-comm-compute-log`: Enable communication and computation logging (disabled by default for performance)

### Logging Categories

**Communication Logs:**
- `[comm][start/end][send_tensors]`: Tensor transmission to next node/master
- `[comm][start/end][recv_tensors]`: Tensor reception from other nodes

**Compute Logs:**  
- `[compute][start/end][transformer_blocks]`: Layer computation timing
- `[compute][start/end][output_linear]`: Final output layer processing

### Log Format

Each log entry contains:
```
[rank][timestamp][category][phase][operation][batch_info, description]
```

- **rank**: Node rank identifier (0 for master, 1+ for workers)
- **timestamp**: ISO 8601 timestamp with millisecond precision
- **category**: `comm` for communication, `compute` for computation
- **phase**: `start` or `end` of operation
- **operation**: Specific function being logged
- **batch_info**: `sbatch_tokens` and `ubatch_tokens` counts
- **description**: Human-readable operation description

### Use Cases

- **Performance profiling**: Identify communication vs compute bottlenecks
- **Gantt chart generation**: Visualize parallel execution timeline
- **Debugging**: Track tensor flow and synchronization issues
- **Load balancing**: Analyze per-node utilization patterns

### Example Output

```
[0][2025-08-02T12:06:35.050Z][comm][start][send_tensors][sbatch_tokens: 0, ubatch_tokens: 512, send the result to the next node or the master]
[0][2025-08-02T12:06:35.051Z][comm][end][send_tensors][sbatch_tokens: 0, ubatch_tokens: 512, send the result to the next node or the master]
[0][2025-08-02T12:06:35.051Z][comm][start][recv_tensors][sbatch_tokens: 0, ubatch_tokens: 512, receive data from other nodes]
[0][2025-08-02T12:06:35.160Z][comm][end][recv_tensors][sbatch_tokens: 0, ubatch_tokens: 512, receive data from other nodes]
[0][2025-08-02T12:06:35.162Z][compute][start][transformer_blocks][sbatch_tokens: 0, ubatch_tokens: 512]
[0][2025-08-02T12:06:35.165Z][compute][end][transformer_blocks][sbatch_tokens: 0, ubatch_tokens: 512]
[0][2025-08-02T12:06:35.190Z][compute][start][output_linear][sbatch_tokens: 0, ubatch_tokens: 512]
[0][2025-08-02T12:06:35.259Z][compute][end][output_linear][sbatch_tokens: 0, ubatch_tokens: 512]
```

This logging system provides comprehensive visibility into distributed inference execution patterns.

## Communication Tensor Dumping

Enable binary dumping of inter-node tensor communications with comprehensive metadata for detailed analysis of distributed inference data flow.

- commits (e48e955)

### New CLI Argument

- `--dump-folder FOLDER`: Specify directory to dump network communication tensors (disabled if unset)

### Binary Dump Format

Each tensor file contains structured binary data with complete shape metadata:

```
[1 byte: element_type][8 bytes: n_embed][8 bytes: n_tokens][8 bytes: tensor_size][tensor_data]
```

**Header Fields:**
- `element_type`: Data type identifier (FLOAT32 = 0, extensible for other types)
- `n_embed`: Embedding dimension/width (uint64_t)  
- `n_tokens`: Token count/height (uint64_t)
- `tensor_size`: Total tensor data size in bytes (uint64_t)
- `tensor_data`: Raw tensor bytes in native format

### Dumping Operations

**Send Path (`llama_send_tensors`):**
- Files: `send_{counter}.bin`
- Shape extracted from `tensors->sub_gf_out->ne[]`
- Captures outbound tensor data before network transmission

**Receive Path (`llama_recv_tensors`):**  
- Files: `recv_{counter}.bin`
- Shape extracted from `dims[]` parameter
- Captures inbound tensor data after network reception

### Usage Examples

```bash
# Enable tensor dumping for distributed inference
./llama-cli --model model.gguf --dump-folder ./tensor_dumps --world 2 --rank 0 [other_args]
```

This feature provides comprehensive visibility into the tensor communication layer for distributed model analysis.

## Multi-Node Perplexity Evaluation

Enable distributed perplexity evaluation using `llama-perplexity` as master coordinator with `llama-cli` worker nodes for large model assessment.

- commits (c949e53)

### Distributed Perplexity Architecture

**Master Node (Rank 0):**
- Runs `llama-perplexity` binary
- Coordinates text processing and perplexity calculations
- Manages input text file distribution and result aggregation

**Worker Nodes (Rank 1+):**
- Run `llama-cli` binaries in distributed mode
- Process assigned model layers during evaluation
- Participate in tensor communication pipeline

### Supported Evaluation Features

- **Text file processing**: Load evaluation datasets via `-f` parameter
- **Distributed inference**: Leverage multi-node processing for large models
- **Standard perplexity metrics**: Calculate perplexity scores across distributed architecture

### Usage Examples

**2-Node Setup:**
```shell
# On rank 0 (master evaluator), run:
./llama-perplexity -m download/qwq-32b-q4_k_m.gguf -f wikitext-2-raw/wiki.test.raw --world 2 --rank 0 --master 192.168.1.2 --next 192.168.1.3

# On rank 1 (worker node), run:
./llama-cli -m download/qwq-32b-q4_k_m.gguf --world 2 --rank 1 --master 192.168.1.2 --next 192.168.1.2
```

## Communication Compression

Enable configurable compression of inter-node tensor communication to reduce bandwidth usage while maintaining flexibility in data type handling.

- Quantization commits (ff310e7–3a97a82)
- Sparsity commits (45a78f2–9e730dd)

### Purpose

This feature allows each rank to specify the data type used when sending tensors across nodes. Each rank is responsible only for compressing its own outgoing data. On the receiving side, the system automatically inspects the metadata to determine whether the payload is quantized and requires dequantization.

A new CLI argument (`--comm_datatype TYPE`) sets the communication data type. Supported values:

- f32 (default, no compression)
- q8_0 (8-bit block quantization)
- q4_0 (4-bit block quantization)
- q2_k (2-bit k quantization)
- f32_sparsity (no quantization, but allows `--comm_sparse_percentage` to select the top-k features for each token based on the specified sparsity percentage)

### Communication Logic

- **Send path**

  - If `f32`, tensors are transmitted directly.
  - If `q8_0`, `q4_0` or `q2_k`, tensors are quantized before transmission.
  - If `f32_sparsity`, tensors are pruned before transmission.

- **Receive path**

  - Metadata indicates whether the payload is raw or quantized.
  - Quantized payloads are automatically dequantized to float32.
  - Sparse payloads are automatically decompressed to float32, with pruned values restored as zeros.

This design ensures that each rank manages compression only for its own outputs, while reception remains transparent.

### Example Usage

#### Rank 0 (send tensors in q8_0 format)

```
./llama-cli --model model.gguf --world 2 --rank 0 --comm_datatype q8_0
```

#### Rank 1 (receive q8_0 data, auto-dequantize to f32, send tensors in q8_0 format)

```
./llama-cli --model model.gguf --world 2 --rank 1 --comm_datatype q8_0
```

By default, communication uses `f32` with no quantization.

#### Sparsity Example

```
./llama-cli --model model.gguf --world 2 --rank 0 --comm_datatype f32_sparsity --comm_sparse_percentage 20
```

### Quantization and Dequantization Timing

To measure the overhead of quantization and dequantization, enable:

```
--enable-comm-compute-log
```

This option produces timestamped logs that mark the start and end of quantization and dequantization operations for profiling and debugging.
