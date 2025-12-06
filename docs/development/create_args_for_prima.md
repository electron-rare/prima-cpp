# Adding New CLI Arguments for `llama_send_tensors` / `llama_recv_tensors`

This note walks through how to add new CLI arguments that flow into the distributed communication path in `src/llama.cpp`, including all touchpoints you must update.

## Key Data Flow

1. CLI parse → `gpt_params` (`common/arg.cpp`)
2. Copy into context params → `llama_context_params` (`common/common.cpp` via `llama_context_params_from_gpt_params`)
3. Propagate into runtime context → `lctx.cparams` (`src/llama.cpp` initialization)
4. Used by send/recv → `llama_send_tensors` / `llama_recv_tensors` (`src/llama.cpp`)

## Files to Update

- `common/arg.cpp`: register the CLI flag(s), set defaults, validate ranges.
- `common/common.h`: add fields to `gpt_params` (string/int/bool as needed).
- `common/common.cpp`:
  - `llama_context_params_from_gpt_params`: copy new fields from `gpt_params` into `llama_context_params` (allocate/copy strings if needed).
  - Ensure `llama_context_default_params` (in `src/llama.cpp`) sets sensible defaults.
- `include/llama.h` and `spm-headers/llama.h`: extend `struct llama_context_params` with the new field(s).
- `src/llama.cpp`:
  - Thread new params through any places that consume them (e.g., `llama_send_tensors` / `llama_recv_tensors`).
  - Validate inputs at use-site (range checks, supported values).
  - If the param affects compression/sparsity/decompression, branch in `llama_send_tensors` and interpret tags in `llama_recv_tensors`.

## Step-by-Step Template

1. **Define parameter storage**
   - Add to `struct gpt_params` in `common/common.h`.
   - Set default values there.

2. **Expose via CLI**
   - In `common/arg.cpp`, add a `llama_arg` entry:
     - Flag names (short/long), help text, value hints.
     - Handler writes into `gpt_params` (string/int/bool handler).
     - Validate ranges here if you want early failure.

3. **Copy into runtime context**
   - In `llama_context_params_from_gpt_params` (`common/common.cpp`):
     - For strings, allocate and `strcpy` into `cparams`.
     - For scalars, direct assignment.

4. **API surface**
   - Add the field to `struct llama_context_params` in both headers:
     - `include/llama.h`
     - `spm-headers/llama.h`
   - Update `llama_context_default_params` in `src/llama.cpp` with defaults (nullptr/0/false, or a literal default).

5. **Use in send/recv**
   - `src/llama.cpp`:
     - Accept the new argument in `llama_send_tensors` (and `llama_recv_tensors` if needed).
     - Pass `lctx.cparams.<field>` when calling `llama_send_tensors` from the main decode loop.
     - Implement behavior (e.g., choosing compression mode, sparse ratio, thresholds).
     - Add validation guards near use; log or error out cleanly.

6. **(Optional) Docs/CHANGES**
   - Document the new flag in `CHANGES.md` and any user-facing README if needed.

## Example: Existing `comm_datatype` / `comm_sparse_percentage`

- **CLI**: `--comm-datatype`, `--comm-sparse-percentage` (`common/arg.cpp`).
- **Params**: Stored in `gpt_params` (`common/common.h`) with defaults.
- **Context copy**: `llama_context_params_from_gpt_params` handles string allocation and scalar copy (`common/common.cpp`).
- **Headers**: `llama_context_params` exposes `const char * comm_datatype; int comm_sparse_percentage;` (`include/llama.h`, `spm-headers/llama.h`).
- **Defaults**: `llama_context_default_params` sets them to `nullptr` / 0 (`src/llama.cpp`).
- **Usage**: `llama_send_tensors` inspects `comm_datatype` and uses `comm_sparse_percentage` when `f32_sparsity` is selected; `llama_recv_tensors` reads the tag and auto-decompresses.

## Validation Tips

- For ranged ints, check on parse and on use; fail fast with a clear message.
- For enums/strings, normalize and validate against a small allowed list before hitting the hot path.
- Keep the wire format self-describing: include a datatype tag in the multipart messages so receivers can branch correctly.

## Quick checklist

- [ ] Field in `gpt_params` with default
- [ ] CLI flag in `common/arg.cpp`
- [ ] Copy into `llama_context_params_from_gpt_params`
- [ ] Field added to both `llama.h` headers
- [ ] Default set in `llama_context_default_params`
- [ ] Passed into `llama_send_tensors`/`llama_recv_tensors`
- [ ] Behavior implemented + input validation
- [ ] Docs updated (optional)
