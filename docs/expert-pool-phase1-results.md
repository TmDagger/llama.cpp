# MoE expert pool: Phase 1 multi-GPU results (fork notes)

Internal summary of the Phase 1 work on the `fix-moe-pool-rail` fork branch
(rebased on upstream master). Not a discussion post; raw material for one.

## What Phase 1 adds

`--moe-expert-cache N` (`-mec`) keeps a persistent per-tensor pool of N experts
in VRAM for MoE expert tensors that offloading placed on the CPU host. Upstream
R1 of the feature only supported a single accelerator; Phase 1 makes it work
with `--split-mode layer` across several GPUs:

- Each tensor's pool is allocated on `model.dev_layer(il)`, the device that runs
  the layer, so every GPU gets its own independent pool set. Layers assigned to
  the CPU host backend are skipped (a VRAM pool there would only add
  cross-device copies).
- `-mec` and `--moe-expert-cache-rail-mb` accept a comma-separated per-device
  list (device order); a single value is broadcast. `-mec 48,0` disables the
  pool on the second device only.
- Per-device budget: `free VRAM (measured after KV and weights) - worst-case
  compute buffer (pp/tg, measured without allocating) - rail`. The rail
  defaults to 512 MiB. This fixes `cudaMalloc` OOM when a large `-mec` filled
  the VRAM needed by `graph_reserve`.
- If the requested slots do not fit, the slot count is scaled down uniformly
  across all of that device's tensors instead of pooling some layers and
  dropping the rest to the host-copy path. Over-requesting `-mec` degrades
  gracefully; the applied scale is logged.
- Fix for a latent bug: DeepSeek-V4 uses separate `ffn_up_exps`,
  `ffn_gate_exps`, `ffn_down_exps` (three tensors per layer); the collector
  overflowed a two-element array.
- Optional per-tensor slot overrides (`--moe-expert-cache-down`,
  `--moe-expert-cache-gate-up`), warm pin (`--moe-expert-cache-warm`), and
  windowed hit-rate telemetry appended to the server `n_gen/tg` lines when
  `GGML_MOE_POOL_STATS=1` is set.

`TENSOR`/`ROW` split (tensor parallelism) is still disabled with a warning;
per-shard pools are future work. No prefetch yet.

## Test setup

- 2 GPUs: RTX 5070 Ti 16 GB on PCIe x16 (CUDA0), RTX 4060 Ti 16 GB on PCIe x4
  (CUDA1). Windows.
- Model: DeepSeek-V4-Flash-Vision-Exp, UD-IQ3_XXS (4 shards).
- Server flags: `-c 262000 --parallel 1 -ngl 99 --split-mode layer -ncmoe 99`
  (all experts on CPU, dense split by layer across both GPUs).
- Decode is single-sequence greedy-ish; numbers are single runs on one machine.

## Decode results

| config | CUDA0 slots | CUDA1 slots | window hit d0/d1 | tg t/s | pp t/s |
|--------|------------:|------------:|------------------|-------:|-------:|
| `-mec 0 -ncmoe 39` (baseline) | - | - | - | 5.36 | 43.7 |
| `-mec 48` | 48 | 48 | 86% / 94% | 9.09 | - |
| `-mec 48,56` | 48 | 56 | 85% / 95% | 9.71-9.90 | - |
| `-mec 48,0` | 48 | 0 | 84% / - | 6.10 | - |
| `-mec 48,128` (auto-scaled) | 48 | ~55 (44%) | - | 9.90 | 45.7 |
| `-mec 128,128` (auto-scaled, final) | ~47 (37%) | ~58 (46%) | 83% / 91% | 10.32 | 49.5 |

Notes:

- Baseline uses `-ncmoe 39` (four extra dense layers on GPU) because `-mec 0`
  with `-ncmoe 99` was slower; both are the same class of stock host-copy path.
- `-mec 48,0` falls back to roughly baseline decode, which confirms that the
  second GPU's pool does real work (the GPU was not idle; monitoring was
  misleading).
- The first post-rebase `-mec 48` run measured 8.97 t/s; later runs with the
  uniform-scaling and compute-buffer fixes were 9.09-10.32 t/s.
- Hit rates are the windowed per-device aggregate printed next to the tg line.
- Prefill is unchanged within noise (40-50 t/s) by construction: a pp ubatch
  selects more distinct experts than the pool has slots, so it takes the stock
  host-copy path and never evicts the decode set.

## Auto-scaling and the compute buffer

- Before the budget fix, `-mec 128,128` failed during `graph_reserve` with
  `cudaMalloc failed: out of memory` on a 1976.50 MiB CUDA0 allocation: the
  pools had consumed the VRAM that the prompt-processing compute buffer needed.
- Measuring the worst-case pp/tg compute buffers without allocating and
  subtracting them per device (plus a 512 MiB rail) removed the OOM.
- `-mec 128,128` then auto-scaled to ~37% (about 47 slots) on CUDA0 and ~46%
  (about 58) on CUDA1 and ran at 10.32 t/s, the best result so far, with no
  spill to shared memory.

## Open items

- `TENSOR`/`ROW` split (per-shard pools).
- Prefetch / overlap of cache misses; imatrix-guided pinning instead of
  first-N warm pin.
- Prefill acceleration (separate track, likely a KTransformers-style activation
  exchange rather than weight streaming).
- Engram table mmap tier (DeepSeek V4.1), if applicable.

## Reproduce (2 GPUs, layer split)

```
llama-server -m <model> -c 262000 -ngl 99 --split-mode layer -ncmoe 99 \
    -mec 48,128 --moe-expert-cache-rail-mb 1024,1024
```

Watch the startup lines for `expert pool budget ... on <dev>` and
`scaled to X%`, and run with `GGML_MOE_POOL_STATS=1` to get the windowed
`moe expert pool: hit ... d0=..% d1=..%` line next to `n_gen/tg`.
