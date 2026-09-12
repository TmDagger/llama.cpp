# RFC: Persistent expert slot pool for MoE CPU offload (`--moe-expert-cache`)

> Draft for a ggml-org/llama.cpp discussions post. Follows up on feature request
> [#20757](https://github.com/ggml-org/llama.cpp/issues/20757). Companion branch:
> `moe-expert-pool`. **Status: validated on consumer hardware (RTX 4090, CUDA):
> perplexity-equivalent, long-session stable, +34% decode on a real code-generation
> workload on top of +84% on the neutral benchmark.**

## Summary

Add `--moe-expert-cache N` (-mec N): for every MoE expert weight tensor that offloading
placed in host memory, keep a persistent pool of `N` expert slots in accelerator memory.
Cache hits serve a decode step with **zero** host-to-device traffic; misses copy the
expert in and evict the least recently used slot. Expert ids are remapped to slot ids
through a per-tensor map table applied with `ggml_get_rows` — **no new ggml op and no
backend-specific code** — so CUDA, Vulkan, Metal and CPU all work unchanged.

Measured on a single RTX 4090 24 GB, Qwen3.8-Flash-Next (qwen4exp, 48 MoE layers, all
experts on CPU, UD-Q3_K_XL, 8 K context, greedy):

| -mec | decode tg512 (t/s) | vs baseline | prefill          |
|-----:|-------------------:|------------:|------------------|
| 0    | 11.55 ± 0.41       | —           | pp512 104.5 ± 9.9 / pp2048 113.4 ± 5.7 |
| 16   | 12.89 ± 0.33       | +12%        | —                |
| 32   | 17.54 ± 0.15       | +52%        | pp512 119.1 ± 10.9 / pp2048 112.9 ± 4.8 |
| 48   | 19.09 ± 0.62       | +65%        | —                |
| 64   | 21.21 ± 0.46       | **+84%**    | —                |

On a real code-generation workload served through `llama-server` (173-token prompt,
512 generated tokens, warm rounds): 12.82 → 14.10 (+10%) at N=32 and → 17.14 (**+34%**)
at N=64; a long multi-task session (2561 tokens with pool-churning 1 K-token
generations interleaved) held anchor outputs **byte-identical across four runs** with
no throughput decay. Greedy decoding with the pool active is **perplexity-equivalent**
to the host-copy path (PPL 3.3096 ± 0.053 vs 3.3262 ± 0.053, deterministic reruns;
difference well inside the error bars).

Prefill is unchanged within noise by construction (below). A backend-level matrix test
passes **bit-exact** on CUDA (RTX 4090) and Metal (AMD dGPU): 5 quant types × dual pools
sharing one routing × ubatch sizes 1/3 × cold / full-eviction / hit+reload /
graph-rebuild. Server-level greedy A/B initially diverged deterministically from the
host-copy path; root cause was found and fixed (details in *Validation* — including one
invalid verification round we are disclosing rather than hiding).

## Problem

With `--cpu-moe` / `--n-cpu-moe` (or the auto-fitter), decode re-streams every selected
expert over PCIe on every token: the selective expert copy added in #15346 removed the
*unused* experts but still copies the same *hot* experts token after token. Routing is
skewed, so for code-generation-style workloads the same small expert set is re-read
thousands of times. #20757 requests a cache for exactly this.

## Why not the other approaches on the table

- **`--prefetch-weights` ([#21067](https://github.com/ggml-org/llama.cpp/pull/21067))**
  overlaps next-layer transfers with compute but cannot know the next layer's routing;
  for MoE it was measured moving 2.06× the bytes with +47.8% TTFT. Complementary, not
  competing: prefetch targets dense/prefill transfer overlap, this targets decode
  residency. The two can share the copy-stream infrastructure.
- **Dynamic caches with SLRU/probation (original #20757 proposal)** let prefill bursts
  evict the decode hot set; a frequency-gated admission filter was needed to recover.
  This design sidesteps the problem structurally: **a ubatch that can touch more
  distinct experts than the pool has slots never goes through the pool at all** — it
  takes the existing selective-copy path. Prefill therefore behaves exactly like
  master (verified: pp512/pp2048 unchanged), and prefill never evicts decode state.
- Prior PRs in this space (#21614/#21620/#24524) were closed for scope/review reasons;
  this branch is deliberately small in backend impact: zero kernel changes, one
  scheduler hook, ~700 lines total of which ~350 are the scheduler core.

## Design

1. `ggml_backend_sched_register_expert_pool(sched, w, backend_id, n_slots, &table)`
   allocates `n_slots * expert_size` (+ a small NaN-safe tail) on the compute backend
   and a host-side I32 `map_table` shaped `[1, n_expert]` (flagged as a graph input).
2. `llm_graph_context::build_lora_mm_id` — the single funnel for all MoE expert
   matmuls — routes through the pool when one is registered for the tensor **and** the
   ubatch cannot select more distinct experts than slots (`n_expert_used * n_tokens ≤
   n_slots`). The remap is `get_rows(table, cont(ids))`: existing ops only.
3. The map-table buffer is registered with the scheduler's pooled-split boundary check,
   so the remap `GET_ROWS` always anchors its own split. In that split's prologue the
   scheduler reads the (original) expert ids back to the host, updates LRU state,
   issues async H2D copies for misses and rewrites the map table — *before* the same
   split's input copies upload the fresh table. (This ordering is the fix described
   below; anchoring the update anywhere later lets the remap consume a stale table.)
4. Safety rails: pool memory is sized per device as `free VRAM - rail`, where free VRAM
   is measured after the KV cache and model weights are already allocated (so KV is not
   counted twice) and the rail is a per-device reserve for compute buffers
   (`--moe-expert-cache-rail-mb`, default 1024 MiB, comma-separated per device); tensors
   beyond the budget keep the selective-copy path; pools are disabled under pipeline
   parallelism; slot-overflow is a hard assert instead of silent corruption.

### Multi-GPU (Phase 1)

With `--split-mode layer`, each layer and its experts are assigned to one device, so
the pool for an offloaded expert tensor is placed on `model.dev_layer(il)` - the same
device that executes the layer. This gives one independent MEC per GPU with no
cross-device pool copies. Both the slot count and the rail are per device: a single
value is broadcast to all devices, a comma-separated list is applied in device order.

```
--mec 48,64 --moe-expert-cache-rail-mb 1024,512
```

Per-device slot counts let a device with more free memory keep a larger hot set. Set a
device's `-mec` to 0 to disable the pool on that device only.

Layers that run on the CPU host backend are skipped (a VRAM pool would only pull the
`MUL_MAT_ID` off the CPU and add cross-device copies). `TENSOR`/`ROW` split (tensor
parallelism) shards expert tensors across devices and is currently disabled with a
warning; per-shard pools are future work.

Runtime telemetry: with `GGML_MOE_POOL_STATS=1` `llama-server` appends a windowed
aggregate of the MoE pool hit rate (total and per device) to its periodic
`n_gen = ... tg = ...` line, and the scheduler logs per-pool hit/miss every 512
updates. This makes it visible whether every device is actually serving its pool.

Slot sizing: `N` must cover the decode working set, not just top-k — `N = top-k` is a
full-miss worst case and measurably *slower* than master (the per-layer id readback
synchronization has nothing to buy). Start at 2–4× top-k and scan; gains are
routing-skew dependent (code workloads benefit most, flat-routing models least).

The engagement criterion is predictable from hardware, without benchmarking: a pool
pays off only where the hit rate clears the **break-even hit rate
`h* = 1 − PCIe_BW / RAM_BW`** (hits are ~free from VRAM, misses stream over PCIe,
and the stock path streams over RAM). Third-party measurement confirming the model
on a PCIe 3.0 x16 / EPYC 8-channel DDR4-3200 system (136 GB/s, h* ≈ 91%), Qwen3.6-35B-A3B
UD-Q6_K_XL with 40 of 48 MoE layers offloaded, 512-token generations:

| N | per-layer hit rate | vs h* | observed |
|--:|-------------------|-------|----------|
| 16 | 60.7–87.4% | all below | slower than stock (2.8× at mec=16) |
| 160 | 94.4–98.6% | all above | +54% over stock |
| 196 | rail skipped layers 35–39 (15 tensors) | mixed execution | slower than N=160 |

(Reported by @MichaelDietzel in the RFC thread. On PCIe 4.0/5.0 rigs h* sits much
lower — a 4090 box measured +12% at N=16 — which is why the crossover must be part of
the documentation rather than a fixed threshold.)

## Validation — including a bug we found, mis-verified once, then actually fixed

The full evidence chain (raw captures, the invalid round, the fix, the bypass control)
is archived in
[memoriaru/llama-cpp-expert-pool-stale-table-fix](https://github.com/memoriaru/llama-cpp-expert-pool-stale-table-fix).

1. **Backend matrix, bit-exact.** `tests/test-expert-pool.cpp` runs the pooled
   `MUL_MAT_ID` and the regular host-copy path in the same process — Q2_K/Q3_K/Q4_K/
   Q6_K/Q8_0 × two pools sharing one routing (fused gate_up + down shape) × ubatch 1/3
   × cold/evict/hit/rebuild. 40/40 on CUDA and Metal.
2. **Server A/B diverged deterministically.** Greedy, 834-token prompt, all-experts-on-
   CPU: `-mec 0` is run-to-run byte-identical (three runs across days and code
   versions); `-mec 32` diverged at byte 17 on a coherent near-tie ("wants" vs "needs")
   and stayed deterministic across environments and code revisions.
3. **One invalid verification round, disclosed.** A first "fix verification" that
   showed mec0/mec32 agreeing was produced against the wrong server instance — the
   `-mec 32` launch had failed (`No such file or directory` in `f32.err`) and both
   curls hit the still-running `-mec 0` server (visible retroactively in the response
   `timings`: `cache_n = 830` on what should have been a cold prompt). Lesson adopted
   in the test guide: verify the instance (startup log, prompt cache counters) before
   trusting an A/B pair.
4. **Root cause.** The remap `GET_ROWS` reads the map table through a `RESHAPE` view;
   views carry no buffer, so the pooled-split boundary check (which keys on src
   buffers) could not see it, and the remap could land in an earlier split than the
   pool-update prologue — consuming the *previous* ubatch's table. Every cache miss
   then read whatever expert last occupied the remapped slot (~20–30% wrong weight
   reads per step on a 512-expert model with 32-slot pools). Small enough to stay
   coherent, deterministic enough to reproduce byte-for-byte.
5. **Fix.** The table is created as `[1, n_expert]` and consumed directly (no view),
   its buffer is registered with the boundary check so the remap anchors its own
   split, and the pool update runs at that remap, before the same split uploads the
   fresh table. The bypass control — pools allocated but the graph on the host path —
   is byte-identical to `-mec 0`; with the pool active the A/B now agrees for 353
   bytes and then flips once on a near-tie (byte 354, deterministic), consistent with
   llama.cpp's known sensitivity of kernel selection to buffer placement (the same
   class of near-tie flips observed when changing `-ngl`/tensor placement). The
   perplexity check quantifies the residual: **PPL 3.3096 ± 0.053 with the pool vs
   3.3262 ± 0.053 without** (8 × 4096-token chunks, deterministic reruns) — the
   difference is well inside the error bars, i.e. quality-equivalent.
6. **Long session.** 2561 generated tokens with two 1 K-token generations churning
   the LRU state between four replays of the same anchor prompt: anchor outputs
   byte-identical (same SHA-1 four times), throughput stable (13.9–15.3 t/s).

## Maintenance footprint

~700 lines: `ggml-backend.cpp` (+~350: pool struct, register/update, split hooks),
`llama-graph.cpp` (+~20 remap), `llama-context.cpp` (+~70 registration & budget),
argument plumbing, `llama-bench` flag, and the self-contained matrix test. No changes
to any backend, no new ops, no allocator changes.

## Limitations / future work

- One ids readback sync per offloaded MoE layer per token — cheap when hits dominate
- Residual near-tie divergence vs the host-copy path: perplexity-equivalent (see
  *Validation* §5); we propose treating it like the variance already accepted for
  `-ngl` changes
- Multi-GPU: per-device pools are supported for `--split-mode layer` (Phase 1), one
  pool set per device placed on `model.dev_layer(il)`; `TENSOR`/`ROW` split is disabled
  with a warning (per-shard pools are future work)
- Platform sensitivity: on PCIe 3.0 hosts with very fast memory the break-even hit
  rate sits high (see the h* formula above) — the documentation carries the formula
  and the stats flag so users can predict engagement before spending GPU hours
- No SSD tier, no prefetch, no imatrix-guided pinning (all mentioned in #20757 and
  composable later)

## Reproduce

```bash
cmake -B build -DGGML_CUDA=ON && cmake --build build -j
./build/bin/test-expert-pool                                  # bit-exact matrix
./build/bin/llama-bench -m <model> -ngl 99 -ncmoe 99 -mec 0,16,32,64 -n 512 -p 0
./build/bin/llama-cli  -m <model> -ngl 99 -cmoe -mec 32 -p <prompt> -n 128 --temp 0
# 2 GPUs, layer split: one independent pool per device
./build/bin/llama-server -m <model> -ngl 99 --split-mode layer -ncmoe 99 -mec 32 \
    --moe-expert-cache-rail-mb 1024,512

# per-pool hit/miss stats (every 512 updates)
GGML_MOE_POOL_STATS=1 ./build/bin/llama-server -m <model> -ngl 99 -ncmoe 99 -mec 32
```

---
*AI usage disclosure: this feature was developed with AI assistance (analysis, code and
benchmarks); the author ran, verified and debugged all results on their own hardware
and has reviewed every line — including writing the fix for the bug the validation
uncovered.*
