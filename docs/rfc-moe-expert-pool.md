# RFC: Persistent expert slot pool for MoE CPU offload (`--moe-expert-cache`)

> Draft for a ggml-org/llama.cpp discussions post. Follows up on feature request
> [#20757](https://github.com/ggml-org/llama.cpp/issues/20757). Companion branch:
> `moe-expert-pool`. **Status: validated on consumer hardware, long-session stability
> test in progress.**

## Summary

Add `--moe-expert-cache N` (-mec N): for every MoE expert weight tensor that offloading
placed in host memory, keep a persistent pool of `N` expert slots in accelerator memory.
Cache hits serve a decode step with **zero** host-to-device traffic; misses copy the
expert in and evict the least recently used slot. Expert ids are remapped to slot ids
through a per-tensor map table applied with `ggml_get_rows` — **no new ggml op and no
backend-specific code** — so CUDA, Vulkan, Metal and CPU all work unchanged.

Measured on a single RTX 4090 24 GB running Qwen3.8-Flash-Next (qwen4exp, 48 MoE
layers, top-8, all experts on CPU, Q3_K_XL, 8 K context):

| -mec | decode tg512 (t/s) | vs baseline | prefill pp512 |
|-----:|-------------------:|------------:|--------------:|
| 0    | 11.55 ± 0.41       | —           | 104.5 ± 9.9   |
| 16   | 12.89 ± 0.33       | +12%        | —             |
| 32   | 17.54 ± 0.15       | +52%        | 119.1 ± 10.9  |
| 48   | 19.09 ± 0.62       | +65%        | —             |
| 64   | 21.21 ± 0.46       | **+84%**    | —             |
| 0 (rerun) | 113.4 pp2048  | prefill baseline | —        |
| 32 (pp2048) | 112.9 ± 4.8 | prefill unchanged | —       |

Greedy A/B on the real model is **byte-identical** between `-mec 0` and `-mec 32`
(independent baseline reruns agree byte-for-byte). A backend matrix test passes
bit-exact on CUDA (RTX 4090) and Metal (AMD dGPU): 5 quant types × dual pools sharing
one routing × ubatch sizes 1/3 × cold / full-eviction / hit+reload / graph-rebuild.

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
  scheduler hook, ~650 lines total of which ~300 are the scheduler core.

## Design

1. `ggml_backend_sched_register_expert_pool(sched, w, backend_id, n_slots, &table)`
   allocates `n_slots * expert_size` (+ a small NaN-safe tail) on the compute backend
   and a host-side I32 `map_table[n_expert]` (flagged as a graph input).
2. `llm_graph_context::build_lora_mm_id` — the single funnel for all MoE expert
   matmuls — routes through the pool when one is registered for the tensor **and** the
   ubatch cannot select more distinct experts than slots (`n_expert_used * n_tokens ≤
   n_slots`). The remap is `get_rows(reshape(table), ids)`: three existing ops.
3. Before each split that reads a pool, the scheduler reads the (original) expert ids
   back to the host (the router is guaranteed to be in a previous split — pooled
   `MUL_MAT_ID` nodes force a split boundary), updates LRU state, issues async H2D
   copies for misses, and rewrites the map table; the regular split input-copy path
   then uploads the table as it would any input.
4. Safety rails: total pool memory is capped at half the device's free memory at load
   time (tensors beyond the budget silently keep the selective-copy path); pools are
   disabled under pipeline parallelism; slot-overflow is a hard assert instead of
   silent corruption.

Slot sizing: `N` must cover the decode working set, not just top-k — `N = top-k` is a
full-miss worst case and measurably *slower* than master (the per-layer id readback
synchronization has nothing to buy). Start at 2–4× top-k and scan; gains are
routing-skew dependent (code workloads benefit most, flat-routing models least).

## Maintenance footprint

~650 lines: `ggml-backend.cpp` (+~300, one pool struct + register/update/split-hook),
`llama-graph.cpp` (+~20 remap), `llama-context.cpp` (+~60 registration & budget),
argument plumbing, `llama-bench` flag, and a self-contained matrix test
(`tests/test-expert-pool.cpp`, skips itself without an accelerator). No changes to any
backend, no new ops, no allocator changes.

## Limitations / future work

- Single accelerator (pools go to the first device; per-layer placement is a TODO)
- One ids readback sync per offloaded MoE layer per token — cheap when hits dominate
- No SSD tier, no prefetch, no imatrix-guided pinning (all mentioned in #20757 and
  composable later)
- Long-session stability run in progress

## Reproduce

```bash
cmake -B build -DGGML_CUDA=ON && cmake --build build -j
./build/bin/test-expert-pool                                  # bit-exact matrix
./build/bin/llama-bench -m <model> -ngl 99 -ncmoe 99 -mec 0,16,32,64 -n 512 -p 0
./build/bin/llama-cli  -m <model> -ngl 99 -cmoe -mec 32 -p <prompt> -n 128 --temp 0
```

---
*AI usage disclosure: this feature was developed with AI assistance (analysis, code and
benchmarks); the author ran, verified and debugged all results on their own hardware
and has reviewed every line.*
