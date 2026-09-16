# MoE expert cache, layer split and telemetry (fork report)

This document summarizes the work on the fork branch `layer-split-moe-telemetry`
(formerly `phase2-bw-layer-split`): the expert cache distribution knobs, the layer
split strategies, and the telemetry channels added to tune them.

All measurements below were taken on DeepSeek-V4-Flash-Vision-Exp (MoE, 256 experts,
top-k = 6) on 2x 16 GB CUDA GPUs with `-ncmoe 99` (experts in host RAM, hot experts
cached in VRAM).

## 1. Summary of what was built

- Per-device expert pools for layer split: layers 0-21 on the first GPU, 22-42 on the
  second; each layer's expert tensors are pooled on the device that runs the layer.
- A miss copies only the missing expert (one expert, not the whole tensor) from host
  RAM into that layer's pool on its own device.
- Layer split strategies: `bw` (VRAM bandwidth), `eq`, `slots` (expert cache slot
  balancing) and `manual`; the auto default is `slots` when `-mec > 0`, else `bw`.
- Expert cache controls: per-device slots, per-tensor overrides, warm pin, whole-layer
  pin (`--mec-whole`) and per-layer slots (`--mec-per-layer`).
- Telemetry: per-layer hit/miss, miss histogram, expert lifetime, evictions, fallbacks,
  copied bytes and pool update time, plus printed recommendations.
- Fixes: hit/miss was counted twice per compute (remap GET_ROWS plus pooled
  MUL_MAT_ID); the `slots` split read one element past the end of its vectors.
- Expert pool hardening (ported from the `moe-expert-pool` branch): the map table is now
  pool-owned on the device (stable pointer, no dependency on the scheduler split input
  copy path), the remap aborts loudly if its pool was not updated, the expert-tensor
  collector is unbounded, and the hot path keeps a single queue sync per ubatch.

## 2. Launch parameters

### 2.1 Layer split

- `--layer-split-strategy {bw,eq,slots,manual}`
  - `bw`: measure per-device VRAM bandwidth at startup (5 rounds, median) and split
    layers proportionally. Slower for decode when the cache is on, because it loads
    the fast card with more layers and its expert cache thrashes.
  - `eq`: equal shares.
  - `slots`: balance the estimated number of expert cache slots per device. This is
    the auto default when `-mec > 0`.
  - `manual`: use `-ts`.
  - Setting `-ts` selects `manual` automatically.
- `--vram-bw GB0,GB1,...`: manual VRAM bandwidth for `bw` (skips the benchmark).

### 2.2 Expert cache

- `-mec, --moe-expert-cache N0,N1,...`: slots per expert tensor, per device (a single
  value is broadcast). Requires host experts (`--cpu-moe` / `--n-cpu-moe` / `-ot`).
- `--moe-expert-cache-rail-mb MiB0,MiB1,...`: per-device VRAM reserve kept free for
  compute buffers and driver headroom. Default 768 MiB. On Windows/WDDM a low headroom
  makes the driver page to host memory and stall; raise this for the affected device.
- `--moe-expert-cache-legacy-kv-estimate`: subtract an estimated max-context KV from
  the budget (off by default; the preallocated KV is already excluded from free VRAM).
- `--moe-expert-cache-down N`, `--moe-expert-cache-gate-up N`: per-tensor slot
  overrides (0 = use `-mec`).
- `--moe-expert-cache-warm N`: pin the first N experts of each pool (index based).
- `--mec-whole, --moe-expert-cache-whole N | N0,N1,... | A-B,...`: keep all experts of
  the given layers resident. A single number means the first N layers; otherwise a list
  with ranges, e.g. `0,1,5` or `0-2,5`. These layers get a dedicated-slot pool
  (slot == expert id) with a fast path that skips the id remap, so there are no
  evictions and no misses. Their full footprint is reserved from the device budget
  first; if it does not fit, a warning is printed and the pin is disabled.
- `--mec-per-layer, --moe-expert-cache-per-layer N0,N1,... | L:N | a`: per-layer slot
  counts. Positional values apply in layer order; `a` marks a dynamic layer (follows
  `-mec`); `0` disables the cache for that layer; `L:N` targets a specific layer.
  Explicit values are reserved from the budget before the dynamic layers scale down.
  `--mec-whole` has priority over `--mec-per-layer` on the same layer (a warning is
  printed).

Budget priority per device: whole layers, then explicit `--mec-per-layer`, then the
dynamic layers scaled uniformly to whatever budget is left.

## 3. Telemetry channels

Two environment variables gate everything:

- `GGML_MOE_POOL_STATS=1`: the per-device window line during generation, the
  per-generation summary and the recommendations.
- `GGML_MOE_POOL_STATS_LAYERS=1`: the verbose per-layer per-step lines.
- `-lv 4` is needed for the low-level `ggml_backend_sched_update_expert_pool` per-pool
  INFO lines (library INFO maps to verbosity 4).

### 3.1 Per-layer per-step lines (need `..._LAYERS=1`)

```
moe L42 hit=81.2%+-5.3% m1tok[0m=33%,1m=30%,...,6m=4%] life=1.80s ev=4674 fb=45 copy=8.70MiB/stp upd=1.30ms/stp
```

- `hit=mean+-sigma`: mean and standard deviation of the per-step hit rate over the
  output period. High sigma means the routing changes a lot between steps.
- `m1tok[0m..Km]`: histogram of how many experts were missed on a step, in percent of
  steps. `K = n_expert_used` from the model (top-k), so for top-k = 6 the buckets are
  `0m` to `6m`. Hits and misses are taken from the first expert tensor of the layer;
  the three tensors of a layer share their routing.
- `life`: average time an expert stay resident before being evicted, over the evictions
  in the period (`n/a` when there were none).
- `ev`: evictions in the period.
- `fb`: ubatch fallbacks in the period (the ubatch was wider than the pool, so it used
  the stock host-copy path; prefill always lands here).
- `copy`: bytes copied host -> device per step, averaged over the period.
- `upd`: host-side pool update time per step, averaged over the period. This is the ids
  readback, the device synchronize, the map table rewrite and issuing the async copies;
  it is not GPU compute and not the PCIe transfer itself.

### 3.2 Per-generation summary (need `GGML_MOE_POOL_STATS=1`)

```
moe summary: 43 layers, 1506 steps
moe L00 hit= 20.8%+-22.1% miss=4.75/stp copy=38.92MiB/stp upd=5.31ms/stp
...
moe dev0 copy=318.77MiB/stp upd=41.14ms/stp
moe dev1 copy=120.16MiB/stp upd=19.82ms/stp
```

Per layer: the mean and sigma over the whole generation, plus the per-step averages for
misses, copied bytes and updater time. Per device: the same copy and upd totals, so it
is easy to see which GPU is overloaded.

### 3.3 Recommendations (printed, no behavior change)

```
moe recommend: --mec-per-layer 191,256,256,61,55,...
moe recommend: --mec-whole 0,1,2
moe recommend: --ts 0.283,0.717 (equalize copy, ...)
moe recommend: --ts 0.335,0.665 (equalize upd ...)
```

- `--mec-per-layer`: nudges each layer's slots by how far its hit rate deviates from the
  device average, divided by the layer's hit chance per slot (`pre = cur * avg / hit`),
  so a slot is worth more where the routing is concentrated. The device total is
  preserved by scaling, clamped to `[n_expert_used, n_expert]`; the residual goes to the
  first layer of the device.
- `--mec-whole` or host path: layers whose hit rate barely beats the random baseline
  `n_slots / n_expert` (LRU does not help them). When pinning them whole would starve
  the remaining layers (`avg_whole < 0.8 * avg_off`), the recommendation switches to
  `--mec-per-layer 0,...` instead: those layers run the stock host path and their slots
  go to the layers where the cache does help.
- Host-pinned layers (`--mec-per-layer 0`) have no pool, so the telemetry cannot see
  them. The suggestions keep them at `0` and exclude them from the redistribution, and a
  `moe note` line lists them. The pin changes the slot distribution, so remove it for a
  clean baseline measurement.
- `--ts`: two variants, one balancing copied bytes and one balancing updater time per
  device. Both assume you optimize the slots first and then move towards the target in
  steps, watching for PCIe saturation.

### 3.4 Low-level logs

- `ggml_backend_sched_update_expert_pool: '<tensor>' hit rate X% (...), N of M slots free`
  (needs `-lv 4`): cumulative per-tensor counters, printed every 512 events.
- `init_expert_pools: expert pool budget ... (free ... - compute ... - rail ... -
  external, ... already consumed)`: how the per-device budget was computed. The external
  reserve for a draft/MTP context is held back only for the part not consumed yet, so a
  re-reserve after the draft has loaded does not subtract it a second time.
- `init_expert_pools: <dev>: requested slots do not fit ... scaled to X%`: the uniform
  downscale that keeps the pools inside the budget.
- A warning when less than about 384 MiB would stay free after the compute buffers,
  pointing at `--moe-expert-cache-rail-mb`.

## 4. How to use it

### 4.1 What to look at

- `hit` per layer: layers with a low hit rate near `n_slots / n_expert` get little from
  the LRU cache; layers with a high hit rate have concentrated routing.
- `m1tok`: the miss histogram shows whether a layer misses a few experts on most steps
  or many on a few steps.
- `life` and `ev`: how stable the resident set is.
- `copy` and `upd` per layer and per device: where the bandwidth and stall cost is.
- `upd` imbalances point at an overloaded GPU and feed the `--ts` suggestion.

### 4.2 What to expect

For this model the early layers (0-2) route almost uniformly: with about 50 of 256
slots the hit rate is near the 20% baseline, and they produce most of the misses and
copies. The later layers concentrate on a small hot set: hit rates of 85-98% with the
same slot count are normal. Per-device `copy` and `upd` can differ by 2-3x when the
split puts the "hard" layers on one card.

On a multi-GPU box, moving the uniform layers to the host path is usually the better
trade. Pinning them whole spends a full `n_expert` slots per layer, which the useful
layers pay for with a lower hit rate; running them on the CPU frees those slots, and the
DDR path is often wider than an already saturated PCIe link (a single PCIe 5.0 x16 card
was measured at ~22-24 GB/s, while part of the traffic also goes to the second card on
PCIe 4.0 x4). For DeepSeek-V4-Flash with `-mec 128`: baseline 10.07 t/s, `--mec-whole
0-2` 9.54 t/s (the whole pin pushed dev0 from 328 to 462 MiB/step of copies), while
`--mec-per-layer 0,0,0` reached 11.74 t/s and roughly halved dev0's copy and update load.
On a single card the difference is smaller.

### 4.3 Tuning scenario

1. Start without knobs: `-mec 128`, auto split (`slots`).
2. Run with `GGML_MOE_POOL_STATS=1` and read the per-generation summary.
3. For layers near the baseline, decide whole vs host path from the suggestion: pin them
   whole with `--mec-whole 0-2`, or, when the summary warns that whole pinning would
   starve the other layers, send them to the host path with `--mec-per-layer 0,0,0`.
   Both move the cache budget to the layers where it helps.
4. Re-run and apply the suggested `--mec-per-layer` (it keeps each card's slot total
   unchanged; host-pinned layers stay `0`). Compare `copy`/`upd` and t/s. To get a clean
   baseline for the active layers, re-measure once without the host pin.
5. Only then look at `--ts`: pick a `-ts` suggestion (copy or upd), set
   `--layer-split-strategy manual` and move towards the target in a few steps. After
   each step re-check the summary for PCIe saturation or a distorted target ratio, which
   means the cache is no longer optimal for the new split.
6. If a device warns about low headroom, raise `--moe-expert-cache-rail-mb` for it
   (comma list, device order).

## 5. Limitations and caveats

- Prefill and any ubatch wider than the pool fall back to the host-copy path
  (`fb`), so prefill never uses the pool and never flushes it.
- Pipeline parallelism disables the expert cache (a single pool would be overwritten
  while a previous copy still reads it). The target is unaffected because `-ncmoe`
  forces tensor overrides, which already disable PP; the draft/MTP context no longer
  inherits the cache settings.
- Pool sizes cannot change at runtime: they are created in `init_expert_pools` before
  any token. All knobs are static, applied at load.
- `--mec-whole` and `--mec-per-layer` values are guaranteed only if they fit the device
  budget; otherwise they are demoted to dynamic with a warning.
- Speculative draft/MTP contexts do not share the target cache; the target reserves the
  draft's tensor bytes (from GGUF metadata) when sizing its pools.

## 6. Backlog

- Adaptive cache slots at runtime (requires re-creating pools and re-reserving graphs).
- Rotating hot-expert lists: a logical hot/cold split of one pool, eviction by
  membership, miss-burst detection, heat maps and switching the hot list on an attention
  shift. The logical split avoids reallocation, so it is feasible; the burst logging is
  the first step.
- Session-wide telemetry aggregation (separate toggle).
