# MoE expert cache: per-layer slots, whole-layer pin and telemetry

This document describes the per-layer expert cache knobs and the telemetry used to
tune them. It is a draft for discussion. The multi-GPU layer split, draft model
placement and everything that needs 2+ GPUs are deliberately out of scope here and
live on a separate branch.

## 1. Summary of what was built

- `-mec, --moe-expert-cache N`: keep a cache of N experts per offloaded MoE weight
  tensor in VRAM; misses copy one expert from host memory.
- `--mec-whole`: keep every expert of the given layers resident (whole-layer pin).
  Those layers use a dedicated slot per expert (slot == expert id), so there is no
  eviction, no miss and no id remap.
- `--mec-per-layer`: per-layer slot counts. Explicit values are reserved from the
  budget first; the remaining (dynamic) layers scale down to fit.
- Telemetry: per-tensor copy volume and pool update time, a per-generation per-layer
  summary, per-step lines and printed `--mec-per-layer` / `--mec-whole`
  recommendations.

## 2. Launch parameters

### 2.1 Expert cache

- `-mec, --moe-expert-cache N`: slots per offloaded expert tensor. Requires host
  experts (`--cpu-moe` / `--n-cpu-moe` / `-ot`). `0` disables the cache.
- `--mec-whole, --moe-expert-cache-whole N | N0,N1,... | A-B,...`: keep all experts
  of the given layers resident. A single number keeps the first N layers; otherwise a
  comma list of indices and inclusive ranges, e.g. `0,1,5` or `0-2,5`. Requires
  `--moe-expert-cache`.
- `--mec-per-layer, --moe-expert-cache-per-layer N0,N1,... | L:N | a`: per-layer slot
  counts. Positional values apply in layer order, `L:N` targets a specific layer, `a`
  marks a dynamic layer (follows `-mec`) and `0` sends the layer to the host path.
  Requires `--moe-expert-cache`.

Budget priority per device: whole layers first, then explicit `--mec-per-layer`
values, then the dynamic layers scaled uniformly to whatever budget is left. If the
whole pin does not fit it is disabled with a warning and the layers become dynamic.

The cache supports exactly one accelerator. With more than one accelerator, or with
pipeline parallelism, it disables itself with a warning.

## 3. Telemetry channels

Two environment variables gate the server telemetry:

- `GGML_MOE_POOL_STATS=1`: the per-generation summary and the recommendations.
- `GGML_MOE_POOL_STATS_LAYERS=1`: the verbose per-step lines.
- `-lv 4` is needed for the low-level `ggml_backend_sched_update_expert_pool` per-tensor
  INFO lines (library INFO maps to verbosity 4).

### 3.1 Per-step lines (need `..._LAYERS=1`)

```
moe L42 hit=81.2%+-5.3% copy=8.70MiB/stp upd=1.30ms/stp
```

- `hit=mean+-sigma`: mean and standard deviation of the per-step hit rate over the
  output period. High sigma means the routing changes a lot between steps.
- `copy`: bytes copied host -> device per step, averaged over the period.
- `upd`: host-side pool update time per step, averaged over the period. This is the
  ids readback, the device synchronize, the map table rewrite and issuing the copies;
  it is not GPU compute and not the PCIe transfer itself.

### 3.2 Per-generation summary (need `GGML_MOE_POOL_STATS=1`)

```
moe summary: 43 layers, 1506 steps
moe L00 hit= 20.8%+-22.1% miss=4.75/stp copy=38.92MiB/stp upd=5.31ms/stp
...
moe dev0 copy=318.77MiB/stp upd=41.14ms/stp
```

Per layer: the mean and sigma over the whole generation, plus the per-step averages
for misses, copied bytes and updater time. Per device: the same copy and update
totals. The copy volume and update time are what make per-layer differences visible:
a hit-rate average can look acceptable while one layer copies far more than another.

### 3.3 Recommendations (printed, no behavior change)

```
moe recommend: --mec-per-layer 191,256,256,61,55,...
moe recommend: --mec-whole 0,1,2
```

- `--mec-per-layer`: nudges each layer's slots by how far its hit rate deviates from
  the average, divided by the layer's hit chance per slot (`pre = cur * avg / hit`),
  so a slot is worth more where the routing is concentrated. The total is preserved by
  scaling, clamped to `[n_expert_used, n_expert]`; the residual goes to the first
  layer.
- `--mec-whole` or host path: layers whose hit rate barely beats the random baseline
  `n_slots / n_expert` (LRU does not help them). When pinning them whole would starve
  the remaining layers (`avg_whole < 0.8 * avg_off`), the recommendation switches to
  `--mec-per-layer 0,...` instead: those layers run the stock host path and their slots
  go to the layers where the cache does help.
- Host-pinned layers (`--mec-per-layer 0`) have no pool, so the telemetry cannot see
  them. The suggestions keep them at `0` and exclude them from the redistribution, and
  a `moe note` line lists them. The pin changes the slot distribution, so remove it for
  a clean baseline measurement.

### 3.4 Low-level logs

- `ggml_backend_sched_update_expert_pool: '<tensor>' hit rate X% (...), N evictions,
  copied Y MiB, update Z ms (W steps), N of M slots free` (needs `-lv 4`): cumulative
  per-tensor counters, printed every 512 events.
- `init_expert_pools: expert pool budget ...`: how the budget was computed.

## 4. How to use it

### 4.1 What to look at

- `hit` per layer: layers with a low hit rate near `n_slots / n_expert` get little from
  the LRU cache; layers with a high hit rate have concentrated routing.
- `copy` and `upd` per layer: where the bandwidth and stall cost is.

### 4.2 What to expect

For a DeepSeek-V4-Flash-like model the early layers route almost uniformly: with about
50 of 256 slots the hit rate is near the 20% baseline, and they produce most of the
misses and copies. The later layers concentrate on a small hot set: hit rates of
85-98% with the same slot count are normal.

### 4.3 Tuning scenario

1. Start with `-mec 128` and no per-layer knobs.
2. Run with `GGML_MOE_POOL_STATS=1` and read the per-generation summary.
3. For layers near the baseline, decide whole vs host path from the suggestion: pin
   them whole with `--mec-whole 0-2`, or, when the summary warns that whole pinning
   would starve the other layers, send them to the host path with
   `--mec-per-layer 0,0,0`. Both move the cache budget to the layers where it helps.
4. Re-run and apply the suggested `--mec-per-layer` (it keeps the slot total
   unchanged; host-pinned layers stay `0`). Compare `copy`/`upd` and t/s.

### 4.4 Uniform-random-layer detection (why `0,0,0`)

A layer whose routing is flat gets almost nothing from the LRU cache. With no
concentration, the chance that the next used expert is already resident is about the
fraction of experts the pool holds, so the expected hit rate is the random baseline
`base = n_slots / n_expert`. The summary prints both numbers per layer, so the
classification is read off the run, not found by trial:

1. `base` per layer: `100 * n_slots / n_expert` from the same `moe L..` line as the
   measured `hit`. It moves with `-mec`, so it is computed per layer rather than fixed.
2. Uniform layer: `hit < 1.5 * base`. The 1.5 factor is a small margin above the
   baseline for noise and one-slot effects; below it the cache buys no hit rate. These
   layers also show a high `+-sigma`, because a flat routing makes the per-step hit rate
   swing, while a concentrated layer stays high and stable.
3. Whole vs host path: pinning a uniform layer whole costs `n_expert` slots. With
   `avg_off = total_slots / n_remaining` and
   `avg_whole = (total_slots - n_expert * n_uniform) / n_remaining`, if
   `avg_whole < 0.8 * avg_off` the pin would starve the layers where the cache does
   help, and the recommendation is the host path (`--mec-per-layer 0,...`); otherwise
   it is `--mec-whole ...`. Host-path layers have no pool and stay `0` in later
   suggestions.

Worked example (DeepSeek-V4-Flash, 256 experts, top-k 6, `-mec 128`): the early layers
report `hit=20.8%+-22.1%` with about 50 of 256 slots, so `base = 19.5%` and
`1.5 * base = 29.3%`; all three are below it, hence uniform. Pinning them whole would
leave the remaining layers far below `0.8 * avg_off`, so the rule picks the host path:
`--mec-per-layer 0,0,0`. Measured 10.07 t/s baseline, 9.54 t/s with `--mec-whole 0-2`,
11.74 t/s with `--mec-per-layer 0,0,0`. The two allocations were compared by trial only
to confirm the rule after it picked `0,0,0`.

## 5. Limitations and caveats

- Single accelerator only. With more than one accelerator the cache disables itself;
  per-device pools and the layer split are a separate follow-up.
- Prefill and any ubatch wider than the pool fall back to the host-copy path, so
  prefill never uses the pool and never flushes it.
- Pipeline parallelism disables the expert cache.
- Pool sizes cannot change at runtime: they are created in `init_expert_pools` before
  any token. All knobs are static, applied at load.
- `--mec-whole` and `--mec-per-layer` values hold only if they fit the budget;
  otherwise they are demoted to dynamic with a warning.

## 6. Backlog

- Adaptive cache slots at runtime (requires re-creating pools and re-reserving graphs).
- Per-device pools and layer split for 2+ GPUs (separate branch).
- Draft/MTP expert cache.
