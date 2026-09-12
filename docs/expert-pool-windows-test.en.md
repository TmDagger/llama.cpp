# MoE Expert Pool (RFC ggml-org#20757) — Windows/CUDA Testing Guide

> 中文版: [expert-pool-windows-test.md](expert-pool-windows-test.md)

Feature branch: `fix-moe-pool-rail` (rebased on upstream master, per-device pools)

Pull on Windows:

```powershell
git clone -b fix-moe-pool-rail http://192.168.0.53:8620/tmdagger/llama.cpp.git
cd llama.cpp
```

## How It Works

`--moe-expert-cache N` / `-mec N`: for every MoE expert weight tensor offloaded to the
CPU (tensors placed on the host by `--cpu-moe`, `--n-cpu-moe`, `-ot` overrides or the
auto-fitter), keep a **persistent pool of N expert slots in accelerator memory**.

- **Decode (small ubatch)**: cache hits are served from VRAM with zero copies (no more
  per-token RAM→VRAM re-copying); on a miss the expert is copied in and the least
  recently used slot is evicted. Expert ids are remapped to slot ids through a per-tensor
  map table applied with `ggml_get_rows` — **no new ggml op, no backend-specific code**,
  so CUDA/Metal/Vulkan/CPU all work out of the box.
- **Prefill (large ubatch)**: whenever a ubatch can touch more distinct experts than the
  pool has slots (`n_expert_used × n_tokens > N`), the graph automatically falls back to
  the existing selective-copy path — prefill behaves exactly like upstream, zero
  regression. This sidesteps the "prefill evicts hot decode experts" problem discussed
  in the issue: prefill simply never goes through the pool.
- The pool only applies to weight tensors that live in a host buffer and are computed on
  an accelerator; CPU-only runs and fully-on-GPU models are unaffected.

Constraints:
- `N` must be ≥ the maximum number of distinct experts a decode ubatch can select, i.e.
  `n_expert_used` (the top-k). Below that the graph always takes the fallback path.
- **`N` = top-k is the worst case, not the minimum useful value**: with flat routing every
  step misses all slots and you pay the pool overhead (per-layer expert-id readback
  synchronization) on top of the same copies the selective path would make. Measured on
  OLMoE (64 experts, top-8, flat-ish routing): `N = 8` was ~35% *slower* than selective
  copy. Aim for `N` covering the decode working set — start at 2–4× top-k and scan up,
  observing the hit rate with `GGML_MOE_POOL_STATS=1` (per-pool hit/miss logged every
  512 updates). Gains are routing-skew dependent (code workloads benefit most,
  flat-routing models least).
- **Mind the PCIe generation**: the pool moves expert compute from the CPU
  (memory-bandwidth speed) to the GPU (misses stream over PCIe). On PCIe 3.0
  (~12 GB/s) with fast multi-channel memory, a miss costs an order of magnitude more
  than on PCIe 4.0/5.0 rigs — low N can be far *slower* than CPU-only mode (community
  measurement: mec=16 was 2.8× slower than mec=0 on PCIe 3.0 + EPYC/8-channel DDR4,
  while a 4090/PCIe 4.0 box saw +12% at mec=16). On such platforms either go
  big-N/high-coverage or don't engage the cache.
- Upper bound is `n_expert - 1`; beyond that just keep the weights in VRAM via `-ngl`.

VRAM budget estimate: `offloaded MoE layers × 2–3 tensors (gate_up+down[+down]) × N ×
bytes-per-expert`. Example: 48 layers, 128 experts, ~24 MB per Q4 expert, N = 32 →
48×2×32×24 MB ≈ 71 GB (does not fit); N = 8 → ~18 GB. The implementation additionally
enforces a per-device VRAM rail: the pool budget is
`free VRAM measured after the KV cache and weights are allocated − worst-case compute
buffer − rail`. The compute buffer (pp/tg) is measured without allocating before the
pools are created, so a large `-mec` can no longer starve `graph_reserve` and fail with
`cudaMalloc failed: out of memory`. If the requested `-mec` does not fit, the slot
count is scaled down **uniformly across all of that device's tensors** until it fits, so
over-requesting degrades gracefully (fewer slots everywhere) instead of pooling some
layers and dropping the rest to the slow host-copy path. `llama-server`/`llama-cli` log
the effective per-device budget and the applied scale. Only if even one slot per tensor
does not fit is the device left entirely on the host path. A fully pooled device is
reported by the `pooled N offloaded MoE expert weight tensors ...` summary line.
The rail defaults to 512 MiB and is configurable per device. The slot count `-mec`
takes the same per-device list form (device order), so a card with more free memory can
keep a larger hot set:

```powershell
# one value: same rail on every device
--moe-expert-cache-rail-mb 1024
# comma list: applied in device order
--moe-expert-cache-rail-mb 1024,512
# per-device slot counts; a device with -mec 0 gets no pool
-mec 48,64
```

Related tuning flags: `--moe-expert-cache-down N`, `--moe-expert-cache-gate-up N`
(per-tensor slot counts; 0 = use `-mec`), `--moe-expert-cache-warm N` (pin the first N
experts of each pool at init; `LLAMA_MOE_POOL_UNPIN_AFTER=N` lifts the pin after N
updates), and `--moe-expert-cache-legacy-kv-estimate` (restore the old KV-subtracting
budget, off by default).

## Windows Build

```powershell
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --config Release --target llama-cli llama-bench test-expert-pool
.\build\bin\Release\test-expert-pool.exe
```

`test-expert-pool` is the deterministic correctness gate: it runs the pooled
`MUL_MAT_ID` and the regular host-copy path **in the same process** on Q2_K weights with
strided expert ids (as produced by `ggml_argsort_top_k`) and requires bit-exact equality
across cold start, full eviction, hit+reload, and graph rebuild. It skips itself when no
accelerator backend is available.

## Test Matrix (suggested order)

### 0. Methodology red lines: own the VRAM exclusively, verify the instance

**Own the VRAM**: stop every other VRAM consumer (production services, other
containers) and verify with `nvidia-smi` before starting. Under memory pressure
llama-server runs through its startup flow despite failed allocations and produces
plausible-looking garbage — a *false divergence*.

**Verify the instance**: before each curl, confirm the target server is the one you
just started — check the startup log timestamp, confirm the previous instance exited,
and check the response `timings`: `prompt_n`/`cache_n` must match expectations (a cold
prompt must not report `cache_n = 830`). Hard-won lesson: one `-mec 32` launch failed
with a path error (`f32.err`: `No such file or directory`) and both curls hit the
still-running mec0 instance, producing a false "fix verified byte-identical" result
while the real divergence was still present. Audit all three before archiving an A/B
pair.

### 1. Correctness

```powershell
# A/B text comparison (greedy, same seed) — outputs must match exactly
.\build\bin\Release\llama-cli.exe -m <model> -ngl 99 -cmoe -p <prompt> -n 128 --temp 0 --simple-io -st > out_base.txt
.\build\bin\Release\llama-cli.exe -m <model> -ngl 99 -cmoe -mec 32 -p <prompt> -n 128 --temp 0 --simple-io -st > out_pool.txt
fc out_base.txt out_pool.txt
```

With `N ≥ 2×top-k` the decode working set is (mostly) cached; both runs should produce
identical text. If they diverge, stop and investigate (check the mul_mat_id tail-padding
interaction with the quant type first). Note: on memory-constrained machines the
*baseline itself* can be nondeterministic across runs (observed on a 4 GB macOS card);
in that case trust `test-expert-pool` as the correctness oracle.

### 2. Decode Speedup (the core payoff)

```powershell
.\build\bin\Release\llama-bench.exe -m <model> -ngl 99 -ncmoe 99 -mec 0,8,16,32,64 -n 128,512 -p 0
```

Watch the tg (token generation) column as `-mec` grows. Expectations from the issue
discussion: gains require `N ≥ 2×top-k`; very large `N` (approaching `n_expert`) flattens
out or regresses (WDDM memory-pressure cliff). Skewed routing (code-generation workloads)
benefits the most; flat-routing models (gpt-oss style) benefit the least.

### 3. Prefill Zero-Regression

```powershell
.\build\bin\Release\llama-bench.exe -m <model> -ngl 99 -ncmoe 99 -mec 0,32 -p 512,2048 -n 0
```

The pp columns of both runs should match within noise (large ubatches always take the
fallback path; the pool is not involved).

### 4. Long-Session Stability

Multi-turn conversation for 2000+ tokens; watch for NaNs/crashes (this exercises the LRU
eviction path). The log should show `pooled X offloaded MoE expert weight tensors`.

### 5. Multi-GPU (two cards, layer split)

With `--split-mode layer` the pools are per device: each GPU pools the experts of the
layers it runs. Watch the startup log for one `expert pool budget ... on <dev>` line per
device, one summary line per device, and confirm with `nvidia-smi` that both cards hold
pools. A tensor that does not fit logs `... exceeds the remaining budget ...`.

```powershell
.\build\bin\Release\llama-server.exe -m <model> -ngl 99 --split-mode layer -ncmoe 99 `
    -mec 48,64 --moe-expert-cache-rail-mb 1024,512
```

Compare against `-mec 0` under the same split; decode tg should improve on both devices.
The previous `expert cache disabled: N accelerator devices present` warning must be gone.
Layer split with `-ncmoe` keeps pipeline parallelism off (`--n-cpu-moe` adds tensor
overrides), which is required for the pool to stay enabled.

To see whether every device is actually serving its pool, run with `GGML_MOE_POOL_STATS=1`:
the server appends a windowed hit-rate aggregate to each periodic `n_gen = ... tg = ...`
line (total and `d0=..% d1=..%`), and the scheduler logs per-pool hit/miss every 512
updates. A device with no `dN=` entry in the window is not serving decode hits.

## Known Limitations (MVP)

- Disabled automatically under pipeline parallelism (warning logged)
- `TENSOR`/`ROW` split (tensor parallelism) is disabled with a warning; per-shard pools
  are future work. `--split-mode layer` and `none` are supported
- Only affects expert tensors actually placed on the host
- One expert-id readback synchronization per offloaded MoE layer per token — cheap when
  hits dominate, a measurable tax when they don't (size `N` accordingly)
- With the pool active, greedy output may deterministically flip a far-out near-tie
  token vs the host-copy path (observed at byte 354 under an 834-token prompt; the
  bypass control with pools allocated but the host path is byte-identical to `-mec 0`).
  Quantify equivalence with `llama-perplexity -mec 0 vs -mec N`, not text A/B alone

## PR Strategy (important)

1. **RFC first, code second**: open a discussion on ggml-org/llama.cpp (see #24528 for
   the cautionary tale of a large PR rejected outright), state the measured benefit
   (benchmarks above) vs. the maintenance burden (~650 lines, of which ~300 core).
2. **AI usage disclosure**: the project rejects fully/predominantly AI-generated PRs.
   Read, understand and rework every line before submitting, and disclose AI assistance
   per CONTRIBUTING.md (maintainers do the same, see #21067).
3. **Relation to #21067** (am17an's open prefetch PR): complementary — that PR overlaps
   transfers with compute for dense models and prefill; this branch keeps hot experts
   resident for decode. The RFC should reference it; the copy-stream infrastructure can
   be shared.
4. Do not submit before local testing passes (per plan).
