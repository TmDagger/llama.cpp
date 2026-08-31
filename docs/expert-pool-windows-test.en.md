# MoE Expert Pool (RFC ggml-org#20757) — Windows/CUDA Testing Guide

> 中文版: [expert-pool-windows-test.md](expert-pool-windows-test.md)

Feature branch: `moe-expert-pool` (based on master 9723942)

Pull on Windows:

```powershell
git clone -b moe-expert-pool https://github.com/memoriaru/llama.cpp.git
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
  copy. Aim for `N` covering the decode working set — start at 2–4× top-k and scan up.
- Upper bound is `n_expert - 1`; beyond that just keep the weights in VRAM via `-ngl`.

VRAM budget estimate: `offloaded MoE layers × 2–3 tensors (gate_up+down[+down]) × N ×
bytes-per-expert`. Example: 48 layers, 128 experts, ~24 MB per Q4 expert, N = 32 →
48×2×32×24 MB ≈ 71 GB (does not fit); N = 8 → ~18 GB. The implementation additionally
caps the total pool memory at half of the device's free memory at load time; tensors
beyond the budget silently keep using the selective-copy path.

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

## Known Limitations (MVP)

- Disabled automatically under pipeline parallelism (warning logged)
- Multi-GPU: pools all go to the first accelerator (TODO: pick per layer)
- Only affects expert tensors actually placed on the host
- One expert-id readback synchronization per offloaded MoE layer per token — cheap when
  hits dominate, a measurable tax when they don't (size `N` accordingly)

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
