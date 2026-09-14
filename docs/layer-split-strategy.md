# Layer split strategy: distributing layers by VRAM bandwidth

When a model is split across several GPUs by layer (`--split-mode layer`, the default),
the per-token cost of a device is dominated by reading its weights from VRAM. To keep
all devices busy for the same wall time, the number of layers placed on a device should
be proportional to its effective VRAM bandwidth.

llama.cpp used to distribute layers by free device memory. This is a poor proxy when the
GPUs differ in bandwidth (e.g. a wide-bus card next to a narrow-bus card): the faster
card is underused. The new `--layer-split-strategy` option selects the distribution.

## Options

- `slots` (default when the expert cache is enabled): distribute layers so that the
  estimated number of expert cache slots per device is equal. This is what matters for
  MoE decode: a device that owns too many layers spreads its cache too thin.
- `bw` (default when the expert cache is disabled): layers proportional to the VRAM
  bandwidth of each device. Good for prompt processing, less so for decode.
- `eq`: equal number of layers per device.
- `manual`: use the proportions from `--tensor-split` / `-ts`.
  Passing `-ts` selects `manual` automatically.

Passing `--layer-split-strategy` explicitly overrides the automatic default.

## Cache-slot split

`slots` estimates, from the loader metadata, the average dense (non-expert) bytes per
offloaded layer and the bytes of one expert slot per layer (sum of the expert tensor
sizes divided by the expert count). It then solves, by binary search, the slot level `s`
such that every device gets as many layers as its free memory (minus the rail) allows
while keeping `s` slots per tensor and covering all offloaded layers:

```
n_d(s) = (free_d - rail_d) / (dense + esz_slot * s),   sum_d n_d(s) = n_gpu_layers
```

If the metadata is not usable it falls back to the free-memory split.
Note that the number of expert tensors per layer differs between models (fused
`gate_up` + `down` versus separate `up`/`gate`/`down`); the formula uses the actual bytes
so no memory is wasted on models with fewer tensors.

## VRAM bandwidth

For `bw`, the bandwidth is taken from `--vram-bw GB0,GB1,...` (GB/s, a single value is
broadcast to all devices). When it is not provided, a short device-to-device copy
benchmark runs once per device at startup (256 MiB buffers, read + write counted).

```
# automatic: 'slots' with the expert cache, 'bw' without
llama-server -m model.gguf -ngl 99 -mec 128

# force the cache-slot balancing
llama-server -m model.gguf -ngl 99 -mec 128 --layer-split-strategy slots

# measure at startup (bw), several rounds are averaged
llama-server -m model.gguf -ngl 99 --layer-split-strategy bw

# provide the bandwidth manually, no benchmark
llama-server -m model.gguf -ngl 99 --layer-split-strategy bw --vram-bw 896,288

# equal shares
llama-server -m model.gguf -ngl 99 --layer-split-strategy eq

# manual proportions
llama-server -m model.gguf -ngl 99 --layer-split-strategy manual -ts 3,1
```

## Interaction with the expert cache

A device that receives many layers owns more MoE expert tensors, so its LRU expert cache
(`--moe-expert-cache`, see `docs/rfc-moe-expert-pool.md`) has to spread over more
tensors and keeps fewer slots per tensor. On a device with a narrow PCIe link (x4) this
can turn missing experts into expensive host transfers. The `bw` distribution does not
model this; `eq` or `manual` let the user trade compute balance for a denser cache.
Monitor with `GGML_MOE_POOL_STATS=1` and `nvidia-smi` to find the best mix.

The strategy only applies to `--split-mode layer`. The device order follows the model's
device list: explicit `--device` order if given, otherwise RPC servers first and then
discrete GPUs (integrated GPUs only if there are no discrete ones).

## Cache hit-rate telemetry

With `GGML_MOE_POOL_STATS=1` the server prints a windowed `hit ... d0=..% d1=..%` line
next to the generation rate. Counts are per expert tensor, not per layer, and a pool is
now updated at most once per graph compute, so the numbers are not double counted
(previously the remap `GET_ROWS` and the pooled `MUL_MAT_ID` both updated the same pool).
Because the whole MoE step needs all selected experts, watch both the hit rate and the
miss count: a small cache can have a high hit rate yet still miss one expert of almost
every token, which is what stalls decode.

