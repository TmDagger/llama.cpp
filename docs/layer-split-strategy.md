# Layer split strategy: distributing layers by VRAM bandwidth

When a model is split across several GPUs by layer (`--split-mode layer`, the default),
the per-token cost of a device is dominated by reading its weights from VRAM. To keep
all devices busy for the same wall time, the number of layers placed on a device should
be proportional to its effective VRAM bandwidth.

llama.cpp used to distribute layers by free device memory. This is a poor proxy when the
GPUs differ in bandwidth (e.g. a wide-bus card next to a narrow-bus card): the faster
card is underused. The new `--layer-split-strategy` option selects the distribution.

## Options

- `bw` (default): layers proportional to the VRAM bandwidth of each device.
- `eq`: equal number of layers per device.
- `manual`: use the proportions from `--tensor-split` / `-ts`.
  Passing `-ts` selects `manual` automatically.

## VRAM bandwidth

For `bw`, the bandwidth is taken from `--vram-bw GB0,GB1,...` (GB/s, a single value is
broadcast to all devices). When it is not provided, a short device-to-device copy
benchmark runs once per device at startup (256 MiB buffers, read + write counted).

```
# measure at startup (default)
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
