# MoE Expert Pool (RFC #20757) — Windows/CUDA 测试指南

特性分支: `moe-expert-pool`（基于 master b9723942 之后）

## 机制概述

`--moe-expert-cache N` / `-mec N`：为每个被 offload 到 CPU 的 MoE 专家权重张量
（`--cpu-moe` / `--n-cpu-moe` / `-ot` / auto-fit 产生的 host 侧专家张量）在显存里
维护一个 **N 个专家槽位的持久池**。

- **decode（小 ubatch）**：命中的专家零拷贝（不再每 token 重复 RAM→VRAM）；未命中的
  专家按 LRU 驱逐一个槽位后拷入。专家 id 通过 `get_rows(map_table, ids)` 重映射到槽位 id，
  不引入任何新 ggml op，全部后端（CUDA/Metal/Vulkan/CPU）天然可用。
- **prefill（大 ubatch）**：当 `n_expert_used × n_tokens > N` 时自动回退到现有
  selective-copy 路径 —— prefill 行为与主线完全一致，零回归（这正是 issue 里 SLRU
  想解决的"prefill 冲刷 decode 缓存"问题的更直接解法：prefill 根本不经过池）。
- 池只用于权重在 host、计算在加速器的张量；CPU-only 或专家全在 VRAM 时自动不生效。

约束: `N` 必须 ≥ decode 时单 ubatch 的最大 distinct 专家数 = `n_expert_used`
（即 top-k）。给小于它的值没有意义（图构建判定会永远走回退路径）。N 的上限是
`n_expert - 1`（再大就等价于全量驻留，直接调 `-ngl` 更好）。

显存占用估算: `n_cpu_moe 层数 × 2 张量(gate_up+down) × N × 单专家字节数`。
例: 48 层、128 experts、Q4 单专家 ~24MB、N=32 → 48×2×32×24MB ≈ 71GB（放不下）；
N=8 → ~18GB。N 从 top-k（如 8）起步往上加，直到显存预算用完。

## Windows 构建

```powershell
git clone -b moe-expert-pool https://github.com/<你的fork>/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --config Release --target llama-cli llama-bench
# 单测（有 CUDA 才有意义）:
cmake --build build --config Release --target test-expert-pool
.\build\bin\Release\test-expert-pool.exe
```

## 测试矩阵（建议顺序）

### 1. 正确性
```powershell
# A/B 输出对比（greedy，同 seed）——两次生成的文本应完全一致
.\build\bin\Release\llama-cli.exe -m <model> -ngl 99 -cmoe -p <长prompt> -n 128 --temp 0 --simple-io > out_base.txt
.\build\bin\Release\llama-cli.exe -m <model> -ngl 99 -cmoe -mec 32 -p <长prompt> -n 128 --temp 0 --simple-io > out_pool.txt
fc out_base.txt out_pool.txt
```
注意：mec 路径下 prefill（prompt 处理）走原路径，decode 走池路径，所以全文应一致。
若不一致立即停止（优先排查 mul_mat_id 尾部 padding 与量化类型的交互）。

### 2. decode 提速（核心收益）
```powershell
.\build\bin\Release\llama-bench.exe -m <model> -ngl 99 -ncmoe 99 -mec 0,8,16,32,64 -n 128,512 -p 0
```
观察 tg（token generation）随 -mec 增长的曲线。社区先验（issue #20757 讨论数据）：
- 热专家命中率随 N 上升，decode 提升应在 N ≥ 2×top-k 后明显
- N 过大（接近 n_expert）时收益趋平甚至回落（WDDM 显存压力悬崖）

### 3. prefill 零回归
```powershell
.\build\bin\Release\llama-bench.exe -m <model> -ngl 99 -ncmoe 99 -mec 0,32 -p 512,2048 -n 0
```
pp 数字两组应基本相同（大 ubatch 全部回退原路径，池不参与）。

### 4. 长会话稳定性
llama-cli 跑 2000+ token 的多轮对话，观察是否出现 NaN/崩溃（LRU 驱逐路径的
压力测试）。日志里确认 `pooled X offloaded MoE expert weight tensors`。

## 已知边界（本 MVP）

- pipeline parallelism（自动开启的 n_copies>1）下池自动禁用（有警告）
- 多 GPU：池全部放在第一个加速器上（TODO: 按层选择）
- `--moe-expert-cache` 只作用于 CPU 侧专家张量；全 VRAM 模型无感知
- warmup 图（n_expert_used = n_expert）自动走回退路径，无需处理

## PR 策略（重要）

1. **先 RFC 后代码**：在 https://github.com/ggml-org/llama.cpp/discussions 发 RFC
   （参考 #24528 的教训：直接发大 PR 会被拒），明确收益数据（上面的 bench）vs
   维护负担（652 行，其中核心 ~300 行）。
2. **AI 使用披露**：llama.cpp 明确拒绝"fully or predominantly AI-generated"的 PR。
   提交前你需要逐行理解并按自己的风格重写注释/命名，PR 描述里按 CONTRIBUTING.md
   要求披露 AI 辅助情况（#21067 的维护者自己也是这么做的）。
3. **与 #21067 的关系**：am17an 的 prefetch PR（open，卡 review）与本分支互补——
   它解决 dense/预取重叠，本分支解决 MoE decode 热专家驻留。RFC 里应引用它并说明
   两者可组合（copy stream 基础设施可以直接复用）。
4. 本地测试通过前不提交（用户的既定计划）。
