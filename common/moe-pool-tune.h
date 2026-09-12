// Auto-sizing helper for the MoE expert slot pool (offline use).
//
// Feed it the per-pool counters from GGML_MOE_POOL_STATS (hits / misses /
// free slots) and it suggests the next N. No runtime dependency: run it over
// server logs or drive a sweep script with it. The engagement rule is the
// break-even hit rate h* = 1 - PCIe_BW / RAM_BW from docs/rfc-moe-expert-pool.md.

#pragma once

struct moe_pool_counters {
    unsigned long long hits = 0;
    unsigned long long misses = 0;
    int n_free = 0;
    int n_slots = 0;
};

struct moe_pool_tune_config {
    float pcie_bw = 0.0f; // GB/s, 0 = unknown (skip the h* gate)
    float ram_bw = 0.0f;  // GB/s, 0 = unknown (skip the h* gate)
    float target_margin = 0.03f; // want h >= h* + margin
    int step = 16; // slot increment / decrement
    int min_slots = 0; // floor for N
    int max_slots = 0; // ceiling for N, 0 = none
};

// break-even hit rate: hits are ~free from VRAM, misses stream over PCIe,
// the stock path streams over RAM
inline float moe_pool_break_even(float pcie_bw, float ram_bw) {
    if (pcie_bw <= 0.0f || ram_bw <= 0.0f || pcie_bw >= ram_bw) {
        return -1.0f; // unknown or degenerate, caller skips the gate
    }
    return 1.0f - pcie_bw / ram_bw;
}

inline double moe_pool_hit_rate(const moe_pool_counters & c) {
    const unsigned long long total = c.hits + c.misses;
    return total == 0 ? -1.0 : (double) c.hits / (double) total;
}

// Suggest the next N: grow while below h* + margin or no free slots remain,
// shrink while comfortably above with headroom, else hold. Returns next N.
inline int moe_pool_tune_next(const moe_pool_counters & c, const moe_pool_tune_config & cfg, int cur_n) {
    const double h = moe_pool_hit_rate(c);
    const float hs = moe_pool_break_even(cfg.pcie_bw, cfg.ram_bw);

    int next = cur_n;
    if (h < 0.0) {
        return next; // no traffic yet, hold
    }
    if (hs >= 0.0f && h < hs + cfg.target_margin) {
        next = cur_n + cfg.step; // below break-even: buy hit rate with slots
    } else if (c.n_free > cfg.step) {
        next = cur_n - cfg.step; // headroom: give VRAM back
    }
    if (cfg.min_slots > 0 && next < cfg.min_slots) {
        next = cfg.min_slots;
    }
    if (cfg.max_slots > 0 && next > cfg.max_slots) {
        next = cfg.max_slots;
    }
    return next < 0 ? 0 : next;
}
