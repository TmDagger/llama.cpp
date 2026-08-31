// tests that the persistent expert pool (ggml_backend_sched_register_expert_pool) serves
// GGML_OP_MUL_MAT_ID results identical to the regular host weight copy path, across cache
// misses, LRU evictions and cache hits, including after the graph is rebuilt
//
// needs an accelerator backend (Metal/CUDA/...); exits with success when there is none

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <vector>

static const int n_out   = 8;  // as->ne[1]
static const int n_in    = 512; // as->ne[0] (multiple of the Q2_K block size)
static const int n_expert = 8;
static const int n_slots  = 6;  // >= n_used * n_tokens so that the pool path is taken
static const int n_used   = 2;  // ids->ne[0]
static const int n_tokens = 3;  // ids->ne[1]

// binds a freshly created tensor to a host buffer and fills it with data
template <typename T>
static ggml_tensor * make_host_tensor(
        ggml_context * ctx, ggml_backend_t cpu, ggml_type type, const int64_t * ne, int ndims,
        const std::vector<T> & data, bool is_input, const char * name) {
    ggml_tensor * t = ndims == 2 ? ggml_new_tensor_2d(ctx, type, ne[0], ne[1])
                                 : ggml_new_tensor_3d(ctx, type, ne[0], ne[1], ne[2]);
    ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(cpu, ggml_nbytes(t));
    t->data = ggml_backend_buffer_get_base(buf);
    t->buffer = buf;
    memcpy(t->data, data.data(), ggml_nbytes(t));
    if (is_input) {
        ggml_set_input(t);
    }
    ggml_format_name(t, "%s", name);
    return t;
}

// a [n_expert, n_tokens] tensor whose first n_used rows per column hold the expert ids,
// strided like the output of ggml_argsort_top_k
struct strided_ids {
    ggml_tensor * wide = nullptr;  // [n_expert, n_tokens] backing store
    ggml_tensor * view = nullptr;  // [n_used, n_tokens] view into it (like argsort_top_k)
};

static strided_ids make_strided_ids(
        ggml_context * ctx, ggml_backend_t cpu, const std::vector<int32_t> & ids_data) {
    strided_ids si;
    const int64_t ne_w[2] = { n_expert, n_tokens };
    std::vector<int32_t> wide_data((size_t) n_expert * n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        for (int e = 0; e < n_expert; e++) {
            wide_data[t * n_expert + e] = e < n_used ? ids_data[t * n_used + e] : n_expert - 1;
        }
    }
    si.wide = make_host_tensor(ctx, cpu, GGML_TYPE_I32, ne_w, 2, wide_data, true, "ids_wide");
    si.view = ggml_view_2d(ctx, si.wide, n_used, n_tokens, si.wide->nb[1], 0);
    ggml_set_input(si.view);
    return si;
}

struct tensors {
    ggml_tensor * w    = nullptr; // [n_in, n_out, n_expert] host weights
    ggml_tensor * pool = nullptr; // [n_in, n_out, n_slots] on the accelerator
    ggml_tensor * table = nullptr; // [n_expert] I32 host map (INPUT)
    strided_ids ids;               // strided [n_used, n_tokens] view, like ggml_argsort_top_k
    ggml_tensor * x    = nullptr; // [n_in, n_tokens] F32 host input
};

// creates a quantized (Q2_K) weight tensor filled with quantized pseudo-random data,
// exercising the same code path as real MoE models with quantized experts
static ggml_tensor * make_quant_host_tensor(
        ggml_context * ctx, ggml_backend_t cpu, const int64_t * ne,
        const std::vector<float> & f32_data, const char * name) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_K, ne[0], ne[1], ne[2]);
    ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(cpu, ggml_nbytes(t));
    t->data = ggml_backend_buffer_get_base(buf);
    t->buffer = buf;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q2_K, ne[0]);
    for (int e = 0; e < ne[2]; e++) {
        for (int r = 0; r < ne[1]; r++) {
            const float * src = &f32_data[((size_t) e * ne[1] + r) * ne[0]];
            ggml_quantize_chunk(GGML_TYPE_Q2_K, src,
                (uint8_t *) t->data + (size_t) e * t->nb[2] + (size_t) r * row_bytes,
                0, 1, ne[0], nullptr);
        }
    }
    ggml_format_name(t, "%s", name);
    return t;
}


static bool run_round(
        ggml_backend_sched_t sched, ggml_context * ctx, tensors & ts,
        const std::vector<int32_t> & ids_data, const char * label) {
    // refresh the ids (first n_used rows of each column of the wide backing store);
    // the split input copy path uploads the backing store to the accelerator
    for (int t = 0; t < n_tokens; t++) {
        for (int e = 0; e < n_used; e++) {
            ((int32_t *) ts.ids.wide->data)[t * n_expert + e] = ids_data[t * n_used + e];
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * x3 = ggml_reshape_3d(ctx, ts.x, n_in, 1, n_tokens);

    // reference: full-size weights, served through the regular host weight copy path
    ggml_tensor * ref = ggml_mul_mat_id(ctx, ts.w, x3, ts.ids.view);

    // pooled: expert ids remapped through the map table into pool slots, mirroring the
    // wrapper chain built by llm_graph_context::build_lora_mm_id (cont of a strided view)
    ggml_tensor * table = ggml_reshape_2d(ctx, ts.table, 1, n_expert);
    ggml_tensor * ids_flat = ggml_cont(ctx, ts.ids.view);
    ggml_tensor * slots = ggml_get_rows(ctx, table, ggml_reshape_1d(ctx, ids_flat, n_used * n_tokens));
    ggml_tensor * ids2 = ggml_reshape_2d(ctx, slots, n_used, n_tokens);
    ggml_tensor * out  = ggml_mul_mat_id(ctx, ts.pool, x3, ids2);

    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, out);

    if (!ggml_backend_sched_reserve(sched, gf)) {
        fprintf(stderr, "%s: reserve failed\n", label);
        return false;
    }

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: compute failed\n", label);
        return false;
    }

    const size_t nbytes = ggml_nbytes(ref);
    std::vector<uint8_t> host_ref(nbytes), host_out(nbytes);
    ggml_backend_tensor_get(ref, host_ref.data(), 0, nbytes);
    ggml_backend_tensor_get(out, host_out.data(), 0, nbytes);

    if (memcmp(host_ref.data(), host_out.data(), nbytes) != 0) {
        fprintf(stderr, "%s: mismatch between pooled and reference outputs\n", label);
        const float * fr = (const float *) host_ref.data();
        const float * fo = (const float *) host_out.data();
        for (size_t i = 0; i < nbytes / sizeof(float); i++) {
            if (fr[i] != fo[i]) {
                fprintf(stderr, "  [%zu] ref = %f, pooled = %f\n", i, fr[i], fo[i]);
                break;
            }
        }
        return false;
    }

    printf("%s: ok (%d experts used)\n", label, n_used * n_tokens);
    return true;
}

int main() {
    // find an accelerator backend to host the pool
    ggml_backend_dev_t accel_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            accel_dev = dev;
            break;
        }
    }
    if (!accel_dev) {
        printf("no accelerator backend found, skipping\n");
        return 0;
    }

    ggml_backend_t accel = ggml_backend_dev_init(accel_dev, nullptr);
    ggml_backend_t cpu   = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!accel || !cpu) {
        fprintf(stderr, "failed to initialize backends\n");
        return 1;
    }
    printf("accelerator: %s\n", ggml_backend_name(accel));

    ggml_backend_t backends[] = { accel, cpu };
    ggml_backend_buffer_type_t bufts[] = {
        ggml_backend_get_default_buffer_type(accel),
        ggml_backend_get_default_buffer_type(cpu),
    };

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 4096, false, true);

    // context for tensors that must stay alive across graph rebuilds
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), NULL, true };
    ggml_context * tctx = ggml_init(ip);

    // deterministic pseudo-random weights and activations
    std::vector<float> w_data((size_t) n_out * n_in * n_expert);
    std::vector<float> x_data((size_t) n_in * n_tokens);
    for (size_t i = 0; i < w_data.size(); i++) w_data[i] = float((i * 1103515245u + 12345u) >> 16) / 4096.0f - 8.0f;
    for (size_t i = 0; i < x_data.size(); i++) x_data[i] = float((i * 214013u + 2531011u) >> 16) / 8192.0f - 4.0f;
    std::vector<int32_t> ids_data(n_used * n_tokens, 0);

    tensors ts;
    // note: ggml mul_mat_id weight layout is [ne0 = input dim, ne1 = output dim, ne2 = experts]
    // use quantized weights like real MoE models
    const int64_t ne_w[3] = { n_in, n_out, n_expert };
    ts.w = make_quant_host_tensor(tctx, cpu, ne_w, w_data, "w");
    const int64_t ne_x[2] = { n_in, n_tokens };
    ts.x = make_host_tensor(tctx, cpu, GGML_TYPE_F32, ne_x, 2, x_data, true, "x");
    ts.ids = make_strided_ids(tctx, cpu, ids_data);

    ts.pool = ggml_backend_sched_register_expert_pool(sched, ts.w, 0, n_slots, &ts.table);
    if (!ts.pool || !ts.table) {
        fprintf(stderr, "failed to register the expert pool\n");
        return 1;
    }

    bool ok = true;

    // graph context, rebuilt once in the middle to exercise pool state across graph rebuilds
    ggml_init_params gp = { 64 * ggml_tensor_overhead() + 4 * ggml_graph_overhead(), NULL, true };
    ggml_context * gctx = ggml_init(gp);

    // round 1: cold pool, experts 0..3 are loaded
    ok = ok && run_round(sched, gctx, ts, {0, 1, 2, 1, 3, 0}, "round1 (cold)");

    // round 2: experts 4..7 miss and evict everything
    ok = ok && run_round(sched, gctx, ts, {4, 5, 6, 4, 7, 5}, "round2 (full eviction)");

    // round 3: expert 4 is a hit, 0 and 1 were evicted and must be reloaded
    ok = ok && run_round(sched, gctx, ts, {4, 0, 1, 4, 0, 1}, "round3 (hit + reload)");

    // rebuild the graph with fresh tensors: pool contents and bookkeeping must survive
    ggml_free(gctx);
    gctx = ggml_init(gp);
    ok = ok && run_round(sched, gctx, ts, {2, 3, 2, 3, 2, 3}, "round4 (after graph rebuild)");

    ggml_free(gctx);
    ggml_free(tctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(accel);
    ggml_backend_free(cpu);

    if (!ok) {
        fprintf(stderr, "expert pool test FAILED\n");
        return 1;
    }
    printf("expert pool test PASSED\n");
    return 0;
}
