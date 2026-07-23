/*
 * tinyllm — Metal Shading Language kernels for GPU-accelerated inference.
 *
 * These kernels run on Apple Silicon GPUs (M1/M2/M3/M4) via Metal.
 * Operations:
 *   - matmul_fp32    : f32 matrix multiply (SGEMM)
 *   - matmul_q4      : Q4_0 dequant + matmul fused
 *   - rms_norm       : RMS normalization
 *   - swiglu         : SiLU-gated linear unit
 *   - rope           : Rotary position embedding
 *   - softmax        : Online softmax (numerically stable)
 *   - moe_gate       : MoE router top-k selection
 *   - flash_attn_tile: Tiled flash attention (forward pass)
 *
 * Compile with:
 *   xcrun -sdk macosx metal -O3 -ffast-math -o tinyllm.metallib tinyllm_ops.metal
 *
 * Architecture: threadgroup-based tiling for optimal GPU occupancy.
 * Each threadgroup = 8x8 threads (64 threads), tile size = 32x32.
 */

#include <metal_stdlib>
using namespace metal;

// ── Constants ────────────────────────────────────────────────────────────────

constant uint TILE_M = 32;
constant uint TILE_N = 32;
constant uint TILE_K = 8;
constant uint THREADS_PER_GROUP = 256;  // 16x16 threadgroup

// ── Matrix Multiply (FP32) ───────────────────────────────────────────────────

/// Tiled SGEMM: C[M×N] += A[M×K] × B[K×N]
/// Each threadgroup computes one TILE_M×TILE_N block of C.
kernel void matmul_fp32(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float       *C [[buffer(2)]],
    constant uint      &M [[buffer(3)]],
    constant uint      &N [[buffer(4)]],
    constant uint      &K [[buffer(5)]],
    uint2 gid          [[threadgroup_position_in_grid]],
    uint2 tid          [[thread_position_in_threadgroup]],
    uint  tpg          [[threads_per_threadgroup]]
) {
    // Tile in shared memory
    threadgroup float As[TILE_M][TILE_K];
    threadgroup float Bs[TILE_K][TILE_N];

    float acc = 0.0f;
    uint row = gid.y * TILE_M + tid.y;
    uint col = gid.x * TILE_N + tid.x;

    // Loop over K dimension in tiles
    for (uint k_block = 0; k_block < K; k_block += TILE_K) {
        // Cooperative load A tile
        if (row < M && (k_block + tid.x) < K) {
            As[tid.y][tid.x] = A[row * K + k_block + tid.x];
        } else {
            As[tid.y][tid.x] = 0.0f;
        }
        // Cooperative load B tile
        if ((k_block + tid.y) < K && col < N) {
            Bs[tid.y][tid.x] = B[(k_block + tid.y) * N + col];
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint k = 0; k < TILE_K; k++) {
            acc += As[tid.y][k] * Bs[k][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ── Q4_0 Dequantize + Matmul (fused) ─────────────────────────────────────────

/// Fused dequantize + matmul for Q4_0 format.
/// Q4_0: 32 weights per block, 1 fp16 scale per block
/// Block layout: [scale_f16 (2B)] [quants (16B)]
kernel void matmul_q4(
    device const uchar  *A_q4 [[buffer(0)]], // Q4_0 quantized weights
    device const float  *B     [[buffer(1)]], // FP32 activations
    device float        *C     [[buffer(2)]],
    constant uint       &M     [[buffer(3)]],
    constant uint       &N     [[buffer(4)]],
    constant uint       &K     [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    uint row = gid.y * TILE_M + tid.y;
    uint col = gid.x * TILE_N + tid.x;

    float acc = 0.0f;
    uint block_size = 32; // Q4_0: 32 weights per block
    uint bytes_per_block = sizeof(half) + block_size / 2; // 2 + 16 = 18 bytes

    if (row < M && col < N) {
        for (uint k = 0; k < K; k += block_size) {
            // Read scale (fp16 → fp32)
            uint block_offset = (row * (K / block_size) + k / block_size) * bytes_per_block;
            half scale_h = *((device half *)(A_q4 + block_offset));
            float scale = float(scale_h);

            // Read and dequantize 4-bit weights
            for (uint b = 0; b < min(block_size, K - k); b++) {
                uint byte_idx = block_offset + sizeof(half) + b / 2;
                uchar packed = A_q4[byte_idx];
                float w = ((b & 1) == 0)
                    ? float(packed & 0x0F) - 8.0f
                    : float(packed >> 4) - 8.0f;
                w *= scale;
                acc += w * B[(k + b) * N + col];
            }
        }
        C[row * N + col] = acc;
    }
}

// ── RMS Normalization ────────────────────────────────────────────────────────

/// RMSNorm: y = x / sqrt(mean(x²) + ε) * weight
kernel void rms_norm(
    device const float *x       [[buffer(0)]],
    device const float *weight  [[buffer(1)]],
    device float       *y       [[buffer(2)]],
    constant uint      &dim     [[buffer(3)]],
    constant float     &eps     [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    // Step 1: compute sum of squares (thread 0 does full reduction)
    // For simplicity: each thread computes its own x², then we parallel-reduce
    threadgroup float shared_sum[THREADS_PER_GROUP];
    float my_x = x[tid];
    shared_sum[tid] = my_x * my_x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction (binary tree)
    for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = sqrt(shared_sum[0] / float(dim) + eps);
    y[tid] = (x[tid] / rms) * weight[tid];
}

// ── SwiGLU Activation ────────────────────────────────────────────────────────

/// SwiGLU: y = (x · sigmoid(x)) · gate
kernel void swiglu(
    device const float *x    [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device float       *y    [[buffer(2)]],
    constant uint      &dim  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float val = x[tid];
    float silu = val / (1.0f + exp(-val)); // SiLU = x * sigmoid(x)
    y[tid] = silu * gate[tid];
}

// ── RoPE (Rotary Position Embedding) ─────────────────────────────────────────

/// Apply rotary position embedding to query/key vectors.
/// freq[i] = 1.0 / theta^(2i/dim)
/// cos/sin tables are precomputed per position.
kernel void rope(
    device float       *q          [[buffer(0)]], // [seq_len, n_heads, head_dim]
    device const float *cos_table  [[buffer(1)]],
    device const float *sin_table  [[buffer(2)]],
    constant uint      &seq_len    [[buffer(3)]],
    constant uint      &n_heads    [[buffer(4)]],
    constant uint      &head_dim   [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    uint s = gid.z;  // sequence position
    uint h = gid.y;  // head index
    uint d = gid.x * THREADS_PER_GROUP + tid.x;  // dimension pair

    if (s >= seq_len || h >= n_heads || d >= head_dim / 2) return;

    uint idx = s * n_heads * head_dim + h * head_dim;
    float cos_val = cos_table[s * (head_dim / 2) + d];
    float sin_val = sin_table[s * (head_dim / 2) + d];

    float x0 = q[idx + 2 * d];
    float x1 = q[idx + 2 * d + 1];

    q[idx + 2 * d]     = x0 * cos_val - x1 * sin_val;
    q[idx + 2 * d + 1] = x0 * sin_val + x1 * cos_val;
}

// ── Online Softmax (numerically stable) ──────────────────────────────────────

/// Online softmax: softmax[i] = exp(x[i] - max) / sum(exp(x[i] - max))
kernel void softmax(
    device const float *x     [[buffer(0)]],
    device float       *y     [[buffer(1)]],
    constant uint      &dim   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    // Find max (parallel reduction)
    threadgroup float shared_max[THREADS_PER_GROUP];
    threadgroup float shared_sum[THREADS_PER_GROUP];
    float val = x[tid];
    shared_max[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float max_val = shared_max[0];
    float exp_val = exp(val - max_val);
    shared_sum[tid] = exp_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    y[tid] = exp_val / shared_sum[0];
}

// ── MoE Router (Top-K Gating) ────────────────────────────────────────────────

/// Compute top-k expert indices and softmax weights for MoE routing.
/// Input:  gate_logits [n_experts]
/// Output: expert_indices [n_active], expert_weights [n_active] (sorted)
kernel void moe_topk(
    device const float *gate_logits    [[buffer(0)]],
    device uint        *expert_indices [[buffer(1)]],
    device float       *expert_weights [[buffer(2)]],
    constant uint      &n_experts     [[buffer(3)]],
    constant uint      &n_active      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    // This kernel runs with 1 threadgroup, THREADS_PER_GROUP threads
    // Each thread processes n_experts/THREADS_PER_GROUP elements

    if (tid >= n_experts) return;

    // Store (value, index) pairs
    threadgroup float shared_vals[THREADS_PER_GROUP];
    threadgroup uint  shared_idx[THREADS_PER_GROUP];

    shared_vals[tid] = gate_logits[tid];
    shared_idx[tid] = tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simple top-k via bitonic sort in threadgroup memory
    // (For n_experts <= 256, this is efficient on GPU)
    for (uint stage = 2; stage <= THREADS_PER_GROUP; stage <<= 1) {
        for (uint step = stage >> 1; step > 0; step >>= 1) {
            uint j = tid ^ step;
            if (j > tid) {
                bool ascending = ((tid & stage) == 0);
                bool swap = ascending
                    ? shared_vals[tid] < shared_vals[j]
                    : shared_vals[tid] > shared_vals[j];
                if (swap && j < THREADS_PER_GROUP) {
                    float tmp_v = shared_vals[tid];
                    uint  tmp_i = shared_idx[tid];
                    shared_vals[tid] = shared_vals[j];
                    shared_idx[tid] = shared_idx[j];
                    shared_vals[j] = tmp_v;
                    shared_idx[j] = tmp_i;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Output top-k
    if (tid < n_active) {
        expert_indices[tid] = shared_idx[tid];
        // Softmax over top-k only
        float max_logit = shared_vals[0];
        float sum_exp = 0.0f;
        for (uint i = 0; i < n_active; i++) {
            sum_exp += exp(shared_vals[i] - max_logit);
        }
        expert_weights[tid] = exp(shared_vals[tid] - max_logit) / sum_exp;
    }
}

// ── Tiled Flash Attention (Forward) ──────────────────────────────────────────

/// Tiled flash attention forward pass (simplified — no causal mask).
/// Q, K, V: [seq_len, n_heads, head_dim]
/// O:       [seq_len, n_heads, head_dim]
/// Uses online softmax for numerical stability.
kernel void flash_attention_fwd(
    device const float *Q        [[buffer(0)]],
    device const float *K        [[buffer(1)]],
    device const float *V        [[buffer(2)]],
    device float       *O        [[buffer(3)]],
    constant uint      &seq_len  [[buffer(4)]],
    constant uint      &n_heads  [[buffer(5)]],
    constant uint      &head_dim [[buffer(6)]],
    constant float     &scale    [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    uint q_idx = gid.y;  // query position
    uint h     = gid.x;  // head index

    if (q_idx >= seq_len || h >= n_heads) return;

    // Load Q tile into threadgroup memory
    threadgroup float Qi[TILE_M];
    uint q_offset = q_idx * n_heads * head_dim + h * head_dim;

    for (uint d = tid.x; d < head_dim; d += THREADS_PER_GROUP) {
        Qi[d] = Q[q_offset + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax accumulator
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    threadgroup float Oi[TILE_M];
    for (uint d = 0; d < TILE_M; d++) Oi[d] = 0.0f;

    float s = scale / sqrt(float(head_dim));

    // Loop over key/value positions
    for (uint k_idx = 0; k_idx < seq_len; k_idx++) {
        uint k_offset = k_idx * n_heads * head_dim + h * head_dim;

        // Compute Q·K for this position
        float score = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            score += Qi[d] * K[k_offset + d];
        }
        score *= s;

        // Update online softmax
        float new_max = max(max_score, score);
        float exp_diff = exp(max_score - new_max);
        sum_exp = sum_exp * exp_diff + exp(score - new_max);

        // Update output
        for (uint d = 0; d < head_dim; d++) {
            Oi[d] = Oi[d] * exp_diff + V[k_offset + d] * exp(score - new_max);
        }

        max_score = new_max;
    }

    // Normalize and write output
    uint o_offset = q_offset;
    for (uint d = 0; d < head_dim; d++) {
        O[o_offset + d] = Oi[d] / sum_exp;
    }
}

// ── Element-wise ops ─────────────────────────────────────────────────────────

kernel void add_vectors(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float       *c [[buffer(2)]],
    constant uint      &n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) c[tid] = a[tid] + b[tid];
}

kernel void mul_scalar(
    device float       *x [[buffer(0)]],
    constant float     &s [[buffer(1)]],
    constant uint      &n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) x[tid] *= s;
}

kernel void gelu(
    device const float *x [[buffer(0)]],
    device float       *y [[buffer(1)]],
    constant uint      &n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float v = x[tid];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    float cdf = 0.5f * (1.0f + tanh(sqrt(2.0f / M_PI_F) * (v + 0.044715f * v * v * v)));
    y[tid] = v * cdf;
}

kernel void silu(
    device const float *x [[buffer(0)]],
    device float       *y [[buffer(1)]],
    constant uint      &n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    float v = x[tid];
    y[tid] = v / (1.0f + exp(-v));
}
