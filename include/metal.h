/*
 * tinyllm — Metal GPU backend header.
 *
 * All functions return 1 on success, 0 on fallback/unavailable.
 * Call tl_metal_init() once at startup, tl_metal_is_available() to check.
 * If Metal is unavailable, use the CPU fallback transparently.
 */
#ifndef TINYLLM_METAL_H
#define TINYLLM_METAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Lifecycle ─────────────────────────────────────────────────────────── */

/** Initialize Metal backend. Auto-detects GPU. Call once at startup. */
int tl_metal_init(void);

/** Shutdown Metal backend. Frees all GPU resources. */
void tl_metal_shutdown(void);

/** Returns 1 if Metal GPU is available and initialized. */
int tl_metal_is_available(void);

/** Returns the Metal GPU device name (e.g., "Apple M3 Pro"). */
const char *tl_metal_device_name_str(void);

/* ── Compute Operations ────────────────────────────────────────────────── */

/** FP32 matrix multiply: C[M×N] = A[M×K] × B[K×N] */
int tl_metal_matmul_fp32(const float *A, const float *B, float *C,
                          int M, int N, int K);

/** Q4_0 quantized matmul (dequant + matmul fused): C = dequant(A_q4) × B */
int tl_metal_matmul_q4(const uint8_t *A_q4, const float *B, float *C,
                        int M, int N, int K);

/** RMS normalization: y = x / sqrt(mean(x²) + eps) * weight */
int tl_metal_rms_norm(const float *x, const float *weight, float *y,
                       int dim, float eps);

/** SwiGLU activation: y = SiLU(x) ⊙ gate */
int tl_metal_swiglu(const float *x, const float *gate, float *y, int dim);

/** Rotary position embedding (in-place on query/key tensor) */
int tl_metal_rope(float *q, const float *cos_table, const float *sin_table,
                   int seq_len, int n_heads, int head_dim);

/** Numerically stable softmax: y = softmax(x) */
int tl_metal_softmax(const float *x, float *y, int dim);

/** MoE top-k routing: selects top n_active experts and computes softmax weights */
int tl_metal_moe_topk(const float *gate_logits, int *expert_indices,
                       float *expert_weights, int n_experts, int n_active);

/** Tiled flash attention forward: O = softmax(Q·K^T/√d)·V */
int tl_metal_flash_attention(const float *Q, const float *K, const float *V,
                              float *O, int seq_len, int n_heads, int head_dim,
                              float scale);

/** GELU activation: y = x·Φ(x) */
int tl_metal_gelu(const float *x, float *y, int n);

/** SiLU activation: y = x·sigmoid(x) */
int tl_metal_silu(const float *x, float *y, int n);

#ifdef __cplusplus
}
#endif

#endif /* TINYLLM_METAL_H */
