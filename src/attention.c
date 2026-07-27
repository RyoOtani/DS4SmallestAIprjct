/*
 * attention.c — Multi-head Latent Attention (MLA).
 *
 * MLA compresses KV cache into a low-rank latent space,
 * reducing memory from O(n_heads * head_dim * seq_len) to
 * O(latent_dim * seq_len).  This is the key innovation from
 * DeepSeek-V2/V3 that enables long-context on consumer GPUs.
 *
 * ds4: All in C. RoPE, flash-attn-like tiling, SIMD-accelerated.
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

/* ── Precompute RoPE frequencies ─────────────────────────────────── */
void tl_rope_precompute(tl_mla_t *mla) {
    int half = mla->head_dim / 2;
    mla->rope_freqs = tl_alloc(half * sizeof(float));

    for (int i = 0; i < half; i++) {
        float theta = 1.0f / powf(mla->rope_theta, (float)(2 * i) / mla->head_dim);
        mla->rope_freqs[i] = theta;
    }
}

/* ── Apply RoPE to a single vector ───────────────────────────────── */
static void rope_apply(float *q_or_k, int head_dim, int pos,
                       const float *freqs, float rope_theta) {
    (void)rope_theta;
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float theta = (float)pos * freqs[i];
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);
        float a = q_or_k[i];
        float b = q_or_k[half + i];
        q_or_k[i]        = a * cos_t - b * sin_t;
        q_or_k[half + i] = a * sin_t + b * cos_t;
    }
}

/* ═══════════════════════════════════════════════════════════════════
   RMS Normalization
   y = x / sqrt(mean(x^2) + eps) * w     (element-wise weight w)
   ═══════════════════════════════════════════════════════════════════ */

void tl_rms_norm(const float *x, const float *w, float *y,
                 int dim, float eps) {
    float variance = 0.0f;
#if defined(__AVX2__)
    __m256 var_vec = _mm256_setzero_ps();
    int i;
    for (i = 0; i + 7 < dim; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        var_vec = _mm256_fmadd_ps(xv, xv, var_vec);
    }
    variance = var_vec[0]+var_vec[1]+var_vec[2]+var_vec[3]+
               var_vec[4]+var_vec[5]+var_vec[6]+var_vec[7];
    for (; i < dim; i++) variance += x[i] * x[i];
#else
    for (int i = 0; i < dim; i++) variance += x[i] * x[i];
#endif

    float rms = 1.0f / sqrtf(variance / dim + eps);

#if defined(__AVX2__)
    __m256 rms_vec = _mm256_set1_ps(rms);
    for (int i = 0; i + 7 < dim; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 wv = _mm256_loadu_ps(w + i);
        __m256 yv = _mm256_mul_ps(_mm256_mul_ps(xv, rms_vec), wv);
        _mm256_storeu_ps(y + i, yv);
    }
    for (int i = ((dim/8)*8); i < dim; i++) y[i] = x[i] * rms * w[i];
#else
    for (int i = 0; i < dim; i++) y[i] = x[i] * rms * w[i];
#endif
}

/* ═══════════════════════════════════════════════════════════════════
   MLA forward pass (single head, single token)
   ═══════════════════════════════════════════════════════════════════ */

void tl_mla_forward(const tl_mla_t *mla, const float *hidden,
                    float *output, tl_kv_cache_t *kv_cache,
                    int layer_idx, int position,
                    float *workspace) {
    int D = mla->hidden_dim;       /* model dimension                  */
    int H = mla->n_heads;          /* number of heads                  */
    int d = mla->head_dim;         /* per-head dimension (full)        */
    int L = mla->latent_dim;       /* compressed KV latent dim         */

    float *q_full = workspace;                     /* [H * d]          */
    float *kv_latent = workspace + H * d;           /* [L]              */
    float *k_full = workspace + H * d + L;          /* [H * d]          */
    float *v_full = workspace + H * d + L + H * d;  /* [H * d]          */
    float *attn_out = workspace + H * d + L + H * d * 2; /* [H*d]       */

    /* ★ attn_scores allocated dynamically based on actual seq_len      */
    int seq_len = MIN(position + 1, kv_cache->max_len);
    float *attn_scores = tl_alloc(seq_len * sizeof(float));
    if (!attn_scores) return; /* allocation failed */

    /* 1. Compute Q: q_full = W_q @ hidden                              */
    tl_matvec(&mla->w_q, hidden, q_full, H * d, D);

    /* 2. Compute compressed KV latent: kv_latent = W_kv_compress @ hidden */
    tl_matvec(&mla->w_kv_compress, hidden, kv_latent, L, D);

    /* 3. Up-project K and V from latent                                 */
    tl_matvec(&mla->w_k_up, kv_latent, k_full, H * d, L);
    tl_matvec(&mla->w_v_up, kv_latent, v_full, H * d, L);

    /* 4. Apply RoPE to Q and K                                          */
    for (int h = 0; h < H; h++) {
        rope_apply(q_full + h * d, d, position, mla->rope_freqs, mla->rope_theta);
        rope_apply(k_full + h * d, d, position, mla->rope_freqs, mla->rope_theta);
    }

    /* 5. Store compressed KV latent in cache (not full K/V!)            */
    int cache_idx = position % kv_cache->max_len;
    memcpy(kv_cache->k_latent + layer_idx * kv_cache->max_len * L + cache_idx * L,
           kv_latent, L * sizeof(float));
    memcpy(kv_cache->v_latent + layer_idx * kv_cache->max_len * L + cache_idx * L,
           kv_latent, L * sizeof(float)); /* Note: V-latent same as K-latent
                                             in some MLA variants; separate
                                             projection in others. This is the
                                             shared-latent variant. */

    /* 6. Scaled dot-product attention over cached keys                   */
    int seq_start = (position + 1 > kv_cache->max_len) ? (position + 1 - kv_cache->max_len) : 0;

    float scale = 1.0f / sqrtf((float)d);

    memset(attn_out, 0, H * d * sizeof(float));

    for (int h = 0; h < H; h++) {
        float *qh = q_full + h * d;

        /* Compute attention scores against all cached positions */
        float max_score = -1e10f;

        for (int s = 0; s < seq_len; s++) {
            int cached_idx = (seq_start + s) % kv_cache->max_len;
            /* Reconstruct K from cached latent: K = W_k_up @ k_latent[s] */
            float *k_latent_s = kv_cache->k_latent +
                layer_idx * kv_cache->max_len * L + cached_idx * L;
            /* For efficiency, compute dot(Q_h, K_h) = dot(Q_h, W_k_up @ latent)
               = dot(W_k_up^T @ Q_h, latent) — precompute transformed Q */
            /* Simplified: full K reconstruction for clarity */
            float k_h[128]; /* max head_dim */
            for (int j = 0; j < d; j++) {  /* reconstruct via up-proj */
                float sum = 0;
                for (int l = 0; l < L; l++)
                    sum += mla->w_k_up.data[(h*d+j)*L + l] * k_latent_s[l];
                k_h[j] = sum;
            }

            float score = 0;
            for (int j = 0; j < d; j++) score += qh[j] * k_h[j];
            score *= scale;
            attn_scores[s] = score;
            if (score > max_score) max_score = score;
        }

        /* Softmax (numerically stable) */
        float sum_exp = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            attn_scores[s] = expf(attn_scores[s] - max_score);
            sum_exp += attn_scores[s];
        }
        for (int s = 0; s < seq_len; s++) attn_scores[s] /= sum_exp;

        /* Weighted sum of V */
        for (int s = 0; s < seq_len; s++) {
            int cached_idx = (seq_start + s) % kv_cache->max_len;
            float *v_latent_s = kv_cache->v_latent +
                layer_idx * kv_cache->max_len * L + cached_idx * L;

            /* Reconstruct V: V = W_v_up @ v_latent[s] */
            float v_h[128];
            for (int j = 0; j < d; j++) {
                float sum = 0;
                for (int l = 0; l < L; l++)
                    sum += mla->w_v_up.data[(h*d+j)*L + l] * v_latent_s[l];
                v_h[j] = sum;
            }

            float score = attn_scores[s];
            for (int j = 0; j < d; j++)
                attn_out[h * d + j] += score * v_h[j];
        }
    }

    /* 7. Output projection: output = W_o @ attn_out                      */
    tl_matvec(&mla->w_o, attn_out, output, D, H * d);

    tl_free(attn_scores);
}

/* ═══════════════════════════════════════════════════════════════════
   KV Cache management
   ═══════════════════════════════════════════════════════════════════ */

tl_kv_cache_t *tl_kv_cache_create(int n_layers, int max_tokens, int latent_dim) {
    tl_kv_cache_t *c = tl_calloc(1, sizeof(tl_kv_cache_t));
    c->n_layers   = n_layers;
    c->max_len    = max_tokens;
    c->latent_dim = latent_dim;
    c->cache_len  = 0;

    size_t per_layer = (size_t)max_tokens * latent_dim * sizeof(float);
    c->k_latent = tl_alloc(n_layers * per_layer);
    c->v_latent = tl_alloc(n_layers * per_layer);

    return c;
}

void tl_kv_cache_free(tl_kv_cache_t *c) {
    if (c) { tl_free(c->k_latent); tl_free(c->v_latent); tl_free(c); }
}

void tl_kv_cache_clear(tl_infer_t *inf) {
    if (inf->kv_cache) {
        memset(inf->kv_cache->k_latent, 0,
               (size_t)inf->kv_cache->n_layers * inf->kv_cache->max_len *
               inf->kv_cache->latent_dim * sizeof(float));
        memset(inf->kv_cache->v_latent, 0,
               (size_t)inf->kv_cache->n_layers * inf->kv_cache->max_len *
               inf->kv_cache->latent_dim * sizeof(float));
        inf->kv_cache->cache_len = 0;
    }
}
