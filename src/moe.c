/*
 * moe.c — Mixture of Experts routing and sparse FFN.
 *
 * MoE: each token is routed to top-k experts via a learned gate.
 * Only those experts' FFNs are computed → massive compute savings.
 *
 * ds4: SIMD-accelerated gate computation, thread-friendly expert dispatch.
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── SwiGLU activation: x * sigmoid(x * beta) ────────────────────── */
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ── MoE gate: route a hidden state to top-k experts ─────────────── */
void tl_moe_gate(const tl_moe_gate_t *gate, const float *hidden,
                 int *expert_indices, float *expert_weights,
                 int n_active) {
    int n_experts = gate->n_experts;
    float *scores = tl_alloc(n_experts * sizeof(float));

    /* Gate: scores = gate_w @ hidden */
    tl_matvec(&gate->gate_w, hidden, scores, n_experts, gate->gate_w.cols);

    /* Top-k selection (naive O(K*N), fine for n_experts ≤ 256) */
    /* Use a simple selection sort for top-k */
    /* Initialize indices */
    for (int i = 0; i < n_experts; i++) expert_indices[i] = -1;

    for (int k = 0; k < n_active; k++) {
        float best_score = -1e10f;
        int   best_idx   = -1;
        for (int i = 0; i < n_experts; i++) {
            /* Skip already selected */
            bool taken = false;
            for (int j = 0; j < k; j++)
                if (expert_indices[j] == i) { taken = true; break; }
            if (!taken && scores[i] > best_score) {
                best_score = scores[i];
                best_idx   = i;
            }
        }
        if (best_idx < 0) break;
        expert_indices[k] = best_idx;
        expert_weights[k] = best_score;
    }

    /* Softmax over selected experts */
    float max_w = expert_weights[0];
    for (int k = 1; k < n_active; k++)
        if (expert_weights[k] > max_w) max_w = expert_weights[k];

    float sum = 0.0f;
    for (int k = 0; k < n_active; k++) {
        expert_weights[k] = expf(expert_weights[k] - max_w);
        sum += expert_weights[k];
    }
    for (int k = 0; k < n_active; k++)
        expert_weights[k] /= sum;

    tl_free(scores);
}

/* ── Single expert FFN (SwiGLU) ───────────────────────────────────── */
void tl_ffn_forward(const tl_ffn_t *ffn, const float *hidden,
                    float *output, float *workspace) {
    int D = ffn->hidden_dim;
    int mid = ffn->inter_dim;

    float *gate_out = workspace;        /* [mid] */
    float *up_out   = workspace + mid;  /* [mid] */
    float *merged   = workspace + mid;  /* reuse after gate applied */

    /* Gate projection: gate_out = W_gate @ hidden */
    tl_matvec(&ffn->w_gate, hidden, gate_out, mid, D);

    /* Up projection: up_out = W_up @ hidden */
    tl_matvec(&ffn->w_up, hidden, up_out, mid, D);

    /* SwiGLU: merged = silu(gate_out) * up_out */
    for (int i = 0; i < mid; i++) {
        merged[i] = silu(gate_out[i]) * up_out[i];
    }

    /* Down projection: output = W_down @ merged */
    tl_matvec(&ffn->w_down, merged, output, D, mid);
}

/* ── Full MoE layer forward ───────────────────────────────────────── */
void tl_moe_forward(const tl_moe_layer_t *moe, const float *hidden,
                    float *output, float *workspace) {
    int D = moe->hidden_dim;
    int n_active = moe->n_active;

    int    *expert_idx = (int*)workspace;
    float  *expert_w   = (float*)(expert_idx + n_active);
    float  *expert_out = (float*)(expert_w + n_active);
    float  *ffn_ws     = expert_out + D;
    /* remaining workspace for FFN internal use */

    /* Route to top-k experts */
    tl_moe_gate(&moe->gate, hidden, expert_idx, expert_w, n_active);

    /* Weighted sum of expert outputs */
    memset(output, 0, D * sizeof(float));

    for (int k = 0; k < n_active; k++) {
        int eid = expert_idx[k];
        if (eid < 0 || eid >= moe->n_experts) continue;

        float *e_out = expert_out + k * D;
        tl_ffn_forward(&moe->experts[eid], hidden, e_out, ffn_ws);

        /* Accumulate: output += expert_w[k] * e_out */
        float wk = expert_w[k];
#if defined(__AVX2__)
        __m256 wv = _mm256_set1_ps(wk);
        int i;
        for (i = 0; i + 7 < D; i += 8) {
            __m256 eo = _mm256_loadu_ps(e_out + i);
            __m256 acc = _mm256_loadu_ps(output + i);
            acc = _mm256_fmadd_ps(wv, eo, acc);
            _mm256_storeu_ps(output + i, acc);
        }
        for (; i < D; i++) output[i] += wk * e_out[i];
#else
        for (int i = 0; i < D; i++) output[i] += wk * e_out[i];
#endif
    }
}

/* ── Load balancing auxiliary loss (for training only) ────────────── */
float tl_moe_load_balance_loss(const tl_moe_layer_t *moe,
                               const float *hidden, int batch_size) {
    int N = moe->n_experts;
    int D = moe->hidden_dim;
    float *freq = tl_calloc(N, sizeof(float));
    float *prob = tl_calloc(N, sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        int idx[TL_MAX_ACTIVE_EXPERTS];
        float w[TL_MAX_ACTIVE_EXPERTS];
        tl_moe_gate(&moe->gate, hidden + b * D, idx, w, moe->n_active);

        for (int k = 0; k < moe->n_active; k++) {
            if (idx[k] >= 0) {
                freq[idx[k]] += 1.0f / moe->n_active;
                prob[idx[k]] += w[k];
            }
        }
    }

    /* Loss = N * sum(f_i * p_i) */
    float loss = 0;
    for (int i = 0; i < N; i++)
        loss += freq[i] * prob[i];

    tl_free(freq); tl_free(prob);
    return loss * N / batch_size;
}
