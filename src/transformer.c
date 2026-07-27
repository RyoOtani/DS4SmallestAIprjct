/*
 * transformer.c — Full transformer forward pass, integrating:
 *   RMS Norm → MLA Attention → Residual → RMS Norm → FFN/MoE → Residual
 *
 * ds4: single-function forward pass, minimal indirection.
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Forward declarations from other modules */
extern void tl_rms_norm(const float *x, const float *w, float *y, int dim, float eps);
extern void tl_mla_forward(const tl_mla_t *mla, const float *hidden,
                           float *output, tl_kv_cache_t *kv_cache,
                           int layer_idx, int position,
                           float *workspace);
extern void tl_ffn_forward(const tl_ffn_t *ffn, const float *hidden,
                           float *output, float *workspace);
extern void tl_moe_forward(const tl_moe_layer_t *moe, const float *hidden,
                           float *output, float *workspace);

#define RMS_EPS 1e-5f

/* ═══════════════════════════════════════════════════════════════════
   Single transformer layer forward
   ═══════════════════════════════════════════════════════════════════ */

static void tl_layer_forward(const tl_layer_t *layer, const float *hidden,
                             float *output, tl_kv_cache_t *kv_cache,
                             int layer_idx, int position,
                             float *workspace) {
    int D = layer->rms_attn_w.rows;  /* hidden_dim — 1D weight stored as [D] */
    if (D <= 0) D = layer->rms_attn_w.cols; /* fallback to cols */
    if (D <= 0) D = 2048; /* ultimate fallback */

    float *normed   = workspace;             /* [D] */
    float *attn_out = workspace + D;         /* [D] */
    float *residual1 = workspace + D * 2;    /* [D] */
    float *normed2   = workspace + D * 3;    /* [D] */
    float *ffn_out   = workspace + D * 4;    /* [D] */
    float *attn_ws   = workspace + D * 5;    /* MLA internal */

    /* 1. Pre-attention RMS Norm */
    tl_rms_norm(hidden, layer->rms_attn_w.data, normed, D, RMS_EPS);

    /* 2. MLA */
    tl_mla_forward(&layer->mla, normed, attn_out, kv_cache,
                   layer_idx, position, attn_ws);

    /* 3. Residual: residual1 = hidden + attn_out */
    for (int i = 0; i < D; i++)
        residual1[i] = hidden[i] + attn_out[i];

    /* 4. Pre-FFN RMS Norm */
    tl_rms_norm(residual1, layer->rms_ffn_w.data, normed2, D, RMS_EPS);

    /* 5. FFN (dense or MoE) */
    if (layer->is_moe) {
        float *moe_ws = attn_ws;  /* reuse MLA workspace */
        tl_moe_forward(&layer->ffn.moe, normed2, ffn_out, moe_ws);
    } else {
        float *ffn_ws = attn_ws;
        tl_ffn_forward(&layer->ffn.dense, normed2, ffn_out, ffn_ws);
    }

    /* 6. Residual: output = residual1 + ffn_out */
    for (int i = 0; i < D; i++)
        output[i] = residual1[i] + ffn_out[i];
}

/* ═══════════════════════════════════════════════════════════════════
   Full model forward pass
     hidden (input) → embedding → [layers] → RMS norm → lm_head → logits
   ═══════════════════════════════════════════════════════════════════ */

void tl_model_forward(const tl_model_t *model, const tl_token_t *tokens,
                      int n_tokens, tl_kv_cache_t *kv_cache,
                      float *hidden, float *logits,
                      float *workspace) {
    int D = model->hidden_dim;
    int vocab = model->vocab_size;

    /* Safety: check embeddings loaded */
    if (!model->tok_embeddings.data) {
        fprintf(stderr, "⚠️  tok_embeddings.data is NULL\n");
        memset(logits, 0, vocab * sizeof(float));
        return;
    }
    if (!model->layers) {
        fprintf(stderr, "⚠️  model->layers is NULL\n");
        memset(logits, 0, vocab * sizeof(float));
        return;
    }

    /* 1. Token embedding lookup (only last token for inference) */
    int last_pos = n_tokens - 1;
    tl_token_t tok = tokens[last_pos];

    if (tok >= 0 && tok < model->tok_embeddings.rows) {
        int emb_cols = model->tok_embeddings.cols;
        float *emb = model->tok_embeddings.data + tok * emb_cols;
        memcpy(hidden, emb, D * sizeof(float));
    } else {
        memset(hidden, 0, D * sizeof(float));
    }

    /* 2. Process through all transformer layers */
    /* Use alternating buffers to avoid extra copy */
    float *buf_in  = workspace;           /* [D] */
    float *buf_out = workspace + D;       /* [D] */
    float *layer_ws = workspace + D * 2;  /* per-layer workspace */

    /* Initial hidden → buf_in */
    memcpy(buf_in, hidden, D * sizeof(float));

    for (int l = 0; l < model->n_layers; l++) {
        tl_layer_forward(&model->layers[l], buf_in, buf_out,
                         kv_cache, l, last_pos, layer_ws);
        /* Swap buffers */
        float *tmp = buf_in; buf_in = buf_out; buf_out = tmp;
    }

    /* 3. Final RMS Norm */
    tl_rms_norm(buf_in, model->rms_final_w.data, hidden, D, RMS_EPS);
    /* (hidden now holds the final normalized output) */

    /* 4. LM head projection: logits = lm_head @ hidden */
    tl_matvec(&model->lm_head, hidden, logits, vocab, D);
}
