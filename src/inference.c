/*
 * inference.c — Inference loop: forward pass + autoregressive generation.
 *
 * ds4: single inference function, streaming token generation,
 *      callback-based architecture for CLI/HTTP integration.
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>

/* Forward declaration */
extern void tl_model_forward(const tl_model_t *model, const tl_token_t *tokens,
                             int n_tokens, tl_kv_cache_t *kv_cache,
                             float *hidden, float *logits, float *workspace);

/* ═══════════════════════════════════════════════════════════════════
   Inference context
   ═══════════════════════════════════════════════════════════════════ */

tl_infer_t *tl_infer_create(tl_model_t *model, tl_tokenizer_t *tok, tl_sampler_t sampler) {
    tl_infer_t *inf = tl_calloc(1, sizeof(tl_infer_t));
    inf->model    = model;
    inf->tokenizer = tok;
    inf->sampler  = sampler;

    int D = model->hidden_dim;
    int V = model->vocab_size;

    /* Allocate scratch buffers */
    inf->hidden_buf    = tl_alloc(D * TL_MAX_BATCH_SIZE * sizeof(float));
    inf->logits_buf    = tl_alloc(V * TL_MAX_BATCH_SIZE * sizeof(float));
    inf->attn_buf      = tl_alloc(D * 16 * sizeof(float)); /* generous */
    inf->ffn_buf       = tl_alloc(V * sizeof(float));       /* for sampling */
    inf->expert_scores = tl_alloc(model->total_experts * sizeof(float));

    /* Generated tokens buffer */
    inf->gen_capacity = TL_MAX_SEQ_LEN;
    inf->generated    = tl_alloc(TL_MAX_SEQ_LEN * sizeof(tl_token_t));
    inf->gen_len      = 0;

    /* KV cache */
    inf->kv_cache = tl_kv_cache_create(model->n_layers, TL_MAX_CACHE_TOKENS,
                                       TL_KV_LATENT_DIM);

    /* Scratchpad */
    inf->scratchpad     = tl_alloc(TL_MAX_SCRATCH_TOKENS * 16);
    inf->scratchpad_len = 0;

    return inf;
}

void tl_infer_free(tl_infer_t *inf) {
    if (!inf) return;
    tl_kv_cache_free(inf->kv_cache);
    tl_free(inf->hidden_buf);
    tl_free(inf->logits_buf);
    tl_free(inf->attn_buf);
    tl_free(inf->ffn_buf);
    tl_free(inf->expert_scores);
    tl_free(inf->generated);
    tl_free(inf->scratchpad);
    tl_free(inf);
}

/* ═══════════════════════════════════════════════════════════════════
   Single forward pass
   ═══════════════════════════════════════════════════════════════════ */

int tl_forward(tl_infer_t *inf, const tl_token_t *tokens, int n_tokens,
               int batch_size) {
    (void)batch_size; /* single-batch for now */

    float *hidden = inf->hidden_buf;
    float *logits = inf->logits_buf;
    float *ws     = inf->attn_buf;

    tl_model_forward(inf->model, tokens, n_tokens, inf->kv_cache,
                     hidden, logits, ws);

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
   Speculative Decoding (draft-verify)
   Uses a cheap draft model (n-gram or small head) to propose k tokens,
   then verifies them in a single forward pass.
   ───────────────────────────────────────────────────────────────────
   Simplified approach:
     - Draft with the model at higher temperature / greedy
     - Propose ~5 tokens at a time
     - Verify all at once (single batch forward)
     - Accept all tokens until first mismatch, then append the
       verified prefix + correct the first rejected token.
   ═══════════════════════════════════════════════════════════════════ */

/* Forward declaration for the draft function */
static int speculative_draft(tl_infer_t *inf, tl_token_t *draft, int n_draft);

/* ── N-gram based draft model ───────────────────────────────────── */
/* Simple draft: use the last N tokens and check if this n-gram has
   been seen before in the generated so far. If so, predict the
   continuation. Falls back to model greedy if no match. */
static int speculative_draft(tl_infer_t *inf, tl_token_t *draft, int n_draft) {
    int lookback = 3; /* n-gram order */
    if (inf->gen_len < lookback) return 0;

    /* Try to find matching n-gram in generated history */
    tl_token_t *gen = inf->generated;
    int glen = inf->gen_len;

    for (int n = 0; n < n_draft; n++) {
        if (glen + n + 1 >= inf->gen_capacity) return n;

        int found = 0;
        /* Search history for matching n-gram */
        for (int i = 0; i < glen + n - lookback; i++) {
            bool match = true;
            for (int j = 0; j < lookback; j++) {
                if (gen[i+j] != gen[glen + n - lookback + j]) { match = false; break; }
            }
            if (match) {
                draft[n] = gen[i + lookback];
                found = 1;
                break;
            }
        }
        if (!found) {
            /* Fall back to model greedy for the first unknown token */
            if (n == 0) return 0;
            return n; /* accept what we have so far */
        }
    }
    return n_draft;
}

/* ── Verify drafts in parallel ──────────────────────────────────── */
/* Runs the model on [prompt + draft_tokens] and checks if each
   predicted token matches the model's argmax at each position.
   Returns the number of accepted tokens. */
static int speculative_verify(tl_infer_t *inf, const tl_token_t *draft, int n_draft) {
    if (n_draft <= 0) return 0;

    int saved_len = inf->gen_len;
    int accepted = 0;

    /* Append draft tokens */
    for (int i = 0; i < n_draft; i++) {
        if (saved_len + i >= inf->gen_capacity) break;
        inf->generated[saved_len + i] = draft[i];
    }

    /* Forward pass on the extended sequence (without modifying KV cache state
       that we'd need to rollback). Since we're using positional cache with
       rolling window, we process one by one with verify. */
    /* Simple sequential verification: run model for each position */
    for (int k = 0; k < n_draft; k++) {
        int pos = saved_len + k;
        if (pos >= inf->gen_capacity) break;

        tl_kv_cache_clear(inf); /* reset cache for this verification */
        /* Re-process full sequence up to this position */
        /* Optimization note: in production, use batch verification with KV cache */
        memcpy(inf->generated, inf->generated, (pos) * sizeof(tl_token_t));
        inf->gen_len = pos;

        tl_forward(inf, inf->generated, pos, 1);

        /* Sample (argmax for verification) */
        float *logits = inf->logits_buf;
        int best = 0;
        for (int i = 1; i < inf->model->vocab_size; i++)
            if (logits[i] > logits[best]) best = i;

        if ((tl_token_t)best == draft[k]) {
            accepted++;
            inf->gen_len = pos + 1;
        } else {
            /* Mismatch: accept up to here, correct with the model's choice */
            inf->generated[pos] = (tl_token_t)best;
            inf->gen_len = pos + 1;
            break;
        }
    }

    return accepted;
}

/* ═══════════════════════════════════════════════════════════════════
   Autoregressive generation (with optional speculative decoding)
   ═══════════════════════════════════════════════════════════════════ */

int tl_generate(tl_infer_t *inf, const tl_token_t *prompt, int prompt_len,
                int max_new_tokens,
                void (*callback)(tl_token_t token, const char *text, int n, void *user),
                void *user) {
    /* Copy prompt into generated buffer */
    memcpy(inf->generated, prompt, prompt_len * sizeof(tl_token_t));
    inf->gen_len = prompt_len;

    tl_token_t eos = inf->tokenizer->eos_id;

    int step = 0;
    while (step < max_new_tokens) {
        /* Try speculative decoding (draft 4 tokens at a time) */
        tl_token_t drafts[8];
        int n_draft = 4;
        int nd = speculative_draft(inf, drafts, n_draft);

        int accepted = 0;
        if (nd > 0) {
            accepted = speculative_verify(inf, drafts, nd);
        }

        /* Update step counter */
        if (accepted > 0) {
            step += accepted;

            /* Callback for each accepted draft token */
            if (callback) {
                for (int k = 0; k < accepted; k++) {
                    char *text = tl_detokenize(inf->tokenizer, &inf->generated[inf->gen_len - accepted + k], 1);
                    callback(inf->generated[inf->gen_len - accepted + k], text, inf->gen_len, user);
                    tl_free(text);
                }
            }

            /* Check for EOS in accepted tokens */
            bool eos_seen = false;
            for (int k = 0; k < accepted; k++) {
                if (inf->generated[inf->gen_len - accepted + k] == eos) {
                    eos_seen = true;
                    break;
                }
            }
            if (eos_seen) break;
            continue; /* speculative batch succeeded, continue */
        }

        /* Standard single-token step (fallback) */
        /* Forward pass */
        tl_forward(inf, inf->generated, inf->gen_len, 1);

        /* Sample next token */
        tl_token_t next = tl_sample(inf);

        /* Stop on EOS */
        if (next == eos) break;

        /* Append */
        if (inf->gen_len < inf->gen_capacity) {
            inf->generated[inf->gen_len++] = next;
        } else {
            break; /* context full */
        }
        step++;

        /* Callback with the new token */
        if (callback) {
            char *text = tl_detokenize(inf->tokenizer, &next, 1);
            callback(next, text, inf->gen_len, user);
            tl_free(text);
        }
    }

    return inf->gen_len - prompt_len; /* number of new tokens generated */
}
