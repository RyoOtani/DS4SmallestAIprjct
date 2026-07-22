/*
 * sampler.c — Token sampling: top-k, top-p (nucleus), temperature.
 *   ds4: compact, fast, no external deps.
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Xorshift128+ PRNG (fast, good enough for sampling) ──────────── */
static uint64_t xorshift128plus(uint64_t *state) {
    uint64_t x = state[0];
    uint64_t y = state[1];
    state[0] = y;
    x ^= x << 23;
    state[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return state[1] + y;
}

static float rand_float(uint64_t *state) {
    /* Generate float in [0, 1) from random bits */
    uint64_t r = xorshift128plus(state);
    return (float)(r >> 40) / (float)(1 << 24);  /* 24 bits of precision */
}

tl_sampler_t tl_sampler_default(void) {
    tl_sampler_t s = {
        .temperature        = TL_DEFAULT_TEMP,
        .top_p              = TL_DEFAULT_TOP_P,
        .top_k              = TL_DEFAULT_TOP_K,
        .repetition_penalty = TL_DEFAULT_REP_PENALTY,
        .seed               = 42,
        .rng_state          = 0x123456789ABCDEF0ULL,
    };
    return s;
}

void tl_sampler_set_seed(tl_sampler_t *s, int seed) {
    s->seed = seed;
    /* SplitMix64 to initialize xorshift state */
    uint64_t z = (uint64_t)(seed);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    s->rng_state = z;
}

/* ═══════════════════════════════════════════════════════════════════
   Sampling
   ═══════════════════════════════════════════════════════════════════ */

/* Structure for (index, logit) pairs during sorting */
typedef struct {
    float    value;
    int      index;
} tl_sort_pair_t;

static int cmp_desc(const void *a, const void *b) {
    float va = ((tl_sort_pair_t*)a)->value;
    float vb = ((tl_sort_pair_t*)b)->value;
    return (va < vb) ? 1 : (va > vb) ? -1 : 0;
}

/* ── In-place nth_element (quickselect) for top-k ─────────────────── */
/* Partitions pairs[start..end) around pivot; returns pivot position. */
static int partition(tl_sort_pair_t *p, int left, int right) {
    float pivot_val = p[left + (right-left)/2].value;
    int i = left-1, j = right+1;
    while (1) {
        do { i++; } while (p[i].value > pivot_val);
        do { j--; } while (p[j].value < pivot_val);
        if (i >= j) return j;
        tl_sort_pair_t tmp = p[i]; p[i] = p[j]; p[j] = tmp;
    }
}

/* Bring top-k values to front (not fully sorted). */
static void topk_select(tl_sort_pair_t *p, int n, int k) {
    if (k >= n) return;
    int left = 0, right = n-1;
    while (left < right) {
        int mid = partition(p, left, right);
        if (mid >= k) right = mid;
        else left = mid + 1;
    }
}

tl_token_t tl_sample(tl_infer_t *inf) {
    tl_sampler_t *s = &inf->sampler;
    tl_model_t *m  = inf->model;
    int vocab = m->vocab_size;
    float *logits = inf->logits_buf;
    float *probs  = inf->ffn_buf;
    tl_sort_pair_t *pairs = (tl_sort_pair_t*)(inf->attn_buf);
    int max_pairs = (int)(inf->attn_buf ? 1 : 0) * (TL_MAX_SEQ_LEN > 0 ? 1 : 1);
    (void)max_pairs;

    /* ── Fast path: greedy (temp → 0) ─────────────────────────── */
    if (s->temperature < 0.01f) {
        int best = 0;
        if (s->repetition_penalty != 1.0f && inf->gen_len > 0) {
            int lookback = (inf->gen_len < 64) ? inf->gen_len : 64;
            for (int i = inf->gen_len - lookback; i < inf->gen_len; i++) {
                tl_token_t prev = inf->generated[i];
                if (prev >= 0 && prev < vocab) {
                    if (logits[prev] > 0) logits[prev] /= s->repetition_penalty;
                    else logits[prev] *= s->repetition_penalty;
                }
            }
        }
        for (int i = 1; i < vocab; i++)
            if (logits[i] > logits[best]) best = i;
        return (tl_token_t)best;
    }

    float temp = s->temperature;

    /* ── Repetition penalty ────────────────────────────────────── */
    float rep_pen = s->repetition_penalty;
    if (rep_pen != 1.0f && inf->gen_len > 0) {
        int lookback = (inf->gen_len < 64) ? inf->gen_len : 64;
        for (int i = inf->gen_len - lookback; i < inf->gen_len; i++) {
            tl_token_t prev = inf->generated[i];
            if (prev >= 0 && prev < vocab) {
                if (logits[prev] > 0) logits[prev] /= rep_pen;
                else logits[prev] *= rep_pen;
            }
        }
    }

    /* ── Softmax + temperature ─────────────────────────────────── */
    float max_logit = -1e10f;
    for (int i = 0; i < vocab; i++)
        if (logits[i] > max_logit) max_logit = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        probs[i] = expf((logits[i] - max_logit) / temp);
        sum += probs[i];
    }
    float inv_sum = (sum > 0) ? 1.0f / sum : 1.0f;
    for (int i = 0; i < vocab; i++) probs[i] *= inv_sum;

    /* ── Top-k filtering (in-place quickselect) ─────────────────── */
    int top_k = s->top_k;
    if (top_k > 0 && top_k < vocab) {
        int nz = 0;
        for (int i = 0; i < vocab; i++) {
            if (probs[i] > 0) {
                pairs[nz].value = probs[i];
                pairs[nz].index = i;
                nz++;
            }
        }
        if (nz > top_k) {
            topk_select(pairs, nz, top_k);
            /* Zero out below top-k */
            memset(probs, 0, vocab * sizeof(float));
            sum = 0.0f;
            for (int i = 0; i < top_k; i++) {
                probs[pairs[i].index] = pairs[i].value;
                sum += pairs[i].value;
            }
            inv_sum = (sum > 0) ? 1.0f / sum : 1.0f;
            for (int i = 0; i < top_k; i++)
                probs[pairs[i].index] *= inv_sum;
        }
    }

    /* ── Top-p (nucleus) filtering ─────────────────────────────── */
    float top_p = s->top_p;
    if (top_p < 1.0f && top_p > 0.0f) {
        int nz = 0;
        for (int i = 0; i < vocab; i++) {
            if (probs[i] > 0) {
                pairs[nz].value = probs[i];
                pairs[nz].index = i;
                nz++;
            }
        }
        if (nz > 1) {
            qsort(pairs, nz, sizeof(tl_sort_pair_t), cmp_desc);
            float cumsum = 0.0f;
            int cutoff = nz;
            for (int i = 0; i < nz; i++) {
                cumsum += pairs[i].value;
                if (cumsum > top_p) { cutoff = i + 1; break; }
            }
            if (cutoff < nz) {
                memset(probs, 0, vocab * sizeof(float));
                sum = 0.0f;
                for (int i = 0; i < cutoff; i++) {
                    probs[pairs[i].index] = pairs[i].value;
                    sum += pairs[i].value;
                }
                inv_sum = (sum > 0) ? 1.0f / sum : 1.0f;
                for (int i = 0; i < cutoff; i++)
                    probs[pairs[i].index] *= inv_sum;
            }
        }
    }

    /* ── Sample from categorical distribution ────────────────────── */
    float r = rand_float(&s->rng_state);
    float csum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        csum += probs[i];
        if (r < csum) return (tl_token_t)i;
    }

    /* Fallback: argmax */
    int best = 0;
    for (int i = 1; i < vocab; i++)
        if (probs[i] > probs[best]) best = i;
    return (tl_token_t)best;
}
