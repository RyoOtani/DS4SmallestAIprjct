/*
 * tokenizer.c — BPE / SentencePiece tokenizer with FIM support.
 *
 * Implements byte-level BPE encoding/decoding using a merge-rank
 * hash table and a trie for reverse lookup.
 *
 * ds4: pure C, no external deps. Suitable for GGUF bundled tokenizers.
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* FNV-1a hash for merge lookup */
#define FNV_OFFSET 2166136261u
#define FNV_PRIME  16777619u

static uint32_t fnv_hash(uint32_t a, uint32_t b) {
    uint32_t h = FNV_OFFSET;
    h ^= (uint8_t)(a);
    h *= FNV_PRIME;
    h ^= (uint8_t)(a >> 8);
    h *= FNV_PRIME;
    h ^= (uint8_t)(b);
    h *= FNV_PRIME;
    h ^= (uint8_t)(b >> 8);
    h *= FNV_PRIME;
    return h;
}

/* ── Merge hash table ────────────────────────────────────────────── */
#define MERGE_TABLE_BITS 18
#define MERGE_TABLE_SIZE (1 << MERGE_TABLE_BITS)
#define MERGE_KEY(a,b)   ((((uint32_t)(uint16_t)(a)) << 16) | ((uint32_t)(uint16_t)(b)))

static int merge_table_find(tl_tokenizer_t *t, tl_token_t a, tl_token_t b) {
    uint32_t key = MERGE_KEY(a, b);
    uint32_t idx = fnv_hash((uint32_t)a, (uint32_t)b) & (MERGE_TABLE_SIZE - 1);

    /* Linear probe */
    for (int i = 0; i < 128; i++) {
        uint32_t probe = (idx + i) & (MERGE_TABLE_SIZE - 1);
        if (t->merges[probe].key == key) return t->merges[probe].rank;
        if (t->merges[probe].key == 0) return -1;
    }
    return -1;
}

static void merge_table_insert(tl_tokenizer_t *t, tl_token_t a, tl_token_t b, int rank) {
    uint32_t key = MERGE_KEY(a, b);
    uint32_t idx = fnv_hash((uint32_t)a, (uint32_t)b) & (MERGE_TABLE_SIZE - 1);

    for (int i = 0; i < 128; i++) {
        uint32_t probe = (idx + i) & (MERGE_TABLE_SIZE - 1);
        if (t->merges[probe].key == 0 || t->merges[probe].key == key) {
            t->merges[probe].key = key;
            t->merges[probe].rank = rank;
            return;
        }
    }
}

/* ── Trie for reverse lookup (token → bytes) ─────────────────────── */

static void trie_insert(tl_tokenizer_t *t, const char *bytes, int len, int token_id) {
    struct tl_trie_node *node = t->trie_root;
    for (int i = 0; i < len; i++) {
        uint8_t b = (uint8_t)bytes[i];
        if (!node->children[b])
            node->children[b] = tl_calloc(1, sizeof(struct tl_trie_node));
        node = node->children[b];
    }
    node->token_id = token_id;
}

static int trie_find(const tl_tokenizer_t *t, const char *bytes, int len,
                     int *out_token_id, int *out_match_len) {
    struct tl_trie_node *node = t->trie_root;
    int last_match = -1;
    int last_len   = 0;

    for (int i = 0; i < len; i++) {
        uint8_t b = (uint8_t)bytes[i];
        if (!node->children[b]) break;
        node = node->children[b];
        if (node->token_id >= 0) {
            last_match = node->token_id;
            last_len   = i + 1;
        }
    }

    if (last_match >= 0) {
        *out_token_id = last_match;
        *out_match_len = last_len;
        return 1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
   BPE Encoding
   ═══════════════════════════════════════════════════════════════════ */

int tl_tokenize(tl_tokenizer_t *t, const char *text, tl_token_t *tokens, int max_tokens) {
    int text_len = (int)strlen(text);
    if (text_len == 0) return 0;

    /* Step 1: Byte-level tokenization (each byte → its token ID) */
    /* Reserve enough space: text_len + 32 (for merges) */
    int cap = text_len + 256;
    tl_token_t *ids = tl_alloc(cap * sizeof(tl_token_t));
    int n_ids = 0;

    for (int i = 0; i < text_len; i++) {
        /* Look up byte as UTF-8 continuation handling */
        uint8_t b = (uint8_t)text[i];

        /* Simple: treat each byte as a base token.
           Real BPE uses a pre-tokenizer (regex/unicode). */
        if (b < 128) {
            ids[n_ids++] = (tl_token_t)b + 3;  /* offset for special tokens */
        } else {
            /* Multi-byte: just use byte value as token for now */
            ids[n_ids++] = (tl_token_t)b + 259;
        }

        if (n_ids >= cap - 2) {
            cap *= 2;
            ids = realloc(ids, cap * sizeof(tl_token_t));
        }
    }

    /* Step 2: Iterative BPE merging (greedy, by rank) */
    /* Simplified: single pass with priority queue style */
    bool merged = true;
    int iterations = 0;

    while (merged && iterations < text_len * 2) {
        merged = false;
        iterations++;

        /* Find the pair with the lowest merge rank */
        int    best_i   = -1;
        int    best_rank = 0x7FFFFFFF;
        tl_token_t best_a = 0, best_b = 0;

        for (int i = 0; i < n_ids - 1; i++) {
            int rank = merge_table_find(t, ids[i], ids[i+1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i   = i;
                best_a   = ids[i];
                best_b   = ids[i+1];
            }
        }

        if (best_i >= 0) {
            /* Apply merge: replace (a, b) with merged token */
            /* The merged token ID is: vocab_size lookup...
               For simplicity, we assign merge rank as token id
               (in real impl, the merge produces a specific token) */
            tl_token_t new_token = (tl_token_t)(t->vocab_size - 1 - best_rank);

            ids[best_i] = new_token;
            /* Shift left */
            memmove(ids + best_i + 1, ids + best_i + 2,
                    (n_ids - best_i - 2) * sizeof(tl_token_t));
            n_ids--;
            merged = true;
        }
    }

    /* Copy to output */
    int result = (n_ids < max_tokens) ? n_ids : max_tokens;
    if (tokens) memcpy(tokens, ids, result * sizeof(tl_token_t));

    tl_free(ids);
    return result;
}

/* ═══════════════════════════════════════════════════════════════════
   FIM (Fill-in-the-Middle) tokenization
   ═══════════════════════════════════════════════════════════════════ */

int tl_tokenize_fim(tl_tokenizer_t *t, const char *prefix, const char *suffix,
                    tl_token_t *tokens, int max_tokens) {
    /* Layout: <FIM_prefix> prefix <FIM_suffix> suffix <FIM_middle>
       Then the model generates in the middle. */
    int n = 0;

    /* Special tokens at boundaries */
    if (n < max_tokens) tokens[n++] = t->fim_prefix_id;

    /* Prefix */
    n += tl_tokenize(t, prefix, tokens ? tokens + n : NULL,
                     max_tokens > 0 ? max_tokens - n : 0);

    if (n < max_tokens) tokens[n++] = t->fim_suffix_id;

    /* Suffix */
    n += tl_tokenize(t, suffix, tokens ? tokens + n : NULL,
                     max_tokens > 0 ? max_tokens - n : 0);

    if (n < max_tokens) tokens[n++] = t->fim_middle_id;

    return n;
}

/* ── Detokenization (tokens → UTF-8 string) ──────────────────────── */
char *tl_detokenize(tl_tokenizer_t *t, const tl_token_t *tokens, int n_tokens) {
    /* Estimate output size: avg 4 bytes per token */
    size_t est = (size_t)n_tokens * 8 + 1;
    char *out = tl_alloc(est);
    size_t pos = 0;

    for (int i = 0; i < n_tokens; i++) {
        tl_token_t tok = tokens[i];

        /* Skip special tokens */
        if (tok == t->bos_id || tok == t->eos_id || tok == t->pad_id ||
            tok == t->fim_prefix_id || tok == t->fim_suffix_id || tok == t->fim_middle_id)
            continue;

        /* Look up in vocab */
        if (tok >= 0 && tok < t->vocab_size && t->vocab[tok]) {
            int len = (int)strlen(t->vocab[tok]);
            if (pos + len + 1 > est) {
                est *= 2;
                out = realloc(out, est);
            }
            memcpy(out + pos, t->vocab[tok], len);
            pos += len;
        }
    }
    out[pos] = '\0';
    return out;
}

/* ── Load tokenizer from GGUF bundled data ───────────────────────── */
tl_tokenizer_t *tl_tokenizer_load(const char *path) {
    /* For now, create a minimal tokenizer.
       In production, this reads from the GGUF file's tokenizer section. */
    tl_tokenizer_t *t = tl_calloc(1, sizeof(tl_tokenizer_t));

    t->merge_capacity = MERGE_TABLE_SIZE;
    t->merges = tl_calloc(MERGE_TABLE_SIZE, sizeof(t->merges[0]));

    /* Dummy vocab: byte-level 256 + specials */
    t->vocab_size = 512;
    t->vocab = tl_calloc(t->vocab_size, sizeof(char*));

    /* Byte tokens */
    for (int i = 0; i < 256; i++) {
        t->vocab[i + 3] = tl_alloc(2);
        t->vocab[i + 3][0] = (char)i;
        t->vocab[i + 3][1] = '\0';
    }

    /* Special tokens */
    t->bos_id = 0;  t->vocab[0] = strdup("<s>");
    t->eos_id = 1;  t->vocab[1] = strdup("</s>");
    t->pad_id = 2;  t->vocab[2] = strdup("<pad>");
    t->unk_id = 3;  t->vocab[3] = strdup("<unk>");

    t->fim_prefix_id = 500; t->vocab[500] = strdup("<fim_prefix>");
    t->fim_suffix_id = 501; t->vocab[501] = strdup("<fim_suffix>");
    t->fim_middle_id = 502; t->vocab[502] = strdup("<fim_middle>");

    /* Init trie */
    t->trie_root = tl_calloc(1, sizeof(struct tl_trie_node));
    for (int i = 0; i < t->vocab_size; i++) {
        if (t->vocab[i])
            trie_insert(t, t->vocab[i], (int)strlen(t->vocab[i]), i);
    }

    return t;
}

void tl_tokenizer_free(tl_tokenizer_t *t) {
    if (!t) return;
    tl_free(t->merges);
    for (int i = 0; i < t->vocab_size; i++) tl_free(t->vocab[i]);
    tl_free(t->vocab);
    tl_free(t->trie_root); /* TODO: recursive trie free */
    tl_free(t->encode_cache);
    tl_free(t);
}
