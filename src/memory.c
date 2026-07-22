/*
 * memory.c — Long-term memory store.
 *
 * ds4: simple key-value store with embedding-based retrieval,
 *   persisted to disk as JSON Lines for crash safety.
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

tl_memory_store_t *tl_memory_create(int emb_dim, const char *filepath) {
    tl_memory_store_t *mem = tl_calloc(1, sizeof(tl_memory_store_t));
    mem->emb_dim  = emb_dim;
    mem->capacity = 1024;
    mem->entries  = tl_calloc(mem->capacity, sizeof(tl_mem_entry_t));
    if (filepath) strncpy(mem->filepath, filepath, sizeof(mem->filepath)-1);
    return mem;
}

void tl_memory_free(tl_memory_store_t *mem) {
    if (!mem) return;
    for (int i = 0; i < mem->n_entries; i++) {
        tl_free(mem->entries[i].key);
        tl_free(mem->entries[i].value);
        tl_free(mem->entries[i].embedding);
    }
    tl_free(mem->entries);
    tl_free(mem);
}

int tl_memory_put(tl_memory_store_t *mem, const char *key, const char *value,
                   const float *emb, float importance) {
    /* Check if key already exists → update */
    for (int i = 0; i < mem->n_entries; i++) {
        if (strcmp(mem->entries[i].key, key) == 0) {
            tl_free(mem->entries[i].value);
            mem->entries[i].value = strdup(value);
            mem->entries[i].importance = importance;
            mem->entries[i].timestamp = (int64_t)tl_time_now();
            mem->entries[i].access_count++;
            return i;
        }
    }

    /* Resize if needed */
    if (mem->n_entries >= mem->capacity) {
        mem->capacity *= 2;
        mem->entries = realloc(mem->entries, mem->capacity * sizeof(tl_mem_entry_t));
    }

    int idx = mem->n_entries++;
    mem->entries[idx].key        = strdup(key);
    mem->entries[idx].value      = strdup(value);
    mem->entries[idx].importance = importance;
    mem->entries[idx].timestamp  = (int64_t)tl_time_now();
    mem->entries[idx].access_count = 1;

    if (emb) {
        mem->entries[idx].embedding = tl_alloc(mem->emb_dim * sizeof(float));
        memcpy(mem->entries[idx].embedding, emb, mem->emb_dim * sizeof(float));
        mem->entries[idx].emb_dim = mem->emb_dim;
    }

    /* Auto-save after each put for durability */
    tl_memory_save(mem);

    return idx;
}

/* Forward declaration */
static float cosine_sim(const float *a, const float *b, int dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a < 1e-8f || norm_b < 1e-8f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

int tl_memory_search(tl_memory_store_t *mem, const float *query_emb, int top_k,
                     tl_mem_entry_t **results) {
    if (!mem || mem->n_entries == 0) return 0;
    if (top_k > mem->n_entries) top_k = mem->n_entries;

    typedef struct { float score; int idx; } scored_t;
    scored_t *scores = tl_alloc(mem->n_entries * sizeof(scored_t));
    bool *taken = tl_calloc(mem->n_entries, sizeof(bool));

    for (int i = 0; i < mem->n_entries; i++) {
        scores[i].score = mem->entries[i].embedding ?
            cosine_sim(query_emb, mem->entries[i].embedding, mem->emb_dim)
            + mem->entries[i].importance * 0.1f
            : mem->entries[i].importance * 0.5f;
        scores[i].idx = i;
    }

    /* Reusable result buffer */
    static tl_mem_entry_t *result_buf = NULL;
    static int result_buf_size = 0;
    if (!result_buf || result_buf_size < top_k) {
        tl_free(result_buf);
        result_buf_size = top_k + 16;
        result_buf = tl_alloc(result_buf_size * sizeof(tl_mem_entry_t));
    }

    for (int k = 0; k < top_k; k++) {
        int best = -1; float best_score = -1e10f;
        for (int i = 0; i < mem->n_entries; i++) {
            if (!taken[i] && scores[i].score > best_score) {
                best_score = scores[i].score;
                best = i;
            }
        }
        if (best < 0) { top_k = k; break; }
        result_buf[k] = mem->entries[scores[best].idx];
        mem->entries[scores[best].idx].access_count++;
        taken[best] = true;
    }

    *results = result_buf;
    tl_free(scores);
    tl_free(taken);
    return top_k;
}

/* ── Persist to disk (JSON Lines) ────────────────────────────────── */
int tl_memory_save(tl_memory_store_t *mem) {
    if (!mem->filepath[0]) return -1;

    FILE *f = fopen(mem->filepath, "w");
    if (!f) return -1;

    for (int i = 0; i < mem->n_entries; i++) {
        tl_mem_entry_t *e = &mem->entries[i];
        /* Escape value — minimal: just replace newlines */
        char *escaped = strdup(e->value);
        for (char *c = escaped; *c; c++)
            if (*c == '\n' || *c == '"' || *c == '\\') *c = ' ';

        fprintf(f, "{\"key\":\"%s\",\"value\":\"%s\",\"importance\":%.3f,"
                   "\"timestamp\":%lld,\"access\":%d}\n",
                e->key, escaped, e->importance,
                (long long)e->timestamp, e->access_count);
        tl_free(escaped);
    }
    fclose(f);
    return 0;
}

/* ── Load from disk ──────────────────────────────────────────────── */
int tl_memory_load(tl_memory_store_t *mem) {
    if (!mem->filepath[0]) return -1;

    char *data = tl_read_file(mem->filepath, NULL);
    if (!data) return 0; /* file doesn't exist yet, OK */

    /* Simple JSON Lines parser */
    char *line = data;
    int count = 0;
    while (*line) {
        char *end = strchr(line, '\n');
        if (!end) end = line + strlen(line);

        /* Extract key, value, importance */
        char key[256] = {0}, value[8192] = {0};
        float importance = 0.5f;

        const char *kp = strstr(line, "\"key\"");
        const char *vp = strstr(line, "\"value\"");
        const char *ip = strstr(line, "\"importance\"");

        if (kp) {
            kp = strchr(kp, ':');
            if (kp) {
                kp++; while (*kp == '"' || *kp == ' ') kp++;
                int i = 0;
                while (*kp && *kp != '"' && i < 255) key[i++] = *kp++;
            }
        }
        if (vp) {
            vp = strchr(vp, ':');
            if (vp) {
                vp++; while (*vp == '"' || *vp == ' ') vp++;
                int i = 0;
                while (*vp && *vp != '"' && i < 8191) value[i++] = *vp++;
            }
        }
        if (ip) {
            ip = strchr(ip, ':');
            if (ip) importance = strtof(ip+1, NULL);
        }

        if (key[0])
            tl_memory_put(mem, key, value, NULL, importance);

        count++;
        if (*end == '\n') line = end + 1;
        else break;
    }
    tl_free(data);
    return count;
}
