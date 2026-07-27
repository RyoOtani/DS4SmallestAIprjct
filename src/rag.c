/*
 * rag.c — Retrieval-Augmented Generation: local vector index.
 *
 * ds4: brute-force cosine similarity is fine for <100k chunks.
 *   Embeddings computed via the model itself (no external API).
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>

tl_rag_index_t *tl_rag_create(int emb_dim) {
    tl_rag_index_t *rag = tl_calloc(1, sizeof(tl_rag_index_t));
    rag->emb_dim  = emb_dim;
    rag->capacity = 1024;
    rag->chunks   = tl_calloc(rag->capacity, sizeof(tl_rag_chunk_t));
    return rag;
}

void tl_rag_free(tl_rag_index_t *rag) {
    if (!rag) return;
    for (int i = 0; i < rag->n_chunks; i++) {
        tl_free(rag->chunks[i].text);
        tl_free(rag->chunks[i].embedding);
    }
    tl_free(rag->chunks);
    tl_free(rag);
}

int tl_rag_add(tl_rag_index_t *rag, const char *text, const float *emb,
               const char *source) {
    /* Resize if needed */
    if (rag->n_chunks >= rag->capacity) {
        rag->capacity *= 2;
        rag->chunks = realloc(rag->chunks, rag->capacity * sizeof(tl_rag_chunk_t));
    }

    int idx = rag->n_chunks++;
    rag->chunks[idx].text = strdup(text);
    strncpy(rag->chunks[idx].source, source, 255);

    if (emb) {
        rag->chunks[idx].embedding = tl_alloc(rag->emb_dim * sizeof(float));
        memcpy(rag->chunks[idx].embedding, emb, rag->emb_dim * sizeof(float));
        rag->chunks[idx].emb_dim = rag->emb_dim;
    }

    return idx;
}

/* ── Cosine similarity ───────────────────────────────────────────── */
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

int tl_rag_search(tl_rag_index_t *rag, const float *query_emb, int top_k,
                  tl_rag_chunk_t **results) {
    if (!rag || rag->n_chunks == 0) return 0;
    if (top_k > rag->n_chunks) top_k = rag->n_chunks;

    /* Temporary scores */
    typedef struct { float score; int idx; } scored_t;
    scored_t *scores = tl_alloc(rag->n_chunks * sizeof(scored_t));
    bool *taken = tl_calloc(rag->n_chunks, sizeof(bool));

    for (int i = 0; i < rag->n_chunks; i++) {
        scores[i].score = rag->chunks[i].embedding ?
            cosine_sim(query_emb, rag->chunks[i].embedding, rag->emb_dim) : 0.0f;
        scores[i].idx = i;
    }

    /* Verify embeddings exist; if not, return empty (semantic search disabled).
       To enable: populate chunk->embedding with float vectors of emb_dim dimensions. */
    if (!rag->chunks[0].embedding && rag->n_chunks > 0) {
        fprintf(stderr, "⚠️  RAG: no embeddings loaded — semantic search disabled.\n");
        fprintf(stderr, "   Populate tl_rag_chunk_t.embedding (float[emb_dim]) to enable.\n");
        *results = NULL;
        tl_free(scores);
        tl_free(taken);
        return 0;
    }

    /* Reusable result buffer (stored on index, but this is a static helper) */
    static tl_rag_chunk_t *result_buf = NULL;
    static int result_buf_size = 0;
    if (!result_buf || result_buf_size < top_k) {
        tl_free(result_buf);
        result_buf_size = top_k + 16;
        result_buf = tl_alloc(result_buf_size * sizeof(tl_rag_chunk_t));
    }

    /* O(N*K) top-k with taken[] array */
    for (int k = 0; k < top_k; k++) {
        int best = -1; float best_score = -1e10f;
        for (int i = 0; i < rag->n_chunks; i++) {
            if (!taken[i] && scores[i].score > best_score) {
                best_score = scores[i].score;
                best = i;
            }
        }
        if (best < 0) { top_k = k; break; }
        result_buf[k] = rag->chunks[scores[best].idx];
        taken[best] = true;
    }

    *results = result_buf;
    tl_free(scores);
    tl_free(taken);
    return top_k;
}

/* ── Index a directory of source files ───────────────────────────── */
int tl_rag_index_dir(tl_rag_index_t *rag, const char *dir_path) {
    DIR *d = opendir(dir_path);
    if (!d) return -1;

    struct dirent *ent;
    char path[1024];
    int count = 0;

    while ((ent = readdir(d))) {
        if (ent->d_name[0] == '.') continue;
        snprintf(path, sizeof(path), "%s/%s", dir_path, ent->d_name);

        struct stat st;
        if (stat(path, &st) != 0) continue;

        if (S_ISREG(st.st_mode)) {
            /* Only index text-like files */
            const char *ext = strrchr(ent->d_name, '.');
            if (ext && (strcmp(ext, ".c") == 0 || strcmp(ext, ".h") == 0 ||
                        strcmp(ext, ".py") == 0 || strcmp(ext, ".md") == 0 ||
                        strcmp(ext, ".txt") == 0 || strcmp(ext, ".rs") == 0 ||
                        strcmp(ext, ".go") == 0 || strcmp(ext, ".js") == 0 ||
                        strcmp(ext, ".ts") == 0)) {
                size_t len;
                char *content = tl_read_file(path, &len);
                if (content) {
                    /* Simple chunking: split by paragraphs or fixed size */
                    /* For now, add whole file as a chunk */
                    tl_rag_add(rag, content, NULL, path);
                    tl_free(content);
                    count++;
                }
            }
        }
    }
    closedir(d);
    return count;
}
