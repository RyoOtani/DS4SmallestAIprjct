/*
 * model.c — GGUF model loader with MoE support.
 *   ds4: reads GGUF directly, no external libs.
 *
 * GGUF format (simplified):
 *   [magic: "GGUF" 4 bytes]
 *   [version: u32le]
 *   [n_tensors: u64le]
 *   [n_metadata_kv: u64le]
 *   [metadata key-value pairs...]
 *   [tensor info entries...]
 *   [padding to ALIGNMENT]
 *   [tensor data...]
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

/* Endianness — assume little-endian (x86_64, ARM). GGUF is always LE. */
#if defined(__BIG_ENDIAN__)
  #error "Big-endian not supported for GGUF loading"
#else
  #define le32toh(x) (x)
  #define le64toh(x) (x)
#endif

#define GGUF_MAGIC           0x46554747  /* "GGUF" in LE */
#define GGUF_ALIGNMENT       32
#define GGUF_DEFAULT_ALIGN   32

/* GGUF value types */
enum {
    GGUF_TYPE_U8  = 0, GGUF_TYPE_I8  = 1, GGUF_TYPE_U16 = 2, GGUF_TYPE_I16 = 3,
    GGUF_TYPE_U32 = 4, GGUF_TYPE_I32 = 5, GGUF_TYPE_F32 = 6, GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STR = 8, GGUF_TYPE_ARR = 9, GGUF_TYPE_U64 = 10, GGUF_TYPE_I64 = 11,
    GGUF_TYPE_F64 = 12,
};

typedef struct {
    FILE   *fp;
    /* Header */
    uint32_t magic, version;
    uint64_t n_tensors, n_meta;

    /* Current offset for reading tensors */
    int64_t  data_offset;

    /* Metadata looked up by key */
    struct { char key[128]; int type; union { uint32_t u32; int32_t i32;
      uint64_t u64; float f32; char str[256]; }; } meta[128];
    int     n_meta_parsed;

    /* Tensor info (read from header) */
    struct {
        char     name[128];
        uint32_t n_dims;
        uint64_t dims[4];
        uint32_t ggml_type;     /* GGML quant type → tl_qtype_t */
        uint64_t offset;        /* byte offset in file */
        uint64_t size;          /* size in bytes */
    } *tensors;
} gguf_ctx_t;

/* ── Little-endian read helpers ──────────────────────────────────── */
static uint32_t read_u32(FILE *f) { uint32_t v; fread(&v,1,4,f); return le32toh(v); }
static uint64_t read_u64(FILE *f) { uint64_t v; fread(&v,1,8,f); return le64toh(v); }
static float    read_f32(FILE *f) { uint32_t v=read_u32(f); float r; memcpy(&r,&v,4); return r; }

static void read_str(FILE *f, char *buf, int max) {
    uint64_t len = read_u64(f);
    if (len >= (uint64_t)max) len = max-1;
    fread(buf, 1, len, f); buf[len] = '\0';
}

/* ── GGUF type → tl_qtype_t ──────────────────────────────────────── */
static tl_qtype_t ggml_to_tl_qtype(uint32_t ggml_type) {
    switch (ggml_type) {
    case 0:  return TL_QTYPE_F32;   /* GGML_TYPE_F32  */
    case 1:  return TL_QTYPE_F16;   /* GGML_TYPE_F16  */
    case 2:  return TL_QTYPE_Q4_0;  /* GGML_TYPE_Q4_0 */
    case 3:  return TL_QTYPE_Q4_1;  /* GGML_TYPE_Q4_1 */
    case 8:  return TL_QTYPE_Q8_0;  /* GGML_TYPE_Q8_0 */
    case 18: return TL_QTYPE_Q6_K;  /* GGML_TYPE_Q6_K */
    default: return TL_QTYPE_F32;
    }
}

/* ── Open and parse header ───────────────────────────────────────── */
static gguf_ctx_t *gguf_open(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { tl_log("cannot open model: %s", path); return NULL; }

    gguf_ctx_t *ctx = tl_calloc(1, sizeof(gguf_ctx_t));
    ctx->fp = f;

    /* Magic */
    ctx->magic = read_u32(f);
    if (ctx->magic != GGUF_MAGIC) { tl_log("not a GGUF file"); fclose(f); tl_free(ctx); return NULL; }

    /* Version */
    ctx->version = read_u32(f);
    tl_log("GGUF v%u detected", ctx->version);

    ctx->n_tensors = read_u64(f);
    ctx->n_meta    = read_u64(f);

    tl_log("  meta: %llu, tensors: %llu", (unsigned long long)ctx->n_meta, (unsigned long long)ctx->n_tensors);

    /* Parse metadata */
    for (uint64_t i = 0; i < ctx->n_meta && ctx->n_meta_parsed < 128; i++) {
        char key[128]; read_str(f, key, sizeof(key));
        uint32_t vtype = read_u32(f);

        int mi = ctx->n_meta_parsed++;
        strncpy(ctx->meta[mi].key, key, sizeof(ctx->meta[mi].key)-1);
        ctx->meta[mi].type = vtype;

        switch (vtype) {
        case GGUF_TYPE_U32: ctx->meta[mi].u32 = read_u32(f); break;
        case GGUF_TYPE_U64: ctx->meta[mi].u64 = read_u64(f); break;
        case GGUF_TYPE_I32: ctx->meta[mi].i32 = (int32_t)read_u32(f); break;
        case GGUF_TYPE_F32: ctx->meta[mi].f32 = read_f32(f); break;
        case GGUF_TYPE_STR: read_str(f, ctx->meta[mi].str, 256); break;
        default: /* skip unknown */
            fseek(f, 1, SEEK_CUR); /* rough skip */
            break;
        }
    }

    /* Parse tensor info */
    ctx->tensors = tl_calloc(ctx->n_tensors, sizeof(ctx->tensors[0]));
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        read_str(f, ctx->tensors[i].name, 128);
        ctx->tensors[i].n_dims = read_u32(f);
        for (uint32_t d = 0; d < ctx->tensors[i].n_dims && d < 4; d++)
            ctx->tensors[i].dims[d] = read_u64(f);
        ctx->tensors[i].ggml_type = read_u32(f);
        ctx->tensors[i].offset = read_u64(f);  /* offset from start of tensor data */
    }

    /* Align to GGUF_ALIGNMENT */
    int64_t cur = ftell(f);
    int64_t aligned = (cur + GGUF_ALIGNMENT - 1) & ~(GGUF_ALIGNMENT - 1);
    fseek(f, aligned, SEEK_SET);
    ctx->data_offset = ftell(f);

    return ctx;
}

static void gguf_close(gguf_ctx_t *ctx) {
    if (ctx) { if (ctx->fp) fclose(ctx->fp); tl_free(ctx->tensors); tl_free(ctx); }
}

/* ── Tensor data loading ─────────────────────────────────────────── */
static int gguf_load_tensor(gguf_ctx_t *ctx, int idx, tl_tensor_t *t) {
    fseek(ctx->fp, ctx->data_offset + ctx->tensors[idx].offset, SEEK_SET);

    int rows = (int)ctx->tensors[idx].dims[0];
    int cols = (int)(ctx->tensors[idx].n_dims >= 2 ? ctx->tensors[idx].dims[1] : 1);

    tl_qtype_t qtype = ggml_to_tl_qtype(ctx->tensors[idx].ggml_type);
    *t = tl_tensor_alloc(rows, cols, qtype);

    size_t elems = (size_t)rows * cols;

    if (qtype == TL_QTYPE_F32) {
        fread(t->data, sizeof(float), elems, ctx->fp);
    } else if (qtype == TL_QTYPE_Q4_0) {
        int n_blocks = (elems + TL_BLOCK_SIZE - 1) / TL_BLOCK_SIZE;
        for (int b = 0; b < n_blocks; b++) {
            float scale; fread(&scale, sizeof(float), 1, ctx->fp);
            t->scales[b] = scale;
            int blk_sz = (b == n_blocks-1) ? (int)(elems - b*TL_BLOCK_SIZE) : TL_BLOCK_SIZE;
            fread(t->qdata + b * TL_BLOCK_SIZE / 2, 1, (blk_sz + 1) / 2, ctx->fp);
        }
        /* Dequantize for working copy */
        tl_dequantize_q4(t->qdata, t->scales, t->data, rows, cols);
    } else {
        /* Other types: read as raw and attempt dequant */
        fread(t->data, 1, ctx->tensors[idx].size, ctx->fp);
    }
    return 0;
}

/* ── Find tensor by name ─────────────────────────────────────────── */
static int gguf_find_tensor(gguf_ctx_t *ctx, const char *name) {
    for (uint64_t i = 0; i < ctx->n_tensors; i++)
        if (strcmp(ctx->tensors[i].name, name) == 0) return (int)i;
    return -1;
}

/* ── Metadata lookup ─────────────────────────────────────────────── */
static int gguf_meta_int(gguf_ctx_t *ctx, const char *key, int def) {
    for (int i = 0; i < ctx->n_meta_parsed; i++)
        if (strcmp(ctx->meta[i].key, key) == 0)
            return (int)(ctx->meta[i].type == GGUF_TYPE_U32 ? ctx->meta[i].u32 :
                         ctx->meta[i].type == GGUF_TYPE_I32 ? ctx->meta[i].i32 : def);
    return def;
}

/* ═══════════════════════════════════════════════════════════════════
   tl_model_load
   ═══════════════════════════════════════════════════════════════════ */

tl_model_t *tl_model_load(const char *path) {
    double t0 = tl_time_now();
    gguf_ctx_t *ctx = gguf_open(path);
    if (!ctx) return NULL;

    tl_model_t *m = tl_calloc(1, sizeof(tl_model_t));

    /* Architecture detection */
    strncpy(m->arch, "unknown", sizeof(m->arch)-1);
    for (int i = 0; i < ctx->n_meta_parsed; i++) {
        if (strstr(ctx->meta[i].key, "architecture"))
            strncpy(m->arch, ctx->meta[i].str, sizeof(m->arch)-1);
        if (strstr(ctx->meta[i].key, "tokenizer.ggml.model"))
            strncpy(m->tokenizer_model, ctx->meta[i].str, sizeof(m->tokenizer_model)-1);
    }

    /* Dimensions from metadata */
    m->hidden_dim   = gguf_meta_int(ctx, "llm.hidden_size", 2048);
    m->n_layers     = gguf_meta_int(ctx, "llm.block_count", 32);
    m->vocab_size   = gguf_meta_int(ctx, "llm.context_length", 32000);
    m->max_seq_len  = gguf_meta_int(ctx, "llm.context_length", TL_MAX_SEQ_LEN);

    /* MoE detection */
    int n_experts    = gguf_meta_int(ctx, "llm.expert_count", 0);
    int n_moe_layers = gguf_meta_int(ctx, "llm.moe_layer_count", 0);

    tl_log("Model: %s", m->arch);
    tl_log("  hidden_dim=%d, layers=%d, vocab=%d, seq_len=%d",
           m->hidden_dim, m->n_layers, m->vocab_size, m->max_seq_len);
    if (n_experts > 0) tl_log("  MoE: %d experts, %d MoE layers", n_experts, n_moe_layers);

    /* Load embeddings */
    int tid;
    tid = gguf_find_tensor(ctx, "token_embd.weight");
    if (tid < 0) tid = gguf_find_tensor(ctx, "model.embed_tokens.weight");
    if (tid >= 0) gguf_load_tensor(ctx, tid, &m->tok_embeddings);

    /* Load layers */
    m->layers = tl_calloc(m->n_layers, sizeof(tl_layer_t));

    char name[256];
    for (int l = 0; l < m->n_layers; l++) {
        tl_layer_t *ly = &m->layers[l];

        /* RMS norm (pre-attn) */
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        tid = gguf_find_tensor(ctx, name);
        if (tid >= 0) gguf_load_tensor(ctx, tid, &ly->rms_attn_w);

        /* MLA attention weights */
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        tid = gguf_find_tensor(ctx, name);
        if (tid >= 0) gguf_load_tensor(ctx, tid, &ly->mla.w_q);

        snprintf(name, sizeof(name), "blk.%d.attn_kv_a.weight", l);
        tid = gguf_find_tensor(ctx, name);
        if (tid >= 0) gguf_load_tensor(ctx, tid, &ly->mla.w_kv_compress);

        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        tid = gguf_find_tensor(ctx, name);
        if (tid >= 0) gguf_load_tensor(ctx, tid, &ly->mla.w_k_up);

        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        tid = gguf_find_tensor(ctx, name);
        if (tid >= 0) gguf_load_tensor(ctx, tid, &ly->mla.w_v_up);

        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        tid = gguf_find_tensor(ctx, name);
        if (tid >= 0) gguf_load_tensor(ctx, tid, &ly->mla.w_o);

        /* RMS norm (pre-ffn) */
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        tid = gguf_find_tensor(ctx, name);
        if (tid >= 0) gguf_load_tensor(ctx, tid, &ly->rms_ffn_w);

        /* MoE or dense FFN? */
        bool is_moe_layer = (n_moe_layers > 0) && (l % (m->n_layers / n_moe_layers) == 0 || n_moe_layers >= m->n_layers);

        if (is_moe_layer && n_experts > 0) {
            ly->is_moe = true;
            ly->ffn.moe.n_experts = n_experts;
            ly->ffn.moe.n_active = TL_MAX_ACTIVE_EXPERTS;

            /* MoE gate */
            snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
            tid = gguf_find_tensor(ctx, name);
            if (tid >= 0) gguf_load_tensor(ctx, tid, &ly->ffn.moe.gate.gate_w);

            /* Expert FFNs */
            ly->ffn.moe.experts = tl_calloc(n_experts, sizeof(tl_ffn_t));
            for (int e = 0; e < n_experts; e++) {
                tl_ffn_t *exp = &ly->ffn.moe.experts[e];

                snprintf(name, sizeof(name), "blk.%d.ffn_gate.%d.weight", l, e);
                tid = gguf_find_tensor(ctx, name);
                if (tid >= 0) gguf_load_tensor(ctx, tid, &exp->w_gate);

                snprintf(name, sizeof(name), "blk.%d.ffn_up.%d.weight", l, e);
                tid = gguf_find_tensor(ctx, name);
                if (tid >= 0) gguf_load_tensor(ctx, tid, &exp->w_up);

                snprintf(name, sizeof(name), "blk.%d.ffn_down.%d.weight", l, e);
                tid = gguf_find_tensor(ctx, name);
                if (tid >= 0) gguf_load_tensor(ctx, tid, &exp->w_down);
            }
            m->total_experts += n_experts;
            m->n_moe_layers++;
        } else {
            ly->is_moe = false;
            snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
            tid = gguf_find_tensor(ctx, name);
            if (tid >= 0) gguf_load_tensor(ctx, tid, &ly->ffn.dense.w_gate);

            snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
            tid = gguf_find_tensor(ctx, name);
            if (tid >= 0) gguf_load_tensor(ctx, tid, &ly->ffn.dense.w_up);

            snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
            tid = gguf_find_tensor(ctx, name);
            if (tid >= 0) gguf_load_tensor(ctx, tid, &ly->ffn.dense.w_down);
        }
    }

    /* Final RMS norm */
    tid = gguf_find_tensor(ctx, "output_norm.weight");
    if (tid < 0) tid = gguf_find_tensor(ctx, "model.norm.weight");
    if (tid >= 0) gguf_load_tensor(ctx, tid, &m->rms_final_w);

    /* LM head (output projection) */
    tid = gguf_find_tensor(ctx, "output.weight");
    if (tid < 0) tid = gguf_find_tensor(ctx, "lm_head.weight");
    if (tid >= 0) gguf_load_tensor(ctx, tid, &m->lm_head);

    gguf_close(ctx);

    double dt = tl_time_now() - t0;
    tl_log("Model loaded in %.2fs", dt);
    return m;
}

void tl_model_free(tl_model_t *m) {
    if (!m) return;
    tl_tensor_free(&m->tok_embeddings);
    tl_tensor_free(&m->rms_final_w);
    tl_tensor_free(&m->lm_head);
    for (int l = 0; l < m->n_layers; l++) {
        tl_layer_t *ly = &m->layers[l];
        tl_tensor_free(&ly->rms_attn_w);
        tl_tensor_free(&ly->rms_ffn_w);
        tl_tensor_free(&ly->mla.w_q);
        tl_tensor_free(&ly->mla.w_kv_compress);
        tl_tensor_free(&ly->mla.w_k_up);
        tl_tensor_free(&ly->mla.w_v_up);
        tl_tensor_free(&ly->mla.w_o);
        tl_free(ly->mla.rope_freqs);
        if (ly->is_moe) {
            tl_tensor_free(&ly->ffn.moe.gate.gate_w);
            for (int e = 0; e < ly->ffn.moe.n_experts; e++) {
                tl_tensor_free(&ly->ffn.moe.experts[e].w_up);
                tl_tensor_free(&ly->ffn.moe.experts[e].w_gate);
                tl_tensor_free(&ly->ffn.moe.experts[e].w_down);
            }
            tl_free(ly->ffn.moe.experts);
        } else {
            tl_tensor_free(&ly->ffn.dense.w_up);
            tl_tensor_free(&ly->ffn.dense.w_gate);
            tl_tensor_free(&ly->ffn.dense.w_down);
        }
    }
    tl_free(m->layers);
    tl_free(m);
}

void tl_model_print_info(const tl_model_t *m) {
    if (!m) return;
    tl_log("====== Model Info ======");
    tl_log("Architecture:    %s", m->arch);
    tl_log("Tokenizer:       %s", m->tokenizer_model);
    tl_log("Hidden dim:      %d", m->hidden_dim);
    tl_log("Layers:          %d", m->n_layers);
    tl_log("Vocab size:      %d", m->vocab_size);
    tl_log("Max seq len:     %d", m->max_seq_len);
    tl_log("MoE layers:      %d / %d", m->n_moe_layers, m->n_layers);
    tl_log("Total experts:   %d", m->total_experts);
    tl_log("Default qtype:   %d", m->default_qtype);
    int64_t mem = tl_memory_usage();
    if (mem > 0) tl_log("Process RSS:     %.1f GB", mem / (1024.0*1024.0*1024.0));
    tl_log("========================");
}
