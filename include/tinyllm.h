/*
 * tinyllm.h — Public API.
 *   ds4 spirit: single #include gives you everything.
 */
#ifndef TINYLLM_H
#define TINYLLM_H

#include "config.h"
#include <stdbool.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════
   Tensor / Matrix types
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float *data;          /* if qtype==F32, else quantized blob      */
    uint8_t *qdata;       /* quantized data (NULL if F32/F16)        */
    float *scales;        /* per-block scales (quantized only)       */
    int     rows, cols;
    tl_qtype_t qtype;
    size_t  byte_size;
    bool    on_device;    /* loaded to GPU? (future)                 */
} tl_tensor_t;

/* ── Token ───────────────────────────────────────────────────────── */
typedef int32_t tl_token_t;

/* ── Batch of token sequences ────────────────────────────────────── */
typedef struct {
    tl_token_t *tokens;          /* [batch_size * seq_len]            */
    int32_t    *positions;       /* position ids                     */
    int         batch_size;
    int         seq_len;
} tl_batch_t;

/* ═══════════════════════════════════════════════════════════════════
   MoE (Mixture of Experts)
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    tl_tensor_t gate_w;          /* [n_experts, hidden_dim]          */
    int         n_experts;
    int         n_active;        /* top-k (default 2)                */
} tl_moe_gate_t;

typedef struct {
    tl_tensor_t w_up;            /* [inter_dim, hidden_dim]          */
    tl_tensor_t w_gate;          /* [inter_dim, hidden_dim]          */
    tl_tensor_t w_down;          /* [hidden_dim, inter_dim]          */
    int         hidden_dim, inter_dim;
} tl_ffn_t;                      /* single expert = SwiGLU FFN       */

typedef struct {
    tl_moe_gate_t gate;
    tl_ffn_t     *experts;       /* [n_experts]                      */
    int           n_experts;
    int           n_active;
    int           hidden_dim;
    int           inter_dim;
} tl_moe_layer_t;                /* one MoE layer = gate + N experts */

/* ═══════════════════════════════════════════════════════════════════
   MLA: Multi-head Latent Attention
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    /* Projections for Q, K, V (compressed)                          */
    tl_tensor_t w_q;             /* [hidden_dim, hidden_dim]         */
    tl_tensor_t w_kv_compress;   /* [latent_dim, hidden_dim]         */
    tl_tensor_t w_k_up;          /* [n_heads*head_dim, latent_dim]   */
    tl_tensor_t w_v_up;          /* [n_heads*head_dim, latent_dim]   */
    tl_tensor_t w_o;             /* [hidden_dim, n_heads*head_dim]   */

    /* RoPE (rotary position embedding) frequencies                  */
    float      *rope_freqs;      /* [head_dim/2]                     */

    int         hidden_dim;
    int         n_heads;
    int         head_dim;
    int         latent_dim;      /* compressed KV latent dim         */
    float       rope_theta;      /* base frequency (default 10000)   */
} tl_mla_t;

/* ── KV cache (MLA: stores compressed latents, not full heads) ──── */
typedef struct {
    float *k_latent;             /* [n_layers][cache_len][latent_dim]*/
    float *v_latent;             /* [n_layers][cache_len][latent_dim]*/
    int    cache_len;            /* current fill                     */
    int    max_len;              /* rolling window size              */
    int    n_layers;
    int    latent_dim;
} tl_kv_cache_t;

/* ═══════════════════════════════════════════════════════════════════
   Transformer Layer
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    tl_tensor_t rms_attn_w;      /* RMS norm weight (pre-attn)       */
    tl_tensor_t rms_ffn_w;       /* RMS norm weight (pre-ffn)        */
    tl_mla_t    mla;             /* attention                        */
    union {
        tl_ffn_t      dense;     /* dense FFN (non-MoE layers)       */
        tl_moe_layer_t moe;      /* MoE layer                        */
    } ffn;
    bool         is_moe;         /* true → use ffn.moe               */
} tl_layer_t;

/* ═══════════════════════════════════════════════════════════════════
   Full Model
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    /* Embedding                                                     */
    tl_tensor_t tok_embeddings;  /* [vocab_size, hidden_dim]         */
    tl_tensor_t rms_final_w;    /* final RMS norm                    */
    tl_tensor_t lm_head;        /* output projection (tied or not)   */

    /* Layers                                                        */
    tl_layer_t *layers;
    int         n_layers;
    int         hidden_dim;
    int         vocab_size;
    int         max_seq_len;

    /* MoE summary                                                   */
    int         total_experts;
    int         n_moe_layers;

    /* Quantization info                                             */
    tl_qtype_t  default_qtype;

    /* Metadata from GGUF                                            */
    char        arch[64];
    char        tokenizer_model[64];
} tl_model_t;

/* ═══════════════════════════════════════════════════════════════════
   Tokenizer
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    /* BPE merge ranks: map from (token_a, token_b) → rank           */
    /* Implemented as hash table for O(1) lookup                     */
    struct {
        uint32_t  key;          /* packed: (a << 16) | b             */
        int32_t   rank;
    } *merges;
    int         n_merges;
    int         merge_capacity;

    /* Vocab: token_id → bytes                                       */
    char      **vocab;
    float      *vocab_scores;
    int         vocab_size;

    /* Reverse: byte sequence → token_id via trie                    */
    struct tl_trie_node {
        int32_t token_id;       /* -1 if not terminal                */
        struct tl_trie_node *children[256];
    } *trie_root;

    /* Special tokens                                                */
    tl_token_t bos_id, eos_id, pad_id, unk_id;
    tl_token_t fim_prefix_id, fim_suffix_id, fim_middle_id;

    /* Token cache for encode (LRU, O(1) amortized)                  */
    struct {
        char     *bytes;
        int       len;
        tl_token_t token;
    } *encode_cache;
    int         encode_cache_size;
} tl_tokenizer_t;

/* ═══════════════════════════════════════════════════════════════════
   Sampler
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float   temperature;
    float   top_p;
    int     top_k;
    float   repetition_penalty;
    int     seed;
    uint64_t rng_state;
} tl_sampler_t;

/* ═══════════════════════════════════════════════════════════════════
   Inference State
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    tl_model_t     *model;
    tl_tokenizer_t *tokenizer;
    tl_sampler_t    sampler;
    tl_kv_cache_t  *kv_cache;

    /* Scratch buffers (reused across forward passes)                */
    float *hidden_buf;           /* [batch * hidden_dim]              */
    float *logits_buf;           /* [batch * vocab_size]              */
    float *attn_buf;             /* attention workspace               */
    float *ffn_buf;              /* FFN workspace                     */
    float *expert_scores;        /* [batch * n_experts] gate output   */

    /* Generated tokens (streaming output)                            */
    tl_token_t *generated;
    int         gen_capacity;
    int         gen_len;

    /* Tool-call scratchpad                                          */
    char   *scratchpad;          /* thinking output (text)            */
    size_t  scratchpad_len;
} tl_infer_t;

/* ═══════════════════════════════════════════════════════════════════
   Tool Definition & Execution
   ═══════════════════════════════════════════════════════════════════ */

typedef enum {
    TL_TOOL_RUN_CMD      = 0,    /* shell command execution          */
    TL_TOOL_READ_FILE    = 1,    /* read file contents               */
    TL_TOOL_WRITE_FILE   = 2,    /* write/modify file                */
    TL_TOOL_SEARCH_CODE  = 3,    /* grep / AST search                */
    TL_TOOL_RUN_TEST     = 4,    /* compile + run test suite         */
    TL_TOOL_WEB_SEARCH   = 5,    /* web search (cached)              */
    TL_TOOL_RAG_RETRIEVE = 6,    /* vector search local docs         */
    TL_TOOL_MEM_STORE    = 7,    /* save to long-term memory         */
    TL_TOOL_MEM_RECALL   = 8,    /* recall from long-term memory     */
    TL_TOOL_SANDBOX_EXEC = 9,    /* execute in Docker/podman         */
    TL_TOOL_BROWSER      = 10,   /* headless browser control         */
} tl_tool_type_t;

typedef struct {
    tl_tool_type_t type;
    char    name[64];
    char    description[256];
    char    params_json[1024];   /* JSON Schema for parameters       */
} tl_tool_def_t;

typedef struct {
    tl_tool_type_t type;
    char   *tool_name;
    char   *params;              /* JSON string                      */
    char   *result;              /* output (filled after execution)  */
    int     exit_code;
    bool    executed;
} tl_tool_call_t;

/* ═══════════════════════════════════════════════════════════════════
   Agent (self-correcting loop)
   ═══════════════════════════════════════════════════════════════════ */

typedef enum {
    TL_AGENT_IDLE,
    TL_AGENT_THINKING,
    TL_AGENT_CALLING_TOOL,
    TL_AGENT_OBSERVING,
    TL_AGENT_DONE,
    TL_AGENT_ERROR
} tl_agent_state_t;

typedef struct {
    tl_infer_t     *infer;
    tl_tool_call_t *tool_calls;
    int             n_tool_calls;
    int             max_tool_calls;
    int             retry_count;
    int             max_retries;
    tl_agent_state_t state;
    char           *task;         /* original user task               */
    char           *plan;         /* CoT plan (scratchpad)            */
} tl_agent_t;

/* ═══════════════════════════════════════════════════════════════════
   RAG: Retrieval-Augmented Generation
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    char    *text;               /* chunk text                       */
    float   *embedding;          /* embedding vector                 */
    int      emb_dim;
    char     source[256];        /* file path or URL                 */
} tl_rag_chunk_t;

typedef struct {
    tl_rag_chunk_t *chunks;
    int       n_chunks;
    int       capacity;
    int       emb_dim;
    /* Simple brute-force cosine-sim search (OK for <100k chunks)    */
} tl_rag_index_t;

/* ═══════════════════════════════════════════════════════════════════
   Long-term Memory Store
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    char    *key;                /* hash / short descriptor           */
    char    *value;              /* content                          */
    float   *embedding;
    int      emb_dim;
    int64_t  timestamp;
    int      access_count;
    float    importance;         /* 0..1 heuristic score              */
} tl_mem_entry_t;

typedef struct {
    tl_mem_entry_t *entries;
    int       n_entries;
    int       capacity;
    int       emb_dim;
    char      filepath[512];     /* persistent storage on disk       */
} tl_memory_store_t;

/* ═══════════════════════════════════════════════════════════════════
   HTTP Server (minimal, no external lib)
   ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    int   fd;
    int   port;
    bool  running;
    tl_infer_t *infer;
    tl_agent_t *agent;
} tl_http_server_t;

/* ═══════════════════════════════════════════════════════════════════
   Internal: used across compilation units
   ═══════════════════════════════════════════════════════════════════ */
tl_tensor_t tl_tensor_alloc(int rows, int cols, tl_qtype_t qtype);
void        tl_tensor_free(tl_tensor_t *t);
void        tl_dequantize_q4(const uint8_t *q, const float *scales,
                             float *out, int rows, int cols);
void        tl_matvec(const tl_tensor_t *W, const float *x, float *y,
                      int rows, int cols);
void        tl_rms_norm(const float *x, const float *w, float *y,
                        int dim, float eps);
tl_kv_cache_t *tl_kv_cache_create(int n_layers, int max_tokens, int latent_dim);
void          tl_kv_cache_free(tl_kv_cache_t *c);

/* ═══════════════════════════════════════════════════════════════════
   API: Lifecycle
   ═══════════════════════════════════════════════════════════════════ */

/* Allocate and load a GGUF model. Returns NULL on failure.
   Auto-detects quantization and allocates appropriately. */
tl_model_t     *tl_model_load(const char *path);

/* Free the model and all its tensors. */
void            tl_model_free(tl_model_t *m);

/* Print model info (architecture, layers, experts, memory usage). */
void            tl_model_print_info(const tl_model_t *m);

/* ═══════════════════════════════════════════════════════════════════
   API: Tokenizer
   ═══════════════════════════════════════════════════════════════════ */

tl_tokenizer_t *tl_tokenizer_load(const char *path);
void            tl_tokenizer_free(tl_tokenizer_t *t);

/* Encode UTF-8 text → token sequence. Returns token count.
   If `tokens` is NULL, just returns the count (for sizing). */
int  tl_tokenize(tl_tokenizer_t *t, const char *text, tl_token_t *tokens, int max_tokens);

/* Decode tokens → UTF-8. Returns malloc'd string (caller frees). */
char *tl_detokenize(tl_tokenizer_t *t, const tl_token_t *tokens, int n_tokens);

/* FIM (Fill-in-the-Middle): encode prefix+suffix with FIM tokens. */
int  tl_tokenize_fim(tl_tokenizer_t *t, const char *prefix, const char *suffix,
                     tl_token_t *tokens, int max_tokens);

/* ═══════════════════════════════════════════════════════════════════
   API: Inference
   ═══════════════════════════════════════════════════════════════════ */

tl_infer_t     *tl_infer_create(tl_model_t *model, tl_tokenizer_t *tok, tl_sampler_t sampler);
void            tl_infer_free(tl_infer_t *inf);

/* Run one forward pass: hidden states → logits for the last position.
   Updates KV cache. Returns 0 on success. */
int  tl_forward(tl_infer_t *inf, const tl_token_t *tokens, int n_tokens,
                int batch_size);

/* Sample next token from logits. */
tl_token_t tl_sample(tl_infer_t *inf);

/* Generate tokens autoregressively until EOS or max_new_tokens.
   Calls callback(token, text, n) for each token (can be NULL).
   Returns total tokens generated. */
int  tl_generate(tl_infer_t *inf, const tl_token_t *prompt, int prompt_len,
                 int max_new_tokens,
                 void (*callback)(tl_token_t token, const char *text, int n, void *user),
                 void *user);

/* Reset KV cache for a new conversation turn. */
void tl_kv_cache_clear(tl_infer_t *inf);

/* ═══════════════════════════════════════════════════════════════════
   API: Sampler
   ═══════════════════════════════════════════════════════════════════ */

tl_sampler_t tl_sampler_default(void);
void         tl_sampler_set_seed(tl_sampler_t *s, int seed);

/* ═══════════════════════════════════════════════════════════════════
   API: Tools
   ═══════════════════════════════════════════════════════════════════ */

/* Execute a tool call. `result` is malloc'd (caller frees).
   Returns exit code (0 = success). */
int  tl_tool_execute(const tl_tool_call_t *call, char **result);

/* Get the list of available tools with their JSON schemas. */
int  tl_tools_list(tl_tool_def_t *defs, int max_defs);

/* ═══════════════════════════════════════════════════════════════════
   API: Agent (self-correcting loop)
   ═══════════════════════════════════════════════════════════════════ */

tl_agent_t     *tl_agent_create(tl_infer_t *infer);
void            tl_agent_free(tl_agent_t *agent);

/* Run the agent loop on a task.
   Steps: plan → think → act(tools) → observe → repeat.
   Returns final output (malloc'd string). */
char           *tl_agent_run(tl_agent_t *agent, const char *task);

/* ═══════════════════════════════════════════════════════════════════
   API: RAG
   ═══════════════════════════════════════════════════════════════════ */

tl_rag_index_t *tl_rag_create(int emb_dim);
void            tl_rag_free(tl_rag_index_t *rag);

/* Add a document chunk (text is copied). Embedding must be provided
   or set to NULL (will be computed via model forward pass). */
int  tl_rag_add(tl_rag_index_t *rag, const char *text, const float *emb, const char *source);

/* Index a directory of source files. Calls tree-sitter externally. */
int  tl_rag_index_dir(tl_rag_index_t *rag, const char *dir_path);

/* Search top-k chunks by cosine similarity. Returns count found. */
int  tl_rag_search(tl_rag_index_t *rag, const float *query_emb, int top_k,
                   tl_rag_chunk_t **results);

/* ═══════════════════════════════════════════════════════════════════
   API: Long-term Memory
   ═══════════════════════════════════════════════════════════════════ */

tl_memory_store_t *tl_memory_create(int emb_dim, const char *filepath);
void               tl_memory_free(tl_memory_store_t *mem);

/* Store a key-value pair with optional embedding. */
int  tl_memory_put(tl_memory_store_t *mem, const char *key, const char *value,
                    const float *emb, float importance);

/* Retrieve top-k entries by embedding similarity. */
int  tl_memory_search(tl_memory_store_t *mem, const float *query_emb, int top_k,
                      tl_mem_entry_t **results);

/* Persist to disk (JSON lines). */
int  tl_memory_save(tl_memory_store_t *mem);
int  tl_memory_load(tl_memory_store_t *mem);

/* ═══════════════════════════════════════════════════════════════════
   API: HTTP Server
   ═══════════════════════════════════════════════════════════════════ */

tl_http_server_t *tl_http_create(int port, tl_infer_t *infer, tl_agent_t *agent);
void              tl_http_free(tl_http_server_t *srv);
int               tl_http_listen(tl_http_server_t *srv);  /* blocking */

/* ═══════════════════════════════════════════════════════════════════
   API: Utility
   ═══════════════════════════════════════════════════════════════════ */

/* Print formatted model info / memory usage to stderr. */
void tl_log(const char *fmt, ...);

/* Allocate with OOM check (exits on failure). */
void *tl_alloc(size_t size);
void *tl_calloc(size_t n, size_t size);
void  tl_free(void *p);

/* Read entire file into malloc'd buffer. Returns NULL on error. */
char *tl_read_file(const char *path, size_t *out_len);

/* Simple SHA256 hash (for caching, dedup). */
void tl_sha256(const uint8_t *data, size_t len, uint8_t out[32]);

/* Get high-resolution monotonic time in seconds. */
double tl_time_now(void);

/* Memory usage of the process in bytes (RSS). */
int64_t tl_memory_usage(void);

#endif /* TINYLLM_H */
