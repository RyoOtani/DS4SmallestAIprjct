/*
 * tinyllm — ds4 philosophy: single binary, minimal code, zero external deps,
 *           UNIX small-tools composition.
 *
 * Configuration: compile-time constants. No runtime config file needed.
 * Auto-adapts on first launch by scanning the environment.
 */
#ifndef TINYLLM_CONFIG_H
#define TINYLLM_CONFIG_H

#include <stdint.h>
#include <stddef.h>

/* ── Model capacity ───────────────────────────────────────────────── */
#define TL_MAX_LAYERS        64      /* max transformer layers       */
#define TL_MAX_EXPERTS       256     /* max MoE experts per layer    */
#define TL_MAX_ACTIVE_EXPERTS 2      /* top-k experts per token      */
#define TL_MAX_SEQ_LEN       8192    /* max context length           */
#define TL_MAX_BATCH_SIZE    8       /* max batch for parallel infer */

/* ── Memory budget (8 GB target) ──────────────────────────────────── */
#define TL_MEM_BUDGET_BYTES  (8ULL * 1024 * 1024 * 1024)

/* ── KV cache (MLA compressed) ────────────────────────────────────── */
#define TL_KV_LATENT_DIM     512     /* MLA latent (compressed) dim  */
#define TL_KV_HEAD_DIM       128     /* per-head dimension (full)    */
#define TL_MAX_KV_HEADS      32
#define TL_MAX_CACHE_TOKENS  4096    /* rolling window for KV cache  */

/* ── Quantization ─────────────────────────────────────────────────── */
#define TL_QTYPE_DEFAULT     TL_QTYPE_Q4_0
typedef enum {
    TL_QTYPE_F32     = 0,
    TL_QTYPE_F16     = 1,
    TL_QTYPE_Q8_0    = 8,
    TL_QTYPE_Q4_0    = 4,
    TL_QTYPE_Q4_1    = 5,
    TL_QTYPE_Q6_K    = 6,
} tl_qtype_t;

#define TL_BLOCK_SIZE        32      /* quant block size (Q4_0 etc)  */

/* ── Sampling defaults ────────────────────────────────────────────── */
#define TL_DEFAULT_TEMP       0.7f
#define TL_DEFAULT_TOP_P      0.9f
#define TL_DEFAULT_TOP_K      50
#define TL_DEFAULT_REP_PENALTY 1.1f

/* ── Agent / self-correction ──────────────────────────────────────── */
#define TL_MAX_TOOL_CALLS     16      /* max tool calls per turn      */
#define TL_MAX_SELF_CORRECT   5       /* max retry iterations         */
#define TL_MAX_SCRATCH_TOKENS 2048    /* scratchpad token budget      */

/* ── HTTP server ──────────────────────────────────────────────────── */
#define TL_HTTP_PORT          8420
#define TL_HTTP_MAX_REQ_SIZE  (16 * 1024 * 1024)

/* ── RAG / memory ─────────────────────────────────────────────────── */
#define TL_RAG_CHUNK_SIZE     512     /* tokens per chunk             */
#define TL_RAG_TOP_K          8       /* top chunks to retrieve       */
#define TL_MEM_MAX_ENTRIES    100000  /* long-term memory cap         */

/* ── Platform detection (auto-adapt) ──────────────────────────────── */
#if defined(__APPLE__)
  #define TL_PLATFORM "darwin"
  #define ACCELERATE_NEW_LAPACK
  #define ACCELERATE_LAPACK_ILP64
  #include <Accelerate/Accelerate.h>   /* Apple Accelerate for BLAS   */
  #define TL_HAS_ACCELERATE 1
#elif defined(__linux__)
  #define TL_PLATFORM "linux"
  #define TL_HAS_ACCELERATE 0
#endif

#if defined(__x86_64__) || defined(_M_X64)
  #define TL_ARCH "x86_64"
  #ifndef TL_HAS_ACCELERATE
    #include <immintrin.h>            /* AVX2/FMA intrinsics          */
  #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
  #define TL_ARCH "aarch64"
  #ifndef TL_HAS_ACCELERATE
    #include <arm_neon.h>             /* NEON intrinsics              */
  #endif
#else
  #define TL_ARCH "unknown"
#endif

#endif /* TINYLLM_CONFIG_H */
