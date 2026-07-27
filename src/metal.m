/*
 * tinyllm — Metal GPU backend for Apple Silicon (M1/M2/M3/M4).
 *
 * Provides GPU-accelerated matrix multiplication and neural network ops
 * via Apple's Metal Performance Shaders.
 *
 * Compilation:
 *   make METAL_ENABLED=1         # enables Metal backend
 *   make METAL_ENABLED=1 clean   # clean + rebuild with Metal
 *
 * This file is only compiled when METAL_ENABLED=1 is set.
 * See Makefile for details.
 */

#include "tinyllm.h"
#include "config.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(__APPLE__) && defined(TL_HAS_METAL)
#include <TargetConditionals.h>
#if TARGET_OS_OSX

/* ── Metal headers ──────────────────────────────────────────────────────── */
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.h>

/* ── Internal state ─────────────────────────────────────────────────────── */
static id<MTLDevice>       tl_metal_device = NULL;
static id<MTLCommandQueue> tl_metal_queue = NULL;
static id<MTLLibrary>      tl_metal_library = NULL;

/* Pipeline state objects (compiled once, reused) */
static id<MTLComputePipelineState> tl_pso_matmul_fp32 = NULL;
static id<MTLComputePipelineState> tl_pso_matmul_q4   = NULL;
static id<MTLComputePipelineState> tl_pso_rms_norm     = NULL;
static id<MTLComputePipelineState> tl_pso_swiglu       = NULL;
static id<MTLComputePipelineState> tl_pso_rope          = NULL;
static id<MTLComputePipelineState> tl_pso_softmax       = NULL;
static id<MTLComputePipelineState> tl_pso_moe_topk      = NULL;
static id<MTLComputePipelineState> tl_pso_flash_attn    = NULL;
static id<MTLComputePipelineState> tl_pso_gelu          = NULL;
static id<MTLComputePipelineState> tl_pso_silu          = NULL;
static id<MTLComputePipelineState> tl_pso_add_vectors   = NULL;
static id<MTLComputePipelineState> tl_pso_mul_scalar    = NULL;

static int tl_metal_available = 0;
static int tl_metal_initialized = 0;
static char tl_metal_device_name[256] = {0};

/* ── Init / Shutdown ────────────────────────────────────────────────────── */

int tl_metal_init(void) {
    if (tl_metal_initialized) return tl_metal_available;

    /* Check Metal availability */
    tl_metal_device = MTLCreateSystemDefaultDevice();
    if (!tl_metal_device) {
        fprintf(stderr, "[tinyllm] Metal: no GPU device found (using CPU fallback)\n");
        tl_metal_available = 0;
        tl_metal_initialized = 1;
        return 0;
    }

    /* Get device name */
    const char *name = [[tl_metal_device name] UTF8String];
    strncpy(tl_metal_device_name, name, sizeof(tl_metal_device_name) - 1);
    printf("[tinyllm] Metal GPU: %s\n", tl_metal_device_name);

    /* Create command queue */
    tl_metal_queue = [tl_metal_device newCommandQueue];
    if (!tl_metal_queue) {
        fprintf(stderr, "[tinyllm] Metal: failed to create command queue\n");
        tl_metal_device = NULL;
        tl_metal_available = 0;
        tl_metal_initialized = 1;
        return 0;
    }

    /* Load Metal library from default (embedded in binary) or from file */
    NSError *error = nil;

    /* Option 1: Load from embedded metallib (linked via -sectcreate) */
    /* Option 2: Load from file alongside binary */
    NSString *libPath = [[NSBundle mainBundle] pathForResource:@"tinyllm"
                                                        ofType:@"metallib"];
    if (libPath) {
        tl_metal_library = [tl_metal_device newLibraryWithFile:libPath error:&error];
    }

    /* Option 3: Load from current directory */
    if (!tl_metal_library) {
        NSString *cwd = [[NSFileManager defaultManager] currentDirectoryPath];
        NSString *fallback = [cwd stringByAppendingPathComponent:@"tinyllm.metallib"];
        tl_metal_library = [tl_metal_device newLibraryWithFile:fallback error:&error];
    }

    /* Option 4: Compile from source at runtime (slower init, always works) */
    if (!tl_metal_library) {
        NSString *srcPath = [[[NSFileManager defaultManager] currentDirectoryPath]
                             stringByAppendingPathComponent:@"src/tinyllm_ops.metal"];
        NSString *source = [NSString stringWithContentsOfFile:srcPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (source) {
            MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
            [opts setFastMathEnabled:YES];
            tl_metal_library = [tl_metal_device newLibraryWithSource:source
                                                             options:opts
                                                               error:&error];
        }
    }

    if (!tl_metal_library) {
        fprintf(stderr, "[tinyllm] Metal: failed to load/compile shader library: %s\n",
                [[error localizedDescription] UTF8String]);
        tl_metal_device = NULL;
        tl_metal_queue = NULL;
        tl_metal_available = 0;
        tl_metal_initialized = 1;
        return 0;
    }

    /* Pre-compile all pipeline states */
    #define LOAD_PSO(name) do { \
        id<MTLFunction> fn = [tl_metal_library newFunctionWithName:@#name]; \
        if (fn) { \
            tl_pso_##name = [tl_metal_device newComputePipelineStateWithFunction:fn error:&error]; \
            if (!tl_pso_##name) { \
                fprintf(stderr, "[tinyllm] Metal: failed to create PSO for '%s'\n", #name); \
            } \
        } \
    } while(0)

    LOAD_PSO(matmul_fp32);
    LOAD_PSO(matmul_q4);
    LOAD_PSO(rms_norm);
    LOAD_PSO(swiglu);
    LOAD_PSO(rope);
    LOAD_PSO(softmax);
    LOAD_PSO(moe_topk);
    LOAD_PSO(flash_attention_fwd);
    LOAD_PSO(gelu);
    LOAD_PSO(silu);
    LOAD_PSO(add_vectors);
    LOAD_PSO(mul_scalar);

    tl_metal_available = 1;
    tl_metal_initialized = 1;

    printf("[tinyllm] Metal: %d compute pipelines loaded\n",
           (tl_pso_matmul_fp32 != NULL) + (tl_pso_matmul_q4 != NULL) +
           (tl_pso_rms_norm != NULL) + (tl_pso_swiglu != NULL) +
           (tl_pso_rope != NULL) + (tl_pso_softmax != NULL) +
           (tl_pso_moe_topk != NULL) + (tl_pso_flash_attn != NULL) +
           (tl_pso_gelu != NULL) + (tl_pso_silu != NULL));

    return 1;
}

void tl_metal_shutdown(void) {
    /* Release all Metal objects (ARC handles this in ObjC, but explicit for clarity) */
    tl_pso_matmul_fp32 = NULL;
    tl_pso_matmul_q4   = NULL;
    tl_pso_rms_norm    = NULL;
    tl_pso_swiglu      = NULL;
    tl_pso_rope        = NULL;
    tl_pso_softmax     = NULL;
    tl_pso_moe_topk    = NULL;
    tl_pso_flash_attn  = NULL;
    tl_pso_gelu        = NULL;
    tl_pso_silu        = NULL;
    tl_pso_add_vectors = NULL;
    tl_pso_mul_scalar  = NULL;
    tl_metal_library   = NULL;
    tl_metal_queue     = NULL;
    tl_metal_device    = NULL;
    tl_metal_available = 0;
    tl_metal_initialized = 0;
}

int tl_metal_is_available(void) {
    if (!tl_metal_initialized) tl_metal_init();
    return tl_metal_available;
}

const char *tl_metal_device_name_str(void) {
    return tl_metal_device_name[0] ? tl_metal_device_name : "none";
}

/* ── Dispatch helper ────────────────────────────────────────────────────── */

static id<MTLBuffer> tl_metal_buffer(const void *data, size_t bytes) {
    if (!data || bytes == 0) return NULL;
    return [tl_metal_device newBufferWithBytes:data length:bytes
                                        options:MTLResourceStorageModeShared];
}

static void tl_metal_dispatch(id<MTLComputePipelineState> pso,
                               id<MTLBuffer> *buffers, int n_buffers,
                               size_t grid_x, size_t grid_y, size_t grid_z) {
    if (!tl_metal_available || !pso) return;

    id<MTLCommandBuffer> cmdBuf = [tl_metal_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    [enc setComputePipelineState:pso];
    for (int i = 0; i < n_buffers; i++) {
        if (buffers[i]) [enc setBuffer:buffers[i] offset:0 atIndex:i];
    }

    MTLSize gridSize = MTLSizeMake(
        MAX(1, grid_x), MAX(1, grid_y), MAX(1, grid_z)
    );
    MTLSize threadGroupSize = MTLSizeMake(
        MIN(256, (NSUInteger)grid_x), 1, 1
    );

    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

/* ── Public API ─────────────────────────────────────────────────────────── */

int tl_metal_matmul_fp32(const float *A, const float *B, float *C,
                          int M, int N, int K) {
    if (!tl_metal_available || !tl_pso_matmul_fp32) return 0;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    id<MTLBuffer> bufA = tl_metal_buffer(A, size_A);
    id<MTLBuffer> bufB = tl_metal_buffer(B, size_B);
    id<MTLBuffer> bufC = [tl_metal_device newBufferWithLength:size_C
                                                       options:MTLResourceStorageModeShared];

    id<MTLBuffer> bufM = tl_metal_buffer(&M, sizeof(int));
    id<MTLBuffer> bufN = tl_metal_buffer(&N, sizeof(int));
    id<MTLBuffer> bufK = tl_metal_buffer(&K, sizeof(int));

    id<MTLBuffer> buffers[] = {bufA, bufB, bufC, bufM, bufN, bufK};
    tl_metal_dispatch(tl_pso_matmul_fp32, buffers, 6, N, M, 1);

    memcpy(C, [bufC contents], size_C);
    return 1;
}

int tl_metal_matmul_q4(const uint8_t *A_q4, const float *B, float *C,
                        int M, int N, int K) {
    if (!tl_metal_available || !tl_pso_matmul_q4) return 0;

    /* Q4_0: 2B scale + 16B quants per 32-weight block */
    int blocks_per_row = K / 32;
    int bytes_per_block = 2 + 16;
    size_t size_A = M * blocks_per_row * bytes_per_block;
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    id<MTLBuffer> bufA = tl_metal_buffer(A_q4, size_A);
    id<MTLBuffer> bufB = tl_metal_buffer(B, size_B);
    id<MTLBuffer> bufC = [tl_metal_device newBufferWithLength:size_C
                                                       options:MTLResourceStorageModeShared];

    id<MTLBuffer> params[] = {
        bufA, bufB, bufC,
        tl_metal_buffer(&M, sizeof(int)),
        tl_metal_buffer(&N, sizeof(int)),
        tl_metal_buffer(&K, sizeof(int)),
    };
    tl_metal_dispatch(tl_pso_matmul_q4, params, 6, N, M, 1);

    memcpy(C, [bufC contents], size_C);
    return 1;
}

int tl_metal_rms_norm(const float *x, const float *weight, float *y,
                       int dim, float eps) {
    if (!tl_metal_available || !tl_pso_rms_norm) return 0;
    size_t size = dim * sizeof(float);

    id<MTLBuffer> bufs[] = {
        tl_metal_buffer(x, size),
        tl_metal_buffer(weight, size),
        tl_metal_buffer(y, size),
        tl_metal_buffer(&dim, sizeof(int)),
        tl_metal_buffer(&eps, sizeof(float)),
    };
    tl_metal_dispatch(tl_pso_rms_norm, bufs, 5, dim, 1, 1);
    return 1;
}

int tl_metal_swiglu(const float *x, const float *gate, float *y, int dim) {
    if (!tl_metal_available || !tl_pso_swiglu) return 0;
    size_t size = dim * sizeof(float);

    id<MTLBuffer> bufs[] = {
        tl_metal_buffer(x, size),
        tl_metal_buffer(gate, size),
        tl_metal_buffer(y, size),
        tl_metal_buffer(&dim, sizeof(int)),
    };
    tl_metal_dispatch(tl_pso_swiglu, bufs, 4, dim, 1, 1);
    return 1;
}

int tl_metal_rope(float *q, const float *cos_table, const float *sin_table,
                   int seq_len, int n_heads, int head_dim) {
    if (!tl_metal_available || !tl_pso_rope) return 0;
    size_t size_q = seq_len * n_heads * head_dim * sizeof(float);
    size_t size_table = seq_len * (head_dim / 2) * sizeof(float);

    id<MTLBuffer> bufQ  = [tl_metal_device newBufferWithBytesNoCopy:q length:size_q
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
    id<MTLBuffer> bufs[] = {
        bufQ,
        tl_metal_buffer(cos_table, size_table),
        tl_metal_buffer(sin_table, size_table),
        tl_metal_buffer(&seq_len, sizeof(int)),
        tl_metal_buffer(&n_heads, sizeof(int)),
        tl_metal_buffer(&head_dim, sizeof(int)),
    };
    tl_metal_dispatch(tl_pso_rope, bufs, 6, head_dim / 2, n_heads, seq_len);
    return 1;
}

int tl_metal_softmax(const float *x, float *y, int dim) {
    if (!tl_metal_available || !tl_pso_softmax) return 0;
    size_t size = dim * sizeof(float);

    id<MTLBuffer> bufs[] = {
        tl_metal_buffer(x, size),
        tl_metal_buffer(y, size),
        tl_metal_buffer(&dim, sizeof(int)),
    };
    tl_metal_dispatch(tl_pso_softmax, bufs, 3, dim, 1, 1);
    return 1;
}

int tl_metal_moe_topk(const float *gate_logits, int *expert_indices,
                       float *expert_weights, int n_experts, int n_active) {
    if (!tl_metal_available || !tl_pso_moe_topk) return 0;

    id<MTLBuffer> bufs[] = {
        tl_metal_buffer(gate_logits, n_experts * sizeof(float)),
        tl_metal_buffer(expert_indices, n_active * sizeof(int)),
        tl_metal_buffer(expert_weights, n_active * sizeof(float)),
        tl_metal_buffer(&n_experts, sizeof(int)),
        tl_metal_buffer(&n_active, sizeof(int)),
    };
    tl_metal_dispatch(tl_pso_moe_topk, bufs, 5, n_experts, 1, 1);
    return 1;
}

int tl_metal_flash_attention(const float *Q, const float *K, const float *V,
                              float *O, int seq_len, int n_heads, int head_dim,
                              float scale) {
    if (!tl_metal_available || !tl_pso_flash_attn) return 0;

    size_t size = seq_len * n_heads * head_dim * sizeof(float);
    id<MTLBuffer> bufQ = tl_metal_buffer(Q, size);
    id<MTLBuffer> bufK = tl_metal_buffer(K, size);
    id<MTLBuffer> bufV = tl_metal_buffer(V, size);
    id<MTLBuffer> bufO = [tl_metal_device newBufferWithLength:size
                                                       options:MTLResourceStorageModeShared];

    id<MTLBuffer> bufs[] = {
        bufQ, bufK, bufV, bufO,
        tl_metal_buffer(&seq_len, sizeof(int)),
        tl_metal_buffer(&n_heads, sizeof(int)),
        tl_metal_buffer(&head_dim, sizeof(int)),
        tl_metal_buffer(&scale, sizeof(float)),
    };
    tl_metal_dispatch(tl_pso_flash_attn, bufs, 8, n_heads, seq_len, 1);

    memcpy(O, [bufO contents], size);
    return 1;
}

int tl_metal_gelu(const float *x, float *y, int n) {
    if (!tl_metal_available || !tl_pso_gelu) return 0;
    size_t size = n * sizeof(float);

    id<MTLBuffer> bufs[] = {
        tl_metal_buffer(x, size), tl_metal_buffer(y, size),
        tl_metal_buffer(&n, sizeof(int)),
    };
    tl_metal_dispatch(tl_pso_gelu, bufs, 3, n, 1, 1);
    return 1;
}

int tl_metal_silu(const float *x, float *y, int n) {
    if (!tl_metal_available || !tl_pso_silu) return 0;
    size_t size = n * sizeof(float);

    id<MTLBuffer> bufs[] = {
        tl_metal_buffer(x, size), tl_metal_buffer(y, size),
        tl_metal_buffer(&n, sizeof(int)),
    };
    tl_metal_dispatch(tl_pso_silu, bufs, 3, n, 1, 1);
    return 1;
}

#else  /* !TARGET_OS_OSX (iOS, etc. — no Metal support) */

int tl_metal_init(void) { return 0; }
void tl_metal_shutdown(void) {}
int tl_metal_is_available(void) { return 0; }
const char *tl_metal_device_name_str(void) { return "not supported (CPU fallback)"; }

/* ── CPU fallback implementations (used when Metal unavailable) ─────── */

int tl_metal_matmul_fp32(const float *A, const float *B, float *C,
                         int M, int N, int K) {
    /* C[M×N] = A[M×K] × B[K×N] — naive triple-loop CPU fallback */
    if (!A || !B || !C) return -1;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    return 0;
}

int tl_metal_matmul_q4(const uint8_t *A_q4, const float *B, float *C,
                        int M, int N, int K) { return 0; }  /* requires dequant */

int tl_metal_rms_norm(const float *x, const float *weight, float *y,
                      int dim, float eps) {
    /* y[i] = x[i] * weight[i] * rsqrt(mean(x²) + eps) */
    if (!x || !y) return -1;
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / (float)dim + eps);
    for (int i = 0; i < dim; i++) {
        float w = weight ? weight[i] : 1.0f;
        y[i] = x[i] * w * inv_rms;
    }
    return 0;
}

int tl_metal_swiglu(const float *x, const float *gate, float *y, int dim) {
    /* y = x * sigmoid(gate) — approximate with silu */
    if (!x || !gate || !y) return -1;
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));  /* sigmoid */
        y[i] = x[i] * sig;
    }
    return 0;
}

int tl_metal_rope(float *q, const float *cos_table, const float *sin_table,
                  int seq_len, int n_heads, int head_dim) { return 0; }
int tl_metal_softmax(const float *x, float *y, int dim) {
    /* y = softmax(x) — numerically stable */
    if (!x || !y) return -1;
    float max_val = x[0];
    for (int i = 1; i < dim; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) { y[i] = expf(x[i] - max_val); sum += y[i]; }
    float inv = 1.0f / (sum + 1e-9f);
    for (int i = 0; i < dim; i++) y[i] *= inv;
    return 0;
}
int tl_metal_moe_topk(const float *gate_logits, int *expert_indices,
                      float *expert_weights, int n_experts, int n_active) { return 0; }
int tl_metal_flash_attention(const float *Q, const float *K, const float *V,
                              float *O, int seq_len, int n_heads, int head_dim,
                              float scale) { return 0; }
int tl_metal_gelu(const float *x, float *y, int n) {
    /* GELU approximation: x * sigmoid(1.702*x) */
    if (!x || !y) return -1;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v * (1.0f / (1.0f + expf(-1.702f * v)));
    }
    return 0;
}
int tl_metal_silu(const float *x, float *y, int n) {
    /* SiLU = x * sigmoid(x) */
    if (!x || !y) return -1;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
    return 0;
}

#endif /* TARGET_OS_OSX */
#endif /* TL_HAS_METAL (real Metal implementation) */

/* ═══════════════════════════════════════════════════════════════
   Fallback stubs — compiled on all platforms.
   Return 0 (not available) so callers gracefully fall back to CPU.
   ═══════════════════════════════════════════════════════════════ */

#else  /* !__APPLE__ — non-Apple platforms: stub */

int tl_metal_init(void) { return 0; }
void tl_metal_shutdown(void) {}
int tl_metal_is_available(void) { return 0; }
const char *tl_metal_device_name_str(void) { return "not apple"; }
int tl_metal_matmul_fp32(const float *A, const float *B, float *C, int M, int N, int K) { return 0; }
int tl_metal_matmul_q4(const uint8_t *A_q4, const float *B, float *C, int M, int N, int K) { return 0; }
int tl_metal_rms_norm(const float *x, const float *weight, float *y, int dim, float eps) { return 0; }
int tl_metal_swiglu(const float *x, const float *gate, float *y, int dim) { return 0; }
int tl_metal_rope(float *q, const float *cos_table, const float *sin_table, int seq_len, int n_heads, int head_dim) { return 0; }
int tl_metal_softmax(const float *x, float *y, int dim) { return 0; }
int tl_metal_moe_topk(const float *gate_logits, int *expert_indices, float *expert_weights, int n_experts, int n_active) { return 0; }
int tl_metal_flash_attention(const float *Q, const float *K, const float *V, float *O, int seq_len, int n_heads, int head_dim, float scale) { return 0; }
int tl_metal_gelu(const float *x, float *y, int n) { return 0; }
int tl_metal_silu(const float *x, float *y, int n) { return 0; }
