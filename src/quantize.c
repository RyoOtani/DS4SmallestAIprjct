/*
 * quantize.c — Quantization & dequantization ops (Q4_0, Q4_1, Q8_0).
 *   ds4: SIMD-accelerated where available (AVX2, NEON, Accelerate).
 *   Falls back to scalar C if no SIMD.
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ═══════════════════════════════════════════════════════════════════
   Q4_0: 4-bit quantization with 32-element blocks, shared scale
   Layout: [half_byte_0, half_byte_1, ...] for each block
           followed by a float16 scale (stored as uint16 at end)
   ═══════════════════════════════════════════════════════════════════ */

/* Dequantize Q4_0 → float32 */
void tl_dequantize_q4(const uint8_t *q, const float *scales,
                      float *out, int rows, int cols) {
    size_t n = (size_t)rows * cols;
    int n_blocks = (int)((n + TL_BLOCK_SIZE - 1) / TL_BLOCK_SIZE);

    for (int b = 0; b < n_blocks; b++) {
        float scale = scales[b];
        int start  = b * TL_BLOCK_SIZE;
        int end    = (start + TL_BLOCK_SIZE < (int)n) ? start + TL_BLOCK_SIZE : (int)n;

#if defined(__AVX2__)
        __m256 scale_vec = _mm256_set1_ps(scale);
        __m256 bias_vec  = _mm256_set1_ps(-8.0f);
        int j;
        for (j = start; j + 16 <= end; j += 16) {
            /* Load 8 bytes containing 16 nibbles */
            __m128i packed = _mm_loadl_epi64(
                (const __m128i*)(q + b * (TL_BLOCK_SIZE/2) + (j - start)/2));
            /* Unpack nibbles: low bytes contain low nibbles, high contain high */
            __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
            __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
            /* Interleave lo/hi into 16 int16 values */
            __m256i indices = _mm256_cvtepi8_epi16(
                _mm_unpacklo_epi8(lo, hi)); /* first 8 lo, first 8 hi */
            /* Convert to float and scale */
            __m256 val = _mm256_cvtepi32_ps(
                _mm256_cvtepi16_epi32(_mm256_castsi256_si128(indices)));
            __m256 res = _mm256_fmadd_ps(val, scale_vec, _mm256_mul_ps(val, bias_vec));
            /* Actually: out = (val - 8.0f) * scale = val*scale - 8.0f*scale */
            res = _mm256_add_ps(_mm256_mul_ps(val, scale_vec),
                                _mm256_mul_ps(bias_vec, scale_vec));
            _mm256_storeu_ps(out + j, res);

            /* Process next 8 nibbles */
            __m256i indices2 = _mm256_cvtepi8_epi16(
                _mm_unpackhi_epi8(lo, hi));
            __m256 val2 = _mm256_cvtepi32_ps(
                _mm256_cvtepi16_epi32(_mm256_castsi256_si128(indices2)));
            __m256 res2 = _mm256_add_ps(_mm256_mul_ps(val2, scale_vec),
                                        _mm256_mul_ps(bias_vec, scale_vec));
            _mm256_storeu_ps(out + j + 8, res2);
        }
        /* Scalar remainder */
        for (; j < end; j++) {
            int k = j - start;
            uint8_t v = q[b * (TL_BLOCK_SIZE/2) + k/2];
            out[j] = (k & 1) ? ((v >> 4) - 8.0f) * scale : ((v & 0xF) - 8.0f) * scale;
        }
#elif defined(__ARM_NEON)
        float32x4_t scale_vec = vdupq_n_f32(scale);
        float32x4_t neg8_vec  = vdupq_n_f32(-8.0f);
        int j;
        for (j = start; j + 8 <= end; j += 8) {
            uint32_t packed;
            memcpy(&packed, q + b * (TL_BLOCK_SIZE/2) + (j - start)/2, 4);
            /* Deinterleave nibbles into 8 floats */
            /* Use table lookup for 4-bit → 8-bit expansion */
            uint8x8_t packed8 = vcreate_u8(packed);
            uint8x8_t lo = vand_u8(packed8, vdup_n_u8(0x0F));
            uint8x8_t hi = vshr_n_u8(packed8, 4);
            uint8x8x2_t inter = vzip_u8(lo, hi);
            int16x8_t widened = vreinterpretq_s16_u16(vmovl_u8(inter.val[0]));
            float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(widened)));
            float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(widened)));
            f1 = vaddq_f32(vmulq_f32(f1, scale_vec), vmulq_f32(neg8_vec, scale_vec));
            f2 = vaddq_f32(vmulq_f32(f2, scale_vec), vmulq_f32(neg8_vec, scale_vec));
            vst1q_f32(out + j, f1);
            vst1q_f32(out + j + 4, f2);
        }
        for (; j < end; j++) {
            int k = j - start;
            uint8_t v = q[b * (TL_BLOCK_SIZE/2) + k/2];
            out[j] = (k & 1) ? ((v >> 4) - 8.0f) * scale : ((v & 0xF) - 8.0f) * scale;
        }
#else
        /* Pure scalar */
        for (int j = start; j < end; j++) {
            int k = j - start;
            uint8_t v = q[b * (TL_BLOCK_SIZE/2) + k/2];
            float val = (k & 1) ? (float)((v >> 4) & 0xF) : (float)(v & 0xF);
            out[j] = (val - 8.0f) * scale;
        }
#endif
    }
}

/* Quantize float32 → Q4_0 (for LoRA merge, model save, etc.) */
void tl_quantize_q4(const float *src, uint8_t *q, float *scales,
                    int rows, int cols) {
    size_t n = (size_t)rows * cols;
    int n_blocks = (int)((n + TL_BLOCK_SIZE - 1) / TL_BLOCK_SIZE);

    for (int b = 0; b < n_blocks; b++) {
        int start = b * TL_BLOCK_SIZE;
        int end   = (start + TL_BLOCK_SIZE < (int)n) ? start + TL_BLOCK_SIZE : (int)n;
        int blk_sz = end - start;

        /* Find max absolute value for scale */
        float amax = 0.0f;
        for (int j = start; j < end; j++) {
            float v = fabsf(src[j]);
            if (v > amax) amax = v;
        }
        float scale = amax / 7.0f;  /* range [-7, 7] leaves one value for symmetry */
        if (scale < 1e-6f) scale = 1.0f;
        scales[b] = scale;

        float inv_scale = 1.0f / scale;

        memset(q + b * (TL_BLOCK_SIZE/2), 0, (blk_sz + 1) / 2);
        for (int j = start; j < end; j++) {
            int k = j - start;
            float v = src[j] * inv_scale + 8.0f;
            int vi = (int)roundf(v);
            if (vi < 0) vi = 0; if (vi > 15) vi = 15;
            if (k & 1)
                q[b * (TL_BLOCK_SIZE/2) + k/2] |= (uint8_t)(vi << 4);
            else
                q[b * (TL_BLOCK_SIZE/2) + k/2] = (uint8_t)vi;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
   Mixed-precision quantization: key layers at 6-bit, rest at 4-bit
   ═══════════════════════════════════════════════════════════════════ */

/* Heuristic: layers near input/output get higher precision */
static int layer_importance_bits(int layer_idx, int n_layers) {
    if (layer_idx == 0 || layer_idx == n_layers - 1) return 6;   /* first/last */
    if (layer_idx <= 2 || layer_idx >= n_layers - 3) return 5;    /* nearby */
    return 4;  /* middle layers */
}

tl_qtype_t tl_choose_qtype(int layer_idx, int n_layers) {
    int bits = layer_importance_bits(layer_idx, n_layers);
    switch (bits) {
    case 6: return TL_QTYPE_Q6_K;
    case 5: return TL_QTYPE_Q8_0;  /* 5-bit approximated as Q8_0 */
    default: return TL_QTYPE_Q4_0;
    }
}

/* ═══════════════════════════════════════════════════════════════════
   Matrix-vector multiply with quantized weights (Q4_0)
   y = W @ x   where W is [rows x cols] in Q4_0, x is [cols]
   ═══════════════════════════════════════════════════════════════════ */

void tl_matvec_q4(const tl_tensor_t *W, const float *x, float *y,
                  int rows, int cols) {
    /* Dequantize on-the-fly per block for cache efficiency */
    int n_blocks = (cols + TL_BLOCK_SIZE - 1) / TL_BLOCK_SIZE;

    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const uint8_t *row_q = W->qdata + r * cols / 2;
        const float *row_s  = W->scales + r * n_blocks;

#if defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (int b = 0; b < n_blocks; b++) {
            float scale = row_s[b];
            __m256 scale_v = _mm256_set1_ps(scale);
            int block_end = (b == n_blocks - 1) ? cols : (b + 1) * TL_BLOCK_SIZE;

            /* Process 16 elements at a time (8 bytes of nibbles = 16 values) */
            for (; c + 16 <= block_end; c += 16) {
                int byte_off = c / 2;
                __m128i packed = _mm_loadl_epi64((const __m128i*)(row_q + byte_off));
                __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
                __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
                __m128i interleaved = _mm_unpacklo_epi8(lo, hi);
                __m256i i16 = _mm256_cvtepi8_epi16(interleaved);
                __m256i i32_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(i16));
                __m256i i32_hi = _mm256_cvtepi16_epi32(
                    _mm256_extracti128_si256(i16, 1));
                __m256 f_lo = _mm256_cvtepi32_ps(i32_lo);
                __m256 f_hi = _mm256_cvtepi32_ps(i32_hi);
                __m256 x_lo = _mm256_loadu_ps(x + c);
                __m256 x_hi = _mm256_loadu_ps(x + c + 8);
                /* Adjust by -8 and multiply by scale */
                f_lo = _mm256_sub_ps(f_lo, _mm256_set1_ps(8.0f));
                f_hi = _mm256_sub_ps(f_hi, _mm256_set1_ps(8.0f));
                acc = _mm256_fmadd_ps(_mm256_mul_ps(f_lo, scale_v), x_lo, acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(f_hi, scale_v), x_hi, acc);
            }
            /* Scalar remainder for this block */
            for (; c < block_end; c++) {
                int k = c % TL_BLOCK_SIZE;
                uint8_t v = row_q[c / 2];
                float wv = (k & 1) ? ((v >> 4) - 8.0f) : ((v & 0xF) - 8.0f);
                sum += wv * scale * x[c];
            }
        }
        float hsum[8];
        _mm256_storeu_ps(hsum, acc);
        sum += hsum[0]+hsum[1]+hsum[2]+hsum[3]+hsum[4]+hsum[5]+hsum[6]+hsum[7];
#elif defined(__ARM_NEON)
        float32x4_t acc = vdupq_n_f32(0);
        int c = 0;
        for (int b = 0; b < n_blocks; b++) {
            float scale = row_s[b];
            float32x4_t scale_v = vdupq_n_f32(scale);
            int block_end = (b == n_blocks - 1) ? cols : (b + 1) * TL_BLOCK_SIZE;
            for (; c + 8 <= block_end; c += 8) {
                uint32_t packed;
                memcpy(&packed, row_q + c/2, 4);
                uint8x8_t nib = vcreate_u8(packed);
                uint8x8_t lo = vand_u8(nib, vdup_n_u8(0x0F));
                uint8x8_t hi = vshr_n_u8(nib, 4);
                uint8x8x2_t inter = vzip_u8(lo, hi);
                int16x8_t i16 = vreinterpretq_s16_u16(vmovl_u8(inter.val[0]));
                float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16)));
                float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16)));
                f1 = vsubq_f32(f1, vdupq_n_f32(8.0f));
                f2 = vsubq_f32(f2, vdupq_n_f32(8.0f));
                float32x4_t x1 = vld1q_f32(x + c);
                float32x4_t x2 = vld1q_f32(x + c + 4);
                acc = vmlaq_f32(acc, vmulq_f32(f1, scale_v), x1);
                acc = vmlaq_f32(acc, vmulq_f32(f2, scale_v), x2);
            }
            for (; c < block_end; c++) {
                int k = c % TL_BLOCK_SIZE;
                uint8_t v = row_q[c / 2];
                float wv = (k & 1) ? ((v >> 4) - 8.0f) : ((v & 0xF) - 8.0f);
                sum += wv * scale * x[c];
            }
        }
        float tmp[4]; vst1q_f32(tmp, acc);
        sum += tmp[0]+tmp[1]+tmp[2]+tmp[3];
#else
        /* Pure scalar */
        for (int b = 0; b < n_blocks; b++) {
            float scale = row_s[b];
            int start = b * TL_BLOCK_SIZE;
            int end   = (start + TL_BLOCK_SIZE < cols) ? start + TL_BLOCK_SIZE : cols;
            for (int c = start; c < end; c++) {
                int k = c - start;
                uint8_t v = row_q[c / 2];
                float wv = (k & 1) ? ((v >> 4) - 8.0f) : ((v & 0xF) - 8.0f);
                sum += wv * scale * x[c];
            }
        }
#endif
        y[r] = sum;
    }
}

/* ── Generic matvec (float weights) ──────────────────────────────── */
void tl_matvec_f32(const float *W, const float *x, float *y,
                   int rows, int cols) {
#ifdef TL_HAS_ACCELERATE
    /* Use BLAS: y = alpha * W * x + beta * y */
    float alpha = 1.0f, beta = 0.0f;
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                rows, cols, alpha, W, cols, x, 1, beta, y, 1);
#else
    memset(y, 0, rows * sizeof(float));
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
#if defined(__AVX2__)
        __m256 s = _mm256_setzero_ps();
        int c;
        for (c = 0; c + 7 < cols; c += 8) {
            __m256 wv = _mm256_loadu_ps(W + r * cols + c);
            __m256 xv = _mm256_loadu_ps(x + c);
            s = _mm256_fmadd_ps(wv, xv, s);
        }
        sum = s[0]+s[1]+s[2]+s[3]+s[4]+s[5]+s[6]+s[7];
        for (; c < cols; c++) sum += W[r*cols+c] * x[c];
#else
        for (int c = 0; c < cols; c++) sum += W[r*cols+c] * x[c];
#endif
        y[r] = sum;
    }
#endif
}

/* ═══════════════════════════════════════════════════════════════════
   Q6_K: 6-bit quantization with 256-element super-blocks.
   Each super-block: 256 values, 6-bit each → 192 bytes of data,
                    16 × float16 scales (one per 16 elements),
                    1 × float16 max_scale (global), 1 × uint8_t max_idx.
   ═══════════════════════════════════════════════════════════════════ */

#define Q6_K_BLOCK_SIZE 256
#define Q6_K_SCALE_BLOCK 16  /* one scale per 16 elements */

/* Dequantize Q6_K → float32 */
void tl_dequantize_q6(const uint8_t *qdata, const float *scales,
                      float *out, int rows, int cols) {
    size_t n = (size_t)rows * cols;
    /* Q6_K: data layout = 6-bit values packed tightly, then scale data */
    /* For simplicity, treat as Q4-like with adjusted precision.
       Real Q6_K has complex packing; this is a simplified fallback. */
    int n_super = (int)((n + Q6_K_BLOCK_SIZE - 1) / Q6_K_BLOCK_SIZE);

    for (int s = 0; s < n_super; s++) {
        int start = s * Q6_K_BLOCK_SIZE;
        int end   = (start + Q6_K_BLOCK_SIZE < (int)n) ? start + Q6_K_BLOCK_SIZE : (int)n;

        /* Each super-block has 16 sub-blocks of 16 elements each */
        for (int sb = 0; sb < Q6_K_BLOCK_SIZE / Q6_K_SCALE_BLOCK && start + sb * Q6_K_SCALE_BLOCK < end; sb++) {
            float scale = scales[s * (Q6_K_BLOCK_SIZE / Q6_K_SCALE_BLOCK) + sb];
            int sb_start = start + sb * Q6_K_SCALE_BLOCK;
            int sb_end   = (sb_start + Q6_K_SCALE_BLOCK < end) ? sb_start + Q6_K_SCALE_BLOCK : end;

            /* 6-bit values packed in 3 bytes per 4 values (3*8/4 = 6 bits each) */
            for (int j = sb_start; j < sb_end; j++) {
                int k = j - start;
                /* Approximate: extract 6 bits from packed storage.
                   Real impl would read from tightly packed 6-bit layout. */
                int byte_off = k * 6 / 8;
                int bit_off  = k * 6 % 8;
                uint32_t word = 0;
                memcpy(&word, qdata + byte_off, 4);
                uint32_t val = (word >> bit_off) & 0x3F; /* 6 bits */
                /* Q6_K range: [-32, 31] centered */
                out[j] = ((float)(int)val - 32.0f + 0.5f) * scale;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
   Fused Q4_0 × vector multiply-add: y += scale * W @ x (in-place)
   Used by speculative decoding verification.
   ═══════════════════════════════════════════════════════════════════ */

void tl_matvec_q4_fused(const tl_tensor_t *W, const float *x,
                        float *y, int rows, int cols, float scale) {
    int n_blocks = (cols + TL_BLOCK_SIZE - 1) / TL_BLOCK_SIZE;

    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const uint8_t *row_q = W->qdata + r * cols / 2;
        const float  *row_s  = W->scales + r * n_blocks;

        for (int b = 0; b < n_blocks; b++) {
            float s = row_s[b];
            int b_start = b * TL_BLOCK_SIZE;
            int b_end   = (b_start + TL_BLOCK_SIZE < cols) ? b_start + TL_BLOCK_SIZE : cols;
            for (int c = b_start; c < b_end; c++) {
                int k = c - b_start;
                uint8_t v = row_q[c / 2];
                float wv = (k & 1) ? ((v >> 4) - 8.0f) : ((v & 0xF) - 8.0f);
                sum += wv * s * x[c];
            }
        }
        y[r] += sum * scale;
    }
}

/* ── Smart dispatch: picks quantized or float path ───────────────── */
void tl_matvec(const tl_tensor_t *W, const float *x, float *y,
               int rows, int cols) {
    if (W->qtype == TL_QTYPE_Q4_0 && W->qdata) {
        tl_matvec_q4(W, x, y, rows, cols);
    } else if (W->qtype == TL_QTYPE_Q6_K && W->qdata) {
        /* For now, use Q4 path with dequant fallback */
        tl_matvec_f32(W->data, x, y, rows, cols);
    } else if (W->data) {
        tl_matvec_f32(W->data, x, y, rows, cols);
    } else {
        memset(y, 0, rows * sizeof(float));
    }
}
