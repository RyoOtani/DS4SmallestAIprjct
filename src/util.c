/*
 * util.c — Memory management, file I/O, hashing, logging, timing.
 *   ds4: no external deps. Pure C + POSIX.
 */
#include "tinyllm.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

/* ── Logging ─────────────────────────────────────────────────────── */
void tl_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

/* ── Memory ──────────────────────────────────────────────────────── */
void *tl_alloc(size_t size) {
    void *p = malloc(size);
    if (!p) { tl_log("OOM: failed to allocate %zu bytes", size); exit(1); }
    memset(p, 0, size);
    return p;
}

void *tl_calloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) { tl_log("OOM: failed to calloc %zu x %zu", n, size); exit(1); }
    return p;
}

void tl_free(void *p) {
    if (p) free(p);
}

/* ── File I/O ────────────────────────────────────────────────────── */
char *tl_read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }

    char *buf = tl_alloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);

    if (rd != (size_t)sz) { tl_free(buf); return NULL; }
    buf[sz] = '\0';
    if (out_len) *out_len = (size_t)sz;
    return buf;
}

/* ── SHA-256 (public domain implementation, compact) ─────────────── */
#ifdef TL_HAS_ACCELERATE
  /* Use CommonCrypto on macOS */
  #include <CommonCrypto/CommonDigest.h>
  void tl_sha256(const uint8_t *data, size_t len, uint8_t out[32]) {
      CC_SHA256(data, (CC_LONG)len, out);
  }
#else
  /* Compact standalone SHA-256 */
  static const uint32_t sha256_k[64] = {
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
  };
  #define ROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
  #define CH(x,y,z)  (((x)&(y))^(~(x)&(z)))
  #define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
  #define BSIG0(x) (ROTR(x,2)^ROTR(x,13)^ROTR(x,22))
  #define BSIG1(x) (ROTR(x,6)^ROTR(x,11)^ROTR(x,25))
  #define SSIG0(x) (ROTR(x,7)^ROTR(x,18)^((x)>>3))
  #define SSIG1(x) (ROTR(x,17)^ROTR(x,19)^((x)>>10))

  void tl_sha256(const uint8_t *data, size_t len, uint8_t out[32]) {
      uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                       0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
      uint8_t block[64]; int blen = 0;
      uint64_t bits = len * 8;

      for (size_t i = 0; i < len; i++) {
          block[blen++] = data[i];
          if (blen == 64) {
              uint32_t w[64];
              for (int j = 0; j < 16; j++)
                  w[j] = ((uint32_t)block[j*4]<<24)|((uint32_t)block[j*4+1]<<16)|
                         ((uint32_t)block[j*4+2]<<8)|(uint32_t)block[j*4+3];
              for (int j = 16; j < 64; j++)
                  w[j] = SSIG1(w[j-2]) + w[j-7] + SSIG0(w[j-15]) + w[j-16];
              uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
              for (int j = 0; j < 64; j++) {
                  uint32_t t1 = hh + BSIG1(e) + CH(e,f,g) + sha256_k[j] + w[j];
                  uint32_t t2 = BSIG0(a) + MAJ(a,b,c);
                  hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
              }
              h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
              blen=0;
          }
      }
      block[blen++]=0x80;
      if(blen>56){while(blen<64)block[blen++]=0;goto SHA256_COMPRESS;}
      while(blen<56)block[blen++]=0;
      SHA256_COMPRESS:;
      for(int i=0;i<8;i++){block[56+i]=(bits>>(56-i*8))&0xff;}
      {uint32_t w[64];for(int j=0;j<16;j++)w[j]=((uint32_t)block[j*4]<<24)|((uint32_t)block[j*4+1]<<16)|((uint32_t)block[j*4+2]<<8)|(uint32_t)block[j*4+3];for(int j=16;j<64;j++)w[j]=SSIG1(w[j-2])+w[j-7]+SSIG0(w[j-15])+w[j-16];uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];for(int j=0;j<64;j++){uint32_t t1=hh+BSIG1(e)+CH(e,f,g)+sha256_k[j]+w[j];uint32_t t2=BSIG0(a)+MAJ(a,b,c);hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;}
      for(int i=0;i<8;i++){out[i*4]=(h[i]>>24)&0xff;out[i*4+1]=(h[i]>>16)&0xff;out[i*4+2]=(h[i]>>8)&0xff;out[i*4+3]=h[i]&0xff;}
  }
#endif

/* ── Timing ──────────────────────────────────────────────────────── */
double tl_time_now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

/* ── Memory usage (RSS in bytes) ─────────────────────────────────── */
int64_t tl_memory_usage(void) {
#if defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return (int64_t)info.resident_size;
    }
    return -1;
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        return (int64_t)ru.ru_maxrss * 1024;  /* Linux: ru_maxrss in KB */
    return -1;
#endif
}

/* ── Tensor allocation helpers ───────────────────────────────────── */
tl_tensor_t tl_tensor_alloc(int rows, int cols, tl_qtype_t qtype) {
    tl_tensor_t t = {0};
    t.rows = rows; t.cols = cols; t.qtype = qtype;
    size_t elems = (size_t)rows * cols;

    switch (qtype) {
    case TL_QTYPE_F32:
        t.data = tl_alloc(elems * sizeof(float));
        t.byte_size = elems * sizeof(float);
        break;
    case TL_QTYPE_F16:
        t.data = tl_alloc(elems * sizeof(float)); /* store as float for now */
        t.byte_size = elems * sizeof(float);
        break;
    case TL_QTYPE_Q4_0: {
        int n_blocks = (elems + TL_BLOCK_SIZE - 1) / TL_BLOCK_SIZE;
        t.qdata = tl_alloc(n_blocks * TL_BLOCK_SIZE / 2);     /* 4-bit data */
        t.scales = tl_alloc(n_blocks * sizeof(float));
        t.data   = tl_alloc(elems * sizeof(float));            /* dequant workspace */
        t.byte_size = n_blocks * (TL_BLOCK_SIZE / 2 + sizeof(float));
        break;
    }
    default: break;
    }
    return t;
}

void tl_tensor_free(tl_tensor_t *t) {
    tl_free(t->data);
    tl_free(t->qdata);
    tl_free(t->scales);
    memset(t, 0, sizeof(*t));
}
