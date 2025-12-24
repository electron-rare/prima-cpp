#ifndef BITSQUEEZE_H
#define BITSQUEEZE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BSQ_INVALID = -1,
    Q8_0 = 0,
    Q4_0 = 1,
    Q2_K = 2,
    TOPK = 3,
    BF16 = 4,
    FP16 = 5,
    FP8  = 6,
    FP4  = 7,
    MXFP8 = 8,
    MXFP4 = 9,
    NVFP4 = 10,
    NF4_DQ = 11,
    NF4 = 12,
    IQ2_XXS = 13,
    IQ2_XS = 14,
    IQ2_S = 15,
    Q2_K_FAST = 16,
    TOPK_IM = 17,
} bsq_method_t;

typedef struct {
    uint64_t num_elements;    /* for 1D formats */
    uint16_t num_tokens;      /* for 2D sparsity */
    uint16_t num_features;    /* for 2D sparsity */
    float    sparse_ratio;    /* only meaningful for TOPK, TOPK_IM */
} bsq_shape_t;

typedef struct bitsqueeze_buffer {
    bsq_method_t method;
    bsq_shape_t  shape;
    void        *payload;
} bitsqueeze_buffer_t;

int bsq_compress_1d(const float *src,
                    uint64_t num_elements,
                    bsq_method_t method,
                    bitsqueeze_buffer_t **out);

int bsq_compress_2d(const float *src,
                    uint16_t num_tokens,
                    uint16_t num_features,
                    float sparse_ratio,
                    bsq_method_t method,
                    bitsqueeze_buffer_t **out,
                    const float *im);

int bsq_decompress(const bitsqueeze_buffer_t *buf,
                   float *dst,
                   uint64_t dst_num_elements);

int bsq_apply(const bitsqueeze_buffer_t *buf,
                   float *dst,
                   uint64_t dst_num_elements);

int64_t bsq_get_packed_size(const bitsqueeze_buffer_t *buf);

bitsqueeze_buffer_t *load_bsq_from_buffer(const void *buffer, int64_t buffer_size);

void bsq_free(bitsqueeze_buffer_t *buf);

#ifdef __cplusplus
}
#endif

#endif
