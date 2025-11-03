#ifndef K_QUANTIZATION_H
#define K_QUANTIZATION_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fp16/fp16.h"

// The setting is refer to https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
// fp16 implementation is refer to https://github.com/Maratyszcza/FP16/tree/master/include/fp16
#define Q2_K_BLOCK_SIZE 16
#define Q2_K_SUPER_BLOCK_SIZE 16
#define WEIGHT_PER_SUPER_BLOCK (Q2_K_BLOCK_SIZE*Q2_K_SUPER_BLOCK_SIZE)

// Q2_K 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// 2.625 bits per weight ((16 * 4 * 2) + (256 * 2) + (16 * 2)) / 256 = 2.625
typedef struct {
    uint16_t super_scale;  // super-block scale for quantized scales (fp16)
    uint16_t super_min;    // super-block min for quantized scales (fp16)
    uint8_t scales[Q2_K_SUPER_BLOCK_SIZE];  // scales and mins, quantized with 4 bits (length: Q2_K_SUPER_BLOCK_SIZE) 
    uint8_t data[WEIGHT_PER_SUPER_BLOCK / 4];   // quants with 2 bits (length: WEIGHT_PER_SUPER_BLOCK / 4)
} super_block_q2_k;

typedef struct {
    uint64_t num_elements;   /* total elements in the original float array */
    uint64_t num_elements_aligned;   /* aligned (padding) total elements for SUPER_BLOCK ELEMENTS */
    uint32_t num_super_blocks;
    super_block_q2_k *super_blocks;
    
} quantized_array_q2_k_t;

quantized_array_q2_k_t *allocate_q2_k_array(uint64_t num_elements);

void free_quantized_q2_k_array(quantized_array_q2_k_t *quantized_array_q2_k);

int64_t get_quantized_q2_k_array_size(const quantized_array_q2_k_t *quantized_array_q2_k);

quantized_array_q2_k_t *load_quantized_q2_k_array_from_buffer(const void *buffer, int64_t buffer_size);

int k_quantize(const float *float_array, uint64_t num_elements, quantized_array_q2_k_t **quantized_array_q2_k);

int k_dequantize(const quantized_array_q2_k_t *quantized_array_q2_k, float *float_array);

#endif
