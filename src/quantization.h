#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* The setting is refer to https://huggingface.co/docs/hub/en/gguf */
#define DEFAULT_Q8_0_BLOCK_SIZE 32
#define DEFAULT_Q4_0_BLOCK_SIZE 32
#define DEFAULT_Q4_K_SUPER_BLOCK_SIZE 8

typedef struct {
    uint8_t  quantized_type; /* 0: q8_0, 1: q4_0, … */
    uint64_t num_elements;   /* total elements in the original float array */
    uint64_t num_blocks;     /* number of blocks (for block‑wised formats) */
    uint64_t block_size;     /* elements per block */
    float  *scales;          /* length = num_blocks (or num_superblocks for kquant formats) */
    int8_t *data;            /* for kquant, here need to contain quantized scale value + quantized value, otherwise it only need to store quantized value*/
} quantized_array_t;

quantized_array_t *allocate_q8_0_array(uint64_t num_elements,
                                       uint64_t block_size);

quantized_array_t *allocate_q4_0_array(uint64_t num_elements,
                                       uint64_t block_size);                                       

void free_quantized_array(quantized_array_t *quantized_array);

int64_t get_quantized_array_size(const quantized_array_t *quantized_array);

int quantize(const float *float_array,
             uint64_t num_elements,
             uint8_t quantized_type,
             quantized_array_t **quantized_array);

int dequantize(const quantized_array_t *quantized_array,
               float *float_array);

#endif
