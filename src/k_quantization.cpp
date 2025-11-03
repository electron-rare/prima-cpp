#include "k_quantization.h"
#include <stdio.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
// The implementation is refer to https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c#L622

quantized_array_q2_k_t *allocate_q2_k_array(uint64_t num_elements) {
    if (!num_elements) return NULL;

    uint64_t num_elements_aligned = (num_elements % WEIGHT_PER_SUPER_BLOCK == 0) ? num_elements : num_elements + (WEIGHT_PER_SUPER_BLOCK - (num_elements % WEIGHT_PER_SUPER_BLOCK));

    uint64_t num_super_blocks = num_elements_aligned / WEIGHT_PER_SUPER_BLOCK;
    
    size_t total = sizeof(quantized_array_q2_k_t) + num_super_blocks * sizeof(super_block_q2_k);
    quantized_array_q2_k_t *qa = (quantized_array_q2_k_t*)calloc(1, total);
    if (!qa) return NULL;
    
    qa->num_elements = num_elements;
    qa->num_elements_aligned = num_elements_aligned;
    qa->num_super_blocks = num_super_blocks;
    qa->super_blocks = (super_block_q2_k*)(qa + 1);

    return qa;
}

void free_quantized_q2_k_array(quantized_array_q2_k_t *quantized_array_q2_k) {
    if (!quantized_array_q2_k) return;
    free(quantized_array_q2_k);
}

int64_t get_quantized_q2_k_array_size(const quantized_array_q2_k_t *quantized_array_q2_k) {
    if (!quantized_array_q2_k) return 0;
    return sizeof(quantized_array_q2_k_t) + quantized_array_q2_k->num_super_blocks * sizeof(super_block_q2_k);
}

quantized_array_q2_k_t *load_quantized_q2_k_array_from_buffer(const void *buffer, int64_t buffer_size) {
    quantized_array_q2_k_t *quantized_array = (quantized_array_q2_k_t*)calloc(1, buffer_size);
    if (!quantized_array) return NULL;
    
    memcpy(quantized_array, buffer, buffer_size);

    quantized_array->super_blocks = (super_block_q2_k*)(quantized_array + 1);
    return quantized_array;
}

static void find_optimal_scale_and_min(int curr_block_index, float *weights, float *scales, float*mins){
    // naive approach
    const float q2_scale = 3.f;
    float min_val = INFINITY;
    float max_val = -INFINITY;
    
    for (int l = 0; l < Q2_K_BLOCK_SIZE; l++) {
        if (weights[l] < min_val) min_val = weights[l];
    }

    for (int l = 0; l < Q2_K_BLOCK_SIZE; l++) {
        weights[l] -= min_val;
    }

    for (int l = 0; l < Q2_K_BLOCK_SIZE; l++) {
        if (weights[l] > max_val) max_val = weights[l];
    }

    scales[curr_block_index] = max_val / q2_scale;
    mins[curr_block_index] = min_val;
}

int k_quantize(const float *float_array, uint64_t num_elements, quantized_array_q2_k_t **quantized_array_q2_k) {
    const float q4_scale = 15.f;
    
    uint8_t L[WEIGHT_PER_SUPER_BLOCK];
    float weights[Q2_K_BLOCK_SIZE];
    float mins[Q2_K_SUPER_BLOCK_SIZE];
    float scales[Q2_K_SUPER_BLOCK_SIZE];
    
    if (!float_array || num_elements == 0 || *quantized_array_q2_k) {
        return 1;
    }
    
    *quantized_array_q2_k = allocate_q2_k_array(num_elements);
    if (!*quantized_array_q2_k) {
        return 1;
    }
    quantized_array_q2_k_t *qa = *quantized_array_q2_k;

    float *float_array_aligned = (float*) calloc(1, sizeof(float) * qa->num_elements_aligned);
    memcpy(float_array_aligned, float_array, (qa->num_elements) * sizeof(float));

    for (uint32_t curr_super_block_index = 0; curr_super_block_index < qa->num_super_blocks; curr_super_block_index++) {
        super_block_q2_k *curr_super_block = &qa->super_blocks[curr_super_block_index];
        
        float max_scale = -INFINITY;
        float max_abs_min = 0.f;

        for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
            for (int l = 0; l < Q2_K_BLOCK_SIZE; l++) {
                weights[l] = float_array_aligned[j * Q2_K_BLOCK_SIZE + l];
            }
            find_optimal_scale_and_min(j, weights, scales, mins);
            if (scales[j] > max_scale) {
                max_scale = scales[j];
            }
            if (fabsf(mins[j]) > max_abs_min) {
                max_abs_min = fabsf(mins[j]);
            }
        }

        if (max_scale > 0) {
            float iscale = q4_scale / max_scale;
            for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
                int l = (int)lrintf(iscale*scales[j]);
                curr_super_block->scales[j] = l;
            }
            curr_super_block->super_scale = fp16_ieee_from_fp32_value(max_scale / q4_scale);
        } else {
            for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) curr_super_block->scales[j] = 0;
            curr_super_block->super_scale = fp16_ieee_from_fp32_value(0.f);
        }

        if (max_abs_min > 0) {
            const float iscale = 7.f / max_abs_min;
            for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
                int l = (int)lrintf(iscale * mins[j]);
                l = MAX(-8, MIN(7, l));
                curr_super_block->scales[j] |= ((l & 0xF) << 4);
            }
            curr_super_block->super_min = fp16_ieee_from_fp32_value(max_abs_min / 7.f);
        } else {
            curr_super_block->super_min = fp16_ieee_from_fp32_value(0.f);
        }

        for (int j = 0; j < Q2_K_SUPER_BLOCK_SIZE; j++) {
            const float temp_scale = fp16_ieee_to_fp32_value(curr_super_block->super_scale) * (curr_super_block->scales[j] & 0xF);
            const float m = fp16_ieee_to_fp32_value(curr_super_block->super_min);
            const int8_t min_q = (curr_super_block->scales[j] >> 4);
            const float temp_min = m * ((int8_t)(min_q << 4) >> 4);
        
            for (int ii = 0; ii < Q2_K_BLOCK_SIZE; ii++) {
                float val = (temp_scale > 0.f) ? (float_array_aligned[j * Q2_K_BLOCK_SIZE + ii] - temp_min) / temp_scale : 0.f;
                int l = (int)lrintf(val);
                l = MAX(0, MIN(3, l));
                L[j * Q2_K_BLOCK_SIZE + ii] = l;
            }
        }

        uint32_t packed_run = WEIGHT_PER_SUPER_BLOCK / 2; // 128
        for (int j = 0; j < WEIGHT_PER_SUPER_BLOCK; j += packed_run) {
            for (int l = 0; l < Q2_K_BLOCK_SIZE * 2; l++) { // l = 0..31
                uint8_t b0 = L[j + l + 0];
                uint8_t b1 = L[j + l + 32];
                uint8_t b2 = L[j + l + 64];
                uint8_t b3 = L[j + l + 96];
                curr_super_block->data[j / 4 + l] = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6);
            }
        }

        float_array_aligned += WEIGHT_PER_SUPER_BLOCK;
    }

    return 0;
}

int k_dequantize(const quantized_array_q2_k_t *quantized_array_q2_k, float *float_array) {
    if (!quantized_array_q2_k || !float_array || quantized_array_q2_k->num_super_blocks == 0) {
        return 1;
    }

    for (uint32_t s = 0; s < quantized_array_q2_k->num_super_blocks; ++s) {
        const super_block_q2_k *curr_super_block = &quantized_array_q2_k->super_blocks[s];
        const float super_scale = fp16_ieee_to_fp32_value(curr_super_block->super_scale);
        const float super_min   = fp16_ieee_to_fp32_value(curr_super_block->super_min);

        float scales[Q2_K_SUPER_BLOCK_SIZE];
        float mins[Q2_K_SUPER_BLOCK_SIZE];

        for(int i = 0; i < Q2_K_SUPER_BLOCK_SIZE; ++i) {
            uint8_t packed_val = curr_super_block->scales[i];
            scales[i] = super_scale * (packed_val & 0x0F);
            
            int8_t min_q = (packed_val >> 4);
            mins[i] = super_min * ((int8_t)(min_q << 4) >> 4);
        }

        const uint8_t *q = curr_super_block->data;

        for (int l = 0; l < 32; ++l) {
            uint8_t packed_byte = q[l];
            
            int idx0 = l;
            int idx1 = l + 32;
            int idx2 = l + 64;
            int idx3 = l + 96;

            float_array[idx0] = mins[idx0/16] + scales[idx0/16] * ((packed_byte >> 0) & 3);
            float_array[idx1] = mins[idx1/16] + scales[idx1/16] * ((packed_byte >> 2) & 3);
            float_array[idx2] = mins[idx2/16] + scales[idx2/16] * ((packed_byte >> 4) & 3);
            float_array[idx3] = mins[idx3/16] + scales[idx3/16] * ((packed_byte >> 6) & 3);
        }

        for (int l = 0; l < 32; ++l) {
            uint8_t packed_byte = q[32 + l];
            
            int idx0 = 128 + l;
            int idx1 = 160 + l;
            int idx2 = 192 + l;
            int idx3 = 224 + l;

            float_array[idx0] = mins[idx0/16] + scales[idx0/16] * ((packed_byte >> 0) & 3);
            float_array[idx1] = mins[idx1/16] + scales[idx1/16] * ((packed_byte >> 2) & 3);
            float_array[idx2] = mins[idx2/16] + scales[idx2/16] * ((packed_byte >> 4) & 3);
            float_array[idx3] = mins[idx3/16] + scales[idx3/16] * ((packed_byte >> 6) & 3);
        }

        float_array += WEIGHT_PER_SUPER_BLOCK;
    }
    return 0;
}