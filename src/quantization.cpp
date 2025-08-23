#include "quantization.h"

static int64_t _get_q8_0_quantized_array_size(const quantized_array_t *quantized_array) {
    if (!quantized_array) return 0;
    return sizeof(uint8_t)        /* quantized_type   */
         + 3 * sizeof(uint64_t)   /* num_elements, num_blocks, block_size */
         + quantized_array->num_blocks * sizeof(float)   /* scales */
         + quantized_array->num_elements * sizeof(int8_t); /* data */
}

static int64_t _get_q4_0_quantized_array_size(const quantized_array_t *quantized_array) {
    if (!quantized_array) return 0;

    const uint64_t num_elements_for_data = (quantized_array->num_elements + 1) / 2;

    return sizeof(quantized_array_t)    /* quantized_type num_elements, num_blocks, block_size */
        + quantized_array->num_blocks * sizeof(float)   /* scales */
        + num_elements_for_data * sizeof(int8_t);   /* packed data */
}

int64_t get_quantized_array_size(const quantized_array_t *quantized_array) {
    if (!quantized_array) return 0;
    switch (quantized_array->quantized_type) {
        case 0: 
            return _get_q8_0_quantized_array_size(quantized_array);
        case 1:
            return _get_q4_0_quantized_array_size(quantized_array);
        default: 
            return 0; /* unknown type */
    }
}

quantized_array_t *allocate_q8_0_array(uint64_t num_elements,
                                       uint64_t block_size)
{
    if (!num_elements || !block_size) return NULL;

    uint64_t num_blocks = (num_elements + block_size - 1) / block_size;

    size_t total = sizeof(quantized_array_t)
                 + num_blocks * sizeof(float)
                 + num_elements * sizeof(int8_t);

    quantized_array_t *qa = (quantized_array_t*)calloc(1, total);
    if (!qa) return NULL;

    /* initialise the header fields */
    qa->quantized_type = 0;          /* q8_0 */
    qa->num_elements   = num_elements;
    qa->num_blocks     = num_blocks;
    qa->block_size     = block_size;

    qa->scales = (float*)(qa + 1);                /* just after the header */
    qa->data   = (int8_t*)(qa->scales + num_blocks);  /* after the scales */

    return qa;
}

quantized_array_t *allocate_q4_0_array(uint64_t num_elements,
                                       uint64_t block_size)
{
    if (!num_elements || !block_size) return NULL;

    uint64_t num_blocks = (num_elements + block_size - 1) / block_size;
    uint64_t num_elements_for_data = (num_elements + 1) / 2;

    size_t total = sizeof(quantized_array_t)
                 + num_blocks * sizeof(float)
                 + num_elements_for_data * sizeof(int8_t);
    
    quantized_array_t *qa = (quantized_array_t*)calloc(1, total);
    if (!qa) return NULL;

    qa->quantized_type = 1;          /* q4_0 */
    qa->num_elements   = num_elements;
    qa->num_blocks     = num_blocks;
    qa->block_size     = block_size;

    qa->scales = (float*)(qa + 1);
    qa->data   = (int8_t*)(qa->scales + num_blocks);

    return qa;
}

void free_quantized_array(quantized_array_t *quantized_array) {
    if (!quantized_array) return;
    free(quantized_array);
}

static int _quantize_q8_0(const float *float_array,
                          quantized_array_t *quantized_array) {
    if (!float_array || !quantized_array) return 1;

    const uint64_t block_size   = quantized_array->block_size;
    const uint64_t num_blocks   = quantized_array->num_blocks;
    const uint64_t num_elements = quantized_array->num_elements;

    for (uint64_t b = 0; b < num_blocks; ++b) {
        const uint64_t start = b * block_size;
        const uint64_t remain = (start + block_size <= num_elements)
                                  ? block_size
                                  : (num_elements - start);

        /* 1) find max‑abs in this block */
        float abs_max = 0.0f;
        for (uint64_t i = 0; i < remain; ++i) {
            float v = fabsf(float_array[start + i]);
            if (v > abs_max) abs_max = v;
        }

        /* 2) compute scale */
        float scale = (abs_max > 0.0f) ? (abs_max / 127.0f) : 0.0f;
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        quantized_array->scales[b] = scale;

        /* 3) quantise */
        for (uint64_t i = 0; i < remain; ++i) {
            float val = float_array[start + i] * inv_scale;
            long qi   = lrintf(val); /* nearest int */
            if (qi < -127) qi = -127;
            if (qi >  127) qi =  127;
            quantized_array->data[start + i] = (int8_t)qi;
        }
    }
    return 0;
}

static int _quantize_q4_0(const float *float_array,
                          quantized_array_t *quantized_array) {
    if (!float_array || !quantized_array) return 1;

    const uint64_t block_size   = quantized_array->block_size;
    const uint64_t num_blocks   = quantized_array->num_blocks;
    const uint64_t num_elements = quantized_array->num_elements;
    uint8_t *data = (uint8_t *)quantized_array->data;
    
    for (uint64_t b = 0; b < num_blocks; ++b) {
        const uint64_t start = b * block_size;
        const uint64_t remain = (start + block_size <= num_elements)
                                  ? block_size
                                  : (num_elements - start);

        /* 1) find max‑abs in this block */
        float abs_max = 0.0f;
        for (uint64_t i = 0; i < remain; ++i) {
            float v = fabsf(float_array[start + i]);
            if (v > abs_max) abs_max = v;
        }

        /* 2) compute scale */
        float scale = (abs_max > 0.0f) ? (abs_max / 7.0f) : 0.0f;
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        quantized_array->scales[b] = scale;

        /* 3) quantise */
        for (uint64_t i = 0; i < remain; ++i) {
            float val = float_array[start + i] * inv_scale;
            long qi   = lrintf(val); /* nearest int */
            if (qi < -7) qi = -7;
            if (qi >  7) qi =  7;

            uint8_t four_bit_qi = ((uint8_t)qi) & 0x0F; 

            int data_index = (start + i) / 2;
            if (i % 2 == 0) {
                data[data_index] = (uint8_t)(four_bit_qi << 4);
            }
            else {
                data[data_index] = (uint8_t)(data[data_index] | four_bit_qi);
            }
        }
    }
    return 0;
}

int quantize(const float *float_array,
             uint64_t num_elements,
             uint8_t quantized_type,
             quantized_array_t **quantized_array) {
    if (!float_array || num_elements == 0 || *quantized_array) return 1;

    switch (quantized_type) {
        case 0: /* q8_0 */
            *quantized_array = allocate_q8_0_array(num_elements,
                                                   DEFAULT_Q8_0_BLOCK_SIZE);
            if (!*quantized_array) return 1;
            return _quantize_q8_0(float_array, *quantized_array);

        case 1: /* q4_0 */
            *quantized_array = allocate_q4_0_array(num_elements,
                                                   DEFAULT_Q4_0_BLOCK_SIZE);
            if (!*quantized_array) return 1;
            return _quantize_q4_0(float_array, *quantized_array);
        default:
            return 1; /* unknown type */
    }
}

static int _dequantize_q8_0(const quantized_array_t *quantized_array, 
                            float *float_array) {
    const uint64_t block_size   = quantized_array->block_size;
    const uint64_t num_blocks   = quantized_array->num_blocks;
    const uint64_t num_elements = quantized_array->num_elements;

    for (uint64_t b = 0; b < num_blocks; ++b) {
        const uint64_t start = b * block_size;
        const uint64_t remain = (start + block_size <= num_elements)
                                  ? block_size
                                  : (num_elements - start);
        const float scale = quantized_array->scales[b];

        for (uint64_t i = 0; i < remain; ++i) {
            float_array[start + i] = scale * (float)quantized_array->data[start + i];
        }
    }
    return 0;
}

static int _dequantize_q4_0(const quantized_array_t *quantized_array, 
                            float *float_array) {
    const uint64_t block_size   = quantized_array->block_size;
    const uint64_t num_blocks   = quantized_array->num_blocks;
    const uint64_t num_elements = quantized_array->num_elements;
    const uint8_t *src_data = (const uint8_t *)quantized_array->data;

    for (uint64_t b = 0; b < num_blocks; ++b) {
        const uint64_t start = b * block_size;
        const uint64_t remain = (start + block_size <= num_elements)
                                  ? block_size
                                  : (num_elements - start);
        const float scale = quantized_array->scales[b];

        for (uint64_t i = 0; i < remain; ++i) {
            int data_index = (start + i) / 2;
            uint8_t packed_qi = src_data[data_index];

            if (i % 2 == 0) {
                uint8_t qi = packed_qi >> 4;
                int8_t signed_qi = (int8_t)(qi << 4) >> 4;
                float_array[start + i] = scale * (float)(signed_qi);
            }
            else {
                uint8_t qi = packed_qi & 0x0F;
                int8_t signed_qi = (int8_t)(qi << 4) >> 4;
                float_array[start + i] = scale * (float)(signed_qi);
            }
        }
    }   
    return 0;
}


int dequantize(const quantized_array_t *quantized_array, float *float_array) {
    if (!quantized_array || !float_array) return 1;

    switch (quantized_array->quantized_type) {
        case 0: /* q8_0 */
            return _dequantize_q8_0(quantized_array, float_array);

        case 1: /* q4_0 */
            return _dequantize_q4_0(quantized_array, float_array);
        default:
            return 1; /* unknown type */
    }
}
