#include "sparsity.h"

sparse_array_t *allocate_sparse_array(uint16_t num_tokens, uint16_t num_features, float sparse_ratio) {
    if (!num_tokens || !num_features) return NULL;
    if (sparse_ratio < 0.0f || sparse_ratio > 1.0f) return NULL;
    
    float raw_sparse = (float)num_features * sparse_ratio;
    uint16_t num_sparse_features = (uint16_t)roundf(raw_sparse);
    
    // clamp to valid range
    if (num_sparse_features > num_features) {
        num_sparse_features = num_features;
    } else if (num_sparse_features == 0 && sparse_ratio > 0.0f) {
        num_sparse_features = 1;  // Avoid total sparsity if ratio positive;
    }

    uint32_t sparse_elements = (uint32_t)num_tokens * num_sparse_features;
    uint64_t total = sizeof(sparse_array_t) + sparse_elements * (sizeof(float) + sizeof(uint16_t));
    sparse_array_t *sparse_array = (sparse_array_t*)calloc(1, total);
    if (!sparse_array) return NULL;

    /* initialise the header fields */
    sparse_array->num_tokens = num_tokens;
    sparse_array->num_features = num_features;
    sparse_array->num_sparse_features = num_sparse_features;
    sparse_array->sparse_indices = (uint16_t*)(sparse_array + 1);    /* just after the header */
    sparse_array->values = (float*)(sparse_array->sparse_indices + sparse_elements);     /* after the sparse_indices */

    return sparse_array;
}                          

void free_sparse_array(sparse_array_t *sparse_array) {
    if (!sparse_array) return;
    free(sparse_array);
}

uint64_t get_sparse_array_size(const sparse_array_t *sparse_array) {
    if (!sparse_array) return 0;

    uint32_t sparse_elements = (uint32_t)sparse_array->num_tokens * sparse_array->num_sparse_features;
    
    return sizeof(sparse_array_t) + sparse_elements * (sizeof(float) + sizeof(uint16_t));
}

sparse_array_t *load_sparse_array_from_buffer(const void *buffer, uint64_t buffer_size) {
    sparse_array_t *sparse_array = (sparse_array_t*)calloc(1, buffer_size);
    if (!sparse_array) return NULL;
    
    memcpy(sparse_array, buffer, buffer_size);

    uint32_t sparse_elements = (uint32_t)sparse_array->num_tokens * sparse_array->num_sparse_features;

    sparse_array->sparse_indices   = (uint16_t*)(sparse_array + 1);
    sparse_array->values = (float*)(sparse_array->sparse_indices + sparse_elements);

    return sparse_array;
}

typedef struct {
    uint16_t index;
    float abs_val;
} sort_entry_t;

static int abs_sort_cmp(const void *a, const void *b) {
    const float abs_a = ((const sort_entry_t *)a)->abs_val;
    const float abs_b = ((const sort_entry_t *)b)->abs_val;
    if (abs_a != abs_b) {
        return (abs_a > abs_b) ? -1 : 1;
    }
    const uint16_t idx_a = ((const sort_entry_t *)a)->index;
    const uint16_t idx_b = ((const sort_entry_t *)b)->index;
    return (int)idx_a - (int)idx_b;
}

int compress(const float *float_array, uint16_t num_tokens, uint16_t num_features, float sparse_ratio, sparse_array_t **sparse_array) {
    if (!float_array || num_tokens == 0 || num_features == 0 || *sparse_array) return 1;

    /* ---- allocate sparse ------------------------------------------ */
    *sparse_array = allocate_sparse_array(num_tokens, num_features, sparse_ratio);
    if (!*sparse_array) return 1;

#pragma omp parallel for
    for (uint16_t cur_token_index = 0; cur_token_index < num_tokens; cur_token_index++) {
        sort_entry_t *entries = (sort_entry_t *)malloc(num_features * sizeof(sort_entry_t));
        
        uint32_t dense_base = (uint32_t)cur_token_index * num_features;
        uint32_t sparse_base = (uint32_t)cur_token_index * (*sparse_array)->num_sparse_features;

        for (uint16_t i = 0; i < num_features; i++) {
            entries[i].index = i;
            entries[i].abs_val = fabsf(float_array[dense_base + i]);
        }
        qsort(entries, num_features, sizeof(sort_entry_t), abs_sort_cmp);
        
        for (uint16_t keep_feature_index = 0; keep_feature_index < (*sparse_array)->num_sparse_features; keep_feature_index++) {
            uint16_t orig_index = entries[keep_feature_index].index;
            (*sparse_array)->sparse_indices[sparse_base + keep_feature_index] = orig_index;
            (*sparse_array)->values[sparse_base + keep_feature_index] = float_array[dense_base + orig_index];
        }

        free(entries);
    }

    return 0;
}

int decompress(const sparse_array_t *sparse_array, float *float_array) {
    if (!float_array || !sparse_array) return 1;

    uint32_t num_elements = (uint32_t)sparse_array->num_tokens * sparse_array->num_features;
    memset(float_array, 0, num_elements * sizeof(float));

    for (uint16_t cur_token_index = 0; cur_token_index < sparse_array->num_tokens; cur_token_index++) {
        uint32_t dense_base = (uint32_t)cur_token_index * sparse_array->num_features;
        uint32_t sparse_base = (uint32_t)cur_token_index * sparse_array->num_sparse_features;

        for (uint16_t keep_feature_index = 0; keep_feature_index < sparse_array->num_sparse_features; keep_feature_index++) {
            uint16_t original_feature_index = sparse_array->sparse_indices[sparse_base + keep_feature_index];
            float_array[dense_base + original_feature_index] = sparse_array->values[sparse_base + keep_feature_index];
        }
    }

    return 0;
}
