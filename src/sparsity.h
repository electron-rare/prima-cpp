#ifndef SPARSITY_H
#define SPARSITY_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/**
 * @brief Represents a sparse array in zero-based COO format for 2D data with shape [num_tokens, num_features].
 *
 * Sparsity is applied along the features dimension. Since each token retains the same number of sparse features,
 * token indices are not stored explicitly. The structure holds the selected feature indices and corresponding values
 * for all tokens in a flattened manner.
 */
typedef struct {
    uint16_t num_tokens;                /* Number of tokens (rows in the 2D shape). */
    uint16_t num_features;              /* Number of features per token (columns in the 2D shape). */
    uint16_t num_sparse_features;       /* Number of retained sparse features per token (must be <= num_features). */
    uint16_t *sparse_indices;           /* Flattened array of selected feature indices; length is (num_tokens * num_sparse_features). */
    float *values;                      /* Flattened array of corresponding sparse values; length is (num_tokens * num_sparse_features). */
} sparse_array_t;

sparse_array_t *allocate_sparse_array(uint16_t num_tokens, uint16_t num_features, float sparse_ratio);                               

void free_sparse_array(sparse_array_t *sparse_array);

uint64_t get_sparse_array_size(const sparse_array_t *sparse_array);

sparse_array_t *load_sparse_array_from_buffer(const void *buffer, uint64_t buffer_size);

int compress(const float *float_array, uint16_t num_tokens, uint16_t num_features,  float sparse_ratio, sparse_array_t **sparse_array);

int decompress(const sparse_array_t *sparse_array, float *float_array);

#endif
