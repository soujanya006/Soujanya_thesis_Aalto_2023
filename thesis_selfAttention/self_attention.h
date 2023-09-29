#ifndef SELF_ATTENTION_H
#define SELF_ATTENTION_H

#include <ap_fixed.h>

typedef ap_fixed<32, 12> float32_t;

#define SEQ_LENGTH 4
#define EMBEDDING_SIZE 5

void self_attention(float32_t key_matrix[SEQ_LENGTH][EMBEDDING_SIZE], 
                    float32_t query_matrix[SEQ_LENGTH][EMBEDDING_SIZE], 
                    float32_t value_matrix[SEQ_LENGTH][EMBEDDING_SIZE],
                    float32_t output_matrix[SEQ_LENGTH][EMBEDDING_SIZE]);

void approx_softmax(float32_t input[SEQ_LENGTH], float32_t output[SEQ_LENGTH]);

#endif
