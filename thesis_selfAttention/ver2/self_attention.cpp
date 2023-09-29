#include "C:\\Users\\Soujanya\Desktop\\thesis_selfAttention111\\self_attention.h"
#include <hls_math.h>

void approx_softmax(float32_t input[SEQ_LENGTH], float32_t output[SEQ_LENGTH]) {
    float32_t sum = 0.0;
    float32_t temp[SEQ_LENGTH];

    softmax_loop1:
    for (int i = 0; i < SEQ_LENGTH; i++) {
        #pragma HLS PIPELINE II=1
        temp[i] = hls::expf(input[i]);
        sum += temp[i];
    }

    softmax_loop2:
    for (int i = 0; i < SEQ_LENGTH; i++) {
        #pragma HLS PIPELINE II=1
        output[i] = temp[i] / sum;
    }
}

void self_attention(float32_t key_matrix[SEQ_LENGTH][EMBEDDING_SIZE], 
                    float32_t query_matrix[SEQ_LENGTH][EMBEDDING_SIZE], 
                    float32_t value_matrix[SEQ_LENGTH][EMBEDDING_SIZE],
                    float32_t output_matrix[SEQ_LENGTH][EMBEDDING_SIZE]) {
    #pragma HLS ARRAY_PARTITION variable=key_matrix complete dim=0
    #pragma HLS ARRAY_PARTITION variable=query_matrix complete dim=0
    #pragma HLS ARRAY_PARTITION variable=value_matrix complete dim=0
    #pragma HLS ARRAY_PARTITION variable=output_matrix complete dim=0

    float32_t attention_scores[SEQ_LENGTH][SEQ_LENGTH];
    #pragma HLS ARRAY_PARTITION variable=attention_scores complete dim=0

    float32_t scaling_factor = hls::sqrtf(EMBEDDING_SIZE);

    attention_scores_calculation:
    for (int i = 0; i < SEQ_LENGTH; i++) {
        float32_t dot_product[SEQ_LENGTH];
        for (int j = 0; j < SEQ_LENGTH; j++) {
            dot_product[j] = 0;
            for (int k = 0; k < EMBEDDING_SIZE; k++) {
                #pragma HLS PIPELINE II=1
                dot_product[j] += query_matrix[i][k] * key_matrix[j][k];
            }
            dot_product[j] /= scaling_factor;
        }
        approx_softmax(dot_product, attention_scores[i]);
    }

    weighted_values_calculation:
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            float32_t sum = 0;
            for (int k = 0; k < SEQ_LENGTH; k++) {
                #pragma HLS PIPELINE II=1
                sum += attention_scores[i][k] * value_matrix[k][j];
            }
            output_matrix[i][j] = sum;
        }
    }
}
