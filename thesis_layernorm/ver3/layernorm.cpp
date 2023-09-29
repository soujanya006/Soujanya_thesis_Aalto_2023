

#include "C:\\Users\\Soujanya\Desktop\\thesis_layernorm\\layernorm.h"
#include <hls_math.h>

#define COLS_UNROLL_FACTOR 2

void layernorm(fixed_point input[ROWS][COLS], fixed_point output[ROWS][COLS]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=2 dim=2

    fixed_point mean[ROWS];
    fixed_point stddev[ROWS];

    // Predefined scale and shift parameters
    fixed_point gamma[COLS] = {1.0, 2.0, 3.0};
    fixed_point beta[COLS] = {0.1, 0.2, 0.3};

    // Calculate mean
    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=1
        fixed_point row_sum = 0;
        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR
            row_sum += input[i][j];
        }
        mean[i] = row_sum / COLS;
    }

    // Calculate stddev
    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=1
        fixed_point row_sq_diff_sum = 0;
        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR
            fixed_point diff = input[i][j] - mean[i];
            row_sq_diff_sum = hls::fma(diff, diff, row_sq_diff_sum);
        }
        stddev[i] = hls::sqrt(row_sq_diff_sum / COLS);
    }

    // Normalize and store output
    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=2
        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR
            fixed_point norm_val = (input[i][j] - mean[i]) / stddev[i];
            output[i][j] = hls::fma(norm_val, gamma[j], beta[j]);
        }
    }
}
