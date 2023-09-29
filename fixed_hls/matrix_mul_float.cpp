#include "matrix_mul_float.h"
#include <hls_stream.h>


void matrix_mul_float(float A[ROW_A][COL_A], float B[ROW_B][COL_B], float C[ROW_A][COL_B]) {


#pragma HLS ARRAY_PARTITION variable=A dim=2
#pragma HLS ARRAY_PARTITION variable=B dim=1
#pragma HLS ARRAY_PARTITION variable=C dim=2


    for(int i = 0; i < ROW_A; i++) {
        for(int j = 0; j < COL_B; j++) {
            C[i][j] = 0;
            for(int k = 0; k < COL_A; k++) {
                #pragma HLS UNROLL
                C[i][j] += A[i][k] * B[k][j];
            }

            }
        }
}
