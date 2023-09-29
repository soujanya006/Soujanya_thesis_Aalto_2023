#include "matrix_mul_fixed.h"
#include <hls_stream.h>



void matrix_mul_fixed(fixed_point_t A[ROW_A][COL_A], fixed_point_t B[ROW_B][COL_B], fixed_point_t C[ROW_A][COL_B]) {
	#pragma HLS ARRAY_PARTITION variable=A dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=B dim=2 complete

	for(int i = 0; i < ROW_A; i++) {
        for(int j = 0; j < COL_B; j++) {
            C[i][j] = 0;
            for(int k = 0; k < COL_A; k++) {
                #pragma HLS PIPELINE II=1
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
