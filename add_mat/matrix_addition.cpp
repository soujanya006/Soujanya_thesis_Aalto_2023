#include "matrix_addition.h"

// Static function which performs the actual addition
static void matrix_add(float32_t matrix1[ROWS][COLS], float32_t matrix2[ROWS][COLS], float32_t result[ROWS][COLS]) {

	#pragma HLS ARRAY_PARTITION variable=matrix1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=matrix2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=result complete dim=0

    for (int i = 0; i < ROWS; i++) {
		#pragma HLS UNROLL
        for (int j = 0; j < COLS; j++) {
			#pragma HLS PIPELINE II=1
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}

// Non-static wrapper function which calls the static function
void matrix_add_wrapper(float32_t matrix1[ROWS][COLS], float32_t matrix2[ROWS][COLS], float32_t result[ROWS][COLS]) {
    matrix_add(matrix1, matrix2, result);
}
