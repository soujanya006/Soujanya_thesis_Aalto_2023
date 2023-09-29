#ifndef MATRIX_ADDITION_H
#define MATRIX_ADDITION_H

#include <ap_fixed.h>
typedef ap_fixed<32,16> float32_t;

#define ROWS 4
#define COLS 5

// Non-static wrapper function
void matrix_add_wrapper(float32_t matrix1[ROWS][COLS], float32_t matrix2[ROWS][COLS], float32_t result[ROWS][COLS]);

#endif
