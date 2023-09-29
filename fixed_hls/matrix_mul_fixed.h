#include <ap_fixed.h>

#define ROW_A 2
#define COL_A 3
#define ROW_B 3
#define COL_B 2

typedef ap_fixed<16,8> fixed_point_t;  // 16 bits in total, 8 bits for integer, 8 bits for fractional part

void matrix_mul_fixed(fixed_point_t A[ROW_A][COL_A], fixed_point_t B[ROW_B][COL_B], fixed_point_t C[ROW_A][COL_B]);
