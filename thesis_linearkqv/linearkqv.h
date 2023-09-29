#ifndef LINEARKQV_H
#define LINEARKQV_H

#include <ap_int.h>
#include <ap_fixed.h>

#define ROWS 4
#define COLS 5
#define WEIGHTS_ROWS 5
#define WEIGHTS_COLS 5



typedef ap_fixed<32, 12> float32_t;

void linear_kqv(float32_t input[ROWS][COLS], float32_t key[ROWS][COLS], float32_t query[ROWS][COLS], float32_t value[ROWS][COLS]);

#endif
