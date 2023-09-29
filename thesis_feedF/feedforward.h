#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include <ap_fixed.h>

#define ROWS 4
#define COLS 5

typedef ap_ufixed<32,12> fixed_point_t;

void feed_forward_network(fixed_point_t input[ROWS][COLS], fixed_point_t output[ROWS][COLS]);


#endif
