


#include "C:\Users\\Soujanya\\Desktop\\thesis_feedF\\feedforward.h"

static fixed_point_t relu(const fixed_point_t& x) {
    if (x > 0) return x;
    else return 0;
}


void feed_forward_network(fixed_point_t input[ROWS][COLS], fixed_point_t output[ROWS][COLS]) {
    // Store weights and bias values on BRAM for better memory usage
    const fixed_point_t weights1[COLS][COLS] = {
        {1.1, 1.1, 1.1, 1.1, 1.1},
        {0.1, 0.2, 0.3, 0.4, 0.5},
        {1.3, 1.3, 1.3, 1.3, 1.3},
        {1.4, 1.4, 1.4, 1.4, 1.4},
        {0.5, 0.4, 0.3, 0.2, 0.1}
    };
    const fixed_point_t bias1[COLS] = {0.1, 0.2, 0.3, 0.4, 0.5};

    const fixed_point_t weights2[COLS][COLS] = {
        {1.5, 1.5, 1.5, 1.5, 1.5},
        {1.4, 1.4, 1.4, 1.4, 1.4},
        {0.1, 0.2, 0.3, 0.4, 0.5},
        {1.2, 1.2, 1.2, 1.2, 1.2},
        {1.1, 1.1, 1.1, 1.1, 1.1}
    };
    const fixed_point_t bias2[COLS] = {0.5, 0.4, 0.3, 0.2, 0.1};

	#pragma HLS ARRAY_PARTITION variable=input complete dim=2
	#pragma HLS ARRAY_PARTITION variable=output complete dim=2


    fixed_point_t layer1[ROWS][COLS];
	#pragma HLS ARRAY_PARTITION variable=layer1 complete dim=2
	#pragma HLS PIPELINE II=2



    L1: for (int i = 0; i < ROWS; i++) {

        L2: for (int j = 0; j < COLS; j++) {
            fixed_point_t sum = 0;

            L3: for (int k = 0; k < COLS; k++) {
                sum += input[i][k] * weights1[k][j];
            }
            layer1[i][j] = relu(sum + bias1[j]);
        }
    }

    L4: for (int i = 0; i < ROWS; i++) {
        L5: for (int j = 0; j < COLS; j++) {
            fixed_point_t sum = 0;
            L6: for (int k = 0; k < COLS; k++) {
                sum += layer1[i][k] * weights2[k][j];
            }
            output[i][j] = relu(sum + bias2[j]);
        }
    }
}
