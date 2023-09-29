#include "C:\Users\\Soujanya\\Desktop\\thesis_feedF\\feedforward.h"

static fixed_point_t relu(fixed_point_t x) {
#pragma HLS INLINE
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}


static void matmul_and_add_bias(fixed_point_t a[ROWS][COLS], const fixed_point_t b[COLS][COLS],
		const fixed_point_t bias[COLS], fixed_point_t c[ROWS][COLS]) {

#pragma HLS ARRAY_PARTITION variable=a cyclic factor=2 dim=2
#pragma HLS ARRAY_PARTITION variable=b cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=c cyclic factor=2 dim=2

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            #pragma HLS PIPELINE II=4
            fixed_point_t tmp = 0;
            for (int k = 0; k < COLS; k++) {
                #pragma HLS UNROLL
                tmp += a[i][k] * b[k][j];
            }
            // Add bias
            c[i][j] = tmp + bias[j];
        }
    }
}

void feed_forward_network(fixed_point_t input[ROWS][COLS], fixed_point_t output[ROWS][COLS]) {
    const fixed_point_t weights1[COLS][COLS] = {
    		{1.1, 1.1, 1.1, 1.1, 1.1},
    		        {0.1, 0.2, 0.3, 0.4, 0.5},
    		        {1.3, 1.3, 1.3, 1.3, 1.3},
    		        {1.4, 1.4, 1.4, 1.4, 1.4},
    		        {0.5, 0.4, 0.3, 0.2, 0.1}
    };
    const fixed_point_t weights2[COLS][COLS] = {
    		{1.5, 1.5, 1.5, 1.5, 1.5},
    		        {1.4, 1.4, 1.4, 1.4, 1.4},
    		        {0.1, 0.2, 0.3, 0.4, 0.5},
    		        {1.2, 1.2, 1.2, 1.2, 1.2},
    		        {1.1, 1.1, 1.1, 1.1, 1.1}
    };

    const fixed_point_t bias1[COLS] = {0.1, 0.2, 0.3, 0.4, 0.5};
    const fixed_point_t bias2[COLS] = {0.5, 0.4, 0.3, 0.2, 0.1};

    fixed_point_t layer1[ROWS][COLS];

    // Matrix multiplication with first set of weights and bias addition
    matmul_and_add_bias(input, weights1, bias1, layer1);

    // Apply ReLU
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
#pragma HLS PIPELINE II=1
            layer1[i][j] = relu(layer1[i][j]);
        }
       }

    // Matrix multiplication with second set of weights and bias addition
    matmul_and_add_bias(layer1, weights2, bias2, output);

}
