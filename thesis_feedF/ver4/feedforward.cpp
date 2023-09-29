

#include "C:\Users\\Soujanya\\Desktop\\thesis_feedF\\feedforward.h"

static fixed_point_t relu(const fixed_point_t& x) {
    #pragma HLS INLINE
    if (x > 0) return x;
    else return 0;
}



void feed_forward_network(fixed_point_t input[ROWS][COLS], fixed_point_t output[ROWS][COLS]) {
    #pragma HLS DATAFLOW
    // ... [Weights and biases declarations remain unchanged]


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


    #pragma HLS ARRAY_PARTITION variable=input block factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=output block factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=weights1 block factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=weights2 block factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=bias1 complete
    #pragma HLS ARRAY_PARTITION variable=bias2 complete

    fixed_point_t layer1[ROWS][COLS];
    #pragma HLS ARRAY_PARTITION variable=layer1 block factor=2 dim=2

    L1: for (int i = 0; i < ROWS; i++) {
        L2: for (int j = 0; j < COLS; j++) {
            fixed_point_t sum = 0;
            L3: for (int k = 0; k < COLS; k++) {
                #pragma HLS PIPELINE
                sum += input[i][k] * weights1[k][j];
            }
            layer1[i][j] = relu(sum + bias1[j]);
        }
    }

    L4: for (int i = 0; i < ROWS; i++) {
        L5: for (int j = 0; j < COLS; j++) {
            fixed_point_t sum = 0;
            L6: for (int k = 0; k < COLS; k++) {
                #pragma HLS PIPELINE
                sum += layer1[i][k] * weights2[k][j];
            }
            output[i][j] = relu(sum + bias2[j]);
        }
    }
}

