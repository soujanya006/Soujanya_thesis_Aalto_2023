#include "linearkqv.h"

void linear_kqv(float32_t input[ROWS][COLS], float32_t key[ROWS][COLS], float32_t query[ROWS][COLS], float32_t value[ROWS][COLS]) {


    const float32_t weights_key[WEIGHTS_ROWS][WEIGHTS_COLS] = { /* your hardcoded weights for key */
    			{0.1, 0.2, 0.3, 0.4, 0.5},
    		    {0.5, 0.4, 0.3, 0.2, 0.1},
    		    {0.6, 0.7, 0.8, 0.9, 1.0},
    		    {1.0, 0.9, 0.8, 0.7, 0.6},
    		    {0.9, 0.8, 0.7, 0.6, 0.5}
    };
    const float32_t weights_query[WEIGHTS_ROWS][WEIGHTS_COLS] = { /* your hardcoded weights for query */
    			{0.2, 0.3, 0.4, 0.5, 0.6},
    		    {0.6, 0.5, 0.4, 0.3, 0.2},
    		    {0.7, 0.8, 0.9, 1.0, 1.1},
    		    {1.1, 1.0, 0.9, 0.8, 0.7},
    		    {1.0, 0.9, 0.8, 0.7, 0.6}
    };
    const float32_t weights_value[WEIGHTS_ROWS][WEIGHTS_COLS] = { /* your hardcoded weights for value */
    		 	 {0.3, 0.4, 0.5, 0.6, 0.7},
    		    {0.7, 0.6, 0.5, 0.4, 0.3},
    		    {0.8, 0.9, 1.0, 1.1, 1.2},
    		    {1.2, 1.1, 1.0, 0.9, 0.8},
    		    {1.1, 1.0, 0.9, 0.8, 0.7}
    };

    const float32_t bias_key[COLS] = { /* your hardcoded bias for key */
    		0.1, 0.2, 0.3, 0.4, 0.5
    };
    const float32_t bias_query[COLS] = { /* your hardcoded bias for query */
    		0.2, 0.3, 0.4, 0.5, 0.6
    };
    const float32_t bias_value[COLS] = { /* your hardcoded bias for value */
    		0.3, 0.4, 0.5, 0.6, 0.7
    };

    ROWS_LOOP: for (int i = 0; i < ROWS; ++i) {
        #pragma HLS UNROLL
        COLS_LOOP: for (int j = 0; j < COLS; ++j) {
            #pragma HLS UNROLL

        	float32_t key_sum = 0;
        	float32_t query_sum = 0;
        	float32_t value_sum = 0;

            WEIGHTS_LOOP: for (int k = 0; k < WEIGHTS_ROWS; ++k) {
                #pragma HLS UNROLL
                key_sum += input[i][k] * weights_key[k][j];
                query_sum += input[i][k] * weights_query[k][j];
                value_sum += input[i][k] * weights_value[k][j];
            }

            key[i][j] = key_sum + bias_key[j];
            query[i][j] = query_sum + bias_query[j];
            value[i][j] = value_sum + bias_value[j];
        }
    }
}
