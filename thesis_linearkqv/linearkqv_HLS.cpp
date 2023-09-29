//////////////////////////////////test code for linear layer to get key query and value , in vitis HLS 

#include <iostream>
#include "linearkqv.h"

int main() {
	float32_t input[ROWS][COLS] = { /* your hardcoded input matrix */
    			{0.1, 0.2, 0.3, 0.4, 0.5},
    		    {0.5, 0.4, 0.3, 0.2, 0.1},
    		    {0.6, 0.7, 0.8, 0.9, 1.0},
    		    {1.0, 0.9, 0.8, 0.7, 0.6}
    };

	float32_t key[ROWS][COLS];
	float32_t query[ROWS][COLS];
	float32_t value[ROWS][COLS];

    linear_kqv(input, key, query, value);

    std::cout << "Key Matrix:" << std::endl;
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            std::cout << static_cast<float>(key[i][j])<< " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Query Matrix:" << std::endl;
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            std::cout << static_cast<float>(query[i][j]) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Value Matrix:" << std::endl;
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            std::cout << static_cast<float>(value[i][j])<< " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

////////////////////////////////// H code for linear layer to get key query and value , in vitis HLS 


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



////////////////////////////////// source code version 1  for linear layer to get key query and value , in vitis HLS 

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


////////////////////////////////// source code version 2  for linear layer to get key query and value , in vitis HLS 

#include "linearkqv.h"

void linear_kqv(float32_t input[ROWS][COLS], float32_t key[ROWS][COLS], float32_t query[ROWS][COLS], float32_t value[ROWS][COLS]) {


	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS INTERFACE axis port=input
	#pragma HLS INTERFACE axis port=key
	#pragma HLS INTERFACE axis port=query
	#pragma HLS INTERFACE axis port=value


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




	#pragma HLS ARRAY_PARTITION variable=weights_key complete dim=0
	#pragma HLS ARRAY_PARTITION variable=weights_query complete dim=0
	#pragma HLS ARRAY_PARTITION variable=weights_value complete dim=0
	#pragma HLS ARRAY_PARTITION variable=bias_key complete dim=0
	#pragma HLS ARRAY_PARTITION variable=bias_query complete dim=0
	#pragma HLS ARRAY_PARTITION variable=bias_value complete dim=0


    ROWS_LOOP: for (int i = 0; i < ROWS; ++i) {
        COLS_LOOP: for (int j = 0; j < COLS; ++j) {
			#pragma HLS PIPELINE II=1


        	float32_t key_sum = 0;
        	float32_t query_sum = 0;
        	float32_t value_sum = 0;

            WEIGHTS_LOOP: for (int k = 0; k < WEIGHTS_ROWS; ++k) {

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

////////////////////////////////// source code version 3  for linear layer to get key query and value , in vitis HLS 


#include "C:\\Users\\Soujanya\\Desktop\\thesis_linearkqv\\linearkqv.h"


void linear_kqv(float32_t input[ROWS][COLS], float32_t key[ROWS][COLS],
		float32_t query[ROWS][COLS], float32_t value[ROWS][COLS]) {

    // Same hardcoded weights and biases here


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

    #pragma HLS ARRAY_PARTITION variable=input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=key complete dim=0
    #pragma HLS ARRAY_PARTITION variable=query complete dim=0
    #pragma HLS ARRAY_PARTITION variable=value complete dim=0

    #pragma HLS PIPELINE
    ROWS_LOOP: for (int i = 0; i < ROWS; ++i) {
        COLS_LOOP: for (int j = 0; j < COLS; ++j) {

        	float32_t key_sum = 0;
        	float32_t query_sum = 0;
        	float32_t value_sum = 0;

            WEIGHTS_LOOP: for (int k = 0; k < WEIGHTS_ROWS; ++k) {
                float32_t input_val = input[i][k];
                key_sum += input_val * weights_key[k][j];
                query_sum += input_val * weights_query[k][j];
                value_sum += input_val * weights_value[k][j];
            }

            key[i][j] = key_sum + bias_key[j];
            query[i][j] = query_sum + bias_query[j];
            value[i][j] = value_sum + bias_value[j];
        }
    }
}
