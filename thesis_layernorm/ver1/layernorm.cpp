
/////version 1




#include "C:\\Users\\Soujanya\Desktop\\thesis_layernorm\\layernorm.h"
#include <hls_math.h>

// Define a constant COLS_UNROLL_FACTOR as a factor of COLS
#define COLS_UNROLL_FACTOR 2





void layernorm(fixed_point input[ROWS][COLS], fixed_point output[ROWS][COLS]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=2 dim=2

    fixed_point mean[ROWS];
    fixed_point stddev[ROWS];

    // Predefined scale and shift parameters
        fixed_point gamma[COLS] = {1.0, 2.0, 3.0};  // Adjust these as needed
        fixed_point beta[COLS] = {0.1, 0.2, 0.3};   // Adjust these as needed

    // Calculate mean
    RowMean:
    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=1

        fixed_point row_sum = 0;

        ColSum:
        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR

            row_sum += input[i][j];
        }

        mean[i] = row_sum / COLS;
    }

    // Calculate stddev
    RowStdDev:
    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=1

        fixed_point row_sq_diff_sum = 0;

        ColSqDiffSum:
        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR

            fixed_point diff = input[i][j] - mean[i];
            //row_sq_diff_sum += diff * diff;
            row_sq_diff_sum = hls::fma(diff, diff, row_sq_diff_sum);
        }

        stddev[i] = hls::sqrt(row_sq_diff_sum / COLS);
    }

    //Normalize and apply scale and shift

    Normalize:
    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=1

        ColNormalize:
        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR

        	fixed_point norm_val = (input[i][j] - mean[i]) / stddev[i];

        	// Apply scaling (multiplication by gamma) and shifting (addition of beta)
        	//output[i][j] = norm_val * gamma[j] + beta[j];
            output[i][j] = hls::fma(norm_val, gamma[j], beta[j]);


        }
    }
}






/// version 2

///////////////////////////////////////////////////////////////////////////

/*

#include "C:\\Users\\Soujanya\Desktop\\thesis_layernorm\\layernorm.h"
#include <hls_math.h>

#define COLS_UNROLL_FACTOR 2

void layernorm(fixed_point input[ROWS][COLS], fixed_point output[ROWS][COLS]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=2 dim=2


	fixed_point mean[ROWS];
	fixed_point stddev[ROWS];

    #pragma HLS ARRAY_PARTITION variable=mean complete
    #pragma HLS ARRAY_PARTITION variable=stddev complete



    fixed_point gamma[COLS] = {1.0, 2.0, 3.0};
    fixed_point beta[COLS] = {0.1, 0.2, 0.3};

    // Calculate mean
    RowMean:
    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=1
        fixed_point row_sum = 0;

        ColSum:
        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR
            row_sum += input[i][j];
        }

        mean[i] = row_sum / COLS;
    }

    // Calculate stddev
    RowStdDev:
    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=1

        fixed_point row_sq_diff_sum = 0;

        ColSqDiffSum:
        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR

            fixed_point diff = input[i][j] - mean[i];
            row_sq_diff_sum = hls::fma(diff, diff, row_sq_diff_sum);
        }

        stddev[i] = hls::sqrt(row_sq_diff_sum / COLS);
    }

    // Normalize and apply scale and shift
    Normalize:
    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=1

        ColNormalize:
        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR

            fixed_point norm_val = (input[i][j] - mean[i]) / stddev[i];
            output[i][j] = hls::fma(norm_val, gamma[j], beta[j]);
        }
    }
}




*/

//////////////////////////version 3



/*

#include "C:\\Users\\Soujanya\Desktop\\thesis_layernorm\\layernorm.h"
#include <hls_math.h>

#define COLS_UNROLL_FACTOR 2

void layernorm(fixed_point input[ROWS][COLS], fixed_point output[ROWS][COLS]) {
    #pragma HLS INLINE off

    fixed_point mean[ROWS];
    fixed_point stddev[ROWS];
    fixed_point gamma[COLS] = {1.0, 2.0, 3.0};
    fixed_point beta[COLS] = {0.1, 0.2, 0.3};

    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output block factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=mean complete
    #pragma HLS ARRAY_PARTITION variable=stddev complete

    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=1
        mean[i] = 0;

        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR
            mean[i] += input[i][j];
        }

        mean[i] /= COLS;
    }

    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=1
        stddev[i] = 0;

        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR
            fixed_point diff = input[i][j] - mean[i];
            stddev[i] = hls::fma(diff, diff, stddev[i]);
        }

        stddev[i] = hls::sqrt(stddev[i] / COLS);
    }

    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=1

        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR
            fixed_point norm_val = (input[i][j] - mean[i]) / stddev[i];
            output[i][j] = hls::fma(norm_val, gamma[j], beta[j]);
        }
    }
}




////////////////ver4


*/





/*

#include "C:\\Users\\Soujanya\Desktop\\thesis_layernorm\\layernorm.h"
#include <hls_math.h>

#define COLS_UNROLL_FACTOR 2

void layernorm(fixed_point input[ROWS][COLS], fixed_point output[ROWS][COLS]) {
    #pragma HLS DATAFLOW

    fixed_point mean[ROWS];
    fixed_point stddev[ROWS];
    fixed_point gamma[COLS] = {1.0, 2.0, 3.0};
    fixed_point beta[COLS] = {0.1, 0.2, 0.3};

    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output block factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=mean complete
    #pragma HLS ARRAY_PARTITION variable=stddev complete

    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=2
        mean[i] = 0;

        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR
            mean[i] += input[i][j];
        }

        mean[i] /= COLS;
    }

    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=2
        stddev[i] = 0;

        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR
            fixed_point diff = input[i][j] - mean[i];
            stddev[i] = hls::fma(diff, diff, stddev[i]);
        }

        stddev[i] = hls::sqrt(stddev[i] / COLS);
    }

    for (int i = 0; i < ROWS; i++) {
        #pragma HLS PIPELINE II=2

        for (int j = 0; j < COLS; j++) {
            #pragma HLS UNROLL factor=COLS_UNROLL_FACTOR
            fixed_point norm_val = (input[i][j] - mean[i]) / stddev[i];
            output[i][j] = hls::fma(norm_val, gamma[j], beta[j]);
        }
    }
}

*/

