
//////////////////////////////////// THIS IS THE TEST FILE FOR VITIS HLS POSITIONAL ENCODING 
#include <iostream>
#include "C:\\Users\\Soujanya\\Desktop\\thesis_positional\\positional_encoding.h"
#include <iomanip>

void print_positional_encoding(float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE]) {
    for (int i = 0; i < SEQ_LENGTH; ++i) {
        for (int j = 0; j < EMBEDDING_SIZE; ++j) {
            std::cout  << static_cast<float>(pos_enc[i][j]) << " ";
        }
        std::cout << std::endl;
    }
}


int main() {
    float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE];
    float32_t output_seq[SEQ_LENGTH][EMBEDDING_SIZE];

    float32_t custom_values[SEQ_LENGTH][EMBEDDING_SIZE] = {
        {0.1, 0.1, 0.1, 0.1, 0.1},
        {0.2, 0.2, 0.2, 0.2, 0.2},
		{0.3, 0.3, 0.3, 0.3, 0.3},
		{0.4, 0.4, 0.4, 0.4, 0.4},


    };



    std::cout<<""<< std::endl;
    std::cout<<""<< std::endl;

    positional_encoding(pos_enc);
    std::cout << "Positional Encoding: " << std::endl;
    print_positional_encoding(pos_enc);

    std::cout<<""<< std::endl;
    std::cout<<""<< std::endl;



    std::cout << "Final added value: " << std::endl;
    add_positional_encoding(custom_values,pos_enc,output_seq);
    print_positional_encoding(output_seq);

    std::cout<<""<< std::endl;
    std::cout<<""<< std::endl;


    return 0;
}


//////////////////////////////////////////THIS IS THE H FILE FOR VITIS HLS POSITIONAL ENCODING 

#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include <ap_fixed.h>
#include <hls_stream.h>
#include <cmath>
#include <vector>

#define SEQ_LENGTH 4
#define EMBEDDING_SIZE 5

typedef ap_fixed<32, 12> float32_t;



void positional_encoding(float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE]);


void add_positional_encoding(float32_t custom_values[SEQ_LENGTH][EMBEDDING_SIZE],
		float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE],
		float32_t output_seq[SEQ_LENGTH][EMBEDDING_SIZE]);


#endif // POSITIONAL_ENCODING_H


//////////////////////////////////////////// source code  first version for vitis hls positional encoding





#include "positional_encoding.h"
#include <hls_math.h>

void positional_encoding(float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE]) {


    const float div_term = 1e-4f;
    const float factor = 2.0f / static_cast<float>(EMBEDDING_SIZE);

    Position_Encoding_Loop1:
    for (int pos = 0; pos < SEQ_LENGTH; ++pos) {
        Position_Encoding_Loop2:
        for (int i = 0; i < EMBEDDING_SIZE; i += 2) {

            float angle_rate = static_cast<float>(pos) * hls::powf(div_term, static_cast<float>(i) * factor);
            pos_enc[pos][i] = static_cast<float32_t>(hls::sinf(angle_rate));
            if (i + 1 < EMBEDDING_SIZE) {
                pos_enc[pos][i + 1] = static_cast<float32_t>(hls::cosf(angle_rate));
            }
        }
    }
}

void add_positional_encoding(float32_t custom_values[SEQ_LENGTH][EMBEDDING_SIZE],
                             float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE],
                             float32_t output_seq[SEQ_LENGTH][EMBEDDING_SIZE]) {





    Add_Position_Encoding_Loop1:
    for (int i = 0; i < SEQ_LENGTH; ++i) {
        Add_Position_Encoding_Loop2:
        for (int j = 0; j < EMBEDDING_SIZE; ++j) {
            output_seq[i][j] = custom_values[i][j] + pos_enc[i][j];
        }
    }
}



//////////////////////////////////////////// source code version 2 for vitis hls positional encoding


#include "C:\\Users\\Soujanya\\Desktop\\thesis_positional\\positional_encoding.h"
#include <hls_math.h>




void positional_encoding(float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE]) {


	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS INTERFACE axis port=pos_enc

	#pragma HLS ARRAY_PARTITION variable=pos_enc cyclic factor=4 dim=2

    const float div_term = 1e-4f;
    const float factor = 2.0f / static_cast<float>(EMBEDDING_SIZE);

    Position_Encoding_Loop1:
    for (int pos = 0; pos < SEQ_LENGTH; ++pos) {
        Position_Encoding_Loop2:
        for (int i = 0; i < EMBEDDING_SIZE; i += 2) {
			#pragma HLS PIPELINE II=1

            float angle_rate = static_cast<float>(pos) * hls::powf(div_term, static_cast<float>(i) * factor);
            pos_enc[pos][i] = static_cast<float32_t>(hls::sinf(angle_rate));
            if (i + 1 < EMBEDDING_SIZE) {
                pos_enc[pos][i + 1] = static_cast<float32_t>(hls::cosf(angle_rate));
            }
        }
    }
}

void add_positional_encoding(float32_t custom_values[SEQ_LENGTH][EMBEDDING_SIZE],
                             float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE],
                             float32_t output_seq[SEQ_LENGTH][EMBEDDING_SIZE]) {




	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS INTERFACE axis port=custom_values
	#pragma HLS INTERFACE axis port=pos_enc
	#pragma HLS INTERFACE axis port=output_seq

	#pragma HLS ARRAY_PARTITION variable=pos_enc cyclic factor=4 dim=2
	#pragma HLS ARRAY_PARTITION variable=custom_values cyclic factor=4 dim=2
	#pragma HLS ARRAY_PARTITION variable=output_seq cyclic factor=4 dim=2



    Add_Position_Encoding_Loop1:
    for (int i = 0; i < SEQ_LENGTH; ++i) {
        Add_Position_Encoding_Loop2:
        for (int j = 0; j < EMBEDDING_SIZE; ++j) {
			#pragma HLS PIPELINE II=1
            output_seq[i][j] = custom_values[i][j] + pos_enc[i][j];
        }
    }
}

///////////////////////////////version 3 




#include "C:\\Users\\Soujanya\\Desktop\\thesis_positional\\positional_encoding.h"
#include <hls_math.h>




void positional_encoding(float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE]) {




    const float div_term = 1e-4f;
    const float factor = 2.0f / static_cast<float>(EMBEDDING_SIZE);

    Position_Encoding_Loop1:
    for (int pos = 0; pos < SEQ_LENGTH; ++pos) {
        Position_Encoding_Loop2:
        for (int i = 0; i < EMBEDDING_SIZE; i += 2) {
			#pragma HLS PIPELINE II=2

            float angle_rate = static_cast<float>(pos) * hls::powf(div_term, static_cast<float>(i) * factor);
            pos_enc[pos][i] = static_cast<float32_t>(hls::sinf(angle_rate));
            if (i + 1 < EMBEDDING_SIZE) {
                pos_enc[pos][i + 1] = static_cast<float32_t>(hls::cosf(angle_rate));
            }
        }
    }
}

void add_positional_encoding(float32_t custom_values[SEQ_LENGTH][EMBEDDING_SIZE],
                             float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE],
                             float32_t output_seq[SEQ_LENGTH][EMBEDDING_SIZE]) {







    Add_Position_Encoding_Loop1:
    for (int i = 0; i < SEQ_LENGTH; ++i) {
        Add_Position_Encoding_Loop2:
        for (int j = 0; j < EMBEDDING_SIZE; ++j) {
			#pragma HLS PIPELINE II=2
            output_seq[i][j] = custom_values[i][j] + pos_enc[i][j];
        }
    }
}


///////////////////version 4 source code  for vitis hls positional encoding 


#include "C:\\Users\\Soujanya\\Desktop\\thesis_positional\\positional_encoding.h"
#include <hls_math.h>

 void positional_encoding(float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE]) {
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE axis port=pos_enc
    #pragma HLS ARRAY_PARTITION variable=pos_enc cyclic factor=4 dim=2

    const float div_term = 1e-4f;
    const float factor = 2.0f / static_cast<float>(EMBEDDING_SIZE);
    float angle_rate_lut[EMBEDDING_SIZE];
    #pragma HLS ARRAY_PARTITION variable=angle_rate_lut complete

    for (int i = 0; i < EMBEDDING_SIZE; i += 2) {
        #pragma HLS UNROLL
        angle_rate_lut[i] = hls::powf(div_term, static_cast<float>(i) * factor);
    }

    Position_Encoding_Loop1:
    for (int pos = 0; pos < SEQ_LENGTH; ++pos) {
        Position_Encoding_Loop2:
        for (int i = 0; i < EMBEDDING_SIZE; i += 2) {
			#pragma HLS PIPELINE II=1
            float angle_rate = static_cast<float>(pos) * angle_rate_lut[i];
            pos_enc[pos][i] = static_cast<float32_t>(hls::sinf(angle_rate));
            if (i + 1 < EMBEDDING_SIZE) {
                pos_enc[pos][i + 1] = static_cast<float32_t>(hls::cosf(angle_rate));
            }
        }
    }
}

void add_positional_encoding(float32_t custom_values[SEQ_LENGTH][EMBEDDING_SIZE],
                             float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE],
                             float32_t output_seq[SEQ_LENGTH][EMBEDDING_SIZE]) {
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE axis port=custom_values
    #pragma HLS INTERFACE axis port=pos_enc
    #pragma HLS INTERFACE axis port=output_seq
    #pragma HLS ARRAY_PARTITION variable=custom_values cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=pos_enc cyclic factor=4 dim=2
    #pragma HLS ARRAY_PARTITION variable=output_seq cyclic factor=4 dim=2

    Add_Position_Encoding_Loop1:
    for (int i = 0; i < SEQ_LENGTH; ++i) {
        Add_Position_Encoding_Loop2:
        for (int j = 0; j < EMBEDDING_SIZE; ++j) {
			#pragma HLS PIPELINE II=1
            output_seq[i][j] = custom_values[i][j] + pos_enc[i][j];
        }
    }
}
