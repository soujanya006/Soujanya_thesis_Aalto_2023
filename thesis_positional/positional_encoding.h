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


