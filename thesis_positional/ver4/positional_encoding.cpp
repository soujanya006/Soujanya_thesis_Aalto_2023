#include "positional_encoding.h"
#include <hls_math.h>

void positional_encoding(float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE]) {

	#pragma HLS DATAFLOW

    const float div_term = 1e-4f;
    const float factor = 2.0f / static_cast<float>(EMBEDDING_SIZE);

    Position_Encoding_Loop1:
    for (int pos = 0; pos < SEQ_LENGTH; ++pos) {
        Position_Encoding_Loop2:
        for (int i = 0; i < EMBEDDING_SIZE; i += 2) {
			#pragma HLS PIPELINE
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


	#pragma HLS DATAFLOW


    Add_Position_Encoding_Loop1:
    for (int i = 0; i < SEQ_LENGTH; ++i) {
        Add_Position_Encoding_Loop2:
        for (int j = 0; j < EMBEDDING_SIZE; ++j) {
			#pragma HLS PIPELINE
            output_seq[i][j] = custom_values[i][j] + pos_enc[i][j];
        }
    }
}
