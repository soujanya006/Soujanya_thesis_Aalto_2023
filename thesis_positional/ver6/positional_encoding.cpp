#include "C:\\Users\\Soujanya\\Desktop\\thesis_positional\\positional_encoding.h"
#include <hls_math.h>



/*



 // A helper function to round to 3 decimal places
float round_to_3(float number) {
    return static_cast<float>(hls::round(number * 1000.0f) / 1000.0f);
}


 */
//sou

void positional_encoding(float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE]) {

#pragma HLS ARRAY_PARTITION variable=pos_enc cyclic factor=4 dim=2

    const float div_term = 1e-4f;

    float angle_rate[EMBEDDING_SIZE];

    Position_Encoding_Loop1:
    for (int pos = 0; pos < SEQ_LENGTH; ++pos) {
        Position_Encoding_Loop2:
        for (int i = 0; i < EMBEDDING_SIZE; i += 2) {
			#pragma HLS PIPELINE II=1
            angle_rate[i] = static_cast<float>(pos) * hls::powf(div_term, static_cast<float>(i) * 2.0f / static_cast<float>(EMBEDDING_SIZE));
            pos_enc[pos][i] = static_cast<float32_t>(hls::sinf(angle_rate[i]));
            if (i + 1 < EMBEDDING_SIZE) {
                pos_enc[pos][i + 1] = static_cast<float32_t>(hls::cosf(angle_rate[i]));
            }
        }
    }
}



void add_positional_encoding(float32_t custom_values[SEQ_LENGTH][EMBEDDING_SIZE],
                             float32_t pos_enc[SEQ_LENGTH][EMBEDDING_SIZE],
                             float32_t output_seq[SEQ_LENGTH][EMBEDDING_SIZE]) {


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
