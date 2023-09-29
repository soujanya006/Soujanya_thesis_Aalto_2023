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
