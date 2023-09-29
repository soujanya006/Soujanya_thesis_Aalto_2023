#include <iostream>
#include "C:\\Users\\Soujanya\Desktop\\thesis_selfAttention111\\self_attention.h"

typedef ap_fixed<32, 12> float32_t;

int main() {
    float32_t key_matrix[SEQ_LENGTH][EMBEDDING_SIZE] = { /* your initialization here */ 

    								{0.124 ,0.169 ,0.214, 0.259, 0.304},
    		    		    		{0.221,  0.303, 0.386,  0.469 ,0.551},
    		    		    		{0.256 ,0.351,  0.446,  0.541 , 0.635},
    		    		    		{0.281 , 0.379,   0.477, 0.575, 0.673}};
    float32_t query_matrix[SEQ_LENGTH][EMBEDDING_SIZE] = { /* your initialization here */ 


    					{0.3049 , 0.2194 ,0.2203, 0.1375, 0.0529},
    		    		{0.5352 ,0.4693 ,0.371 , 0.2940, 0.0964},
    		    		{0.577, 0.5507 , 0.4028, 0.36149 ,0.1110},
    		    	    {0.5643 , 0.5327 , 0.39753, 0.3528 , 0.11636}};
    float32_t value_matrix[SEQ_LENGTH][EMBEDDING_SIZE] = { /* your initialization here */ 

    					{1, 0.05899 ,0.05959, 0.079399, 0.09649},
    		    		{0.10475 ,2, 0.11533 ,0.13692, 0.17655},
    		    		{0.1202, 0.1198, 3, 0.1584, 0.2097},
    		    		{0.1261, 0.122 , 0.1454,  4, 0.228}};
    float32_t output_matrix[SEQ_LENGTH][EMBEDDING_SIZE];

    self_attention(key_matrix, query_matrix, value_matrix, output_matrix);

    std::cout << "Output matrix:" << std::endl;
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            std::cout << static_cast<float>(output_matrix[i][j]) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
