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

