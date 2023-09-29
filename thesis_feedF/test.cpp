#include <iostream>
#include "C:\Users\\Soujanya\\Desktop\\thesis_feedF\\feedforward.h"

int main() {
    fixed_point_t output[ROWS][COLS];

    // Hardcoded input matrix
    fixed_point_t input[ROWS][COLS] = {
        {1, 3, .14, .15, .16},
        {.2, 4, .2, .2, 2},
        {3, .3, 6, .3, .3},
        {0, 0, 0, 3, 0}
    };
    
    feed_forward_network(input, output);

    // Printing the output
    for(int i = 0; i < ROWS; i++) {
        for(int j = 0; j < COLS; j++) {
            std::cout << static_cast<double>(output[i][j]) << " ";
        }
        std::cout << std::endl;
    }

    return 0;

}







