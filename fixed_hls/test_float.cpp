#include <iostream>
#include "matrix_mul_float.h"

int main() {
    float A_float[ROW_A][COL_A] = {{1.2, 2.3, 3.4}, {4.5, 5.6, 6.7}};
    float B_float[ROW_B][COL_B] = {{0.7, 0.8}, {0.9, 1.0}, {1.1, 1.2}};
    float C_float[ROW_A][COL_B];

    matrix_mul_float(A_float, B_float, C_float);

    // Print float result
    std::cout << "Result using float:" << std::endl;
    for(int i = 0; i < ROW_A; i++) {
        for(int j = 0; j < COL_B; j++) {
            std::cout << C_float[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
