#include <iostream>
#include "matrix_mul_fixed.h"

int main() {
    fixed_point_t A_fixed[ROW_A][COL_A] = {{1.2, 2.3, 3.4}, {4.5, 5.6, 6.7}};
    fixed_point_t B_fixed[ROW_B][COL_B] = {{0.7, 0.8}, {0.9, 1.0}, {1.1, 1.2}};
    fixed_point_t C_fixed[ROW_A][COL_B];

    matrix_mul_fixed(A_fixed, B_fixed, C_fixed);

    // Print fixed-point result
    std::cout << "Result using ap_fixed:" << std::endl;
    for(int i = 0; i < ROW_A; i++) {
        for(int j = 0; j < COL_B; j++) {
            std::cout << C_fixed[i][j].to_double() << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
