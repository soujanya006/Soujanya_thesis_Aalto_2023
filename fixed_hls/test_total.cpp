#include <iostream>
#include <cmath>
#include "matrix_mul_float.h"
#include "matrix_mul_fixed.h"

int main() {
    float A[ROW_A][COL_A] = {{1.2, 2.3, 3.4}, {4.5, 5.6, 6.7}};
    float B[ROW_B][COL_B] = {{0.7, 0.8}, {0.9, 1.0}, {1.1, 1.2}};
    float C_float[ROW_A][COL_B];
    fixed_point_t A_fixed[ROW_A][COL_A] = {{1.2, 2.3, 3.4}, {4.5, 5.6, 6.7}};
    fixed_point_t B_fixed[ROW_B][COL_B] = {{0.7, 0.8}, {0.9, 1.0}, {1.1, 1.2}};
    fixed_point_t C_fixed[ROW_A][COL_B];

    matrix_mul_float(A, B, C_float);
    matrix_mul_fixed(A_fixed, B_fixed, C_fixed);

    float SSE = 0.0f; // Sum of Squared Errors
    float max_error = 0.0f; 

    for(int i = 0; i < ROW_A; i++) {
        for(int j = 0; j < COL_B; j++) {
            float error = C_float[i][j] - C_fixed[i][j].to_float();
            SSE += error * error;
            if(fabs(error) > max_error) {
                max_error = fabs(error);
            }
        }
    }

    float RMSE = std::sqrt(SSE / (ROW_A * COL_B));
    std::cout << "RMSE: " << RMSE << std::endl;
    std::cout << "Max Error: " << max_error << std::endl;

    return 0;
}
