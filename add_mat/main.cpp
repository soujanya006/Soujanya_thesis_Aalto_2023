#include <iostream>
#include "matrix_addition.h"

int main() {
    float32_t matrix1[ROWS][COLS] = { /* Your values here */


    		{0.12499996 ,0.16999994 ,0.21499993, 0.2599999 , 0.3049999},
    		        {0.2211043,  0.3037452 , 0.3863861,  0.46902695 ,0.5516679},
    		        {0.25645804 ,0.3513321,  0.4462061,  0.5410801 , 0.63595414},
    		        {0.2813279 , 0.379367,   0.47740608, 0.57544523, 0.6734843}
    };
    float32_t matrix2[ROWS][COLS] = { /* Your values here */


    		{0.3049999 , 0.21949989 ,0.22039995, 0.13759995, 0.05299998},
    		    		{0.53525925 ,0.46936825 ,0.3715092 , 0.29408222, 0.09648724},
    		    		{0.57772917, 0.5507468 , 0.40281922, 0.36149833 ,0.11103243},
    		    	    {0.5643749 , 0.5327983 , 0.39753163, 0.3528948 , 0.11636797}
    };
    float32_t result[ROWS][COLS];

    // Call the wrapper function
    matrix_add_wrapper(matrix1, matrix2, result);

    // Print the result matrix
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            std::cout << static_cast<float>(result[i][j] )<< " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
