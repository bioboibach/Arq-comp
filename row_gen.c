#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    /*  
        argv[1] = height da matrixA 
        argv[2] = width da matrixA

        argv[3] = height da matrixB
        argv[4] = width da matrixB
    */

    FILE* f_matrixA = fopen("floats_256_2.0f.dat", "wb");
    FILE* f_matrixB = fopen("floats_256_5.0f.dat", "wb");
    FILE* f_result1 = fopen("result1.dat", "wb");
    FILE* f_result2 = fopen("result2.dat", "wb");

    int a_row_count = atof(argv[1]) * atof(argv[2]);
    int b_row_count = atof(argv[3]) * atof(argv[4]);

    float* a = (float*)malloc(a_row_count * sizeof(float));
    float* b = (float*)malloc(b_row_count * sizeof(float));

    for (int count = 0; count < a_row_count; count++) {
        a[count] = 2;
    }

    for (int count = 0; count < b_row_count; count++){
        b[count] = 5;
    }

    fwrite(a, sizeof(float), a_row_count, f_matrixA);
    fwrite(b, sizeof(float), b_row_count, f_matrixB);

    fclose(f_matrixA);
    fclose(f_matrixB);
    fclose(f_result1);
    fclose(f_result2);
    free(a);
    free(b);
    return 0;
}
