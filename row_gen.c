#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    /*  
        argv[1] = height da matrixA 
        argv[2] = width da matrixA
    */

    FILE* f_matrixA = fopen("floats_256_2.0f.dat", "wb");
    FILE* f_matrixB = fopen("floats_256_5.0f.dat", "wb");
    FILE* f_result1 = fopen("result1.dat", "wb");
    FILE* f_result2 = fopen("result2.dat", "wb");

    int row_count = atof(argv[1]) * atof(argv[2]);

    float* a = (float*)malloc(row_count * sizeof(float));
    float* b = (float*)malloc(row_count * sizeof(float));

    for (int count = 0; count < row_count; count++) {
        a[count] = 2+count;
        b[count] = 5+count;
    }

    fwrite(a, sizeof(float), row_count, f_matrixA);
    fwrite(b, sizeof(float), row_count, f_matrixB);

    fclose(f_matrixA);
    fclose(f_matrixB);
    fclose(f_result1);
    fclose(f_result2);
    free(a);
    free(b);
    return 0;
}