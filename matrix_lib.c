#include<stdio.h>
#include "matrix_lib.h"


int scalar_matrix_mult(float scalar_value, struct matrix* matrix) {
    if (matrix == NULL) {
        printf("matrix struct given is NULL pointer");
        return 0;
    }
    int i, limit = matrix->height * matrix->width;
    for (i = 0; i < limit; i++) {
        matrix->rows[i] *= scalar_value;
    }
    return 1;
}



int matrix_matrix_mult(struct matrix* matrixA, struct matrix* matrixB, struct matrix* matrixC) {
    if ((matrixA == NULL) || (matrixB == NULL) || (matrixC == NULL)) {
        printf("one of the matrix struct given is NULL pointer");
        return 0;
    }

    int i, j, k = 0;
    int a_limit = matrixA->height * matrixA->width;
    int b_limit = matrixB->height * matrixB->width;
    int aw = matrixA->width;
    int bw = matrixB->width;
    

    for (i = 0; i < a_limit; k = (i / aw) * bw){

        for(j = 0; j < b_limit; j++){
            //printf("i = %d | j = %d | k = %d\n", i, j, k);
            matrixC->rows[k] += matrixA->rows[i] * matrixB->rows[j];
            k++;
            if((j + 1) % bw == 0){ // se B fim da linha
                i++; // A anda
                k = (i / aw) * bw; // c = inicio da linha de a
            }
        }
    }
    return 1;
}