#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"

__global__ void compute_scalar_matrix_mult(int n, float *matrix,float scalar){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i< n ; i+=stride){
        matrix[i] *= scalar;
    }
}

__global__ void compute_matrix_matrix_mult(int n, struct matrix *matrixA,struct matrix *matrixB,struct matrix *matrixC){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i< n;i +=stride){
        //resto do algoritimo normal
    }
}

int scalar_matrix_mult(float scalar_value, struct matrix* matrix) {
    if (matrix == NULL) {
        printf("matrix struct given is NULL pointer");
        return 0;
    }
    
    
    return 1;
}

int matrix_matrix_mult(struct matrix* matrixA, struct matrix* matrixB, struct matrix* matrixC) {
    if ((matrixA == NULL) || (matrixB == NULL) || (matrixC == NULL)) {
        printf("one of the matrix struct given is NULL pointer");
        return 0;
    }

    else if(matrixA->width != matrixB->height){
        printf("Matrices A and B are not compatible for multiplication");
        return 0;
    }       
    return 1;
}