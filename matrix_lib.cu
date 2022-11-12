#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"

int threadsPerBlock = 256;
int maxBlocksPerGrid = 4096;

__global__ void compute_scalar_matrix_mult(int n, float *matrix,float scalar){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i< n ; i+=stride){
        matrix[i] *= scalar;
    }
}

__global__ void compute_matrix_matrix_mult(int matrixSize, ){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i< matrixSize;i+=stride){
        
    }
}

int set_grid_size(int threads_per_block, int max_blocks_per_grid) {
    if (threads_per_block > 1024 || max_blocks_per_grid > 65535) 
        return 0;
    
    threadsPerBlock = threads_per_block;
    maxBlocksPerGrid = max_blocks_per_grid;
    return 1;
}

int scalar_matrix_mult(float scalar_value, struct matrix* matrix) {
    if (matrix == NULL) {
        printf("matrix struct given is NULL pointer");
        return 0;
    }
    int blockSize = threads_per_block;
    int numBlocks = (matrix->height * matrix->width + blockSize - 1) / blockSize;

    compute_scalar_matrix_mult<<<numBlocks,blockSize>>>(matrix->height * matrix->width,matrix->d_rows,scalar_value);
    cudaMemcpy(matrix->h_rows, matrix->d_rows, matrix->height*matrix->width, cudaMemcpyDeviceToHost);
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