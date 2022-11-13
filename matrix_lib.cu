#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"

int threadsPerBlock = 256;
int maxBlocksPerGrid = 4096;

__global__ void compute_scalar_matrix_mult(int n, float *matrix_row,float scalar){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    //printf("index: %d\n", index);

    for(i = index; i < n ; i += stride){
        matrix_row[i] = matrix_row[i] * scalar;
    }

}

__global__ void compute_matrix_matrix_mult(float * matrixA, float * matrixB , float * matrixC,unsigned long int heightA,
    unsigned long int heightB,unsigned long int heightC, unsigned long int widthA,
    unsigned long int widthB , unsigned long int widthC){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    /*
    for (int i = index; i < widthC*heightC; i+=stride) {
        if (i % widthA == 0) j = 0; // se fim da linha de A, B = 0
        k = (i / widthA) * widthB; //inicio da linha de A
        for (count = 0; count < widthB ; count++){ // anda ate fim da linha de B
            matrixC[k] += matrixA[i] * matrixB[j];
            j++;
            k++; // C na msm coluna de B
        }
    }
    */
}

int set_grid_size(int threads_per_block, int max_blocks_per_grid) {
    if (threads_per_block > 1024 || max_blocks_per_grid > 65535) 
        return 0;
    
    threadsPerBlock = threads_per_block;
    maxBlocksPerGrid = max_blocks_per_grid;
    return 1;
}

int scalar_matrix_mult(float scalar_value, struct matrix* matrix) {
    cudaError_t gpuError;
    if (matrix == NULL) {
        printf("matrix struct given is NULL pointer");
        return 0;
    }
    int matrix_row_len = matrix->height * matrix->width;

    int blockSize = threadsPerBlock;
    int numBlocks = (matrix_row_len + blockSize - 1) / blockSize;

    compute_scalar_matrix_mult<<<numBlocks,blockSize>>>(matrix_row_len,matrix->d_rows,scalar_value);
    cudaDeviceSynchronize();

    gpuError = cudaMemcpy(matrix->h_rows, matrix->d_rows, matrix_row_len * sizeof(float), cudaMemcpyDeviceToHost);
       if (gpuError != cudaSuccess){
        printf("memcpy error");
        exit(0);
       }
    return 1;
}

int matrix_matrix_mult(struct matrix* matrixA, struct matrix* matrixB, struct matrix* matrixC) {
    if ((matrixA == NULL) || (matrixB == NULL) || (matrixC == NULL)) {
        printf("one of the matrix struct given is NULL pointer");
        return 0;
    }else if(matrixA->width != matrixB->height){
        printf("Matrices A and B are not compatible for multiplication");
        return 0;
    }

    int blockSize = threadsPerBlock;
    int numBlocks = (matrixC->height * matrixC->width + blockSize - 1) / blockSize;

    compute_matrix_matrix_mult<<<numBlocks,blockSize>>>(matrixA->d_rows,matrixB->d_rows,
        matrixC->d_rows,matrixA->height, matrixB->height,
         matrixC->height,matrixA->width,matrixB->width,matrixC->width);

    cudaMemcpy(matrixC->h_rows, matrixC->d_rows, matrixC->height*matrixC->width, cudaMemcpyDeviceToHost);

    return 1;
}