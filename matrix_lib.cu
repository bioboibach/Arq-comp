#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"

int threadsPerBlock = 256;
int maxBlocksPerGrid = 4096;

__global__ void compute_scalar_matrix_mult(int n, float *matrix_row,float scalar){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < n ; i += stride){
        matrix_row[i] = matrix_row[i] * scalar;
    }
}

__global__ void compute_matrix_matrix_mult(float * matrixA, float * matrixB , float * matrixC,unsigned long int ah,
    unsigned long int bh,unsigned long int ch, unsigned long int aw,
    unsigned long int bw , unsigned long int cw) {
        
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int indexA, indexB, indexC, counter,auxA,auxB;

     for (indexC = index; indexC < ch * cw; indexC += stride) {
        auxA = ((indexC / aw) * bw);
        auxB = (indexC % ah);
        for (counter = 0; counter < aw ; counter++){ 
            indexA = auxA + counter;
            indexB = counter * bw + auxB;
            matrixC[indexC] += matrixA[indexA] * matrixB[indexB];
        }
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
    cudaError_t gpuError;
    if (matrix == NULL) {
        printf("matrix struct given is NULL pointer");
        return 0;
    }
    int matrix_row_len = matrix->height * matrix->width;

    int blockSize = threadsPerBlock;
    int numBlocks = (matrix_row_len + blockSize - 1) / blockSize;

    if (numBlocks > maxBlocksPerGrid) numBlocks = maxBlocksPerGrid;

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

    if (numBlocks > maxBlocksPerGrid) numBlocks = maxBlocksPerGrid;

    compute_matrix_matrix_mult<<<numBlocks,blockSize>>>(matrixA->d_rows,matrixB->d_rows,
        matrixC->d_rows,matrixA->height, matrixB->height,
        matrixC->height,matrixA->width,matrixB->width,matrixC->width);
    cudaDeviceSynchronize();

    cudaMemcpy(matrixC->h_rows, matrixC->d_rows, matrixC->height*matrixC->width * sizeof(float), cudaMemcpyDeviceToHost);

    return 1;
}


int memo_opt_matrix_matrix_mult(struct matrix* matrixA, struct matrix* matrixB, struct matrix* matrixC) {
    if ((matrixA == NULL) || (matrixB == NULL) || (matrixC == NULL)) {
        printf("one of the matrix struct given is NULL pointer");
        return 0;
    }

    int i;
    int j = 0;
    int k = 0;
    int count = 0;

    int row_len_A = matrixA->height * matrixA->width;

    int aw = matrixA->width;
    int bw = matrixB->width;
    
    for (i = 0; i < row_len_A; i++) {

        if (i % aw == 0) j = 0; // se fim da linha de A, B = 0
        k = (i / aw) * bw; //inicio da linha de A

        for (count = 0; count < bw ; count++){ // anda ate fim da linha de B

            matrixC->h_rows[k] += matrixA->h_rows[i] * matrixB->h_rows[j];

            j++;
            k++; // C na msm coluna de B
        }
    }
    return 1;
}