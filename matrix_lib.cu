#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"
#include <immintrin.h>

void* compute_scalar_matrix_mult(){
    
    int i, limit = p->limit;
    struct matrix * matrix = p->matrix;
    __m256 scalar = _mm256_set1_ps(p->scalar_value);
    for (i = p->i; i < limit; i+=8) {
        __m256 vec = _mm256_load_ps(matrix->rows+i);
        __m256 res = _mm256_mul_ps(vec,scalar);
      _mm256_store_ps(matrix->rows+i,res);
    }
}

__global__
void compute_scalar_matrix_mult(int n, float *matrix,float scalar){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i< n ; i+=stride){
        matrix[i] *= scalar;
    }
}

__global__
void compute_matrix_matrix_mult(int n, struct matrix *matrixA,struct matrix *matrixB,struct matrix *matrixC){
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

            matrixC->rows[k] += matrixA->rows[i] * matrixB->rows[j];

            j++;
            k++; // C na msm coluna de B
        }
    }
    return 1;
}


int avx_matrix_matrix_mult(struct matrix* matrixA, struct matrix* matrixB, struct matrix* matrixC) {
    if ((matrixA == NULL) || (matrixB == NULL) || (matrixC == NULL)) {
        printf("one of the matrix struct given is NULL pointer");
        return 0;
    }

    else if(matrixA->width != matrixB->height){
        printf("Matrices A and B are not compatible for multiplication");
        return 0;
    }
  
    int i;
    int j = 0; 
    int k = 0;
    int count = 0;

    int row_len_A = matrixA->height * matrixA->width;
    
    int aw = matrixA->width;
    int bw = matrixB->width;

    for (i = 0; i < row_len_A; i += 1) {

        if (i % aw == 0) j = 0;
        k = (i / aw) * bw;

        for(count = 0; count < bw; count += 8){
            __m256 vecA = _mm256_set1_ps(matrixA->rows[i]);
            __m256 vecB = _mm256_load_ps(matrixB->rows + j);
            __m256 vecC = _mm256_load_ps(matrixC->rows + k);
            __m256 res = _mm256_fmadd_ps(vecA,vecB,vecC);
            _mm256_store_ps(matrixC->rows + k,res);
            
            k += 8;
            j += 8;
        }
    }
    return 1;
}


void* compute_matrix_matrix(){
    struct matrix * matrixA = p->matrixA;
    struct matrix * matrixB = p->matrixB;
    struct matrix * matrixC = p->matrixC;

    int aw = matrixA->width;
    int bw = matrixB->width;
    
    int slice_start_a = (matrixA->height * matrixA->width / NUM_THREADS) * (p->par_count);
    int slice_end_a = (matrixA->height * matrixA->width / NUM_THREADS) * (p->par_count + 1);

    int slice_start_c = (matrixA->height * matrixB->width / NUM_THREADS) * (p->par_count);
    int slice_end_c = (matrixA->height * matrixB->width / NUM_THREADS) * (p->par_count + 1);

    int count = 0;
    int i = slice_start_a;
    int j = 0;
    int k = slice_start_c;

    for (i = slice_start_a; i < slice_end_a; i += 1){

        if (i % aw == 0) j = 0;
        k = (i / aw) * bw;

        for(count = 0; count < bw; count += 8) {

            __m256 vecA = _mm256_set1_ps(matrixA->rows[i]);
            __m256 vecB = _mm256_load_ps(matrixB->rows+j);
            __m256 vecC = _mm256_load_ps(matrixC->rows+k);
            __m256 res = _mm256_fmadd_ps(vecA,vecB,vecC);
            _mm256_store_ps(matrixC->rows+k,res);

            j += 8;
            k += 8;
        }
    }
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