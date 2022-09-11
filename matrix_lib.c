#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"
#include <immintrin.h>

int nuke_scalar_matrix_mult(float scalar_value, struct matrix* matrix) {
    if (matrix == NULL) {
        printf("matrix struct given is NULL pointer");
        return 0;
    }
    int i, limit = matrix->height * matrix->width;
    __m256 scalar = _mm256_set1_ps(scalar_value);
    for (i = 0; i < limit; i+=8) {
        __m256 vec = _mm256_load_ps(matrix->rows+i);
        __m256 res = _mm256_mul_ps(vec,scalar);
      _mm256_store_ps(matrix->rows+i,res);
    }
    return 1;
}

int scalar_matrix_mult(float scalar_value, struct matrix* matrix) {
    if (matrix == NULL) {
        printf("matrix struct given is NULL pointer");
        return 0;
    }
    int i, limit = matrix->height * matrix->width;
    for (i = 0; i < limit; i+=1) {
        matrix->rows[i] *= scalar_value;
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
  

    int i, j, k = 0;
    int a_limit = matrixA->height * matrixA->width;
    int b_limit = matrixB->height * matrixB->width;
    int aw = matrixA->width;
    int bw = matrixB->width; 
  
  
    for (i = 0; i < a_limit; k = (i / aw) * bw){
      
        for(j = 0; j < b_limit; j+=8){
            __m256 vecA = _mm256_set1_ps(matrixA->rows[i]);
            __m256 vecB = _mm256_load_ps(matrixB->rows+j);
            __m256 vecC = _mm256_load_ps(matrixC->rows+k);
            __m256 res = _mm256_fmadd_ps(vecA,vecB,vecC);
            _mm256_store_ps(matrixC->rows+k,res);
            k+=8;
            if((j+8) % bw == 0){ // se B fim da linha
                i++; // A anda
                k = (i / aw) * bw; // c = inicio da linha de a
            }
        }
    }
    return 1;
}


int memo_opt_matrix_matrix_mult(struct matrix* matrixA, struct matrix* matrixB, struct matrix* matrixC) {
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