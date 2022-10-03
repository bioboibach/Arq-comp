#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib.h"
#include <immintrin.h>
#include <pthread.h>

int NUM_THREADS = 1;

typedef struct __ParamsMM{
    struct matrix * matrixA;
    struct matrix * matrixB;
    struct matrix * matrixC;
    int i;
    int a_limit;
    int b_limit;
    int aw;
    int bw;
}ParamsMM;

void set_number_threads(int num_threads){
    NUM_THREADS = num_threads;
}

int scalar_matrix_mult(float scalar_value, struct matrix* matrix) {
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
            if((j+8) % bw == 0){
                i++;
                k = (i / aw) * bw;
            }
        }
    }
    return 1;
}

int p_matrix_matrix_mult(struct matrix* matrixA, struct matrix* matrixB, struct matrix* matrixC) {
    if ((matrixA == NULL) || (matrixB == NULL) || (matrixC == NULL)) {
        printf("one of the matrix struct given is NULL pointer");
        return 0;
    }

    else if(matrixA->width != matrixB->height){
        printf("Matrices A and B are not compatible for multiplication");
        return 0;
    }

    int a_limit = matrixA->height * matrixA->width;
    int b_limit = matrixB->height * matrixB->width;

    ParamsMM pm[NUM_THREADS];
    pthread_t threads[NUM_THREADS];

    for(int counter = 0;counter < NUM_THREADS;counter++){
        pm[counter].matrixA = matrixA;
        pm[counter].matrixB = matrixB;
        pm[counter].matrixC = matrixC;
        pm[counter].aw = matrixA->width;
        pm[counter].bw = matrixB->width;
        pm[counter].a_limit = (a_limit/(NUM_THREADS))*(counter+1);
        pm[counter].b_limit = b_limit;
        pm[counter].i = (a_limit/NUM_THREADS)*counter;
    }

    for(int counter = 0;counter < NUM_THREADS;counter++)
        pthread_create(&threads[counter],NULL,compute_matrix_matrix,(void *)&pm[counter]);
    
    for(int t=0; t < NUM_THREADS; t++)
        pthread_join(threads[t],NULL); 


    return 1;
}



void * compute_matrix_matrix(void * params){
    ParamsMM * p = (ParamsMM) params;
    struct matrix * matrixA = p->matrixA;
    struct matrix * matrixB = p->matrixB;
    struct matrix * matrixC = p->matrixC;
    
    int a_limit = p->a_limit;
    int b_limit = p->b_limit;
    int aw = p->aw;
    int bw = p->bw;
    int i,j,k;

    for (i = p->i; i < a_limit; k = (i / aw) * bw){
      
        for(j = 0; j < b_limit; j+=8){
            __m256 vecA = _mm256_set1_ps(matrixA->rows[i]);
            __m256 vecB = _mm256_load_ps(matrixB->rows+j);
            __m256 vecC = _mm256_load_ps(matrixC->rows+k);
            __m256 res = _mm256_fmadd_ps(vecA,vecB,vecC);
            _mm256_store_ps(matrixC->rows+k,res);
            k+=8;
            if((j+8) % bw == 0){
                i++;
                k = (i / aw) * bw;
            }
        }
    }
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
            if((j + 1) % bw == 0){
                i++;
                k = (i / aw) * bw;
            }
        }
    }
    return 1;
}