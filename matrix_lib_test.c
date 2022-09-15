#include "matrix_lib.h"
#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

typedef struct matrix Matrix;

float *matrix_from(char *from, int size) {
  float *matrix = aligned_alloc(32, sizeof(float) * size);
  FILE *f = fopen(from, "rb");
  fread(matrix, sizeof(float), size, f);

  fclose(f);
  return matrix;
}

void save_matrix(char *where, float *matrix, int size) {
  FILE *f = fopen(where, "w+b");
  int count;
  fwrite(matrix, sizeof(float), size, f);
  fclose(f);
  return;
}

void print_matrix(float *matrix_row, int row_size) {

  for (int count = 0; count < row_size; count++) {
    printf("%.2f ", matrix_row[count]);
    if ((count + 1) % 8 == 0)
      printf("\n");
    if (count == 256)
      break;
  }
  if (row_size > 256) {
    printf("Print lenght limit of 256 values reached\n\n");
  }
  return;
}

//optimize

int check_matrix(Matrix *correct_m, Matrix *questionable_m, int row_len){
for(int count = 0; count < row_len; count++) {
  if(correct_m->rows[count] != questionable_m->rows[count]) {
    printf("\n\n\nMatrix is wrong\n\n\n");
    return 0;
  }
}
  printf("\n\n\nMatrix is fine\n\n\n");
  return 1;
}

int main(int argc, char *argv[]) {
  for (int count = 2; count < 6; count++){
    //printf("%d\n", atoi(argv[count]));
    if (atoi(argv[count]) % 8 != 0){
      printf("Matrix nao multipla de 8, %d\n", (count -2));
      return 0;
    }
  }
  return 0;

  int a_row_len, b_row_len, c_row_len, error_count = 0;
  float scalar_value;
  Matrix *matrixA, *matrixB, *matrixC, *matrix_check;
  struct timeval start, stop, overall_t1, overall_t2;

  gettimeofday(&overall_t1, NULL);

  matrixA = malloc(sizeof(Matrix));
  matrixB = malloc(sizeof(Matrix));
  matrixC = malloc(sizeof(Matrix));
  matrix_check = malloc(sizeof(Matrix));

  scalar_value = atof(argv[1]);

  matrixA->height = atoi(argv[2]);
  matrixA->width = atoi(argv[3]);

  matrixB->height = atoi(argv[4]);
  matrixB->width = atoi(argv[5]);

  matrixC->height = matrixA->height;
  matrixC->width = matrixB->width;

  matrix_check->height = matrixA->height;
  matrix_check->width = matrixB->width;
  
  a_row_len = matrixA->height * matrixA->width;
  b_row_len = matrixB->height * matrixB->width;
  c_row_len = matrixC->height * matrixC->width;

  matrixA->rows = matrix_from(argv[6], a_row_len);
  matrixB->rows = matrix_from(argv[7], b_row_len);

  matrixC->rows = (float *)aligned_alloc(32,sizeof(float) * c_row_len);
  matrix_check->rows = (float *)aligned_alloc(32,sizeof(float) * c_row_len);
  __m256 zero = _mm256_set1_ps(0.0f);
  for (int count = 0; count < c_row_len; count+=8){
    _mm256_store_ps(matrixC->rows+count,zero);
    _mm256_store_ps(matrix_check->rows+count,zero);
  }

  // printing all matrices
  printf("====== Matrix A ======\n");
  print_matrix(matrixA->rows, a_row_len);
  printf("====== Matrix B ======\n");
  print_matrix(matrixB->rows, b_row_len);
  printf("====== Matrix C ======\n");
  print_matrix(matrixC->rows, c_row_len);

  // executing and timing scalar_matrix_mult
  printf("Executing scalar_matrix_mult . . . \n");

  gettimeofday(&start, NULL);
  error_count += scalar_matrix_mult(scalar_value, matrixA);
  gettimeofday(&stop, NULL);

  // printing scalar_matrix_mult result and time
  printf("====== Scalar * Matrix A ======\n");
  print_matrix(matrixA->rows, a_row_len);
  printf("scalar_matrix_mult elapsed time: %.4f ms\n",
  timedifference_msec(start, stop));

  // executing and timing matrix_matrix_mult
  printf("Executing matrix_matrix_mult . . .\n");

  gettimeofday(&start, NULL);
  error_count += matrix_matrix_mult(matrixA, matrixB, matrixC);
  gettimeofday(&stop, NULL);
  memo_opt_matrix_matrix_mult(matrixA, matrixB, matrix_check);

  // printing matrix_matrix_mult and time
  printf("====== MatrixA * MatrixB  ======\n");
  print_matrix(matrixC->rows, c_row_len);
  printf("matrix_matrix_mult elapsed time: %.4f ms\n",
  timedifference_msec(start, stop));

  save_matrix(argv[8], matrixA->rows, a_row_len);
  save_matrix(argv[9], matrixC->rows, c_row_len);
  
  error_count += check_matrix(matrixC, matrix_check, c_row_len);
  
  error_count = abs(error_count - 3);
  printf("====================\n Errors detected: %d\n====================\n", error_count);

  
  free(matrixA->rows);
  free(matrixB->rows);
  free(matrixC->rows);
  free(matrix_check->rows);
  free(matrixA);
  free(matrixB);
  free(matrixC);
  free(matrix_check);

  gettimeofday(&overall_t2, NULL);
  printf("Overall time: %.4f ms\n",
  timedifference_msec(overall_t1, overall_t2));

  return 0;
}
