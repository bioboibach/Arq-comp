// Bernardo Bach - 1613231
// Eduardo Luna - 2111484


#include "matrix_lib.h"
#include "timer.h"
#include <stdio.h>
#include <stdlib.h>


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
    if (count == 255)
      break;
  }
  if (row_size > 256) {
    printf("=====  Print lenght limit of 256 values reached  =====\n");
  }
  return;
}

int check_matrix_result(Matrix *correct_m, Matrix *questionable_m, int row_len){
for(int count = 0; count < row_len; count++) {
  if(correct_m->rows[count] != questionable_m->rows[count]) {
    printf("Matrix is wrong\n\n");
    return 0;
  }
}
  printf("\nMatrix is correct!\n\n");
  return 1;
}

int check_input(int height_A, int width_A, int height_B, int width_B, int in_num_threads) {

  if (height_A % 8 != 0 || width_A % 8 != 0 || height_B % 8 != 0 || width_B % 8 != 0) {
    printf("Error: Invalid matrix size\n");
    return 0;
  }

  else if (width_A != height_B) {
    printf("Error: Matrices not compatible\n");
    return 0;
  }

  else if (height_A % in_num_threads != 0){
    printf("Error: Number of threads greater than matrix A height\n");
    return 0;
  }

  else return 1;

}

int main(int argc, char *argv[]) {

  if (!check_input(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]))) return 0;
  
  int a_row_len, b_row_len, c_row_len, error_count = 0;
  float scalar_value;
  Matrix *matrixA, *matrixB, *matrixC, *matrix_check;
  struct timeval start, stop, overall_t1, overall_t2;

  gettimeofday(&overall_t1, NULL);


//adicionar inidices 7,8 e modificar  6

  set_number_threads(atoi(argv[6]));
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

  matrixA->h_rows = matrix_from(argv[9], a_row_len);
  
  if(cudaMalloc(&matrixA->d_rows,sizeof(float) * a_row_len) != cudaSuccess)
    printf("Cuda Malloc on array A  returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
  
  matrixB->h_rows = matrix_from(argv[10], b_row_len);
  
  if(cudaMalloc(&matrixB->d_rows,sizeof(float) * b_row_len) != cudaSuccess)
    printf("Cuda Malloc on array A  returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
  
  matrixC->h_rows = (float *)malloc(sizeof(float) * c_row_len);
  
  if(cudaMalloc(&matrixC->d_rows,sizeof(float) * c_row_len) != cudaSuccess)
    printf("Cuda Malloc on array A  returned error %s (code %d)\n", cudaGetErrorString(cudaError), cudaError);
  
  matrix_check->h_rows = (float *)malloc(sizeof(float) * c_row_len);
  
  // printing all matrices
  printf("====== Matrix A ======\n");
  print_matrix(matrixA->rows_h, a_row_len);

  printf("\n====== Matrix B ======\n");
  print_matrix(matrixB->rows_h, b_row_len);

  printf("\n====== Matrix C ======\n");
  print_matrix(matrixC->rows_h, c_row_len);

  // executing and timing scalar_matrix_mult
  printf("\nExecuting scalar_matrix_mult . . . \n");

  gettimeofday(&start, NULL);
  error_count += scalar_matrix_mult(scalar_value, matrixA);
  gettimeofday(&stop, NULL);

  // printing scalar_matrix_mult result and time
  printf("====== Scalar * Matrix A ======\n");
  print_matrix(matrixA->h_rows, a_row_len);
  printf("\n===========================================\nscalar_matrix_mult elapsed time: %.4f ms\n===========================================\n\n\n",
  timedifference_msec(start, stop));

  // executing and timing matrix_matrix_mult
  printf("Executing matrix_matrix_mult . . .\n");

  gettimeofday(&start, NULL);
  error_count += matrix_matrix_mult(matrixA, matrixB, matrixC);
  gettimeofday(&stop, NULL);

  // printing matrix_matrix_mult and time
  printf("====== MatrixA * MatrixB  ======\n");
  print_matrix(matrixC->h_rows, c_row_len);
  printf("\n===============================================\nmatrix_matrix_mult elapsed time: %.4f ms\n===============================================\n\n\n",
  timedifference_msec(start, stop));

  save_matrix(argv[9], matrixA->h_rows, a_row_len);
  save_matrix(argv[10], matrixC->h_rows, c_row_len);
  
  printf("Checking matrix . . .\n");
  //avx_matrix_matrix_mult(matrixA, matrixB, matrix_check);
  matrix_matrix_mult(matrixA, matrixB, matrix_check);
  error_count += check_matrix_result(matrixC, matrix_check, c_row_len);
  
  error_count = abs(error_count - 3);
  printf("====================\n Errors detected: %d\n====================\n", error_count);

  free(matrixA->h_rows);
  free(matrixB->h_rows);
  free(matrixC->h_rows);
  free(matrix_check->h_rows);
  
  freeCuda(matrixA->d_rows);
  freeCuda(matrixB->d_rows);
  freeCuda(matrixC->d_rows);
  
  free(matrixA);
  free(matrixB);
  free(matrixC);
  free(matrix_check);

  gettimeofday(&overall_t2, NULL);
  printf("Overall time: %.4f ms\n", timedifference_msec(overall_t1, overall_t2));

  return 0;
}