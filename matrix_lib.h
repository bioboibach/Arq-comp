struct matrix
{
    unsigned long int height;
    unsigned long int width;
    float *rows;
};

int scalar_matrix_mult(float scalar_value, struct matrix *matrix);

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC);

int avx_matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC);

void set_number_threads(int num_threads);