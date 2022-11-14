struct matrix {
    unsigned long int height;
    unsigned long int width;
    float *h_rows;
    float *d_rows;
    int alloc_mode;
};

int scalar_matrix_mult(float scalar_value, struct matrix *matrix);

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB, struct matrix *matrixC);

int set_grid_size(int threads_per_block, int max_blocks_per_grid);

int memo_opt_matrix_matrix_mult(struct matrix* matrixA, struct matrix* matrixB, struct matrix* matrixC);