const char *dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

/* #define min(a,b) (((a)<(b))?(a):(b)) */
#define min(x,y) ((y) ^ ((x ^ (y)) & (-(x < y)))) /* The awkward parenthesis supresses compiler warnings :/ */


/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block (int lda, int M, int N, int K, double *A, double *B,  double *restrict C)
{
    /* For each row i of A */
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
        {
            double bkj = B[k + j * lda];
            for (int i = 0; i < M; ++i)
            {
                /* For each column j of B */

                /* Compute C(i,j) */
                // double *restrict cij = C + i + j * lda;
                double   *restrict cij = C + i + j * lda;
                double aik = A [ i + k * lda];

                *cij +=  aik * bkj;

                // C[i + j * lda] = cij;

            }
        }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm (int lda, double *A, double *B,  double *restrict C)
{
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
        /* Accumulate block dgemms into block of C */
    {
        int N = min (BLOCK_SIZE, lda - j);
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
            double *bkj_block = B + k + j * lda;  //this had no significant impact
            /* For each block-row of A */
            int K = min (BLOCK_SIZE, lda - k);
            for (int i = 0; i < lda; i += BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min (BLOCK_SIZE, lda - i);

                /* Perform individual block dgemm */
                // do_block(lda, M, N, K, A + i + k * lda, bkj_block, C + i + j * lda);
                for (int jj = 0; jj < N; ++jj)
                    for (int kk = 0; kk < K; ++kk)
                    {
                        double bkkjj = bkj_block[kk + jj * lda];
                        for (int ii = 0; ii < M; ++ii)
                        {
                            /* For each column jj of B */

                            /* Compute C(ii,jj) */
                            // double *restrict ciijj = C + ii + jj * lda;
                            double   *restrict ciijj = C + (i + ii) + (j + jj) * lda ;
                            double aiikk = A [i + ii + (k + kk) * lda ];

                            *ciijj +=  aiikk * bkkjj;

                            // C[ii + jj * lda] = cijj;

                        }
                    }
            }
        }
    }
}
