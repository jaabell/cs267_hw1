const char *dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 86 
#endif


/* #define min(a,b) (((a)<(b))?(a):(b)) */

/* non-branching min function */
#define min(x,y) ((y) ^ ((x ^ (y)) & (-(x < y)))) /* The awkward parenthesis supresses compiler warnings :/ */

static double buf2[BLOCK_SIZE *BLOCK_SIZE];

#define REGBLOCK 2

inline static void do_block (int lda, int M, int N, int K, double *restrict A, double *restrict B, double *restrict C)
{
    for (int j = 0; j < N; j += REGBLOCK)
        for (int k = 0; k < K; k += REGBLOCK)
        {
            double *restrict BB = B + j * lda + k;
            double b00 = BB[0];
            double b10 = BB[1];
            double b01 = BB[lda];
            double b11 = BB[1 + lda];

            double *restrict AA = A + k * lda;
            double *restrict CC = C + j * lda;
            //            __builtin_assume_aligned(AA, 64);
            //            __builtin_assume_aligned(CC, 64);
            int i = 0;

            for (i = 0; i < M; i++)
            {
                CC[i] += b00 * AA[i] + b10 * AA[i + lda];
                CC[i + lda] += b01 * AA[i] + b11 * AA[i + lda];
            }

        }
}

void square_dgemm (int lda, double *__restrict__  A, double   *__restrict__  B,  double *__restrict__ C)
{
    static double *restrict Bbuf = buf2;


    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
        int N = min (BLOCK_SIZE, lda - j);
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
            int K = min (BLOCK_SIZE, lda - k);

            for (int jj = 0; jj < N; ++jj)
                for (int kk = 0; kk < K; ++kk)
                {
                    Bbuf[kk + K * jj] = B [k + kk + (j + jj) * lda ];
                }
            for (int i = 0; i < lda; i += BLOCK_SIZE)
            {
                int M = min (BLOCK_SIZE, lda - i);

                // =======
                //If block has even dimensions, use fast register multiply
                // =======
                if (N % 2 == 0 && K % 2 == 0)
                {
                    do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                }
                // =======
                // Else, use the fast multiply on whatever subblock it can be used, then add the missing terms the naive way
                // =======
                else if (N % 2 == 1 && K % 2 == 0) //know N is even.. can do unrollage
                {
                    do_block(lda, M, N - 1, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);

                    int jj = N - 1;
                    double *restrict CC = C + (i + (j + jj) * lda);
                    for (int kk = 0; kk < K; ++kk)
                    {
                        double *restrict AA = A + (i + (k + kk) * lda);
                        double bb = Bbuf[kk + K * jj];
                        int ii = 0;
                        for (ii = 0; ii < M; ii++)
                        {
                            CC[ii] += AA[ii] * bb;
                        }
                    }

                }
                else if (N % 2 == 0 && K % 2 == 1) //know N is even.. can do unrollage
                {
                    do_block(lda, M, N, K - 1, A + i + k * lda, B + k + j * lda, C + i + j * lda);

                    int kk = K - 1;
                    for (int jj = 0; jj < N; ++jj)
                    {
                        double *restrict CC = C + (i + (j + jj) * lda);

                        double *restrict AA = A + (i + (k + kk) * lda);
                        double bb = Bbuf[kk + K * jj];
                        int ii = 0;
                        for (ii = 0; ii < M; ii++)
                        {
                            CC[ii] += AA[ii] * bb;
                        }
                    }
                }
                else if (N % 2 == 1 && K % 2 == 1)
                {
                    do_block(lda, M, N - 1, K - 1, A + i + k * lda, B + k + j * lda, C + i + j * lda);

                    int jj = N - 1;
                    double *restrict CC = C + (i + (j + jj) * lda);
                    int ii = 0;
                    for (ii = 0; ii < M; ii++)
                    {
                        for (int kk = 0; kk < K; ++kk)
                        {
                            double *restrict AA = A + (i + (k + kk) * lda);
                            double bb = Bbuf[kk + K * jj];
                            CC[ii] += AA[ii] * bb;
                        }

                        int kk = K - 1;
                        for (int jj = 0; jj < N - 1; ++jj)
                        {
                            double *restrict CC = C + (i + (j + jj) * lda);
                            double *restrict AA = A + (i + (k + kk) * lda);
                            double bb = Bbuf[kk + K * jj];

                            CC[ii] += AA[ii] * bb;
                        }
                    }
                }
                // =======
            }
        }
    }
}
