const char *dgemm_desc = "Simple blocked dgemm.";

#include <emmintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 86
#endif

#define ALIGNMENT_BOUNDARY 16

/* #define min(a,b) (((a)<(b))?(a):(b)) */

/* non-branching min function */
#define min(x,y) ((y) ^ ((x ^ (y)) & (-(x < y)))) /* The awkward parenthesis supresses compiler warnings :/ */

// static double buf2[BLOCK_SIZE *BLOCK_SIZE];

#define REGBLOCK 2

inline static void do_block (int lda, int M, int N, int K, double *restrict A, double *restrict B, double *restrict C)
{
    for (int j = 0; j < N; j += REGBLOCK)
        for (int k = 0; k < K; k += REGBLOCK)
        {
            // double *restrict BB = B + j * lda + k;
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

            for (i = 0; i < M; ++i)
            {
                // double a0 = AA[i];
                // double a1 = AA[i + lda];
                // double c1 = CC [i];
                // double c2 = CC [i + lda];
                CC[i]       += b00 * AA[i] + b10 * AA[i + lda];
                CC[i + lda] += b01 * AA[i] + b11 * AA[i + lda];
                // c1 = c1 + b00 * a0;
                // c1 = c1 + b10 * a1;
                // c2 = c2 + b01 * a0;
                // c2 = c2 + b11 * a1;
                // CC[i] = c1;
                // CC[i + lda] = c2;
                // __m128d c1 = _mm_loadu_pd( CC + 0 * lda + i ); //load unaligned block in C
                // __m128d c2 = _mm_loadu_pd( CC + 1 * lda + i );
                // for ( int ll = 0; ll < 2; ll++ )
                // {
                //     __m128d a = _mm_load_pd( AA + ll * lda + i ); //load aligned i-th column of A
                //     __m128d b1 = _mm_load1_pd( BB + ll + 0 * lda ); //load i-th row of B
                //     __m128d b2 = _mm_load1_pd( BB + ll + 1 * lda );
                //     c1 = _mm_add_pd( c1, _mm_mul_pd( a, b1 ) ); //rank-1 update
                //     c2 = _mm_add_pd( c2, _mm_mul_pd( a, b2 ) );
                // }
                // _mm_storeu_pd( CC + 0 * lda + i, c1 ); //store unaligned block in C
                // _mm_storeu_pd( CC + 1 * lda + i, c2 );
            }

            //


        }
}

void square_dgemm (int lda_, double *__restrict__  A_, double   *__restrict__  B_,  double *__restrict__ C_)
{
    int lda = (lda_ / ALIGNMENT_BOUNDARY + 1) * ALIGNMENT_BOUNDARY ;

    // printf("\nOld size %d, new size %d\n\n", lda_, lda);

    // printf("sizeof(void*) = %d\n", sizeof(void *));
    // printf("sizeof(double*) = %d\n", sizeof(double *));

    double *restrict A = (double *) memalign ( (size_t )ALIGNMENT_BOUNDARY, (size_t )lda * lda * sizeof(double));
    double *restrict B = (double *) memalign ( ALIGNMENT_BOUNDARY, lda * lda * sizeof(double));
    double *restrict C = (double *) memalign ( ALIGNMENT_BOUNDARY, lda * lda * sizeof(double));

    for (int i = 0; i < lda; ++i)
        for (int j = 0; j < lda; ++j)
        {
            if ((i < lda_) && (j < lda_))
            {
                A[i + lda * j] = A_[i + lda_ * j];
                B[i + lda * j] = B_[i + lda_ * j];
            }
            else
            {
                A[i + lda * j] = 0.0;
                B[i + lda * j] = 0.0;
            }
            C[i + lda * j] = 0.0;
        }

    // printf("Done resizing\n");

    // printf("Address of A at (%p)\n", A);
    // printf("Address of B at (%p)\n", B);
    // printf("Address of C at (%p)\n", C);

    // double *restrict Bbuf = buf2;

    // int M = BLOCK_SIZE;
    // int N = BLOCK_SIZE;
    // int K = BLOCK_SIZE;

    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
        int N = min (BLOCK_SIZE, lda - j);
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
            int K = min (BLOCK_SIZE, lda - k);

            // for (int jj = 0; jj < N; ++jj)
            //     for (int kk = 0; kk < K; ++kk)
            //     {
            //         Bbuf[kk + K * jj] = B [k + kk + (j + jj) * lda ];
            //     }
            for (int i = 0; i < lda; i += BLOCK_SIZE)
            {
                int M = min (BLOCK_SIZE, lda - i);

                // =======
                //If block has even dimensions, use fast register multiply
                // =======
                // if (N % 2 == 0 && K % 2 == 0)
                // {
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                // do_block(lda, M, N, K, A + i + k * lda, Bbuf, C + i + j * lda);
                // }
                // =======
                // // Else, use the fast multiply on whatever subblock it can be used, then add the missing terms the naive way
                // // =======
                // else if (N % 2 == 1 && K % 2 == 0) //know N is even.. can do unrollage
                // {
                //     do_block(lda, M, N - 1, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);

                //     int jj = N - 1;
                //     double *restrict CC = C + (i + (j + jj) * lda);
                //     for (int kk = 0; kk < K; ++kk)
                //     {
                //         double *restrict AA = A + (i + (k + kk) * lda);
                //         double bb = Bbuf[kk + K * jj];
                //         int ii = 0;
                //         for (ii = 0; ii < M; ii++)
                //         {
                //             CC[ii] += AA[ii] * bb;
                //         }
                //     }

                // }
                // else if (N % 2 == 0 && K % 2 == 1) //know N is even.. can do unrollage
                // {
                //     do_block(lda, M, N, K - 1, A + i + k * lda, B + k + j * lda, C + i + j * lda);

                //     int kk = K - 1;
                //     for (int jj = 0; jj < N; ++jj)
                //     {
                //         double *restrict CC = C + (i + (j + jj) * lda);

                //         double *restrict AA = A + (i + (k + kk) * lda);
                //         double bb = Bbuf[kk + K * jj];
                //         int ii = 0;
                //         for (ii = 0; ii < M; ii++)
                //         {
                //             CC[ii] += AA[ii] * bb;
                //         }
                //     }
                // }
                // else if (N % 2 == 1 && K % 2 == 1)
                // {
                //     do_block(lda, M, N - 1, K - 1, A + i + k * lda, B + k + j * lda, C + i + j * lda);

                //     int jj = N - 1;
                //     double *restrict CC = C + (i + (j + jj) * lda);
                //     int ii = 0;
                //     for (ii = 0; ii < M; ii++)
                //     {
                //         for (int kk = 0; kk < K; ++kk)
                //         {
                //             double *restrict AA = A + (i + (k + kk) * lda);
                //             double bb = Bbuf[kk + K * jj];
                //             CC[ii] += AA[ii] * bb;
                //         }

                //         int kk = K - 1;
                //         for (int jj = 0; jj < N - 1; ++jj)
                //         {
                //             double *restrict CC = C + (i + (j + jj) * lda);
                //             double *restrict AA = A + (i + (k + kk) * lda);
                //             double bb = Bbuf[kk + K * jj];

                //             CC[ii] += AA[ii] * bb;
                //         }
                //     }
                // }
                // =======
            }
        }
    }

    // printf("Outputting\n");
    for (int i = 0; i < lda_; ++i)
        for (int j = 0; j < lda_; ++j)
        {
            C_[i + lda_ * j] = C[i + lda * j];
        }

    // printf("Freeing address at (%p)\n", A);
    free((void *)A);
    // printf("Freeing address at (%p)\n", B);
    free((void *)B);
    // printf("Freeing address at (%p)\n", C);
    free((void *)C);
}
