const char *dgemm_desc = "Simple blocked dgemm.";

#include <emmintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>

//Knobs
#define BLOCK_SIZE 72
#define PAD_NEXT_MULTIPLE 4
#define ALIGNMENT_BOUNDARY 16
#define REGBLOCK 2

/* #define min(a,b) (((a)<(b))?(a):(b)) */

/* non-branching min function */
#define min(x,y) ((y) ^ ((x ^ (y)) & (-(x < y)))) /* The awkward parenthesis supresses compiler warnings :/ */

static double buf2[BLOCK_SIZE *BLOCK_SIZE];

inline static void do_block (int lda, int M, int N, int K, double *restrict A, double *restrict B, double *restrict C)
{
    for (int j = 0; j < N; j += REGBLOCK)
        for (int k = 0; k < K; k += REGBLOCK)
        {
            double *restrict BB = B + j * K + k;
            double *restrict AA = A + k * lda;
            double *restrict CC = C + j * lda;

            int i = 0;

            for (i = 0; i < M; i += 2)
            {
                __m128d c1 = _mm_load_pd( CC + i );
                __m128d c2 = _mm_load_pd( CC +  lda + i );
                __m128d a  = _mm_load_pd( AA +  i );
                __m128d b1 = _mm_load1_pd( BB  );
                __m128d b2 = _mm_load1_pd( BB +  K );
                c1 = _mm_add_pd( c1, _mm_mul_pd( a, b1 ) );
                c2 = _mm_add_pd( c2, _mm_mul_pd( a, b2 ) );
                a  = _mm_load_pd( AA +  lda + i );
                b1 = _mm_load1_pd( BB + 1 );
                b2 = _mm_load1_pd( BB + 1 + K );
                c1 = _mm_add_pd( c1, _mm_mul_pd( a, b1 ) );
                c2 = _mm_add_pd( c2, _mm_mul_pd( a, b2 ) );
                _mm_storeu_pd( CC +  i, c1 );
                _mm_storeu_pd( CC +  lda + i, c2 );
            }
        }
}

void square_dgemm (int lda_, double *__restrict__  A_, double   *__restrict__  B_,  double *__restrict__ C_)
{
    // Pad to new size lda = next size that is multiple of PAD_NEXT_MULTIPLE
    int lda = (lda_ / PAD_NEXT_MULTIPLE + 1) * PAD_NEXT_MULTIPLE ;
    // int lda = nextpow2(lda_);

    //Pad matrices A and C with zeros, B will be buffered
    double *restrict A = (double *) memalign ( (size_t )ALIGNMENT_BOUNDARY, (size_t )lda * lda * sizeof(double));
    double *restrict C = (double *) memalign ( ALIGNMENT_BOUNDARY, lda * lda * sizeof(double));
    double *restrict B = B_;
    for (int i = 0; i < lda; ++i)
        for (int j = 0; j < lda; ++j)
        {
            if ((i < lda_) && (j < lda_))
            {
                A[i + lda * j] = A_[i + lda_ * j];
            }
            else
            {
                A[i + lda * j] = 0.0;
            }
            C[i + lda * j] = 0.0;
        }


    double *restrict Bbuf = buf2;

    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
        int N = min (BLOCK_SIZE, lda - j);
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
            int K = min (BLOCK_SIZE, lda - k);

            //Put a block of B into Buffer
            for (int jj = 0; jj < N; ++jj)
                for (int kk = 0; kk < K; ++kk)
                {
                    Bbuf[kk + K * jj] = B [k + kk + (j + jj) * lda_ ];
                }

            //Multiply inner loop
            for (int i = 0; i < lda; i += BLOCK_SIZE)
            {
                int M = min (BLOCK_SIZE, lda - i);

                do_block(lda, M, N, K, A + i + k * lda, Bbuf, C + i + j * lda);
            }
        }
    }

    //Copy padded C into output C
    for (int i = 0; i < lda_; ++i)
        for (int j = 0; j < lda_; ++j)
        {
            C_[i + lda_ * j] = C[i + lda * j];
        }

    // De-allocate memory :/ this is slow
    free((void *)A);
    free((void *)C);
}
