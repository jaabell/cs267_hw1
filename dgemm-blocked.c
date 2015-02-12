const char *dgemm_desc = "Simple blocked dgemm.";

#include <emmintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>

//Knobs
#define BLOCK_SIZE 16
#define PAD_NEXT_MULTIPLE 16
#define ALIGNMENT_BOUNDARY 16
#define REGBLOCK 2

/* #define min(a,b) (((a)<(b))?(a):(b)) */

//For clarifying prefetch directives
#define PREFETCH_READ  0
#define PREFETCH_WRITE 1
#define PREFETCH_NO_TEMPORAL_LOCALITY 0
#define PREFETCH_LOW_TEMPORAL_LOCALITY 1
#define PREFETCH_MED_TEMPORAL_LOCALITY 2
#define PREFETCH_HIGH_TEMPORAL_LOCALITY 3


/* non-branching min function */
#define min(x,y) ((y) ^ ((x ^ (y)) & (-(x < y)))) /* The awkward parenthesis supresses compiler warnings :/ */

// static double buf2[BLOCK_SIZE *BLOCK_SIZE];

inline static void do_block (const int lda, const double *__restrict__ A, const double *__restrict__ B, double *__restrict__ C)
{
    for (int j = 0; j < BLOCK_SIZE; j += REGBLOCK)
        for (int k = 0; k < BLOCK_SIZE; k += REGBLOCK)
        {
            const double *__restrict__ BB = B + j * BLOCK_SIZE + k;
            const double *__restrict__ AA = A + k * BLOCK_SIZE;
            double *__restrict__ CC = C + j * BLOCK_SIZE;

            int i = 0;

            for (i = 0; i < BLOCK_SIZE; i += REGBLOCK)
            {

                __m128d c1 = _mm_load_pd( CC + i );
                __m128d c2 = _mm_load_pd( CC +  BLOCK_SIZE + i );

                for (int ll = 0; ll < REGBLOCK; ll++)
                {
                    __m128d a  = _mm_load_pd( AA +  i + BLOCK_SIZE * ll );

                    __m128d b1 = _mm_load1_pd( BB  + ll);
                    __m128d b2 = _mm_load1_pd( BB +  BLOCK_SIZE + ll );
                    __m128d ab1;
                    __m128d ab2;
                    ab1 = _mm_mul_pd( a, b1 );
                    ab2 = _mm_mul_pd( a, b2 );

                    c1 = _mm_add_pd( c1, ab1 );
                    c2 = _mm_add_pd( c2, ab2 );
                }
                _mm_storeu_pd( CC +  i, c1 );
                _mm_storeu_pd( CC +  BLOCK_SIZE + i, c2 );
            }
        }
}

void square_dgemm (int lda_, double *__restrict__  A_, double   *__restrict__  B_,  double *__restrict__ C_)
{
    // Pad to new size lda = next size that is multiple of PAD_NEXT_MULTIPLE
    int lda = (lda_ / PAD_NEXT_MULTIPLE + 1) * PAD_NEXT_MULTIPLE ;
    // int lda = nextpow2(lda_);

    //Pad matrices A and C with zeros, B will be buffered
    double *__restrict__ A = (double *) memalign ( ALIGNMENT_BOUNDARY, (size_t )lda * lda * sizeof(double));
    double *__restrict__ B = (double *) memalign ( ALIGNMENT_BOUNDARY, (size_t )lda * lda * sizeof(double));
    double *__restrict__ C = (double *) memalign ( ALIGNMENT_BOUNDARY, (size_t )lda * lda * sizeof(double));

    //Load A and B into padded arrays and initialize C to zero.
    int linear_index = 0;
    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
        for (int i = 0; i < lda; i += BLOCK_SIZE)
        {
            for (int jj = 0; jj < BLOCK_SIZE; ++jj)
                for (int ii = 0; ii < BLOCK_SIZE; ++ii)
                {
                    if (((i + ii ) < lda_) && ((j + jj) < lda_))
                    {
                        A[linear_index] = *(A_ + (i + ii) + (j + jj) * lda_ );
                        B[linear_index] = *(B_ + (i + ii) + (j + jj) * lda_ );
                    }
                    else
                    {
                        A[linear_index] = 0.0;
                        B[linear_index] = 0.0;
                    }
                    C[linear_index] = 0.0;
                    linear_index++;
                }
        }
    }

    //Main loop
    linear_index = 0;
    int JSTRIDE = (lda / BLOCK_SIZE) * BLOCK_SIZE;
    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
        // int N = min (BLOCK_SIZE, lda - j);
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
            __builtin_prefetch (B + k * BLOCK_SIZE + j * JSTRIDE, PREFETCH_READ, PREFETCH_HIGH_TEMPORAL_LOCALITY);
            //Multiply inner loop
            for (int i = 0; i < lda; i += BLOCK_SIZE)
            {
                // int M = min (BLOCK_SIZE, lda - i);
                __builtin_prefetch (A + i * BLOCK_SIZE + k * JSTRIDE, PREFETCH_READ, PREFETCH_NO_TEMPORAL_LOCALITY);
                __builtin_prefetch (C + i * BLOCK_SIZE + j * JSTRIDE, PREFETCH_WRITE, PREFETCH_NO_TEMPORAL_LOCALITY);
                do_block(lda,
                         A + i * BLOCK_SIZE + k * JSTRIDE,
                         B + k * BLOCK_SIZE + j * JSTRIDE,
                         C + i * BLOCK_SIZE + j * JSTRIDE);
            }
        }
    }

    //Copy padded C into output C_
    linear_index = 0;
    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
        for (int i = 0; i < lda; i += BLOCK_SIZE)
        {
            for (int jj = 0; jj < BLOCK_SIZE; ++jj)
                for (int ii = 0; ii < BLOCK_SIZE; ++ii)
                {
                    if (((i + ii ) < lda_) && ((j + jj) < lda_))
                    {
                        *(C_ + (i + ii) + (j + jj) * lda_ ) = C[linear_index];
                    }
                    linear_index++;
                }
        }
    }

    free(A);
    free(B);
    free(C);
}