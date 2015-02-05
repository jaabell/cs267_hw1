const char *dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

// Number of registers
#define NR 4

// #include <stdlib.h>

/* #define min(a,b) (((a)<(b))?(a):(b)) */

/* non-branching min function */
#define min(x,y) ((y) ^ ((x ^ (y)) & (-(x < y)))) /* The awkward parenthesis supresses compiler warnings :/ */


// static double buf1[BLOCK_SIZE *BLOCK_SIZE];
static double buf2[BLOCK_SIZE *BLOCK_SIZE];
// static double buf3[BLOCK_SIZE *BLOCK_SIZE];

void square_dgemm (int lda, double *A, double *B,  double *restrict C)
{
    // static double *restrict Abuf = buf1;
    static double *restrict Bbuf = buf2;
    // static double *restrict Cbuf = buf3;


    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
        int N = min (BLOCK_SIZE, lda - j);
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
            // double *bkj_block = B + k + j * lda;  //this had no significant impact
            int K = min (BLOCK_SIZE, lda - k);

            for (int jj = 0; jj < N; ++jj)
                for (int kk = 0; kk < K; ++kk)
                {
                    Bbuf[kk + K * jj] = B [k + kk + (j + jj) * lda ];
                }

            for (int i = 0; i < lda; i += BLOCK_SIZE)
            {
                int M = min (BLOCK_SIZE, lda - i);


                // for (int ii = 0; ii < M; ++ii)
                //     for (int jj = 0; jj < N; ++jj)
                //     {
                //         Cbuf[ii + M * jj] = C [(i + ii) + (j + jj) * lda ];
                //     }
                // do_block inlined
                for (int jj = 0; jj < N; ++jj){
                    double *CC=C+(i+(j+jj)*lda);
                    for (int kk = 0; kk < K; ++kk)
                    {
                        // double bkkjj = bkj_block[kk + jj * lda];
                        double *AA=A+(i+(k+kk)*lda);
                        double bb=Bbuf[kk+K*jj];
                        int ii=0;
                        for (ii = 0; ii+7 < M; ii+=8)
                        {
                            // double   *restrict ciijj = C + (i + ii) + (j + jj) * lda ;
                            // double aiikk = A [i + ii + (k + kk) * lda ];
                            // *ciijj +=  aiikk * bkkjj;
                            // *ciijj += Abuf[ii + M * kk] * Bbuf[kk + K * jj];
                            // Cbuf[ii + M * jj] += Abuf[ii + M * kk] * Bbuf[kk + K * jj];
                            // Cbuf[ii + M * jj] += aiikk * Bbuf[kk + K * jj];
                            CC[ii] +=AA[ii]*bb;
                            CC[ii+1]+=AA[ii+1]*bb;
                            CC[ii+2]+=AA[ii+2]*bb;
                            CC[ii+3]+=AA[ii+3]*bb;
                            CC[ii+4]+=AA[ii+4]*bb;
                            CC[ii+5]+=AA[ii+5]*bb;
                            CC[ii+6]+=AA[ii+6]*bb;
                            CC[ii+7]+=AA[ii+7]*bb;
                            //C [(i + ii) + (j + jj) * lda ] += A [i + ii + (k + kk) * lda ] * Bbuf[kk + K * jj];
                        }
                        for(;ii<M;ii++)
                            CC[ii]+=AA[ii]*bb;
                    }
                }
                // for (int ii = 0; ii < M; ++ii)
                //     for (int jj = 0; jj < N; ++jj)
                //     {
                //         C [(i + ii) + (j + jj) * lda ] = Cbuf[ii + M * jj];
                //     }

            }
        }
    }
}
