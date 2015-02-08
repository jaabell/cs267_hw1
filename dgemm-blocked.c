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

#define REGBLOCK 2

inline static void do_block (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C)
{
    for(int j=0;j<N;j+=REGBLOCK)
        for(int k=0;k<K;k+=REGBLOCK){
            double *restrict BB=B+j*lda+k;
            double b00=BB[0];
            double b10=BB[1];
            double b01=BB[lda];
            double b11=BB[1+lda];
            
            double *restrict AA=A+k*lda;
            double *restrict CC=C+j*lda;
//            __builtin_assume_aligned(AA, 64);
//            __builtin_assume_aligned(CC, 64);
            int i=0;
            //for(i=0;i+3<M;i+=4)//REGBLOCK)
//            for(i=0;i<M;i++)

            for(i=0;i<M;i++)
            {
                CC[i]+=b00*AA[i]+b10*AA[i+lda];
                CC[i+lda]+=b01*AA[i]+b11*AA[i+lda];
                
               
                //int MM=min(M-i,REGBLOCK);
                //int NN=min(N-j,REGBLOCK);
                //int KK=min(K-k,REGBLOCK);
                
                /*double a0=AA[i];
                double a1=AA[i+lda];
                CC[i]+=a0*b00+a1*b10;
                CC[i+lda]+=a0*b01+a1*b11;
                a0=AA[i+1];
                a1=AA[i+1+lda];
                CC[i+1]+=a0*b00+a1*b10;
                CC[i+1+lda]+=a0*b01+a1*b11;
                a0=AA[i+2];
                a1=AA[i+2+lda];
                CC[i+2]+=a0*b00+a1*b10;
                CC[i+2+lda]+=a0*b01+a1*b11;
                a0=AA[i+3];
                a1=AA[i+3+lda];
                CC[i+3]+=a0*b00+a1*b10;
                CC[i+3+lda]+=a0*b01+a1*b11;*/
                /*for(int jj=0;jj<NN;jj++)
                {
                    for(int kk=0;kk<KK;kk++){
                        double b=BB[kk+jj*lda];
                        for(int ii=0;ii<MM;ii++)
                            CC[ii+jj*lda]+=AA[ii+kk*lda]*BB[kk+jj*lda];
                    }
                    }*/
            }
            for(;i<M;i++)
            {
                double a0=AA[i];
                double a1=AA[i+lda];
                CC[i]+=a0*b00+a1*b10;
                CC[i+lda]+=a0*b01+a1*b11;
            }
        }
}

void square_dgemm (int lda, double* __restrict__  A, double *  __restrict__  B,  double * __restrict__ C)
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

                if (N%2==0&&K%2==0)
                    do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
                else{
                // for (int ii = 0; ii < M; ++ii)
                //     for (int jj = 0; jj < N; ++jj)
                //     {
                //         Cbuf[ii + M * jj] = C [(i + ii) + (j + jj) * lda ];
                //     }
                // do_block inlined
                for (int jj = 0; jj < N; ++jj){
                    double *restrict CC=C+(i+(j+jj)*lda);
                    for (int kk = 0; kk < K; ++kk)
                    {
                        // double bkkjj = bkj_block[kk + jj * lda];
                        double *restrict AA=A+(i+(k+kk)*lda);
                        double bb=Bbuf[kk+K*jj];
                        int ii=0;
                        for (ii = 0; ii < M; ii++)
                        {
                            // double   *restrict ciijj = C + (i + ii) + (j + jj) * lda ;
                            // double aiikk = A [i + ii + (k + kk) * lda ];
                            // *ciijj +=  aiikk * bkkjj;
                            // *ciijj += Abuf[ii + M * kk] * Bbuf[kk + K * jj];
                            // Cbuf[ii + M * jj] += Abuf[ii + M * kk] * Bbuf[kk + K * jj];
                            // Cbuf[ii + M * jj] += aiikk * Bbuf[kk + K * jj];
                            CC[ii] +=AA[ii]*bb;/*
                            CC[ii+1]+=AA[ii+1]*bb;
                            CC[ii+2]+=AA[ii+2]*bb;
                            CC[ii+3]+=AA[ii+3]*bb;
                            CC[ii+4]+=AA[ii+4]*bb;
                            CC[ii+5]+=AA[ii+5]*bb;
                            CC[ii+6]+=AA[ii+6]*bb;
                            CC[ii+7]+=AA[ii+7]*bb;*/
                            //C [(i + ii) + (j + jj) * lda ] += A [i + ii + (k + kk) * lda ] * Bbuf[kk + K * jj];
                        }
                        for(;ii<M;ii++)
                            CC[ii]+=AA[ii]*bb;
                    }
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
