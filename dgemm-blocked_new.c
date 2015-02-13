const char *dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

// Number of registers
#define NR 4

#include <stdlib.h>
#include <string.h>
/* #define min(a,b) (((a)<(b))?(a):(b)) */

/* non-branching min function */
#define min(x,y) ((y) ^ ((x ^ (y)) & (-(x < y)))) /* The awkward parenthesis supresses compiler warnings :/ */


static double buf1[BLOCK_SIZE *BLOCK_SIZE];
static double buf2[BLOCK_SIZE *BLOCK_SIZE];
static double buf3[BLOCK_SIZE *BLOCK_SIZE];

#define REGBLOCK 2
#define MAXD 800*800

inline static void do_block (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C)
{
    for(int j=0;j<N;j+=REGBLOCK)
        for(int k=0;k<K;k+=REGBLOCK){
            double *restrict BB=B+j*K+k;
            double b00=BB[0];
            double b10=BB[1];
            double b01=BB[K];
            double b11=BB[1+K];
            
            double *restrict AA=A+k*M;//lda;
            double *restrict CC=C+j*lda;
//            __builtin_assume_aligned(AA, 64);
//            __builtin_assume_aligned(CC, 64);
            int i=0;

            for(i=0;i<M;i++)
            {
                CC[i]+=b00*AA[i]+b10*AA[i+M];
                CC[i+lda]+=b01*AA[i]+b11*AA[i+M];
            }
        }
}


inline void copy_block(double *restrict A,double *restrict ABuf,int M,int K,int lda,int lda_)
{
    for (int kk = 0; kk < K; ++kk)
        memcpy(ABuf+lda_*kk,A+kk*lda,sizeof(double)*M);
}


void square_even_dgemm (int lda, double* __restrict__  A, double *  __restrict__  B,  double * __restrict__ C)
{
    static double *restrict Abuf = buf1;
    static double *restrict Bbuf = buf2;
    static double *restrict Cbuf = buf3;


    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
        int N = min (BLOCK_SIZE, lda - j);
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
            int K = min (BLOCK_SIZE, lda - k);
            copy_block(B+k+j*lda,Bbuf,K,N,lda,K);
            for (int i = 0; i < lda; i += BLOCK_SIZE)
            {
                int M = min (BLOCK_SIZE, lda - i);
                copy_block(A+i+k*lda,Abuf,M,K,lda,M);
                //copy_block(C+i+j*lda,Cbuf,M,N,lda,M);
                //memset(CBuf,0,sizeof(CBuf));
                do_block(lda, M, N, K, Abuf, Bbuf, C + i + j*lda);
            }
        }
    }
}

double A_[MAXD],B_[MAXD],C_[MAXD];


void square_dgemm (int lda, double* A, double *B,  double *C)
{
    if (lda%2==0)
        square_even_dgemm(lda,A,B,C);
    else
    {
        int l=lda+1;
        for(int j=0;j<lda;j++){
            memcpy(A_+j*l,A+j*lda,sizeof(double)*lda);
            memcpy(B_+j*l,B+j*lda,sizeof(double)*lda);
            memcpy(C_+j*l,C+j*lda,sizeof(double)*lda);
            A_[lda+j*l]=B_[lda+j*l]=C_[lda+j*l]=0;
        }
        for(int i=0;i<l;i++)
            A_[i+lda*l]=B_[i+lda*l]=C_[i+lda*l]=0;
        square_even_dgemm(l,A_,B_,C_);
        for(int j=0;j<lda;j++)
            memcpy(C+j*lda,C_+j*l,sizeof(double)*lda);
    }
}
