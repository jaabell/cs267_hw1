const char *dgemm_desc = "Simple blocked dgemm.";


#define BLOCK_SIZEL1 32
#define BLOCK_SIZEL2 128
#define BLOCK_SIZEReg 2


#define min(a,b) (((a)<(b))?(a):(b))

double AL1[BLOCK_SIZEL1 *BLOCK_SIZEL1] __attribute__((aligned(16)));
double BL1[BLOCK_SIZEL1 *BLOCK_SIZEL1] __attribute__((aligned(16)));
double AL2[BLOCK_SIZEL2 *BLOCK_SIZEL2] __attribute__((aligned(16)));
double BL2[BLOCK_SIZEL2 *BLOCK_SIZEL2] __attribute__((aligned(16)));
double CL1[BLOCK_SIZEL1 *BLOCK_SIZEL1] __attribute__((aligned(16)));


static inline void load_l1_block(int M, int N, double *from, double *to)
{
    int j = 0;
    for (; j < N; j++)
    {
        int i = 0;
        for (; i < M; i++)
        {
            *(to + i + j * BLOCK_SIZEL1) = *(from + i + j * BLOCK_SIZEL2);
        }
        for (; i < BLOCK_SIZEL1; i++)
        {
            *(to + i + j * BLOCK_SIZEL1) = 0.0;
        }
    }
    for (; j < BLOCK_SIZEL1; j++)
        for (int i = 0; i < BLOCK_SIZEL1; i++)
        {
            *(to + i + j * BLOCK_SIZEL1) = 0.0;
        }
}

static inline void load_l1_block2(int lda, int M, int N, double *from, double *to)
{
    int j = 0;
    for (; j < N; j++)
    {
        int i = 0;
        for (; i < M; i++)
        {
            *(to + i + j * BLOCK_SIZEL1) = *(from + i + j * lda);
        }
        for (; i < BLOCK_SIZEL1; i++)
        {
            *(to + i + j * BLOCK_SIZEL1) = 0.0;
        }
    }
    for (; j < BLOCK_SIZEL1; j++)
        for (int i = 0; i < BLOCK_SIZEL1; i++)
        {
            *(to + i + j * BLOCK_SIZEL1) = 0.0;
        }
}

static inline void save_l1_block(int lda, int M, int N, double *from, double *to)
{
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < M; i++)
        {
            *(to + i + j * lda) = *(from + i + j * BLOCK_SIZEL1);
        }
    }
}

static inline void load_l2_block(int lda, int M, int N, double *from, double *to)
{
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < M; i++)
        {
            *(to + i + j * BLOCK_SIZEL2) = *(from + i + j * lda);
        }
    }
}





static inline void do_block_l1 (int lda, int M, int N, int K, double *A, double *B, double *C)
{
    /* For each row i of A */
    for (int j = 0; j < N; j += BLOCK_SIZEReg)
        /* For each column j of B */
        for (int k = 0; k < K; k += BLOCK_SIZEReg)
        {
            double *restrict BB = B + j * BLOCK_SIZEL1 + k;
            register double b00 = BB[0];
            register double b10 = BB[1];
            register double b01 = BB[BLOCK_SIZEL1];
            register double b11 = BB[1 + BLOCK_SIZEL1];

            double *restrict AA = A + k * BLOCK_SIZEL1;
            double *restrict CC = C + j * BLOCK_SIZEL1;
            //            __builtin_assume_aligned(AA, 64);
            //            __builtin_assume_aligned(CC, 64);
            int i = 0;

            for (i = 0; i < BLOCK_SIZEL1; i++)
            {
                register double a0 = *(AA + i);
                register double a1 = *(AA + i + BLOCK_SIZEL1);
                *(CC + i) += b00 * a0 + b10 * a1;
                *(CC + i + BLOCK_SIZEL1) += b01 * a0 + b11 * a1;
            }
            /* Compute C(i,j) */
            // double cij = C[i + j * lda];
            // for (int i = 0; i < M; ++i)
            // {
            //     cij += A[i + k * BLOCK_SIZEL1] * B[k + j * BLOCK_SIZEL1];
            // }
            // C[i + j * lda] = cij;
        }
}





static inline void do_block_l2 (int lda, int M, int N, int K, double *A, double *B, double *C)
{
    for (int j = 0; j < N; j += BLOCK_SIZEL1)
    {
        int N_ = min (BLOCK_SIZEL1, N - j);
        for (int k = 0; k < K; k += BLOCK_SIZEL1)
        {
            int K_ = min (BLOCK_SIZEL1, K - k);
            load_l1_block(K_, N_, B + k + j * BLOCK_SIZEL2, BL1);
            for (int i = 0; i < M; i += BLOCK_SIZEL1)
            {
                int M_ = min (BLOCK_SIZEL1, M - i);
                load_l1_block(M_, K_, A + i + k * BLOCK_SIZEL2, AL1);
                load_l1_block2(lda, M_, N_, C + i + j * lda, CL1);
                do_block_l1(lda, M_, N_, K_, AL1, BL1, CL1);
                save_l1_block(lda, M_, N_, CL1, C + i + j * lda);
            }
        }
    }
}



void square_dgemm (int lda, double *A, double *B, double *C)
{
    for (int j = 0; j < lda; j += BLOCK_SIZEL2)
    {
        int N = min (BLOCK_SIZEL2, lda - j);
        for (int k = 0; k < lda; k += BLOCK_SIZEL2)
        {
            int K = min (BLOCK_SIZEL2, lda - k);
            load_l2_block(lda, K, N, B + k + j * lda, BL2);
            for (int i = 0; i < lda; i += BLOCK_SIZEL2)
            {
                int M = min (BLOCK_SIZEL2, lda - i);
                load_l2_block(lda, M, K, A + i + k * lda, AL2);
                do_block_l2(lda, M, N, K, AL2, BL2, C + i + j * lda);
            }
        }
    }
}

