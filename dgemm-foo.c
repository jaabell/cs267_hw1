const char *dgemm_desc = "Simple blocked dgemm.";


#define BLOCK_SIZEReg 4
#define BLOCK_SIZEL1 16 // Should be multiple of BLOCK_SIZEReg
#define BLOCK_SIZEL2 96


#define min(a,b) (((a)<(b))?(a):(b))

double AL1[BLOCK_SIZEL1 *BLOCK_SIZEL1] __attribute__((aligned(16)));
double BL1[BLOCK_SIZEL1 *BLOCK_SIZEL1] __attribute__((aligned(16)));
double AL2[BLOCK_SIZEL2 *BLOCK_SIZEL2] __attribute__((aligned(16)));
double BL2[BLOCK_SIZEL2 *BLOCK_SIZEL2] __attribute__((aligned(16)));
double CL1[BLOCK_SIZEL1 *BLOCK_SIZEL1] __attribute__((aligned(16)));


static inline void load_l1_block(int K, int M, int N, double *from, double *to)
{
    int j = 0;
    for (; j < N; j++)
    {
        int i = 0;
        for (; i < M; i++)
        {
            *(to + i + j * BLOCK_SIZEL1) = *(from + i + j * K);
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

static inline void save_l1_block(int K, int M, int N, double *from, double *to)
{
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < M; i++)
        {
            *(to + i + j * K) = *(from + i + j * BLOCK_SIZEL1);
        }
    }
}

static inline void load_l2_block(int K, int M, int N, double *from, double *to)
{
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < M; i++)
        {
            *(to + i + j * BLOCK_SIZEL2) = *(from + i + j * K);
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
            // =========================================
            //FOR BLOCK_SIZEReg == 2
            // =========================================

#if BLOCK_SIZEReg == 2
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
#endif
            // =========================================
            //FOR BLOCK_SIZEReg == 4
            // =========================================
#if BLOCK_SIZEReg == 4


            double *restrict BB = B + j * BLOCK_SIZEL1 + k;
            double *restrict AA = A + k * BLOCK_SIZEL1;
            double *restrict CC = C + j * BLOCK_SIZEL1;

            register double b00 = BB[0];
            register double b10 = BB[1];
            register double b20 = BB[2];
            register double b30 = BB[3];
            register double b01 = BB[0 + BLOCK_SIZEL1];
            register double b11 = BB[1 + BLOCK_SIZEL1];
            register double b21 = BB[2 + BLOCK_SIZEL1];
            register double b31 = BB[3 + BLOCK_SIZEL1];
            register double b02 = BB[0 + 2 * BLOCK_SIZEL1];
            register double b12 = BB[1 + 2 * BLOCK_SIZEL1];
            register double b22 = BB[2 + 2 * BLOCK_SIZEL1];
            register double b32 = BB[3 + 2 * BLOCK_SIZEL1];
            register double b03 = BB[0 + 3 * BLOCK_SIZEL1];
            register double b13 = BB[1 + 3 * BLOCK_SIZEL1];
            register double b23 = BB[2 + 3 * BLOCK_SIZEL1];
            register double b33 = BB[3 + 3 * BLOCK_SIZEL1];

            int i = 0;

            for (i = 0; i < BLOCK_SIZEL1; i++)
            {
                register double ai0 = *(AA + i);
                register double ai1 = *(AA + i + BLOCK_SIZEL1);
                register double ai2 = *(AA + i + 2 * BLOCK_SIZEL1);
                register double ai3 = *(AA + i + 3 * BLOCK_SIZEL1);

                *(CC + i) += ai0 * b00 + ai1 * b10 + ai2 * b20 + ai3 * b30;
                *(CC + i + BLOCK_SIZEL1) += ai0 * b01 + ai1 * b11 + ai2 * b21 + ai3 * b31;
                *(CC + i + 2 * BLOCK_SIZEL1) += ai0 * b02 + ai1 * b12 + ai2 * b22 + ai3 * b32;
                *(CC + i + 3 * BLOCK_SIZEL1) += ai0 * b03 + ai1 * b13 + ai2 * b23 + ai3 * b33;
            }
#endif

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
            load_l1_block(BLOCK_SIZEL2, K_, N_, B + k + j * BLOCK_SIZEL2, BL1);
            for (int i = 0; i < M; i += BLOCK_SIZEL1)
            {
                int M_ = min (BLOCK_SIZEL1, M - i);
                load_l1_block(BLOCK_SIZEL2, M_, K_, A + i + k * BLOCK_SIZEL2, AL1);
                load_l1_block(lda, M_, N_, C + i + j * lda, CL1);
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

