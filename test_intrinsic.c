#include <emmintrin.h>
#include <stdio.h>

int main(void)
{

    double C[4] = {0, 0, 0, 0};
    double A[4] = {1, 1, 1, 1};
    double B[4] = {2, 0, 0, 2};

    __m128d c1 = _mm_loadu_pd( C + 0 * 2 ); //load unaligned block in C
    __m128d c2 = _mm_loadu_pd( C + 1 * 2 );
    for ( int i = 0; i < 2; i++ )
    {
        __m128d a = _mm_load_pd( A + i * 2 ); //load aligned i-th column of A
        __m128d b1 = _mm_load1_pd( B + i + 0 * 2 ); //load i-th row of B
        __m128d b2 = _mm_load1_pd( B + i + 1 * 2 );
        c1 = _mm_add_pd( c1, _mm_mul_pd( a, b1 ) ); //rank-1 update
        c2 = _mm_add_pd( c2, _mm_mul_pd( a, b2 ) );
    }
    _mm_storeu_pd( C + 0 * 2, c1 ); //store unaligned block in C
    _mm_storeu_pd( C + 1 * 2, c2 );

    for (int i = 0; i < 4; i++)
    {
        printf("A[%d] = %g \n", i, A[i]);
    }
    for (int i = 0; i < 4; i++)
    {
        printf("B[%d] = %g \n", i, B[i]);
    }
    for (int i = 0; i < 4; i++)
    {
        printf("C[%d] = %g \n", i, C[i]);
    }

    return 0;
}