#include <emmintrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>

void main()
{
	int n = 7;
	double *A = malloc(n*n*sizeof(double*));
	double *B = malloc(n*n*sizeof(double*));
	double *C = malloc(n*n*sizeof(double*));
	
	
	for(int i = 0; i < n*n; i++)
	{
		A[i] = (double) i;
		B[i] = (double) i;
		C[i] = 0;
	}
	
    for (int i = 0;  i < n;  i += 4) 
    {
        for (int j = 0;  j < n;  j++) 
        {
            __m256d c = {0,0,0,0};
            for (int k = 0;  k < n;  k++) 
            {
            
            	__m256d a = _mm256_loadu_pd(A+i+k*n);
            	__m256d b = _mm256_broadcast_sd(B+k+j*n);
                c = _mm256_add_pd(c, _mm256_mul_pd(a,b));
            }
            
            _mm256_storeu_pd(C+i+j*n, c);
        }
    }
		
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			printf("%f ",A[i*n+j]);
		}
		printf("\n");
	}

	printf("\n");	
	
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			printf("%f ",B[i*n+j]);
		}
		printf("\n");
	}

	printf("\n");

	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			printf("%f ", C[i*n+j]);
		}
		printf("\n");
	}

	free(A);
	free(B);
	free(C);
}
