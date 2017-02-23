#include <emmintrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>

void main()
{
	int n = 4;
	double *A = malloc(n*n*sizeof(double*));	// A nxn matrix
	double *B = malloc(n*n*sizeof(double*));	// B nxn matrix
	double *C = malloc(n*n*sizeof(double*));	// C nxn matrix
	
	
	// Populate A and B matrix
	// Needed for example
	// A = {0,1,2,3
	//      4,5,6,7
	//	8,9,10,11
	//	12,13,14,15}
        // B = {0,1,2,3
        //      4,5,6,7
        //      8,9,10,11
        //      12,13,14,15}
	for(int i = 0; i < n*n; i++)
	{
		A[i] = (double) i;
		B[i] = (double) i;
		C[i] = 0;
	}
	

	__m256d c1 = _mm256_loadu_pd(C+0*n);	// load first row of C
	__m256d c2 = _mm256_loadu_pd(C+1*n);	// load second row of C
	__m256d c3 = _mm256_loadu_pd(C+2*n);	// load third row of C
	__m256d c4 = _mm256_loadu_pd(C+3*n);	// load fourth row of C


	for(int i = 0; i < n; i++)
	{
		__m256d a = _mm256_loadu_pd(A+i*n);		// Load ith row of A; a = <ai0, ai1, ai2, ai3>
		__m256d b1 = _mm256_broadcast_sd(B+i+0*n);	// create vector b1 = <B+i+0*n, B+i+0*n, B+i+0*n, B+i+0*n> 
		__m256d b2 = _mm256_broadcast_sd(B+i+1*n);	// create vector b2 = <B+i+1*n, B+i+1*n, B+i+1*n, B+i+1*n>
		__m256d b3 = _mm256_broadcast_sd(B+i+2*n);	// create vector b3 = <B+i+2*n, B+i+2*n, B+i+2*n, B+i+2*n>
		__m256d b4 = _mm256_broadcast_sd(B+i+3*n);	// create vector b4 = <B+i+3*n, B+i+3*n, B+i+3*n, B+i+3*n>

		c1 = _mm256_add_pd(c1, _mm256_mul_pd(a,b1));	// c1 = c1 + a * b1
		c2 = _mm256_add_pd(c2, _mm256_mul_pd(a,b2));	// c2 = c2 + a * b2
		c3 = _mm256_add_pd(c3, _mm256_mul_pd(a,b3));	// c3 = c3 + a * b3
		c4 = _mm256_add_pd(c4, _mm256_mul_pd(a,b4));	// c4 = c4 + a * b4
	}
	
	_mm256_storeu_pd(C+0*n, c1);
	_mm256_storeu_pd(C+1*n, c2);
	_mm256_storeu_pd(C+2*n, c3);
	_mm256_storeu_pd(C+3*n, c4);
	
	
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
