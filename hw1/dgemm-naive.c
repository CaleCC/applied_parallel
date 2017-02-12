#include <stdlib.h>
#include <math.h>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/*
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines

CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

const char* dgemm_desc = "Naive, three-loop dgemm.";


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm ( int n, double* A, double* B, double* C )
{

        double * AT = malloc( n*n*sizeof( double* ) );

        double *C2;
        double *B2;
        double *A2;

        /**
        *       Transpose the matrix
        *
        *       [ 1 2 3 ]    [ 1 4 7 ]
        *       [ 4 5 6 ] -> [ 2 5 8 ]
        *       [ 7 8 9 ]    [ 3 6 9 ]
        **/
        for( int i = 0; i < n; ++i )
        {
                for( int j = 0; j < n; ++j )
                {
                        AT[ i*n + j ] = A[ i + n*j ];
                }
        }

        // Calculate appropriate block size
        //int BLOCK_SIZE = ( int ) floor( sqrt( n ) );
        int BLOCK_SIZE = 82;
        /** For each row i of A */
        for( int i = 0; i < n; i += BLOCK_SIZE )
        {
                /* For each column j of B */
                for( int j = 0; j < n; j += BLOCK_SIZE )
                {

                        for( int k = 0; k < n; k += BLOCK_SIZE )
                        {

                                int L = MIN( BLOCK_SIZE, ( n-i ) );
                                int M = MIN( BLOCK_SIZE, ( n-j ) );
                                int N = MIN( BLOCK_SIZE, ( n-k ) );

                                C2 = C + i + j*n;
                                A2 = AT + k + i*n;
                                B2 = B + k + j*n;

                                for( int ii = 0; ii < L; ++ii )
                                {

                                        for( int jj = 0; jj < M; ++jj )
                                        {
                                                /* Compute C(i,j) */
                                                double cij = C2[ ii + jj*n ];

                                                for( int kk = 0; kk < N; ++kk )
                                                {
                                                        cij += A2[ ii + kk*n ] * B2[ kk + jj*n ];
                                                }
                                                C2[ ii + jj*n ] = cij;
                                        }
                                }
                        }
                }
        }

        free(AT);
}
