#include <stdlib.h>
#include <math.h>

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

        // double *C2;
        // double *B2;
        // double *A2;

        for( int i = 0; i < n; ++i )
        {
                for( int j = 0; j< n; ++j )
                {
                        AT[ i*n + j ] = A[ i + n*j ];
                }
        }

        //int BLOCK_SIZE = ( int ) floor( sqrt( n ) );

        /** For each row i of A */
        for( int i = 0; i < n; ++i )
        {
                /* For each column j of B */
                for( int k = 0; k < n; ++k )
                {
                        /* Compute C(i,j) */
                        //double cij = C[ i + j*n ];

                        for( int j = 0; j < n; j++ )
                        {
                                C[ i + j*n ] += AT[ i*n + k ] * B[ k + j*n ];
                        }
                        //C[ i + j*n ] = cij;
                }
        }

        free(AT);
}
