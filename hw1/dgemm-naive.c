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

static void do_block(int n, int L2, int M2, int N2, double* A3, double* B3, double* C3){
  for(int i3 = 0; i3 < L2; ++i3){
    for(int j3 = 0; j3 < M2; j3++){
      for(int k3 = 0; k3 < N2; k3++){
        C3[ i3 + j3*n ] += A3[ k3 + i3*n ] * B3[ k3 + j3*n  ];

      }
    }
  }
}


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
        double *C3;
        double *B3;
        double *A3;

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
        int BLOCK_SIZE = 180;
        //int BLOCK_SIZE = 82;
        /** For each row i of A */
        for( int i = 0; i < n; i += BLOCK_SIZE )
        {
                /* For each column j of B */
                for( int k = 0; k < n; k += BLOCK_SIZE )
                {

                        for( int j = 0; j < n; j += BLOCK_SIZE )
                        {

                                int L = MIN( BLOCK_SIZE, ( n-i ) );
                                int M = MIN( BLOCK_SIZE, ( n-j ) );
                                int N = MIN( BLOCK_SIZE, ( n-k ) );

                                 C2 = C + i + j*n;
                                 A2 = AT + k + i*n;
                                 B2 = B + k + j*n;
                                 int BLOCK_SIZE_2 = 64;
                                for( int ii = 0; ii < L; ii += BLOCK_SIZE_2 )
                                {

                                        for( int jj = 0; jj < M; jj+= BLOCK_SIZE_2 )
                                        {
                                                /* Compute C(i,j) */
                                                //double cij = C2[ ii + jj*n ];

                                                for( int kk = 0; kk < N; kk+=BLOCK_SIZE_2 )
                                                {

                                                        int L2 = MIN(BLOCK_SIZE_2, (L- ii));
                                                        int M2 = MIN(BLOCK_SIZE_2, (M - jj));
                                                        int N2 = MIN(BLOCK_SIZE_2, (N - kk));
                                                        A3 = A2 + kk + ii*n;
                                                        B3 = B2 + kk + jj*n;
                                                        C3 = C2 + ii + jj*n;
                                                        for(int i3 = 0; i3 < L2; ++i3){
                                                          for(int j3 = 0; j3 < M2; j3++){
                                                            for(int k3 = 0; k3 < N2; k3++){
                                                              C3[ i3 + j3*n ] += A3[ k3 + i3*n ] * B3[ k3 + j3*n  ];

                                                            }
                                                          }
                                                        }
                                                        //do_block(n, L2, M2, N2, A2 + kk + ii*n, B2 + kk + jj*n, C2 + ii + jj*n);
                                                        //C2[ ii + jj*n ] += A2[ kk + ii*n ] * B2[ kk + jj*n ];
                                                }
                                                //C2[ ii + jj*n ] = cij;
                                        }
                                }
                        }
                }
        }

        free(AT);
}
