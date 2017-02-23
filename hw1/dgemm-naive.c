#include<stdlib.h>
#include<stdio.h>
const char* dgemm_desc = "Naive, three-loop dgemm.";

//reference git wiki https://github.com/flame/how-to-optimize-gemm/wiki
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define BLOCK_SIZE 128
#define A(i,j) A[(j)*n + (i)]
#define B(i,j) B[(j)*n + (i)]
#define C(i,j) C[(j)*n + (i)]
//void Mymulti(int, double*,  double*, double *)

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

void Mymulti(int n, double *A, double* B,double *C){
  /* So, this routine computes a 4x4 block of matrix A
            C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).
            C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).
            C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).
            C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).
      Notice that this routine is called with c = C( i, j ) in the
      previous routine, so these are actually the elements
            C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 )
            C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 )
            C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 )
            C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 )

      in the original matrix C
      In this version, we use registers for elements in the current row
      of B as well */

      int p;

      for ( p=0; p<n; p++ ){
        /* First row */
        C( 0, 0 ) += A( 0, p ) * B( p, 0 );
        C( 0, 1 ) += A( 0, p ) * B( p, 1 );
        C( 0, 2 ) += A( 0, p ) * B( p, 2 );
        C( 0, 3 ) += A( 0, p ) * B( p, 3 );

        /* Second row */
        C( 1, 0 ) += A( 1, p ) * B( p, 0 );
        C( 1, 1 ) += A( 1, p ) * B( p, 1 );
        C( 1, 2 ) += A( 1, p ) * B( p, 2 );
        C( 1, 3 ) += A( 1, p ) * B( p, 3 );

        /* Third row */
        C( 2, 0 ) += A( 2, p ) * B( p, 0 );
        C( 2, 1 ) += A( 2, p ) * B( p, 1 );
        C( 2, 2 ) += A( 2, p ) * B( p, 2 );
        C( 2, 3 ) += A( 2, p ) * B( p, 3 );

        /* Fourth row */
        C( 3, 0 ) += A( 3, p ) * B( p, 0 );
        C( 3, 1 ) += A( 3, p ) * B( p, 1 );
        C( 3, 2 ) += A( 3, p ) * B( p, 2 );
        C( 3, 3 ) += A( 3, p ) * B( p, 3 );
      }
}

void square_dgemm ( int n, double* A, double* B, double* C )
{

  int i = 0;
  int j = 0;
  //for each columns of C
    //printf("debig 1\n");
  printf("d0");
  for(j = 0; j < n; j+=4){
    //for each row of C
    for(i = 0 ; i < n; i+=4){
      //printf("debig 1\n");
      printf("d0");
      Mymulti(n, &A(i,0),&B(0,j),&C(i,j));
    }
  }
}
