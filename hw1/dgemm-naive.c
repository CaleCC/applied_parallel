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
typedef union
{
  __m128d v;
  double d[2];
} v2df_t;

void Mymulti(int n, double *A, double* B,double *C){
  int p;
 register double
   /* hold contributions to
      C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ) */
      c_00_reg,   c_01_reg,   c_02_reg,   c_03_reg,
   /* holds A( 0, p ) */
      a_0p_reg;
 double
   /* Point to the current elements in the four columns of B */
   *bp0_pntr, *bp1_pntr, *bp2_pntr, *bp3_pntr;

 bp0_pntr = &B( 0, 0 );
 bp1_pntr = &B( 0, 1 );
 bp2_pntr = &B( 0, 2 );
 bp3_pntr = &B( 0, 3 );

 c_00_reg = 0.0;
 c_01_reg = 0.0;
 c_02_reg = 0.0;
 c_03_reg = 0.0;

 for ( p=0; p<n; p+=4 ){
   a_0p_reg = A( 0, p );

   c_00_reg += a_0p_reg * *bp0_pntr++;
   c_01_reg += a_0p_reg * *bp1_pntr++;
   c_02_reg += a_0p_reg * *bp2_pntr++;
   c_03_reg += a_0p_reg * *bp3_pntr++;

   a_0p_reg = A( 0, p+1 );

   c_00_reg += a_0p_reg * *bp0_pntr++;
   c_01_reg += a_0p_reg * *bp1_pntr++;
   c_02_reg += a_0p_reg * *bp2_pntr++;
   c_03_reg += a_0p_reg * *bp3_pntr++;

   a_0p_reg = A( 0, p+2 );

   c_00_reg += a_0p_reg * *bp0_pntr++;
   c_01_reg += a_0p_reg * *bp1_pntr++;
   c_02_reg += a_0p_reg * *bp2_pntr++;
   c_03_reg += a_0p_reg * *bp3_pntr++;

   a_0p_reg = A( 0, p+3 );

   c_00_reg += a_0p_reg * *bp0_pntr++;
   c_01_reg += a_0p_reg * *bp1_pntr++;
   c_02_reg += a_0p_reg * *bp2_pntr++;
   c_03_reg += a_0p_reg * *bp3_pntr++;
 }

 C( 0, 0 ) += c_00_reg;
 C( 0, 1 ) += c_01_reg;
 C( 0, 2 ) += c_02_reg;
 C( 0, 3 ) += c_03_reg;
}


#define X(i) x[ (i)*n ]

void AddDot( int n, double *x,   double *y, double *gamma )
{
  /* compute gamma := x' * y + gamma with vectors x and y of length n.
     Here x starts at location x with increment (stride) incx and y starts at location y and has (implicit) stride of 1.
  */

  int p;

  for ( p=0; p<n; p++ ){
    *gamma += X( p ) * y[ p ];
  }
}

void square_dgemm ( int n, double* A, double* B, double* C )
{

  int i = 0;
  int j = 0;
  //for each columns of C
    //printf("debig 1\n");
  int p = n-n%4;
  for(j = 0; j < p; j+=4){
    //for each row of C
    for(i = 0 ; i < p; i+=4){
      Mymulti(n, &A(i,0),&B(0,j),&C(i,j));
    }
  }


  for(j = p; j <n; j++){
    for(i = 0; i < p; i++){
      AddDot(n, &A(i,0),&B(0,j),&C(i,j));
    }
  }


  for(j = 0; j <n; j++){
    for(i = p; i < n; i++){
      AddDot(n, &A(i,0),&B(0,j),&C(i,j));
    }
  }

}
