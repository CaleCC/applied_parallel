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
      printf("d1");
   v2df_t
     c_00_c_10_vreg,    c_01_c_11_vreg,    c_02_c_12_vreg,    c_03_c_13_vreg,
     c_20_c_30_vreg,    c_21_c_31_vreg,    c_22_c_32_vreg,    c_23_c_33_vreg,
     a_0p_a_1p_vreg,
     a_2p_a_3p_vreg,
     b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;
    printf("d2");
   double
     /* Point to the current elements in the four columns of B */
     *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

   b_p0_pntr = &B( 0, 0 );
   b_p1_pntr = &B( 0, 1 );
   b_p2_pntr = &B( 0, 2 );
   b_p3_pntr = &B( 0, 3 );

   c_00_c_10_vreg.v = _mm_setzero_pd();
   c_01_c_11_vreg.v = _mm_setzero_pd();
   c_02_c_12_vreg.v = _mm_setzero_pd();
   c_03_c_13_vreg.v = _mm_setzero_pd();
   c_20_c_30_vreg.v = _mm_setzero_pd();
   c_21_c_31_vreg.v = _mm_setzero_pd();
   c_22_c_32_vreg.v = _mm_setzero_pd();
   c_23_c_33_vreg.v = _mm_setzero_pd();
   printf("d3");
   for ( p=0; p<n; p++ ){
     a_0p_a_1p_vreg.v = _mm_load_pd( (double *) &A( 0, p ) );
     a_2p_a_3p_vreg.v = _mm_load_pd( (double *) &A( 2, p ) );

     b_p0_vreg.v = _mm_loaddup_pd( (double *) b_p0_pntr++ );   /* load and duplicate */
     b_p1_vreg.v = _mm_loaddup_pd( (double *) b_p1_pntr++ );   /* load and duplicate */
     b_p2_vreg.v = _mm_loaddup_pd( (double *) b_p2_pntr++ );   /* load and duplicate */
     b_p3_vreg.v = _mm_loaddup_pd( (double *) b_p3_pntr++ );   /* load and duplicate */

     /* First row and second rows */
     c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;
     c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;
     c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;
     c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;

     /* Third and fourth rows */
     c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;
     c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;
     c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;
     c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;
   }

   C( 0, 0 ) += c_00_c_10_vreg.d[0];  C( 0, 1 ) += c_01_c_11_vreg.d[0];
   C( 0, 2 ) += c_02_c_12_vreg.d[0];  C( 0, 3 ) += c_03_c_13_vreg.d[0];

   C( 1, 0 ) += c_00_c_10_vreg.d[1];  C( 1, 1 ) += c_01_c_11_vreg.d[1];
   C( 1, 2 ) += c_02_c_12_vreg.d[1];  C( 1, 3 ) += c_03_c_13_vreg.d[1];

   C( 2, 0 ) += c_20_c_30_vreg.d[0];  C( 2, 1 ) += c_21_c_31_vreg.d[0];
   C( 2, 2 ) += c_22_c_32_vreg.d[0];  C( 2, 3 ) += c_23_c_33_vreg.d[0];

   C( 3, 0 ) += c_20_c_30_vreg.d[1];  C( 3, 1 ) += c_21_c_31_vreg.d[1];
   C( 3, 2 ) += c_22_c_32_vreg.d[1];  C( 3, 3 ) += c_23_c_33_vreg.d[1];
}

void square_dgemm ( int n, double* A, double* B, double* C )
{

  int i = 0;
  int j = 0;
  //for each columns of C
    //printf("debig 1\n");
  for(j = 0; j < n; j+=4){
    //for each row of C
    for(i = 0 ; i < n; i+=4){
      //printf("debig 1\n");
      printf("d0");
      Mymulti(n, &A(i,0),&B(0,j),&C(i,j));
    }
  }
}
