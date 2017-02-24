/*
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= icc

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines

CC = cc
OPT = -O3 -xhost
CFLAGS = -Wall -std=gnu99  -g -msse3 -fast -mavx -unroll-aggressive $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/



#include<stdlib.h>
#include<stdio.h>
const char* dgemm_desc = "Naive, three-loop dgemm.";

//reference git wiki https://github.com/flame/how-to-optimize-gemm/wiki
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define A(i,j) A[(j)*lda + (i)]
#define B(i,j) B[(j)*ldb + (i)]
#define C(i,j) C[(j)*ldc + (i)]
//void Mymulti(int, double*,  double*, double *)

#include <emmintrin.h>
#include <immintrin.h>

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2

typedef union
{
  //__m128d vector;
  __m256d vector2;
  //double data[2];
  double data2[4];
} v2df_t;

void Mymulti(int k, double *A,int lda, double* B,int  ldb,
                                       double *C, int ldc){

   int p;
   v2df_t
    //  c_00_c_10_vreg,    c_01_c_11_vreg,    c_02_c_12_vreg,    c_03_c_13_vreg,
    //  c_20_c_30_vreg,    c_21_c_31_vreg,    c_22_c_32_vreg,    c_23_c_33_vreg,

    //  a_0p_a_1p_vreg,
    //  a_2p_a_3p_vreg,
     //
    //
    //  b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

     c_00_10_20_30_vreg,
     c_02_12_22_32_vreg,
     c_03_13_23_33_vreg,
     c_01_11_21_31_vreg,
     a_0p_1p_2p_3p_vreg,
     b_p0_p1_vreg,
     b_p2_p3_vreg;


  //  c_00_c_10_vreg.vector = _mm_setzero_pd();
  //  c_01_c_11_vreg.vector = _mm_setzero_pd();
  //  c_02_c_12_vreg.vector = _mm_setzero_pd();
  //  c_03_c_13_vreg.vector = _mm_setzero_pd();
  //  c_20_c_30_vreg.vector = _mm_setzero_pd();
  //  c_21_c_31_vreg.vector = _mm_setzero_pd();
  //  c_22_c_32_vreg.vector = _mm_setzero_pd();
  //  c_23_c_33_vreg.vector = _mm_setzero_pd();


   c_00_10_20_30_vreg.vector2 = _mm256_setzero_pd();
   c_02_12_22_32_vreg.vector2 = _mm256_setzero_pd();
   c_03_13_23_33_vreg.vector2 = _mm256_setzero_pd();
   c_01_11_21_31_vreg.vector2 = _mm256_setzero_pd();



   //c_00_c_10_vreg.vector = _mm256_setzero_pd();
   //c_01_c_11_vreg.vector = _mm256_setzero_pd();
   //c_02_c_12_vreg.vector = _mm256_setzero_pd();
   //c_03_c_13_vreg.vector = _mm256_setzero_pd();
   //c_20_c_30_vreg.vector = _mm256_setzero_pd();
   //c_21_c_31_vreg.vector = _mm256_setzero_pd();
   //c_22_c_32_vreg.vector = _mm256_setzero_pd();
   //c_23_c_33_vreg.vector = _mm256_setzero_pd();

   for ( p=0; p<k; p++ ){
     a_0p_1p_2p_3p_vreg.vector2 =  _mm256_load_pd( (double *) A );
     //a_0p_a_1p_vreg.vector = _mm256_load_pd( (double *) A );
     //a_2p_a_3p_vreg.vector = _mm256_load_pd( (double *) ( A+2 ) );
    //  a_0p_a_1p_vreg.vector = _mm_load_pd( (double *) A );
    //  a_2p_a_3p_vreg.vector = _mm_load_pd( (double *) ( A+2 ) );
     A += 4;

    //  b_p0_vreg.vector = _mm_loaddup_pd( (double *) B );	   /* load and duplicate */
    //  b_p1_vreg.vector = _mm_loaddup_pd( (double *) (B+1) );   /* load and duplicate */
    //  b_p2_vreg.vector = _mm_loaddup_pd( (double *) (B+2) );   /* load and duplicate */
    //  b_p3_vreg.vector = _mm_loaddup_pd( (double *) (B+3) );   /* load and duplicate */

    //  b_p0_vreg.vector = _mm256_broadcast_sd( (double *) B );	   /* load and duplicate */
    //  b_p1_vreg.vector = _mm256_broadcast_sd( (double *) (B+1) );   /* load and duplicate */
    //  b_p2_vreg.vector = _mm256_broadcast_sd( (double *) (B+2) );   /* load and duplicate */
    //  b_p3_vreg.vector = _mm256_broadcast_sd( (double *) (B+3) );   /* load and duplicate */
    b_p0_p1_vreg.vector2 = _mm256_broadcast_sd((double *) B);
    b_p2_p3_vreg.vector2 = _mm256_broadcast_sd((double *) (B+2));

     B += 4;
     /*first row and second rows*/
     c_00_10_20_30_vreg.vector2 += a_0p_1p_2p_3p_vreg.vector2 * b_p0_p1_vreg.vector2;
     c_01_11_21_31_vreg.vector2 += a_0p_1p_2p_3p_vreg.vector2 * b_p2_p3_vreg.vector2;

     /*third and forth rows*/
     c_02_12_22_32_vreg.vector2 += a_0p_1p_2p_3p_vreg.vector2 * b_p0_p1_vreg.vector2;
     c_03_13_23_33_vreg.vector2 += a_0p_1p_2p_3p_vreg.vector2 * b_p2_p3_vreg.vector2;




    //  /* First row and second rows */
    //  c_00_c_10_vreg.vector += a_0p_a_1p_vreg.vector * b_p0_vreg.vector;
    //  c_01_c_11_vreg.vector += a_0p_a_1p_vreg.vector * b_p1_vreg.vector;
    //  c_02_c_12_vreg.vector += a_0p_a_1p_vreg.vector * b_p2_vreg.vector;
    //  c_03_c_13_vreg.vector += a_0p_a_1p_vreg.vector * b_p3_vreg.vector;
     //
    //  /* Third and fourth rows */
    //  c_20_c_30_vreg.vector += a_2p_a_3p_vreg.vector * b_p0_vreg.vector;
    //  c_21_c_31_vreg.vector += a_2p_a_3p_vreg.vector * b_p1_vreg.vector;
    //  c_22_c_32_vreg.vector += a_2p_a_3p_vreg.vector * b_p2_vreg.vector;
    //  c_23_c_33_vreg.vector += a_2p_a_3p_vreg.vector * b_p3_vreg.vector;
   }

  //  C( 0, 0 ) += c_00_c_10_vreg.data[0];  C( 0, 1 ) += c_01_c_11_vreg.data[0];
  //  C( 0, 2 ) += c_02_c_12_vreg.data[0];  C( 0, 3 ) += c_03_c_13_vreg.data[0];
   //
  //  C( 1, 0 ) += c_00_c_10_vreg.data[1];  C( 1, 1 ) += c_01_c_11_vreg.data[1];
  //  C( 1, 2 ) += c_02_c_12_vreg.data[1];  C( 1, 3 ) += c_03_c_13_vreg.data[1];
   //
  //  C( 2, 0 ) += c_20_c_30_vreg.data[0];  C( 2, 1 ) += c_21_c_31_vreg.data[0];
  //  C( 2, 2 ) += c_22_c_32_vreg.data[0];  C( 2, 3 ) += c_23_c_33_vreg.data[0];
   //
  //  C( 3, 0 ) += c_20_c_30_vreg.data[1];  C( 3, 1 ) += c_21_c_31_vreg.data[1];
  //  C( 3, 2 ) += c_22_c_32_vreg.data[1];  C( 3, 3 ) += c_23_c_33_vreg.data[1];


   C( 0, 0 ) += c_00_10_20_30_vreg.data2[0]; C( 0, 1) += c_01_11_21_31_vreg.data2[0];
   C( 0, 2 ) += c_02_12_22_32_vreg.data2[0]; C( 0, 3) += c_03_13_23_33_vreg.data2[0];

   C( 1, 0 ) += c_00_10_20_30_vreg.data2[1]; C( 1, 1) += c_01_11_21_31_vreg.data2[1];
   C( 1, 2 ) += c_02_12_22_32_vreg.data2[1]; C( 1, 3) += c_03_13_23_33_vreg.data2[1];

   C( 2, 0 ) += c_00_10_20_30_vreg.data2[2]; C( 2, 1) += c_01_11_21_31_vreg.data2[2];
   C( 2, 2 ) += c_02_12_22_32_vreg.data2[2]; C( 2, 3) += c_03_13_23_33_vreg.data2[2];

   C( 3, 0 ) += c_00_10_20_30_vreg.data2[3]; C( 3, 1) += c_01_11_21_31_vreg.data2[3];
   C( 3, 2 ) += c_02_12_22_32_vreg.data2[3]; C( 3, 3) += c_03_13_23_33_vreg.data2[3];
}

//the naive way
#define a(i) a[ (i)*len ]

void AddDot( int n, double *a,   double *b, double *c ,int len)
{
  register double a_reg, c_reg = *c;
  double *b_pntr = &b[0];

  for (int p = 0; p < n; p++ )
  {
    a_reg = a(p);
    c_reg += a_reg * *b_pntr++;
  }
  *c = c_reg;
}

/* Block sizes */
#define mc 128
#define kc 128
#define nb 1000//size of packing for B


void PackMatrixA(int k, double *A, int lda, double *a_to){
  int j;

  for(j = 0; j < k; j++)
  {
    double *a_ij_pntr = &A(0,j);
    *a_to++ = *a_ij_pntr;
    *a_to++ = *(a_ij_pntr+1);
    *a_to++ = *(a_ij_pntr+2);
    *a_to++ = *(a_ij_pntr+3);

  }
}

void PackMatrixB(int k, double *B,int ldb, double *b_to){
  double
  *b_i0_pntr = &B( 0, 0 ), *b_i1_pntr = &B( 0, 1 ),
  *b_i2_pntr = &B( 0, 2 ), *b_i3_pntr = &B( 0, 3 );

  for(int i = 0; i < k; i++){
    *b_to++ = *b_i0_pntr++;
    *b_to++ = *b_i1_pntr++;
    *b_to++ = *b_i2_pntr++;
    *b_to++ = *b_i3_pntr++;
  }
}
void InnerKernel(int m, int n, int k,double*A, int lda,
                                     double*B, int ldb,
                                     double*C, int ldc,int first_pack){
 // int i,j;
  int p = n-n%4;
  int q = m-m%4;
  double packedA[q*k];
  double packedB[kc*nb];
  for(int j = 0; j < p; j+=4)
  {
    //for each row of C
    if(first_pack)
    {
      PackMatrixB(k, &B(0,j),ldb,&packedB[j*k]);
    }
    //for each col of C
    for(int i = 0 ; i < q; i+=4)
    {
      if(j == 0)
      {
       	PackMatrixA(k, &A(i,0), lda, &packedA[i*k]);
      }
       	Mymulti(k, &packedA[i*k],4,&packedB[j*k],ldb,&C(i,j),ldc);
    }
  }



  for(int j = p; j <n; j++){
    for(int i = 0; i < q; i++){
      AddDot(k, &A(i,0),&B(0,j),&C(i,j),n);
    }
  }


  for(int j = 0; j <n; j++){
    for(int i = q; i < m; i++){
      AddDot(k, &A(i,0),&B(0,j),&C(i,j),n);
    }
  }
}

void square_dgemm ( int n, double* A, double* B, double* C )
{

  int lda = n;
  int ldb = n;
  int ldc = n;

  for(int p = 0; p<n; p+=kc){
    int pb = MIN(n-p, kc);
    for(int i = 0; i<n; i+=mc){
      int ib = MIN(n - i, mc);
      InnerKernel(ib, n, pb, &A(i,p),n, &B(p,0),n,&C(i,0),n,i==0);
    }
  }
}
