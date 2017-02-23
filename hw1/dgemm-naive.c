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


void Mymulti(int n, double *A,int lda, double* B,int  ldb,
                                       double *C, int ldc){
   __m256d c1 = _mm256_loadu_pd(C(0,0));	// load first row of C
 	__m256d c2 = _mm256_loadu_pd(C(1,0));	// load second row of C
 	__m256d c3 = _mm256_loadu_pd(C(2,0));	// load third row of C
 	__m256d c4 = _mm256_loadu_pd(C(3,0));	// load fourth row of C

  for(int i = 0, i < 4, i++){
    __m256d a = _mm256_loadu_pd(A(i,0));		// Load ith row of A; a = <ai0, ai1, ai2, ai3>
    __m256d b1 = _mm256_broadcast_sd(B(i,0));	// create vector b1 = <B+i+0*n, B+i+0*n, B+i+0*n, B+i+0*n>
    __m256d b2 = _mm256_broadcast_sd(B(i,1));	// create vector b2 = <B+i+1*n, B+i+1*n, B+i+1*n, B+i+1*n>
    __m256d b3 = _mm256_broadcast_sd(B(i,2));	// create vector b3 = <B+i+2*n, B+i+2*n, B+i+2*n, B+i+2*n>
    __m256d b4 = _mm256_broadcast_sd(B(i,3));	// create vector b4 = <B+i+3*n, B+i+3*n, B+i+3*n, B+i+3*n>

    c1 = _mm256_add_pd(c1, _mm256_mul_pd(a,b1));	// c1 = c1 + a * b1
    c2 = _mm256_add_pd(c2, _mm256_mul_pd(a,b2));	// c2 = c2 + a * b2
    c3 = _mm256_add_pd(c3, _mm256_mul_pd(a,b3));	// c3 = c3 + a * b3
    c4 = _mm256_add_pd(c4, _mm256_mul_pd(a,b4));	// c4 = c4 + a * b4

  }

  _mm256_storeu_pd(C(0,0), c1);
  _mm256_storeu_pd(C(1,0), c2);
  _mm256_storeu_pd(C(2,0), c3);
  _mm256_storeu_pd(C(3,0), c4);
  // /* So, this routine computes a 4x4 block of matrix A
  //          C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).
  //          C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).
  //          C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).
  //          C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).
  //    Notice that this routine is called with c = C( i, j ) in the
  //    previous routine, so these are actually the elements
  //          C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 )
  //          C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 )
  //          C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 )
  //          C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 )
  //
  //    in the original matrix C
  //    In this version, we use pointer to track where in four columns of B we are */
  //
  // int p;
  // register double
  //   /* hold contributions to
  //      C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 )
  //      C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 )
  //      C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 )
  //      C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 )   */
  //      c_00_reg,   c_01_reg,   c_02_reg,   c_03_reg,
  //      c_10_reg,   c_11_reg,   c_12_reg,   c_13_reg,
  //      c_20_reg,   c_21_reg,   c_22_reg,   c_23_reg,
  //      c_30_reg,   c_31_reg,   c_32_reg,   c_33_reg,
  //   /* hold
  //      A( 0, p )
  //      A( 1, p )
  //      A( 2, p )
  //      A( 3, p ) */
  //      a_0p_reg,
  //      a_1p_reg,
  //      a_2p_reg,
  //      a_3p_reg;
  // double
  //   /* Point to the current elements in the four columns of B */
  //   *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

  // b_p0_pntr = &B( 0, 0 );
  // b_p1_pntr = &B( 0, 1 );
  // b_p2_pntr = &B( 0, 2 );
  // b_p3_pntr = &B( 0, 3 );
  //
  // c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
  // c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
  // c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
  // c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;
  //
  // for ( p=0; p<n; p++ ){
  //   a_0p_reg = A( 0, p );
  //   a_1p_reg = A( 1, p );
  //   a_2p_reg = A( 2, p );
  //   a_3p_reg = A( 3, p );
  //
  //   /* First row */
  //   c_00_reg += a_0p_reg * *b_p0_pntr;
  //   c_01_reg += a_0p_reg * *b_p1_pntr;
  //   c_02_reg += a_0p_reg * *b_p2_pntr;
  //   c_03_reg += a_0p_reg * *b_p3_pntr;
  //
  //   /* Second row */
  //   c_10_reg += a_1p_reg * *b_p0_pntr;
  //   c_11_reg += a_1p_reg * *b_p1_pntr;
  //   c_12_reg += a_1p_reg * *b_p2_pntr;
  //   c_13_reg += a_1p_reg * *b_p3_pntr;
  //
  //   /* Third row */
  //   c_20_reg += a_2p_reg * *b_p0_pntr;
  //   c_21_reg += a_2p_reg * *b_p1_pntr;
  //   c_22_reg += a_2p_reg * *b_p2_pntr;
  //   c_23_reg += a_2p_reg * *b_p3_pntr;

    /* Four row */
  //   c_30_reg += a_3p_reg * *b_p0_pntr++;
  //   c_31_reg += a_3p_reg * *b_p1_pntr++;
  //   c_32_reg += a_3p_reg * *b_p2_pntr++;
  //   c_33_reg += a_3p_reg * *b_p3_pntr++;
  // }
  //
  // C( 0, 0 ) += c_00_reg;   C( 0, 1 ) += c_01_reg;   C( 0, 2 ) += c_02_reg;   C( 0, 3 ) += c_03_reg;
  // C( 1, 0 ) += c_10_reg;   C( 1, 1 ) += c_11_reg;   C( 1, 2 ) += c_12_reg;   C( 1, 3 ) += c_13_reg;
  // C( 2, 0 ) += c_20_reg;   C( 2, 1 ) += c_21_reg;   C( 2, 2 ) += c_22_reg;   C( 2, 3 ) += c_23_reg;
  // C( 3, 0 ) += c_30_reg;   C( 3, 1 ) += c_31_reg;   C( 3, 2 ) += c_32_reg;   C( 3, 3 ) += c_33_reg;
}

//the naive way
#define X(i) x[ (i)*len ]

void AddDot( int n, double *x,   double *y, double *gamma ,int len)
{
  /* compute gamma := x' * y + gamma with vectors x and y of length n.
     Here x starts at location x with increment (stride) incx and y starts at location y and has (implicit) stride of 1.
  */

  int p;

  for ( p=0; p<n; p++ ){
    *gamma += X( p ) * y[ p ];
  }
}

/* Block sizes */
#define mc 128
#define kc 128
#define nb 1000//size of packing for B
void PackMatrixA(int k, double *A, int lda, double *a_to){
  int j;

  for(j = 0; j < k; j++){
    double *a_ij_pntr = &A(0,j);
    *a_to++ = *a_ij_pntr;
    *a_to++ = *(a_ij_pntr+1);
    *a_to++ = *(a_ij_pntr+2);
    *a_to++ = *(a_ij_pntr+3);

  }
}

void PackMatrixB(int k, double *B,int ldb, double *b_to){
  int i;
  double
  *b_i0_pntr = &B( 0, 0 ), *b_i1_pntr = &B( 0, 1 ),
  *b_i2_pntr = &B( 0, 2 ), *b_i3_pntr = &B( 0, 3 );

  for(i = 0; i < k; i++){
    *b_to++ = *b_i0_pntr++;
    *b_to++ = *b_i1_pntr++;
    *b_to++ = *b_i2_pntr++;
    *b_to++ = *b_i3_pntr++;
  }
}




void InnerKernel(int m, int n, int k,double*A, int lda,
                                     double*B, int ldb,
                                     double*C, int ldc,int first_pack){
  int i,j;
  int p = n-n%4;
  int q = m-m%4;
  double packedA[q*k];
  double packedB[kc*nb];
  for(j = 0; j < p; j+=4){
    //for each row of C
    if(first_pack){
      PackMatrixB(k, &B(0,j),ldb,&packedB[j*k]);
    }
    for(i = 0 ; i < q; i+=4){
      if(j == 0){
        PackMatrixA(k, &A(i,0), lda, &packedA[i*k]);
      }
      Mymulti(k, &packedA[i*k],4,&B(0,j),k,&C(i,j),ldc);
    }
  }



  for(j = p; j <n; j++){
    for(i = 0; i < q; i++){
      AddDot(k, &A(i,0),&B(0,j),&C(i,j),n);
    }
  }


  for(j = 0; j <n; j++){
    for(i = q; i < m; i++){
      AddDot(k, &A(i,0),&B(0,j),&C(i,j),n);
    }
  }
}

void square_dgemm ( int n, double* A, double* B, double* C )
{

  int i = 0;
  int j = 0;
  int lda = n;
  int ldb=n;
  int ldc = n;
  int p,pb,ib;

  for(p = 0; p<n; p+=kc){
    pb = MIN(n-p, kc);
    for(i = 0; i<n; i+=mc){
      ib = MIN(n - i, mc);
      InnerKernel(ib, n, pb, &A(i,p),n, &B(p,0),n,&C(i,0),n,i==0);
    }
  }
  //for each columns of C
    //printf("debig 1\n");


}
