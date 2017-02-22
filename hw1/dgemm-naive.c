#include<stdlib.h>
#include<stdio.h>
const char* dgemm_desc = "Naive, three-loop dgemm.";


#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define BLOCK_SIZE 128
#define A(i,j) A[(j)*n + (i)]
#define B(i,j) B[(j)*n + (i)]
#define C(i,j) C[(j)*n + (i)]
//void Mymulti(int, double*,  double*, double *)
void Mymulti(int n, double *A, double* B,double *C){
  int l;
  //printf("debig 2\n");
  register double
    c_00_reg, c_01_reg, c_02_reg, c_03_reg,
    c_10_reg, c_11_reg, c_12_reg, c_13_reg,
    c_20_reg, c_21_reg, c_22_reg, c_23_reg,
    c_30_reg, c_31_reg, c_32_reg, c_33_reg,
    a_0l_reg,//A(0,l)
    a_1l_reg,
    a_2l_reg,
    a_3l_reg;
  double
    *bl0_pntr, *bl1_pntr, *bl2_pntr, *bl3_pntr;

  bl0_pntr = &B(0,0);
  bl1_pntr = &B(0,1);
  bl2_pntr = &B(0,2);
  bl3_pntr = &B(0,3);
  //set the reg to zero
  c_00_reg = 0.0;
  c_01_reg = 0.0;
  c_02_reg = 0.0;
  c_03_reg = 0.0;
  c_10_reg = 0.0;
  c_11_reg = 0.0;
  c_12_reg = 0.0;
  c_13_reg = 0.0;
  c_20_reg = 0.0;
  c_21_reg = 0.0;
  c_22_reg = 0.0;
  c_23_reg = 0.0;  //printf("debig 3\n");
  c_30_reg = 0.0;
  c_31_reg = 0.0;
  c_32_reg = 0.0;
  c_33_reg = 0.0;
  for(l = 0; l < n; l++){
    a_0l_reg = A(0,l);
    a_1l_reg = A(1,l);
    a_2l_reg = A(2,l);
    a_3l_reg = A(3,l);
    //first row of C
    c_00_reg += a_0l_reg * *(bl0_pntr);
    c_01_reg += a_0l_reg * *(bl1_pntr);
    c_02_reg += a_0l_reg * *(bl2_pntr);
    c_03_reg += a_0l_reg * *(bl3_pntr);


    c_10_reg += a_1l_reg * *(bl0_pntr);
    c_11_reg += a_1l_reg * *(bl1_pntr);
    c_12_reg += a_1l_reg * *(bl2_pntr);
    c_13_reg += a_1l_reg * *(bl3_pntr);


    c_20_reg += a_2l_reg * *(bl0_pntr);
    c_21_reg += a_2l_reg * *(bl1_pntr);
    c_22_reg += a_2l_reg * *(bl2_pntr);
    c_23_reg += a_2l_reg * *(bl3_pntr);


    c_30_reg += a_3l_reg * *(bl0_pntr);
    c_31_reg += a_3l_reg * *(bl1_pntr);
    c_32_reg += a_3l_reg * *(bl2_pntr);
    c_33_reg += a_3l_reg * *(bl3_pntr);

    bl0_pntr++;
    bl1_pntr++;
    bl2_pntr++;
    bl3_pntr++;
  }
  C(0,0) += c_00_reg; C(0,1) += c_01_reg; C(0,2) += c_02_reg; C(0,3) += c_03_reg;
  C(1,0) += c_10_reg; C(1,1) += c_11_reg; C(1,2) += c_12_reg; C(1,3) += c_13_reg;
  C(2,0) += c_20_reg; C(2,1) += c_21_reg; C(2,2) += c_22_reg; C(2,3) += c_23_reg;
  C(3,0) += c_30_reg; C(3,1) += c_31_reg; C(3,2) += c_32_reg; C(3,3) += c_33_reg;
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
      Mymulti(n, &A(i,0),&B(0,j),&C(i,j));
    }
  }
}
