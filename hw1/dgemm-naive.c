#include<stdlib.h>
const char* dgemm_desc = "Naive, three-loop dgemm.";


#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define BLOCK_SIZE 128
#define A(i,j) A[(j)*n + (i)]
#define B(i,j) B[(j)*n + (i)]
#define C(i,j) C[(j)*n + (i)]
//void Mymulti(int, double*,  double*, double *)
void Mymulti(int n, double *A, double* B,double *C){
  int l;
  register double
    c_00_reg, c_01_reg, c_02_reg, c_03_reg,
    a_0l_reg;//A(0,l)
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

  for(l = 0; l < n; l += 4){
    a_0l_reg = A(0,l);

    c_00_reg += a_0l_reg * *(bl0_pntr);
    c_01_reg += a_0l_reg * *(bl1_pntr);
    c_02_reg += a_0l_reg * *(bl2_pntr);
    c_03_reg += a_0l_reg * *(bl3_pntr);

    a_0l_reg = A(0,l+1);

    c_00_reg += a_0l_reg * *(bl0_pntr+1);
    c_01_reg += a_0l_reg * *(bl1_pntr+1);
    c_02_reg += a_0l_reg * *(bl2_pntr+1);
    c_03_reg += a_0l_reg * *(bl3_pntr+1);

    a_0l_reg = A(0,l+2);

    c_00_reg += a_0l_reg * *(bl0_pntr+2);
    c_01_reg += a_0l_reg * *(bl1_pntr+2);
    c_02_reg += a_0l_reg * *(bl2_pntr+2);
    c_03_reg += a_0l_reg * *(bl3_pntr+2);

    a_0l_reg = A(0,l+3);

    c_00_reg += a_0l_reg * *(bl0_pntr+3);
    c_01_reg += a_0l_reg * *(bl1_pntr+3);
    c_02_reg += a_0l_reg * *(bl2_pntr+3);
    c_03_reg += a_0l_reg * *(bl3_pntr+3);

    bl0_pntr+=4;
    bl1_pntr+=4;
    bl2_pntr+=4;
    bl3_pntr+=4;
  }
  C(0,0) += c_00_reg;
  C(0,1) += c_01_reg;
  C(0,2) += c_02_reg;
  c(0,3) += c_03_reg;
}

void square_dgemm ( int n, double* A, double* B, double* C )
{
  int i = 0;
  int j = 0;
  //for each columns of C
  for(j = 0; j < n; j+=4){
    //for each row of C
    for(i = 0 ; i < n; i+=1){
      Mymulti(n, &A(i,0),&B(0,j),&C(i,j));
    }
  }
}
