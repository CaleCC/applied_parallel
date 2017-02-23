# On Trestles we will check versus your performance versus Intel MKL library's BLAS.

CC = icc
OPT = -O3 -xhost
CFLAGS = -Wall -std=gnu99  -g -msse3 -fast -mavx -unroll-aggressive$(OPT)
#CFLAGS = -Wall  -g -msse3 $(OPT)
#MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
#LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
LDLIBS = -lrt  -lm
targets = benchmark-naive
objects = benchmark.o dgemm-naive.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark.o : benchmark.c
	$(CC) -c $(CFLAGS) $<
dgemm-naive.o : dgemm-naive.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects) *.stdout
