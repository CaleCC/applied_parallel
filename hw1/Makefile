# On Trestles we will check versus your performance versus Intel MKL library's BLAS.

CC = icc
OPT = -O3 -xhost
CFLAGS = -Wall -std=gnu99  -g -msse3 -fast -mavx -unroll-aggressive -opt-prefetch-distance=64,8 $(OPT)
#CFLAGS = -Wall  -g -msse3 $(OPT)
#MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
#LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
LDLIBS = -lrt  -I$(MKLROOT)/include -Wl,-L$(MKLROOT)/lib/intel64/ -lmkl_intel_lp64 -lmkl_core -lgomp -lmkl_gnu_thread -lpthread -lm -ldl

targets = benchmark-blocked
objects = benchmark.o dgemm-blocked.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark.o : benchmark.c
	$(CC) -c $(CFLAGS) $<
dgemm-blocked.o : dgemm-blocked.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects) *.stdout
