# on Hopper, we will benchmark you against Cray LibSci, the default vendor-tuned BLAS. The Cray compiler wrappers handle all the linking. If you wish to compare with other BLAS implementations, check the NERSC documentation.
# This makefile is intended for the GNU C compiler. On Hopper, the Portland compilers are default, so you must instruct the Cray compiler wrappers to switch to GNU: type "module swap PrgEnv-pgi PrgEnv-gnu"

CC = cc 
OPT1 = -g -pg
OPT =  -O2   \
	-march=sandybridge \
	-mtune=sandybridge \
	-ftree-vectorize \
	-ftree-slp-vectorize \
	-ffast-math  \
	-funroll-loops \
	-fvariable-expansion-in-unroller \
	-floop-block \
	-floop-strip-mine 
	# -funsafe-loop-optimizations \
	# -faggressive-loop-optimizations
#-fstrict-aliasing
# CFLAGS = -Wall -std=gnu99 $(OPT) 
CFLAGS = -Wall -std=gnu11 $(OPT) 
LDFLAGS = -Wall
# librt is needed for clock_gettime
LDLIBS = -lrt 
MYLAPACK = /home/jaabell/Repositories/essi_dependencies/lib/liblapack.a \
	/home/jaabell/Repositories/essi_dependencies/lib/libblas.a -lgfortran -lm 


targets = benchmark-naive benchmark-blocked benchmark-blas test_intrinsic
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o  test_intrinsic.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(MYLAPACK) $(LDLIBS) 
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(MYLAPACK) $(LDLIBS) 
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(MYLAPACK) $(LDLIBS) 
test_intrinsic : test_intrinsic.o
	$(CC) -o $@ $^ $(MYLAPACK) $(LDLIBS) 

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
