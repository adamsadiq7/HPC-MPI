stencil: stencil.c
	mpicc -std=c99 -Wopenmp-simd -xHost -g -pg -Ofast -Wall $^ -o $@ 