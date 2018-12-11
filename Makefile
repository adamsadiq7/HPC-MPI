stencil: stencil.c
	mpicc -std=c99 -Wopenmp-simd -Ofast -Wall $^ -o $@ 