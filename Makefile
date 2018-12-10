stencil: stencil.c
	mpicc -std=c99 -Wall -Ofast -Wopenmp-simd $^ -o $@