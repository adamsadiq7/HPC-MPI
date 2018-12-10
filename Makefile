stencil: stencil.c
	mpicc -std=c99 -Wall -Ofast -qopenmp-stubs $^ -o $@