stencil: stencil.c
	mpicc -std=c99 -qopenmp-stubs -xHost -g -pg -Ofast -Wall $^ -o $@ 