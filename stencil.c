
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(const int nx, const int ny, float *  restrict image, float *  restrict tmp_image, int rank);
void init_image(const int nx, const int ny, float *  image, float *  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
double wtime(void);

#define MASTER 0

int main(int argc, char *argv[]) {

  int rank;                 /* rank of process */
  int size;                 /* number of processes started */

  MPI_Init( &argc, &argv );
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* check whether the initialisation was successful */
  // MPI_Initialized(&flag);
  // if ( flag != TRUE )
  //   MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  // }

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Allocate the image
  float *image = malloc(sizeof(float)*nx*ny);
  float *tmp_image = malloc(sizeof(float)*nx*ny);
  float *temp_image = malloc(sizeof(float) * nx * (ny/16));

  float sectionSize = nx*(ny/16);

  // Set the input image
  init_image(nx, ny, image, tmp_image);

  // MPI_Scatter(image, sectionSize, MPI_FLOAT, temp_image, sectionSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (rank == MASTER){
      float *temp_image = malloc(sizeof(float) * nx * (ny/16));
  }

  MPI_Scatter(image, sectionSize, MPI_FLOAT, temp_image, sectionSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {
    // printf("iteration %d\n",t);
    stencil(nx, ny/16, image, tmp_image, rank);
    stencil(nx, ny/16, tmp_image, image, rank);
  }

  // if (world_rank == 0) {
  //   sub_avgs = malloc(sizeof(float) * world_size);
  // }
  // MPI_Gather(&sub_avg, 1, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // printf("size is %d\n", size);

  double toc = wtime();

  // printf("------------------------------------\n");
  // printf(" runtime: %lf s\n", toc-tic);
  // printf("------------------------------------\n");

  output_image(OUTPUT_FILE, nx, ny, image);
  free(image);
}


float *getRow(float *segment, float *image, int start, int finish){
  int j = 0;
  for (int i = start; i<= finish; i++){
      segment[j] = image[i];
      j++;
  }

  return segment;
}

void stencil(const int nx, const int ny, float *  restrict image, float *  restrict tmp_image, int rank) {

    

    // printf("rank %d\n", rank);
    if (rank == 0)
    {
      MPI_Status *status;
      int start = (ny - 1) * nx;
      int finish = (ny-1) * nx + (nx-1);

      printf("rank 0\n");
      float *send1 = (float *)malloc(sizeof(float) * nx);
      send1 = getRow(send1, image, start, finish);

      float *receive1 = (float *)malloc(sizeof(float) * nx);

      MPI_Send(send1, nx, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(receive1, nx, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, status);

      printf("Sending to 1\n");

      free(send1);
      free(receive1);
    }
    else if (rank == 1)
    {
      MPI_Status *status;
      printf("rank 1\n");

      int start = (ny - 1) * nx;
      int finish = (ny-1) * nx + (nx-1);

      float *send2 = (float *)malloc(sizeof(float) * nx);
      send2 = getRow(send2, image, start, finish);

      float *receive2 = (float *)malloc(sizeof(float) * nx);
      MPI_Recv(receive2, nx, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, status); 
      MPI_Send(send2, nx, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
      

      printf("Received %f\n", receive2[0]);

      free(receive2);
      free(send2);
    }

    // else if ( rank == 15 ){

    // }
    else{
      int i = 0;
    }




    // //Corner cases cmonnnnn
    // tmp_image[0] = image[0] * 0.6f + (image[nx] + image[1]) * 0.1f; //comment   
    // tmp_image[nx-1] = image[nx-1] * 0.6f + (image[nx*2-1]+ image[nx-2]) * 0.1f;
    // tmp_image[nx*ny-(nx)] = image[nx*ny-(nx)] * 0.6f + (image[nx*ny-(nx*2)] + image[nx*ny-(nx-1)]) * 0.1f;
    // tmp_image[nx*ny-1] = image[nx*ny-1] * 0.6f + (image[nx*ny-(nx+1)] + image[nx*ny-2]) * 0.1f;

    // //top cases

    // for (int j = 1; j<nx-1; ++j){
    //   tmp_image[j] = image[j] * 0.6f + (image[j-1] + image[j+1] + image[j+nx]) * 0.1f;
    // }

    // //bottom cases

    // for (int j = 1; j<nx-1; ++j){
    //   tmp_image[nx*ny-nx+j] = image[nx*ny-(nx)+j] * 0.6f + (image[nx*ny-(nx)+j-1] + image[nx*ny-(nx)+j+1] + image[nx*ny-(2*nx)+j]) * 0.1f;
    // }

    // //1. left cases

    // for (int j = 1; j<nx-1; ++j){
    //   tmp_image[ny*j] = image[ny*j] * 0.6f + (image[(nx*j)+1] + image[nx*(j-1)] + image[nx*(j+1)]) * 0.1f;
    // }
    
    // //2. right cases

    // for (int j = 1; j<nx-1; ++j){
    //   tmp_image[nx*(j+1)-1] = image[nx*(j+1)-1] * 0.6f + (image[nx*j-1] + image[nx*(j+2)-1] + image[nx*(j+1)-2]) * 0.1f;
    // }

    // //3. middle cases

    // #pragma omp simd
    // for (int j = 0; j < (nx*(nx-2)); j+=nx) {
    //   for(int i = 1; i<ny-1;++i){
    //     tmp_image[j+i+nx] = image[j+i+nx] * 0.6f + (image[j+i+nx+1] + image[j+i+nx-1] + image[j+i] + image[j+i+(nx*2)]) * 0.1f;
    //   }
    // }

}

// Create the input image
void init_image(const int nx, const int ny, float *  image, float *  tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
      tmp_image[j+i*ny] = 0.0;
    }
  }

  // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
          image[jj+ii*ny] = 100.0;
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, float *image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}
