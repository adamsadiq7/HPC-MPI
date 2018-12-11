#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void stencil(const int nx, const int ny, float *image, float *tmp_image, int rank);
void init_image(const int nx, const int ny, float *image, float *tmp_image);
void output_image(const char *file_name, const int nx, const int ny, float *image);
double wtime(void);
int main(int argc, char *argv[])
{

  // Check usage
  if (argc != 4)
  {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments

  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Allocate the images

  //Figuring out which processors are involved in the computation
  MPI_Init(&argc, &argv);

  int size;
  int rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //printf("rank %d called Init\n", rank);
  float *image;
  float *tmp_image;
  if (rank == 0)
  {
    image = malloc(sizeof(float) * ny * nx);

    tmp_image = malloc(sizeof(float) * ny * nx);

    // Set the input image
    init_image(nx, ny, image, tmp_image);
  }
  int sectionSize = ny * nx / 16;

  float *bufferImg = malloc((ny * nx / 16) * sizeof(float));
  float *bufferTempImg = malloc((ny * nx / 16) * sizeof(float));

  MPI_Scatter(image, sectionSize, MPI_FLOAT, bufferImg, sectionSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
  

  
  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t <  niters; ++t)
  { 
    
    printf("iteration : %d and rank %d\n", t, rank);
    

    stencil(nx, ny / 16, bufferImg, bufferTempImg, rank);
    stencil(nx, ny / 16, bufferTempImg, bufferImg, rank);
  }
  printf("finished stencil\n");
  double toc = wtime();
  

  MPI_Finalize();
  // Output
  // printf("------------------------------------\n");
  // printf(" runtime: %lf s\n", toc - tic);
  // printf("------------------------------------\n");

  // output_image(OUTPUT_FILE, nx, ny, image);
  // free(image);
  
}

float *extractElements(float *subArray, float *array, int start, int end)
{

  for (int i = start; i <= end; i++)
  {
    subArray[i - start] = array[i];
  }

  return subArray;
}
void stencil(const int nx, const int ny, float *restrict image, float *restrict tmp_image, int rank)
{

  int sectionSize = 16 * nx * 16 * ny / 16;

  if (rank == 0)
  {
    
    //sending the last row of the array to rank 1;
    int start = (ny - 1) * nx;
    int end = (ny - 1) * nx + nx - 1;

    float *lastRowSend = (float *) malloc(nx * sizeof(float));
    lastRowSend = extractElements(lastRowSend, image, start, end);

    float *lastRowRecv = (float *) malloc(nx * sizeof(float));
    MPI_Status *status;

    MPI_Sendrecv(lastRowSend, nx, MPI_FLOAT, rank + 1, 0, lastRowRecv, nx, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, status);
    
    free(lastRowSend);
    free(lastRowRecv);
  }
  else if (rank==15){
    //sending the first row of the array to rank 14
    int start = 0;
    int end = nx - 1;

    float *firstRowSend = (float *) malloc(nx * sizeof(float));
    firstRowSend = extractElements(firstRowSend, image, start, end);

    float *firstRowRecv = (float *) malloc(nx * sizeof(float));

    MPI_Status *status;

    //MPI_Sendrecv( firstRowSend , nx, MPI_FLOAT, 1, 0 , firstRowRecv , nx, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, status);
    MPI_Recv(firstRowRecv, nx , MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, status);
    MPI_Send(firstRowSend, nx , MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD );

    free(firstRowSend);
    free(firstRowRecv);
  }
  else{

    float *firstRowRecv = (float *) malloc(nx * sizeof(float));
    float *lastRowRecv = (float *)  malloc(nx * sizeof(float));

    float *firstRowSend = (float *) malloc(nx * sizeof(float));
    float *lastRowSend = (float *)  malloc(nx * sizeof(float));

    int firstRowStart = 0;
    int firstRowEnd = nx - 1;

    int lastRowStart = (ny - 1) * nx;
    int lastRowEnd = (ny - 1) * nx + nx - 1;
    
    firstRowSend = extractElements(firstRowSend, image, firstRowStart, firstRowEnd);
    
    lastRowSend = extractElements(lastRowSend, image, lastRowStart, lastRowEnd);
    

    //Sending and receving data from each rank above and below in the image
    MPI_Status *status;
  

    //MPI_Sendrecv( firstRowSend , nx, MPI_FLOAT, rank - 1, 0 , firstRowRecv , nx, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, status);
    printf("deadlock \n");
     //MPI_Sendrecv( lastRowSend , nx, MPI_FLOAT, 2, 0 , lastRowRecv , nx, MPI_FLOAT, 2, 0, MPI_COMM_WORLD, status);
    MPI_Send(firstRowSend, nx, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD );
    MPI_Recv(firstRowRecv, nx, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, status);

    MPI_Send(lastRowSend, nx, MPI_FLOAT,rank+1, 0, MPI_COMM_WORLD );
    MPI_Recv(lastRowRecv, nx, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, status);
    //int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
    //MPI_Recv(firstRowRecv, nx , MPI_FLOAT, 2, 0, MPI_COMM_WORLD, status);
    printf("finish 1\n");
    
    free(firstRowRecv);
    free(firstRowSend);
    free(lastRowRecv);
    free(lastRowSend);
  }
  


  

  //   //manually amending the values of the corners
  //  tmp_image[0]                   = 0.6f * image[0]                  + 0.1f*image[1 + ny*0]                  + 0.1f*image[0 + ny*1];
  //  tmp_image[nx-1 + ny*0]         = 0.6f * image[nx-1 + ny*0]        + 0.1f*image[nx-2 + ny*0]               + 0.1f*image[nx-1 + ny*1];
  //  tmp_image[0 + ny*(nx-1)]       = 0.6f * image[0 + ny*(ny-1)]      + 0.1f*image[0 +ny*(ny-2)]              + 0.1f*image[(1 + ny*(ny-1))];
  //  tmp_image[nx-1 + (ny)*(ny-1)]  = 0.6f * image[nx-1 + (ny)*(ny-1)] + 0.1f*image[nx-1 + (ny)*(nx-2)]        + 0.1f*image[nx-2 +(nx-1)*(ny)];

  //   //top row
  //   for(int j = 1; j<nx-1; ++j){
  //     tmp_image[j+ny*0] = 0.1f*image[j-1 + ny*0] + 0.6f*image[j+ny*0]  + 0.1f*image[j+1 + ny*0] + 0.1f*image[j+ny*1];
  //   }

  //   //first column
  //   for(int i = 1; i< ny-1 ; ++i){
  //    tmp_image[0+ny*i] = 0.6f*image[0+ny*i] + 0.1f*image[1+ ny*i] + 0.1f*image[0+ny*(i-1)] + 0.1f*image[0 + ny*(i+1)];
  //   }

  //   //editing the values of the (ny-1)*(nx-1) pisxels
  //   for(int i = 1 ; i<ny-1; ++i){
  //    for(int j = 1 ; j<nx-1; ++j){
  //      int base = j+ny*i;
  //      __assume_aligned(image,64);
  //      __assume_aligned(tmp_image,64);
  //      #pragma omp simd

  //      tmp_image[base] = image[base-1]*0.1f   + image[base]*0.6f + image[base+1]*0.1f + image[base -ny]*0.1f + image[base +ny]*0.1f;
  //    }
  //   }
  //   //last column
  //   for(int i = 1; i< ny-1 ; ++i){
  //     int base  = nx-1 + ny*i;
  //     tmp_image[base] = 0.6f*image[base] + 0.1f*image[base-1] + 0.1f*image[base -ny] + 0.1f*image[base + ny];
  //   }

  //   //last row
  //   for(int j = 1; j<nx-1; ++j){
  //    tmp_image[j + ny*(nx-1)] = 0.6f*image[j+ ny*(nx-1)] + 0.1f*image[(j-1)+ ny*(nx-1)] + 0.1f*image[(j+1)+ ny*(nx-1)] + 0.1f*image[j+ ny*(nx-2)];
  //   }
  
}

// Create the input image
void init_image(const int nx, const int ny, float *image, float *tmp_image)
{
  // Zero everything
  for (int j = 0; j < ny; ++j)
  {
    for (int i = 0; i < nx; ++i)
    {

      image[j + ny * i] = 0.0;
      tmp_image[j + ny * i] = 0.0;
    }
  }

  // Checkerboard

  for (int i = 0; i < 8; ++i)
  {
    for (int j = 0; j < 8; ++j)
    {
      for (int ii = i * ny / 8; ii < (i + 1) * ny / 8; ++ii)
      {
        for (int jj = j * nx / 8; jj < (j + 1) * nx / 8; ++jj)
        {
          if ((i + j) % 2)
            image[jj + ii * ny] = 100.0;
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char *file_name, const int nx, const int ny, float *image)
{

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp)
  {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int i = 0; i < ny; ++i)
  {
    for (int j = 0; j < nx; ++j)
    {
      if (image[j + i * ny] > maximum)
        maximum = image[j + i * ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int i = 0; i < ny; ++i)
  {
    for (int j = 0; j < nx; ++j)
    {
      //fputc((char)(255.0*image[j+ny*i]/maximum), fp);
      fputc((char)(255.0 * image[j + ny * i] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
