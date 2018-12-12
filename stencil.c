#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0



void stencil(const int nx, const int ny, float * image, float * tmp_image,int rank);
void init_image(const int nx, const int ny, float * image, float * tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
double wtime(void);
int main(int argc, char *argv[]) {

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments

  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Allocate the images
  float *image =malloc(sizeof(float)*nx*ny);

  float *tmp_image = malloc(sizeof(float)*nx*ny);

  // Initialising MPI
  MPI_Init(&argc, &argv);
  int rank;
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == MASTER){
    image =malloc(sizeof(float)*ny*nx);
    tmp_image = malloc(sizeof(float)*ny*nx);
    
    // Set the input image
    init_image(nx, ny, image, tmp_image);

  }

  int sectionSize = nx*ny/16;
  float * buffer = malloc(sizeof(float) * sectionSize);
  float * bufferTmp = malloc(sizeof(float) * sectionSize);

  MPI_Scatter(image, sectionSize, MPI_FLOAT, buffer, sectionSize, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
 

  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {
    stencil(nx, ny/16, buffer, bufferTmp,rank);
    stencil(nx, ny/16, bufferTmp, buffer,rank);
  }
  double toc = wtime();

  float * result;

  if(rank == MASTER){
    result = malloc(sizeof(float)*ny*nx);
  }
  MPI_Gather(bufferTmp, sectionSize, MPI_FLOAT,result ,sectionSize, MPI_FLOAT,MASTER, MPI_COMM_WORLD);


  if(rank==MASTER){
    output_image(OUTPUT_FILE, nx, ny, result);
    // for(int i = 0 ; i< ny ; i++){
    //   for(int j =0 ; j< nx ; j++){
    //     printf("%f \n", result[j+i*nx]);
    //   }
    // }
  }
  MPI_Finalize();


  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");




  
  
  free(image);
}


float * extractRow(float* inputArray,float * outputArray, int start,int end){
  for( int i =start ; i<= end; i++){
    outputArray[i-start] = inputArray[i];
  }

  return outputArray;
}
void stencil(const int nx, const int ny,  float *restrict image, float *restrict tmp_image, int rank) {
  float * firstRowSend = malloc(sizeof(float)* nx );
  float * firstRowRecv = malloc(sizeof(float)* nx );  

  float * lastRowSend = malloc(sizeof(float)* nx );
  float * lastRowRecv = malloc(sizeof(float)* nx );
  
  int firstRowStart, firstRowEnd, lastRowStart, lastRowEnd;


//------- INDEX OF THE ROWS -------//

  firstRowStart = 0;
  firstRowEnd = nx-1;
  lastRowStart = (ny-1)* nx;
  lastRowEnd   = (ny-1)* nx + (nx-1);

//--------INDEX OF THE ROWS ------// 
  MPI_Status * status;


  if(rank == MASTER){

    lastRowSend = extractRow(image,lastRowSend, lastRowStart, lastRowEnd);

    MPI_Sendrecv(lastRowSend, nx, MPI_FLOAT, rank + 1, 0, lastRowRecv, nx, MPI_FLOAT, rank +1, 0, MPI_COMM_WORLD, status);


    // MPI_Send(lastRowSend, nx, MPI_FLOAT,rank +1, 0, MPI_COMM_WORLD);
    // MPI_Recv(lastRowRecv, nx, MPI_FLOAT, rank+1, MASTER, MPI_COMM_WORLD, status);

    //Corner cases cmonnnnn
    tmp_image[0] = image[0] * 0.6f + (image[nx] + image[1]) * 0.1f; //comment
    tmp_image[nx-1] = image[nx-1] * 0.6f + (image[nx*2-1]+ image[nx-2]) * 0.1f;
    tmp_image[nx*ny-(nx)] = image[nx*ny-(nx)] * 0.6f + (image[nx*ny-(nx*2)] + image[nx*ny-(nx-1)] + lastRowRecv[0]) * 0.1f;
    tmp_image[nx*ny-1] = image[nx*ny-1] * 0.6f + (image[nx*ny-(nx+1)] + image[nx*ny-2] + lastRowRecv[nx-1]) * 0.1f;

    //top cases

    for (int j = 1; j<nx-1; ++j){
      tmp_image[j] = image[j] * 0.6f + (image[j-1] + image[j+1] + image[j+nx]) * 0.1f;
    }

    //bottom cases
    
    for (int j = 1; j<nx-1; ++j){
      tmp_image[nx*ny-nx+j] = image[nx*ny-(nx)+j] * 0.6f + (image[nx*ny-(nx)+j-1] + image[nx*ny-(nx)+j+1] + image[nx*ny-(2*nx)+j] +lastRowRecv[j]) * 0.1f;
    }

    //1. left cases

    for (int j = 1; j<ny-1; ++j){
      tmp_image[nx*j] = image[nx*j] * 0.6f + (image[(nx*j)+1] + image[nx*(j-1)] + image[nx*(j+1)]) * 0.1f;
    }
    
    //2. right cases

    for (int j = 1; j<ny-1; ++j){
      tmp_image[nx*(j+1)-1] = image[nx*(j+1)-1] * 0.6f + (image[nx*j-1] + image[nx*(j+2)-1] + image[nx*(j+1)-2]) * 0.1f;
    }

    //3. middle cases

    //#pragma omp simd
    for (int j = 0; j < (nx*(ny-2)); j+=nx) {
      for(int i = 1; i<nx-1;++i){
        tmp_image[j+i+nx] = image[j+i+nx] * 0.6f + (image[j+i+nx+1] + image[j+i+nx-1] + image[j+i] + image[j+i+(nx*2)]) * 0.1f;
      }
    }

    //  for(int i = 0 ; i< ny ; i++){
    //   for(int j =0 ; j< nx ; j++){
    //     int base = j+i*nx;

    //     tmp_image[base] = image[base] *0.6f;
    //     if(i>0)     tmp_image[base] += image[j + (i-1)*nx] * 0.1f; 
    //     if(j>0)     tmp_image[base] += image[j-1 +i*nx] * 0.1f;
    //     if(i<ny-1)  tmp_image[base] += image[j + (i+1)*nx] *0.1f;
    //     if(j<nx-1)  tmp_image[base] += image[j+1 + i*nx] * 0.1f;
    //     if(i==ny-1) tmp_image[base] += lastRowRecv[j] * 0.1f;
    //   }
    // }

  }
  else if(rank == 15){

    firstRowSend = extractRow(image,firstRowSend, firstRowStart, firstRowEnd);

    MPI_Sendrecv(firstRowSend, nx, MPI_FLOAT, rank - 1, 0, firstRowRecv, nx, MPI_FLOAT, rank -1, 0, MPI_COMM_WORLD, status);

    // MPI_Send(firstRowSend, nx, MPI_FLOAT,rank -1, 0, MPI_COMM_WORLD);
    // MPI_Recv(firstRowRecv, nx, MPI_FLOAT, rank-1 , MASTER, MPI_COMM_WORLD, status);

    //Corner cases cmonnnnn
    tmp_image[0] = image[0] * 0.6f + (image[nx] + image[1] + firstRowRecv[0]) * 0.1f; //comment
    tmp_image[nx-1] = image[nx-1] * 0.6f + (image[nx*2-1]+ image[nx-2] + firstRowRecv[nx-1]) * 0.1f;
    tmp_image[nx*ny-(nx)] = image[nx*ny-(nx)] * 0.6f + (image[nx*ny-(nx*2)] + image[nx*ny-(nx-1)]) * 0.1f;
    tmp_image[nx*ny-1] = image[nx*ny-1] * 0.6f + (image[nx*ny-(nx+1)] + image[nx*ny-2]) * 0.1f;

    //top cases

    for (int j = 1; j<nx-1; ++j){
      tmp_image[j] = image[j] * 0.6f + (image[j-1] + image[j+1] + image[j+nx] + firstRowRecv[j]) * 0.1f;
    }

    //bottom cases
    
    for (int j = 1; j<nx-1; ++j){
      tmp_image[nx*ny-nx+j] = image[nx*ny-(nx)+j] * 0.6f + (image[nx*ny-(nx)+j-1] + image[nx*ny-(nx)+j+1] + image[nx*ny-(2*nx)+j]) * 0.1f;
    }

    //1. left cases

    for (int j = 1; j<ny-1; ++j){
      tmp_image[nx*j] = image[nx*j] * 0.6f + (image[(nx*j)+1] + image[nx*(j-1)] + image[nx*(j+1)]) * 0.1f;
    }
    
    //2. right cases

    for (int j = 1; j<ny-1; ++j){
      tmp_image[nx*(j+1)-1] = image[nx*(j+1)-1] * 0.6f + (image[nx*j-1] + image[nx*(j+2)-1] + image[nx*(j+1)-2]) * 0.1f;
    }

    //3. middle cases

    //#pragma omp simd
    for (int j = 0; j < (nx*(ny-2)); j+=nx) {
      for(int i = 1; i<nx-1;++i){
        tmp_image[j+i+nx] = image[j+i+nx] * 0.6f + (image[j+i+nx+1] + image[j+i+nx-1] + image[j+i] + image[j+i+(nx*2)]) * 0.1f;
      }
    }

    //  for(int i = 0 ; i< ny ; i++){
    //   for(int j =0 ; j< nx ; j++){
    //     int base = j+i*nx;

    //     tmp_image[base] = image[base] *0.6f;

    //     if(i==0)    tmp_image[base] += firstRowRecv[j]*0.1f; 
    //     if(i>0)     tmp_image[base] += image[j + (i-1)*nx] * 0.1f; 
    //     if(j>0)     tmp_image[base] += image[j-1 +i*nx] * 0.1f;
    //     if(i<ny-1)  tmp_image[base] += image[j + (i+1)*nx] *0.1f;
    //     if(j<nx-1)  tmp_image[base] += image[j+1 + i*nx] * 0.1f;
        
    //   }
    // }

  }
  else{
    
    firstRowSend = extractRow(image, firstRowSend, firstRowStart, firstRowEnd );
    lastRowSend  = extractRow(image, lastRowSend, lastRowStart, lastRowEnd );

    MPI_Sendrecv(firstRowSend, nx, MPI_FLOAT, rank - 1, 0, firstRowRecv, nx, MPI_FLOAT, rank -1, 0, MPI_COMM_WORLD, status);

    // MPI_Send(firstRowSend, nx, MPI_FLOAT,  rank -1, 0, MPI_COMM_WORLD);
    // MPI_Recv(firstRowRecv, nx, MPI_FLOAT, rank -1, MASTER, MPI_COMM_WORLD, status);
   
    MPI_Sendrecv(lastRowSend, nx, MPI_FLOAT, rank + 1, 0, lastRowRecv, nx, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, status);

    // MPI_Send(lastRowSend, nx, MPI_FLOAT,  rank +1, 0, MPI_COMM_WORLD);
    // MPI_Recv(lastRowRecv, nx, MPI_FLOAT, rank +1, MASTER, MPI_COMM_WORLD, status);

    //Corner cases cmonnnnn
    tmp_image[0] = image[0] * 0.6f + (image[nx] + image[1] + firstRowRecv[0]) * 0.1f; //comment
    tmp_image[nx-1] = image[nx-1] * 0.6f + (image[nx*2-1]+ image[nx-2] + firstRowRecv[nx-1]) * 0.1f;
    tmp_image[nx*ny-(nx)] = image[nx*ny-(nx)] * 0.6f + (image[nx*ny-(nx*2)] + image[nx*ny-(nx-1)] + lastRowRecv[0]) * 0.1f;
    tmp_image[nx*ny-1] = image[nx*ny-1] * 0.6f + (image[nx*ny-(nx+1)] + image[nx*ny-2] + lastRowRecv[nx-1]) * 0.1f;

    //top cases

    for (int j = 1; j<nx-1; ++j){
      tmp_image[j] = image[j] * 0.6f + (image[j-1] + image[j+1] + image[j+nx] + firstRowRecv[j]) * 0.1f;
    }

    //bottom cases
    
    for (int j = 1; j<nx-1; ++j){
      tmp_image[nx*ny-nx+j] = image[nx*ny-(nx)+j] * 0.6f + (image[nx*ny-(nx)+j-1] + image[nx*ny-(nx)+j+1] + image[nx*ny-(2*nx)+j] +lastRowRecv[j]) * 0.1f;
    }

    //1. left cases

    for (int j = 1; j<ny-1; ++j){
      tmp_image[nx*j] = image[nx*j] * 0.6f + (image[(nx*j)+1] + image[nx*(j-1)] + image[nx*(j+1)]) * 0.1f;
    }
    
    //2. right cases

    for (int j = 1; j<ny-1; ++j){
      tmp_image[nx*(j+1)-1] = image[nx*(j+1)-1] * 0.6f + (image[nx*j-1] + image[nx*(j+2)-1] + image[nx*(j+1)-2]) * 0.1f;
    }

    //3. middle cases

    //#pragma omp simd
    for (int j = 0; j < (nx*(ny-2)); j+=nx) {
      for(int i = 1; i<nx-1;++i){
        tmp_image[j+i+nx] = image[j+i+nx] * 0.6f + (image[j+i+nx+1] + image[j+i+nx-1] + image[j+i] + image[j+i+(nx*2)]) * 0.1f;
      }
    }

    // for(int i = 0 ; i< ny ; i++){
    //   for(int j =0 ; j< nx ; j++){
    //     int base = j+i*nx;

    //     tmp_image[base] = image[base] *0.6f;

    //     if(i==0)    tmp_image[base] += firstRowRecv[j]*0.1f; 
    //     if(i>0)     tmp_image[base] += image[j + (i-1)*nx] * 0.1f; 
    //     if(j>0)     tmp_image[base] += image[j-1 +i*nx] * 0.1f;
    //     if(i<ny-1)  tmp_image[base] += image[j + (i+1)*nx] *0.1f;
    //     if(j<nx-1)  tmp_image[base] += image[j+1 + i*nx] * 0.1f;
    //     if(i == ny-1)tmp_image[base] += lastRowRecv[j] * 0.1f;
    //   }
    // }
  }
}

// Create the input image
void init_image(const int nx, const int ny, float * image, float * tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
     
     image[j+ny*i] = 0.0;
     tmp_image[j+ny*i] = 0.0;
    }
  }

  // Checkerboard
   
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      for (int ii = i*ny/8; ii < (i+1)*ny/8; ++ii) {
        for (int jj = j*nx/8; jj < (j+1)*nx/8; ++jj) {
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
  for (int i = 0; i < ny; ++i) {
    for (int j = 0; j < nx; ++j) {
     if (image[j+i*ny] > maximum)
       maximum = image[j+i*ny];

    }
  }

  // Output image, converting to numbers 0-255
  for (int i = 0; i < ny; ++i) {
    for (int j = 0; j < nx; ++j) {
      //fputc((char)(255.0*image[j+ny*i]/maximum), fp);
     	fputc((char)(255.0*image[j+ny*i]/maximum),fp);
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
