#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// Define output file name - should be final
#define OUTPUT_FILE "stencil.pgm"



void stencil(const int nx, const int ny, float * image, float * tmp_image,int rank,int size);
void init_image(const int nx, const int ny, float * image, float * tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
float *find_row(float *inputArray, float *outputArray, int start, int end);
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

  float *tmp_image = malloc(sizeof(float)*nx*ny);;

  // Initialising MPI
  MPI_Init(&argc, &argv);
  int rank;
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0){
    image =malloc(sizeof(float)*ny*nx);
    tmp_image = malloc(sizeof(float)*ny*nx);
    
    // Set the input image
    init_image(nx, ny, image, tmp_image);

  }

  int sectionSize = nx*ny/size;
  float * buffer = malloc(sizeof(float) * sectionSize);
  float * bufferTmp = malloc(sizeof(float) * sectionSize);

  int segmentSize = ny / size;
  int remainderSize = ny % size;

  int start, end;

  start = rank * nx * segmentSize;
  if (rank != size - 1)
  {
    end = rank * nx * segmentSize + (nx * segmentSize) - 1;
  }
  else
  { 
    end = nx * ny - 1;
  }

  buffer = find_row(image, buffer, start, end);
  
  //MPI_Scatter(image, sectionSize, MPI_FLOAT, buffer, sectionSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
 

  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {
    stencil(nx, ny/size, buffer, bufferTmp,rank,size);
    stencil(nx, ny/size, bufferTmp, buffer,rank,size);
  }
  double toc = wtime();



  float * result;

  result = malloc(sizeof(float)*ny*nx);
  
  MPI_Gather(bufferTmp, sectionSize, MPI_FLOAT,result ,sectionSize, MPI_FLOAT,0, MPI_COMM_WORLD);


  if(rank==0){
    output_image(OUTPUT_FILE, nx, ny, result);
  
  }
  MPI_Finalize();


  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");




  
  
  free(image);
}


float * find_row(float* values,float * offset_array, int start,int end){
  for( int i =start ; i<= end; i++){
    offset_array[i-start] = values[i];
  }

  return offset_array;
}
void stencil(const int nx, const int ny,  float *restrict image, float *restrict tmp_image, int rank,int size) {


  float * send_first_row = malloc(sizeof(float)* nx );
  float * receive_first_row = malloc(sizeof(float)* nx );  

  float * send_last_row = malloc(sizeof(float)* nx );
  float * receive_last_row = malloc(sizeof(float)* nx );
  
  //int start, end, bottom_start, bottom_end;


  int start = 0;
  int end = nx-1;
  int bottom_start = (ny-1)* nx;
  int bottom_end   = (ny-1)* nx + (nx-1);

  MPI_Status status;


  if(rank == 0){

    send_last_row = find_row(image,send_last_row, bottom_start, bottom_end);
    

    MPI_Ssend(send_last_row, nx, MPI_FLOAT,rank +1, 0, MPI_COMM_WORLD);
    MPI_Recv(receive_last_row, nx, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);

     for(int i = 0 ; i< ny ; i++){
      for(int j =0 ; j< nx ; j++){
        

        tmp_image[j+i*nx] = image[j+i*nx] *0.6f;
        if(i>0)     tmp_image[j+i*nx] += image[j + (i-1)*nx] * 0.1f; 
        if(j>0)     tmp_image[j+i*nx] += image[j-1 +i*nx] * 0.1f;
        if(i<ny-1)  tmp_image[j+i*nx] += image[j + (i+1)*nx] *0.1f;
        if(j<nx-1)  tmp_image[j+i*nx] += image[j+1 + i*nx] * 0.1f;
        if(i==ny-1) tmp_image[j+i*nx] += receive_last_row[j] * 0.1f;
      }
    }

  }
  else if(rank == size - 1){

    send_first_row = find_row(image,send_first_row, start, end);
    MPI_Recv(receive_first_row, nx, MPI_FLOAT, rank-1 , 0, MPI_COMM_WORLD, &status);   
    MPI_Ssend(send_first_row, nx, MPI_FLOAT,rank -1, 0, MPI_COMM_WORLD);
    

    
    

     for(int i = 0 ; i< ny ; i++){
      for(int j =0 ; j< nx ; j++){
        
        tmp_image[j+i*nx] = image[j+i*nx] *0.6f;

        if(i==0)    tmp_image[j+i*nx] += receive_first_row[j]*0.1f; 
        if(i>0)     tmp_image[j+i*nx] += image[j + (i-1)*nx] * 0.1f; 
        if(j>0)     tmp_image[j+i*nx] += image[j-1 +i*nx] * 0.1f;
        if(i<ny-1)  tmp_image[j+i*nx] += image[j + (i+1)*nx] *0.1f;
        if(j<nx-1)  tmp_image[j+i*nx] += image[j+1 + i*nx] * 0.1f;
        
      }
    }

  }
  else if(rank % 2 == 1){
    
    send_first_row = find_row(image, send_first_row, start, end );
    send_last_row  = find_row(image, send_last_row, bottom_start, bottom_end );

    MPI_Recv(receive_first_row, nx, MPI_FLOAT, rank -1, 0, MPI_COMM_WORLD, &status);
    MPI_Ssend(send_first_row, nx, MPI_FLOAT,  rank -1, 0, MPI_COMM_WORLD);
    

    MPI_Recv(receive_last_row, nx, MPI_FLOAT, rank +1, 0, MPI_COMM_WORLD, &status);
    MPI_Ssend(send_last_row, nx, MPI_FLOAT,  rank +1, 0, MPI_COMM_WORLD);
    
   
    

    for(int i = 0 ; i< ny ; i++){
      for(int j =0 ; j< nx ; j++){
        

        tmp_image[j+i*nx] = image[j+i*nx] *0.6f;

        if(i==0)    tmp_image[j+i*nx] += receive_first_row[j]*0.1f; 
        if(i>0)     tmp_image[j+i*nx] += image[j + (i-1)*nx] * 0.1f; 
        if(j>0)     tmp_image[j+i*nx] += image[j-1 +i*nx] * 0.1f;
        if(i<ny-1)  tmp_image[j+i*nx] += image[j + (i+1)*nx] *0.1f;
        if(j<nx-1)  tmp_image[j+i*nx] += image[j+1 + i*nx] * 0.1f;
        if(i == ny-1)tmp_image[j+i*nx] += receive_last_row[j] * 0.1f;
      }
    }


  }else if(rank % 2==0){
    send_first_row = find_row(image, send_first_row, start, end );
    send_last_row  = find_row(image, send_last_row, bottom_start, bottom_end );

    MPI_Ssend(send_first_row, nx, MPI_FLOAT,  rank -1, 0, MPI_COMM_WORLD);
    MPI_Recv(receive_first_row, nx, MPI_FLOAT, rank -1, 0, MPI_COMM_WORLD, &status);
    
    
    MPI_Ssend(send_last_row, nx, MPI_FLOAT,  rank +1, 0, MPI_COMM_WORLD);
    MPI_Recv(receive_last_row, nx, MPI_FLOAT, rank +1, 0, MPI_COMM_WORLD, &status);
    
    
   
    

    for(int i = 0 ; i< ny ; i++){
      for(int j =0 ; j< nx ; j++){
        

        tmp_image[j+i*nx] = image[j+i*nx] *0.6f;

        if(i==0)    tmp_image[j+i*nx] += receive_first_row[j]*0.1f; 
        if(i>0)     tmp_image[j+i*nx] += image[j + (i-1)*nx] * 0.1f; 
        if(j>0)     tmp_image[j+i*nx] += image[j-1 +i*nx] * 0.1f;
        if(i<ny-1)  tmp_image[j+i*nx] += image[j + (i+1)*nx] *0.1f;
        if(j<nx-1)  tmp_image[j+i*nx] += image[j+1 + i*nx] * 0.1f;
        if(i == ny-1)tmp_image[j+i*nx] += receive_last_row[j] * 0.1f;
      }
    }

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
