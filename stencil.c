#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// Define output file name - should be final
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0


void stencil(const int nx, const int ny, float * image, float * temp_image,int rank,int noRanks);
void init_image(const int nx, const int ny, float * image, float * temp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
double wtime(void);
int main(int argc, char *argv[]) {

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Allocate the images
  float *image =malloc(sizeof(float)*nx*ny);

  float *temp_image = malloc(sizeof(float)*nx*ny);;

  // Initialising MPI
  MPI_Init(&argc, &argv);
  int rank;
  int noRanks;
  MPI_Comm_size(MPI_COMM_WORLD, &noRanks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0){
    image =malloc(sizeof(float)*ny*nx);
    temp_image = malloc(sizeof(float)*ny*nx);
    
    // Set the input image
    init_image(nx, ny, image, temp_image);

  }

  int sectionSize = nx * (ny/noRanks);

  //changing remainder:
  int remainder_section_size = (ny % noRanks ) * nx;
  
  float * bufferTmp;
  float * buffer;

  //array of our group noRanks
  if(rank == noRanks-1) {
    buffer = (float*) malloc(sizeof(float) * (sectionSize + remainder_section_size));
    bufferTmp = (float*)malloc(sizeof(float) * (sectionSize + remainder_section_size));
  } else {
    buffer = (float*)malloc(sizeof(float) * sectionSize);
    bufferTmp = (float*)malloc(sizeof(float) * sectionSize);
  }

  int* scounts = (int*) malloc(sizeof(int)  * noRanks) ;
  int* stride = (int*) malloc(sizeof(int) * noRanks);

  for(int i = 0; i < noRanks - 1; i++) {
    stride[i] = i * sectionSize;
    scounts[i] = sectionSize;
  }

  stride[noRanks-1] = (noRanks-1) * sectionSize;
  scounts[noRanks-1] = sectionSize + remainder_section_size;


  //float * buffer = malloc(sizeof(float) * sectionSize);
  //float * bufferTmp = malloc(sizeof(float) * sectionSize);

  //MPI_Scatter(image, sectionSize, MPI_FLOAT, buffer, sectionSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(image, scounts, stride, MPI_FLOAT, buffer ,scounts[rank],MPI_FLOAT, 0, MPI_COMM_WORLD);

  int noRows = (rank == noRanks-1) ? ny/noRanks + ny%noRanks : ny/noRanks;

  
  
  

  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {
    stencil(nx, noRows, buffer, bufferTmp,rank,noRanks);
    stencil(nx, noRows, bufferTmp, buffer,rank,noRanks);
  }
  double toc = wtime();



  float * final;

  final = malloc(sizeof(float)*ny*nx);
  
  //MPI_Gather(bufferTmp, sectionSize, MPI_FLOAT,final ,sectionSize, MPI_FLOAT,0, MPI_COMM_WORLD);
  MPI_Gatherv(bufferTmp, scounts[rank], MPI_FLOAT, final, scounts, stride, MPI_FLOAT, 0, MPI_COMM_WORLD);


  if(rank==0){
    output_image(OUTPUT_FILE, nx, ny, final);
  
  }

  MPI_Finalize();


  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");




  
  
  free(image);
}


float * getHalo(float* image,float * output, int start,int finish){
  for( int i =start ; i<= finish; i++){
    output[i-start] = image[i];
  }

  return output;
}
void stencil(const int nx, const int ny,  float *restrict image, float *restrict temp_image, int rank,int noRanks) {


  float * topRowSend = malloc(sizeof(float)* nx );
  float * topRowReceive = malloc(sizeof(float)* nx );  

  float * bottomRowSend = malloc(sizeof(float)* nx );
  float * bottomRowReceive = malloc(sizeof(float)* nx );
  
  //int start, finish, bottomStart, bottomFinish;


  int start = 0;
  int finish = nx-1;
  int bottomStart = (ny-1)* nx;
  int bottomFinish   = (ny-1)* nx + (nx-1);

  MPI_Status status;


  if(rank == 0){

    bottomRowSend = getHalo(image,bottomRowSend, bottomStart, bottomFinish);
    

    MPI_Ssend(bottomRowSend, nx, MPI_FLOAT,rank +1, 0, MPI_COMM_WORLD);
    MPI_Recv(bottomRowReceive, nx, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);

     for(int i = 0 ; i< ny ; i++){
      for(int j =0 ; j< nx ; j++){
        

        temp_image[j+i*nx] = image[j+i*nx] *0.6f;
        if(i>0)     temp_image[j+i*nx] += image[j + (i-1)*nx] * 0.1f; 
        if(j>0)     temp_image[j+i*nx] += image[j-1 +i*nx] * 0.1f;
        if(i<ny-1)  temp_image[j+i*nx] += image[j + (i+1)*nx] *0.1f;
        if(j<nx-1)  temp_image[j+i*nx] += image[j+1 + i*nx] * 0.1f;
        if(i==ny-1) temp_image[j+i*nx] += bottomRowReceive[j] * 0.1f;
      }
    }

  }
  else if(rank == noRanks - 1){

    if(noRanks%2 == 0) {
      topRowSend = getHalo(image,topRowSend, start, finish);
      MPI_Recv(topRowReceive, nx, MPI_FLOAT, rank-1 , 0, MPI_COMM_WORLD, &status);   
      MPI_Ssend(topRowSend, nx, MPI_FLOAT,rank -1, 0, MPI_COMM_WORLD);
    }
    else{
      topRowSend = getHalo(image,topRowSend, start, finish);
      MPI_Ssend(topRowSend, nx, MPI_FLOAT,rank -1, 0, MPI_COMM_WORLD);
      MPI_Recv(topRowReceive, nx, MPI_FLOAT, rank-1 , 0, MPI_COMM_WORLD, &status);   
    }


    
    

     for(int i = 0 ; i< ny ; i++){
      for(int j =0 ; j< nx ; j++){
        
        temp_image[j+i*nx] = image[j+i*nx] *0.6f;

        if(i==0)    temp_image[j+i*nx] += topRowReceive[j]*0.1f; 
        if(i>0)     temp_image[j+i*nx] += image[j + (i-1)*nx] * 0.1f; 
        if(j>0)     temp_image[j+i*nx] += image[j-1 +i*nx] * 0.1f;
        if(i<ny-1)  temp_image[j+i*nx] += image[j + (i+1)*nx] *0.1f;
        if(j<nx-1)  temp_image[j+i*nx] += image[j+1 + i*nx] * 0.1f;
        
      }
    }

  }
  else if(rank % 2 == 1){
    
    topRowSend = getHalo(image, topRowSend, start, finish );
    bottomRowSend  = getHalo(image, bottomRowSend, bottomStart, bottomFinish );

    MPI_Recv(topRowReceive, nx, MPI_FLOAT, rank -1, 0, MPI_COMM_WORLD, &status);
    MPI_Ssend(topRowSend, nx, MPI_FLOAT,  rank -1, 0, MPI_COMM_WORLD);
    

    MPI_Recv(bottomRowReceive, nx, MPI_FLOAT, rank +1, 0, MPI_COMM_WORLD, &status);
    MPI_Ssend(bottomRowSend, nx, MPI_FLOAT,  rank +1, 0, MPI_COMM_WORLD);
    
   
    

    for(int i = 0 ; i< ny ; i++){
      for(int j =0 ; j< nx ; j++){
        

        temp_image[j+i*nx] = image[j+i*nx] *0.6f;

        if(i==0)    temp_image[j+i*nx] += topRowReceive[j]*0.1f; 
        if(i>0)     temp_image[j+i*nx] += image[j + (i-1)*nx] * 0.1f; 
        if(j>0)     temp_image[j+i*nx] += image[j-1 +i*nx] * 0.1f;
        if(i<ny-1)  temp_image[j+i*nx] += image[j + (i+1)*nx] *0.1f;
        if(j<nx-1)  temp_image[j+i*nx] += image[j+1 + i*nx] * 0.1f;
        if(i == ny-1)temp_image[j+i*nx] += bottomRowReceive[j] * 0.1f;
      }
    }


  }else if(rank % 2==0){
    topRowSend = getHalo(image, topRowSend, start, finish );
    bottomRowSend  = getHalo(image, bottomRowSend, bottomStart, bottomFinish );

    MPI_Ssend(topRowSend, nx, MPI_FLOAT,  rank -1, 0, MPI_COMM_WORLD);
    MPI_Recv(topRowReceive, nx, MPI_FLOAT, rank -1, 0, MPI_COMM_WORLD, &status);
    
    
    MPI_Ssend(bottomRowSend, nx, MPI_FLOAT,  rank +1, 0, MPI_COMM_WORLD);
    MPI_Recv(bottomRowReceive, nx, MPI_FLOAT, rank +1, 0, MPI_COMM_WORLD, &status);
    
    
   
    

    for(int i = 0 ; i< ny ; i++){
      for(int j =0 ; j< nx ; j++){
        

        temp_image[j+i*nx] = image[j+i*nx] *0.6f;

        if(i==0)    temp_image[j+i*nx] += topRowReceive[j]*0.1f; 
        if(i>0)     temp_image[j+i*nx] += image[j + (i-1)*nx] * 0.1f; 
        if(j>0)     temp_image[j+i*nx] += image[j-1 +i*nx] * 0.1f;
        if(i<ny-1)  temp_image[j+i*nx] += image[j + (i+1)*nx] *0.1f;
        if(j<nx-1)  temp_image[j+i*nx] += image[j+1 + i*nx] * 0.1f;
        if(i == ny-1)temp_image[j+i*nx] += bottomRowReceive[j] * 0.1f;
      }
    }

  }


 }

// Create the input image
void init_image(const int nx, const int ny, float * image, float * temp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
     
     image[j+ny*i] = 0.0;
     temp_image[j+ny*i] = 0.0;
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
  // This is used to rescale the image
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
