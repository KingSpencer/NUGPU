
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);               
 
 
#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \
 
 
#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);     
  
#define TILE_SIZE 16
#define CUDA_TIMING
#define DEBUG
#define HIST_SIZE 256
#define SCAN_SIZE HIST_SIZE*2
#define WARP_SIZE 32

unsigned char *input_gpu;
unsigned char *output_gpu;
float *cdf;
float *hist_array;

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,  &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
    #if defined(DEBUG) || defined(_DEBUG)
        if (result != cudaSuccess) {
            fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
            exit(-1);
        }
    #endif
        return result;
}
                
// Add GPU kernel and functions
// HERE!!!
__global__ void kernel_hist(unsigned char *input, 
                       float *hist, unsigned int height, unsigned int width){

    //int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    //int y = blockIdx.y*TILE_SIZE+threadIdx.y;
    int interval_x = (width - 1) / TILE_SIZE + 1;
    int interval_y = (height - 1) / TILE_SIZE + 1;
    int y = threadIdx.y*interval_y+blockIdx.y;
    int x = threadIdx.x*interval_x+blockIdx.x;
    //int location =  y*TILE_SIZE*gridDim.x+x;
    //int location =  y*width+x;
    int location = y*width + x;

    int block_loc = threadIdx.y*TILE_SIZE+threadIdx.x;
    // HIST_SIZE 256
    __shared__ unsigned int hist_shared[HIST_SIZE];

    /*if (block_loc<HIST_SIZE) hist_shared[block_loc]=0;
    __syncthreads();*/
    hist_shared[block_loc]=0;
    __syncthreads();

    if (x<width && y<height) atomicAdd(&(hist_shared[input[location]]),1);
    __syncthreads();

    //if (block_loc<HIST_SIZE) {
    atomicAdd(&(hist[block_loc]),(float)hist_shared[block_loc]);
    //}
}

__global__ void kernel_hist3(unsigned char *input, 
                       float *hist, unsigned int height, unsigned int width){

    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;
    // int interval_x = (width - 1) / TILE_SIZE + 1;
    // int interval_y = (height - 1) / TILE_SIZE + 1;
    // int y = threadIdx.y*interval_y+blockIdx.y;
    // int x = threadIdx.x*interval_x+blockIdx.x;
    // int location =  y*TILE_SIZE*gridDim.x+x;
    // int location =  y*width+x;
    int location = y*width + x;

    int block_loc = threadIdx.y*TILE_SIZE+threadIdx.x;
    // HIST_SIZE 256
    __shared__ unsigned int hist_shared[HIST_SIZE];

    /*if (block_loc<HIST_SIZE) hist_shared[block_loc]=0;
    __syncthreads();*/
    hist_shared[block_loc]=0;
    __syncthreads();

    if (x<width && y<height) atomicAdd(&(hist_shared[input[location]]),1);
    __syncthreads();

    //if (block_loc<HIST_SIZE) {
    atomicAdd(&(hist[block_loc]),(float)hist_shared[block_loc]);
    //}
}

__global__ void kernel_hist2 (float*histo, unsigned char* data,int size,int BINS,int R){

  __shared__ int Hs[2048]; //(BINS+1*)R

  //Warp index
  const int warpid=(int)(threadIdx.x / WARP_SIZE);
  const int lane=threadIdx.x % WARP_SIZE;
  const int warps_block=blockDim.x /WARP_SIZE;

  // Offset to per-block
  const int off_rep = (BINS +1) * (threadIdx.x % R);

  //Constants for interleaved read access
  const int begin = (size / warps_block) * warpid + WARP_SIZE * blockIdx.x +lane;
  const int end = (size / warps_block) * (warpid+1);
  const int step = WARP_SIZE * gridDim.x;

  //Initializetion
  for (int pos =threadIdx.x; pos< (BINS+1)*R;pos+=blockDim.x) Hs[pos]=0;

  __syncthreads();

  for (int i=begin;i<end;i+=step){
    atomicAdd(&Hs[off_rep+data[i]],1);
  }

  __syncthreads();

  //Merge
  for (int pos=threadIdx.x;pos<BINS;pos+=blockDim.x){
    int sum=0;
    for (int base =0;base<(BINS+1)*R;base+=BINS+1)
      sum+=Hs[base+pos];
    atomicAdd(histo +pos, sum);
  }
}

__global__ void kernel_hist_global(unsigned char *input, 
                       float *hist, unsigned int height, unsigned int width){

    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    int location =  y*TILE_SIZE*gridDim.x+x;
    // int block_loc = threadIdx.y*TILE_SIZE+threadIdx.x;
    // HIST_SIZE 256
    //__shared__ unsigned int hist_shared[HIST_SIZE];

    if (x<width && y<height) atomicAdd(&(hist[input[location]]),1);
    __syncthreads();

}



__global__ void kernel_cdf(float *hist_array, int size){

    __shared__ float p[HIST_SIZE];
    int tid=blockIdx.x*blockDim.x+threadIdx.x;

    if (tid<HIST_SIZE){
        p[tid]=hist_array[tid] / (float)size;
    }
    __syncthreads();

    for (int i=1; i<=HIST_SIZE;i=i<<1){
        int ai=(threadIdx.x+1)*i*2-1;
        if (ai<HIST_SIZE) p[ai]+=p[ai-i];
        __syncthreads();
    }

    for (int i=HIST_SIZE/2;i>0;i=i>>1){
        int bi=(threadIdx.x+1)*i*2-1;
        if (bi+i<HIST_SIZE) p[bi+i]+=p[bi];
        __syncthreads();
    }

    if (tid<HIST_SIZE) hist_array[tid]=p[threadIdx.x];

}

__global__ void kernel_equlization( 
                               unsigned char *input,
                               float * cdf, unsigned int height, unsigned int width){
    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    int location =  y*TILE_SIZE*gridDim.x+x;
    float min=cdf[0]; float down=0.0F; float up=255.0F;

    if (x<width && y<height) {
        float value=255.0F * (cdf[input[location]]-min) / (1.0F-min);
        if (value<down) value=down;
        if (value>up) value=up;
        input[location]=(unsigned char) value; 

    }

}

void histogram_gpu(unsigned char *data, 
                   unsigned int height, 
                   unsigned int width){
    
    //float cdf_cpu[HIST_SIZE]={0};
    //unsigned int hist_cpu[HIST_SIZE]={0};

    int gridXSize = 1 + (( width - 1) / TILE_SIZE);
    int gridYSize = 1 + ((height - 1) / TILE_SIZE);
    
    int XSize = gridXSize*TILE_SIZE;
    int YSize = gridYSize*TILE_SIZE;
    
    // Both are the same size (CPU/GPU).
    int size = XSize*YSize;

    int block_size2=256;
    int grid_size2=1+((width*height-1) / block_size2);
    
    // Allocate arrays in GPU memory
    float Ktime00;
    TIMER_CREATE(Ktime00);
    TIMER_START(Ktime00);
    cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char));
    cudaMalloc((void**)&hist_array  , HIST_SIZE*sizeof(float));

    // cudaMallocManaged((void**)&input_gpu   , size*sizeof(unsigned char));
    // cudaMallocManaged((void**)&hist_array  , HIST_SIZE*sizeof(float));

    // cudaHostRegister((void*)&input_gpu   , size*sizeof(unsigned char),0);
    // cudaHostRegister((void*)&hist_array  , HIST_SIZE*sizeof(float),0);

    TIMER_END(Ktime00);
    printf("CUDA_MALLOC Execution Time: %f ms\n", Ktime00);
    // cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char));
    // checkCuda(cudaMalloc((void**)&cdf         , size*sizeof(float)));
    // init output_gpu to 0
    //checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
    cudaMemset(hist_array , 0 , HIST_SIZE*sizeof(float));
    // checkCuda(cudaMemset(cdf        , 0 , HIST_SIZE*sizeof(float)));
    
    // Copy data to GPU
    cudaMemcpy(input_gpu, 
        data, 
        size*sizeof(char), 
        cudaMemcpyHostToDevice);

    checkCuda(cudaDeviceSynchronize());

    // Execute algorithm

    dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    //hist2
    dim3 dimGrid2(grid_size2);
    dim3 dimBlock2(block_size2);

    dim3 dimCdfGrid(1);
    dim3 dimCdfBlock(HIST_SIZE);

    // Kernel Call
    #if defined(CUDA_TIMING)
        float Ktime;
        TIMER_CREATE(Ktime);
        TIMER_START(Ktime);
    #endif
        
        float Ktime0;
        TIMER_CREATE(Ktime0);
        TIMER_START(Ktime0);
        kernel_hist3<<<dimGrid, dimBlock>>>(input_gpu, 
                                      hist_array,
                                      height,
                                      width);
       // kernel_hist2<<<dimGrid2, dimBlock2>>>(hist_array,
       //                                   input_gpu,
       //                                   size,
       //                                   255,
       //                                   8);
        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
        TIMER_END(Ktime0);
        printf("HIST Kernel Execution Time: %f ms\n", Ktime0);

        float Ktime1;
        TIMER_CREATE(Ktime1);
        TIMER_START(Ktime1);
        kernel_cdf<<<dimCdfGrid,dimCdfBlock>>>( 
                                      hist_array,
                                      height*width);
        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
        TIMER_END(Ktime1);
        printf("CDF Kernel Execution Time: %f ms\n", Ktime1);
   

// ///////////////////////////////////////////        
//         checkCuda(cudaMemcpy(hist_cpu,
//          hist_array,
//          HIST_SIZE*sizeof(unsigned int),
//          cudaMemcpyDeviceToHost));
//         checkCuda(cudaDeviceSynchronize());
//         cdf_cpu[0]=hist_cpu[0]/ ((float) height*width);
//         for (int i=1;i<HIST_SIZE;i++){
//          cdf_cpu[i]=cdf_cpu[i-1]+hist_cpu[i]/ ((float) height*width);
//         }
//         checkCuda(cudaMemcpy(cdf,
//          cdf_cpu,
//          HIST_SIZE*sizeof(float),
//          cudaMemcpyHostToDevice));
//         checkCuda(cudaDeviceSynchronize());
// ///////////////////////////////////////////



        float Ktime2;
        TIMER_CREATE(Ktime2);
        TIMER_START(Ktime2);

        kernel_equlization<<<dimGrid, dimBlock>>>(
                                      input_gpu,
                                      hist_array, 
                                      height,
                                      width);

        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
        TIMER_END(Ktime2);
        printf("EQUALIZATION Kernel Execution Time: %f ms\n", Ktime2);
    
    #if defined(CUDA_TIMING)
        TIMER_END(Ktime);
        printf("Kernel Execution Time: %f ms\n", Ktime);
    #endif
        
    // Retrieve results from the GPU
    checkCuda(cudaMemcpy(data, 
            input_gpu, 
            size*sizeof(unsigned char), 
            cudaMemcpyDeviceToHost));

    // Free resources and end the program
    // checkCuda(cudaFree(output_gpu));
    checkCuda(cudaFree(input_gpu));
    checkCuda(cudaFree(hist_array));

}

void histogram_gpu_warmup(unsigned char *data, 
                   unsigned int height, 
                   unsigned int width){
                         
    //float cdf_cpu[HIST_SIZE]={0};
    //unsigned int hist_cpu[HIST_SIZE]={0};

    int gridXSize = 1 + (( width - 1) / TILE_SIZE);
    int gridYSize = 1 + ((height - 1) / TILE_SIZE);
    
    int XSize = gridXSize*TILE_SIZE;
    int YSize = gridYSize*TILE_SIZE;
    
    // Both are the same size (CPU/GPU).
    int size = XSize*YSize;

    int block_size2=256;
    int grid_size2=1+((width*height-1) / block_size2);
    
    // Allocate arrays in GPU memory
    float Ktime00;
    TIMER_CREATE(Ktime00);
    TIMER_START(Ktime00);

    cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char));
    cudaMalloc((void**)&hist_array  , HIST_SIZE*sizeof(float));

    // cudaMallocManaged((void**)&input_gpu   , size*sizeof(unsigned char));
    // cudaMallocManaged((void**)&hist_array  , HIST_SIZE*sizeof(float));

    // cudaHostRegister((void*)&input_gpu   , size*sizeof(unsigned char),0);
    // cudaHostRegister((void*)&hist_array  , HIST_SIZE*sizeof(float),0);

    TIMER_END(Ktime00);
    //printf("CUDA_MALLOC Execution Time: %f ms\n", Ktime00);
    // cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char));
    // checkCuda(cudaMalloc((void**)&cdf         , size*sizeof(float)));
    // init output_gpu to 0
    //checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
    cudaMemset(hist_array , 0 , HIST_SIZE*sizeof(float));
    // checkCuda(cudaMemset(cdf        , 0 , HIST_SIZE*sizeof(float)));
    
    // Copy data to GPU
    cudaMemcpy(input_gpu, 
        data, 
        size*sizeof(char), 
        cudaMemcpyHostToDevice);

    checkCuda(cudaDeviceSynchronize());

    // Execute algorithm

    dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    //hist2
    dim3 dimGrid2(grid_size2);
    dim3 dimBlock2(block_size2);

    dim3 dimCdfGrid(1);
    dim3 dimCdfBlock(HIST_SIZE);

    // Kernel Call
    #if defined(CUDA_TIMING)
        float Ktime;
        TIMER_CREATE(Ktime);
        TIMER_START(Ktime);
    #endif
        
        float Ktime0;
        TIMER_CREATE(Ktime0);
        TIMER_START(Ktime0);
        kernel_hist3<<<dimGrid, dimBlock>>>(input_gpu, 
                                      hist_array,
                                      height,
                                      width);
        // kernel_hist2<<<dimGrid2, dimBlock2>>>(hist_array,
        //                                  input_gpu,
        //                                  size,
        //                                  255,
        //                                  8);
        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
        TIMER_END(Ktime0);
        //printf("HIST Kernel Execution Time: %f ms\n", Ktime0);

        float Ktime1;
        TIMER_CREATE(Ktime1);
        TIMER_START(Ktime1);
        kernel_cdf<<<dimCdfGrid,dimCdfBlock>>>( 
                                      hist_array,
                                      height*width);
        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
        TIMER_END(Ktime1);
        //printf("CDF Kernel Execution Time: %f ms\n", Ktime1);
   

// ///////////////////////////////////////////        
//         checkCuda(cudaMemcpy(hist_cpu,
//          hist_array,
//          HIST_SIZE*sizeof(unsigned int),
//          cudaMemcpyDeviceToHost));
//         checkCuda(cudaDeviceSynchronize());
//         cdf_cpu[0]=hist_cpu[0]/ ((float) height*width);
//         for (int i=1;i<HIST_SIZE;i++){
//          cdf_cpu[i]=cdf_cpu[i-1]+hist_cpu[i]/ ((float) height*width);
//         }
//         checkCuda(cudaMemcpy(cdf,
//          cdf_cpu,
//          HIST_SIZE*sizeof(float),
//          cudaMemcpyHostToDevice));
//         checkCuda(cudaDeviceSynchronize());
// ///////////////////////////////////////////



        float Ktime2;
        TIMER_CREATE(Ktime2);
        TIMER_START(Ktime2);

        kernel_equlization<<<dimGrid, dimBlock>>>(
                                      input_gpu,
                                      hist_array, 
                                      height,
                                      width);

        checkCuda(cudaPeekAtLastError());                                     
        checkCuda(cudaDeviceSynchronize());
        TIMER_END(Ktime2);
        //printf("EQUALIZATION Kernel Execution Time: %f ms\n", Ktime2);
    
    #if defined(CUDA_TIMING)
        TIMER_END(Ktime);
        //printf("Kernel Execution Time: %f ms\n", Ktime);
    #endif
        
    // Retrieve results from the GPU
    checkCuda(cudaMemcpy(data, 
            input_gpu, 
            size*sizeof(unsigned char), 
            cudaMemcpyDeviceToHost));

    // Free resources and end the program
    // checkCuda(cudaFree(output_gpu));
    checkCuda(cudaFree(input_gpu));
    checkCuda(cudaFree(hist_array));

}

