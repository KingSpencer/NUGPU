#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void initialize(int N, float *a, float *b, float *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		c[i] = 0;
		a[i] = 1 + i;
		b[i] = 1 - i;
	}
}

__global__ void addVectors(int N, float *a, float *b, float *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N){
		c[i] = a[i] + b[i];
	}
}

int main (int argc, char **argv){
	
	if (argc != 2) exit (1);
	int N = atoi(argv[1]);
	int block_size = 512;
	int grid_size = (N-1) / block_size + 1;

	float *a, *b, *c;
	cudaMallocManaged (&a, N*sizeof(float));
	cudaMallocManaged (&b, N*sizeof(float)); 
	cudaMallocManaged (&c, N*sizeof(float));

	initialize<<<grid_size, block_size>>>(N,a,b,c);
	cudaDeviceSynchronize();
	addVectors<<<grid_size, block_size>>>(N,a,b,c);
	cudaDeviceSynchronize();

	for (int i = 0; i < 5; i++) {
		printf("%f\n", c[i]);
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}
