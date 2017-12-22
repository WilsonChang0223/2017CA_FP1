#include <iostream>
#include <cstdlib>
#include "mm.h"
using namespace std;

//declare variables for GPU
int *devInputA, *devInputB , *devOut_share , *devOut_nonshare;

void initGPU()
{   
	//allocate memory space(VRAM) on GPU
	cudaMalloc(&devInputA, sizeof(int)* SIZE * SIZE  );
	cudaMalloc(&devInputB, sizeof(int)* SIZE * SIZE );
	cudaMalloc(&devOut_share, sizeof(int)* SIZE *SIZE  );
	cudaMalloc(&devOut_nonshare, sizeof(int)* SIZE *SIZE  );
	
	//copy data from DRAM to VRAM
	cudaMemcpy(devInputA, MAT_A, sizeof(int)* SIZE * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(devInputB, MAT_B, sizeof(int)* SIZE * SIZE, cudaMemcpyHostToDevice);
	
}

void MatrixMul_CPU(int* A, int* B, int* C)
{
	for(int i=0;i<SIZE;i++){
		for(int j=0;j<SIZE;j++){
			for(int k=0;k<SIZE;k++){
				C[i*SIZE+j] += A[i*SIZE+k] * B[k*SIZE+j];
			}
		}
	}
}
__global__ void MatrixMul_GPUnonshared(int* A, int* B, int* C)
{
	int tx = threadIdx.x; int ty = threadIdx.y;
	int bx = blockIdx.x; int by = blockIdx.y;
	
	int Row = by * TILE_SIZE + ty;
	int Col = bx * TILE_SIZE + tx;
	int Pvalue = 0;
	
	for (int k = 0; k < SIZE; ++k)
		Pvalue += A[Row*SIZE+k] * B[k*SIZE+Col];

	__syncthreads();
	
	C[Row*SIZE+Col] = Pvalue;
}

__global__ void MatrixMul_GPUshared(int* A, int* B, int* C)
{
	int tx = threadIdx.x; int ty = threadIdx.y;
	int bx = blockIdx.x; int by = blockIdx.y;
	
	//allocate shared memory
	__shared__ int A_share[TILE_SIZE][TILE_SIZE];
	__shared__ int B_share[TILE_SIZE][TILE_SIZE];
	
	int Row = by * TILE_SIZE + ty;
	int Col = bx * TILE_SIZE + tx;
	int Pvalue = 0;
	
	for (int m = 0; m < SIZE/TILE_SIZE; ++m) {
		//load data from GPU global memory to GPU shared memory
		A_share[ty][tx] = A[Row*SIZE + m*TILE_SIZE + tx];
		B_share[ty][tx] = B[Col + (m*TILE_SIZE + ty)*SIZE];
		__syncthreads();
		
		for (int k = 0; k < TILE_SIZE; ++k)
			Pvalue += A_share[ty][k] * B_share[k][tx];
		__syncthreads();
	}
	__syncthreads();
	C[Row*SIZE+Col] = Pvalue;
}

int main()
{
	int ExecTime_CPU,ExecTime_GPUnonshared, ExecTime_GPUshared;
	timespec time_begin, time_end;     
	init();
	initGPU();
	dim3 threadsPerBlock(TILE_SIZE,TILE_SIZE);
	dim3 numBlocks(SIZE/TILE_SIZE,SIZE/TILE_SIZE);
	
	clock_gettime(CLOCK_REALTIME, &time_begin);
	MatrixMul_CPU(MAT_A,MAT_B,outCPU);
	clock_gettime(CLOCK_REALTIME, &time_end);
	ExecTime_CPU = timespec_diff_ns(time_begin, time_end);
	cout << "ExecTime_CPU is = "  <<  ExecTime_CPU  << "ns" << endl;

	clock_gettime(CLOCK_REALTIME, &time_begin);
	MatrixMul_GPUnonshared<<<numBlocks,threadsPerBlock>>>(devInputA,devInputB,devOut_nonshare); 
	clock_gettime(CLOCK_REALTIME, &time_end);
	ExecTime_GPUnonshared = timespec_diff_ns(time_begin, time_end);
	cout << "ExecTime_GPUnonshared is = "  <<  ExecTime_GPUnonshared  << "ns" << endl;
	
	clock_gettime(CLOCK_REALTIME, &time_begin);
	MatrixMul_GPUshared<<<numBlocks,threadsPerBlock>>>(devInputA,devInputB,devOut_share); 
	clock_gettime(CLOCK_REALTIME, &time_end);
	ExecTime_GPUshared = timespec_diff_ns(time_begin, time_end);
	cout << "ExecTime_GPUshared is = "  <<  ExecTime_GPUshared  << "ns" << endl;

	//copy data from VRAM to DRAM 
	cudaMemcpy(outGPU_nonshare, devOut_nonshare , sizeof(int) * SIZE * SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(outGPU_share   , devOut_share    , sizeof(int) * SIZE * SIZE, cudaMemcpyDeviceToHost);
	
	cudaFree(&devInputA);
	cudaFree(&devInputB);
	cudaFree(&devOut_share);
	cudaFree(&devOut_nonshare);
	
	if(checker())
		cout << "You pass the check" << endl;
	else 
		cout << "There is something wrong" << endl;
	
	return 0;
}

