#include<utility>
#include<stdio.h>
#include<assert.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <utility>
#include <time.h>
#include <iomanip>

// HEADER FILES
#include "Helper/fillThreadsPerBlock.h"
#include "Helper/level.h"
#include "Helper/jacobi.h"
#include "Helper/residual.h"
#include "Helper/setGPU.h"
#include "Helper/srjSchemes.h"

// #define RUN_CPU_FLAG 1
#define RUN_GPU_FLAG 1
// #define RUN_SHARED_FLAG 1

// Determine which header files to include based on which directives are active
#ifdef RUN_CPU_FLAG
#include "jacobi-1D-cpu.h"
#endif
#ifdef RUN_GPU_FLAG
#include "jacobi-1D-gpu.h"
#endif
#ifdef RUN_SHARED_FLAG
#include "jacobi-1D-shared-srj-shifted.h"
#endif

int main(int argc, char *argv[])
{
    /* Inputs and Settings */
    const int nDim = atoi(argv[1]); 
	const int threadsPerBlock = atoi(argv[2]);
	const int numCycles = atoi(argv[3]);
	const int levelSRJ = atoi(argv[4]);
    
    /* Set the correct GPU to use (Endeavour GPUs: "TITAN V" OR "GeForce GTX 1080 Ti") */
	std::string gpuToUse = "TITAN V"; // "GeForce GTX 1080 Ti"; 
    setGPU(gpuToUse);
    
	/* Initialize initial condition and rhs */
	const int Mcopies = 2;
    int nGrids = nDim + 2;
    double * initX = new double[nGrids * Mcopies];
    double * rhs = new double[nGrids * Mcopies];
    int stride;
	for (int Mcopy = 0; Mcopy < Mcopies; Mcopy++) {
		stride = Mcopy * nGrids;
		for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
			if (iGrid == 0 || iGrid == nGrids-1) {
				initX[iGrid + stride] = 0.0f;
			}
			else {
				initX[iGrid + stride] = 1.0f;
			}
			rhs[iGrid + stride] = 1.0f;
		}
	}
    
	/* Load SRJ schemes from Python txt files */
	int numSchemes = 25;
	int numSchemeParams = 9710;
	double * srjSchemes = new double[numSchemeParams];
	int indexPointer[numSchemes] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	loadSRJSchemes(srjSchemes, numSchemeParams);
	loadIndexPointer(indexPointer, numSchemes);

	/* CPU SRJ Jacobi*/
#ifdef RUN_CPU_FLAG
	printf("===============CPU SRJ============================\n");
    double cpuJacobiResidual;
    double * solutionJacobiCpu = new double[nGrids];
	// solutionJacobiCpu = jacobiCpuSRJ(initX, rhs, nGrids, srjSchemes, indexPointer, numSchemes, numCycles, levelSRJ);
	solutionJacobiCpu = jacobiCpuSRJHeuristic(initX, rhs, nGrids, srjSchemes, indexPointer, numSchemes, numCycles);
	cpuJacobiResidual = residual1DPoisson(solutionJacobiCpu, rhs, nGrids);
	printf("Residual of the Jacobi CPU solution is %f\n", cpuJacobiResidual);
#endif 

	/* GPU SRJ Jacobi */
#ifdef RUN_GPU_FLAG 
	printf("===============GPU SRJ============================\n");
    double gpuJacobiResidual;
    double * solutionJacobiGpu = new double[nGrids];
	float gpuSRJTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
 	solutionJacobiGpu = jacobiGpuSRJ(initX, rhs, nGrids, srjSchemes, indexPointer, numSchemes, threadsPerBlock, numCycles, levelSRJ, Mcopies);
 	// solutionJacobiGpu = jacobiGpuSRJHeuristic(initX, rhs, nGrids, srjSchemes, indexPointer, numSchemes, threadsPerBlock, numCycles);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuSRJTime, start, stop);
	gpuJacobiResidual = residual1DPoisson(solutionJacobiGpu, rhs, nGrids);
	printf("Residual of the Jacobi GPU solution is %.15f\n", gpuJacobiResidual);
	printf("Time needed for SRJ GPU: %f ms\n", gpuSRJTime);
/*	for (int i = 0; i < Mcopies * nGrids; i++) {
		printf("solutionJacobiGpu[%d] = %f\n", i, solutionJacobiGpu[i]);	
	}
*/
#endif 
	
	/* Shared SRJ Jacobi */
#ifdef RUN_SHARED_FLAG
	printf("===============SHARED SRJ============================\n");
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    double sharedJacobiResidual;
    double * solutionJacobiShared = new double[nGrids];
	int overlap = 0;
	float sharedSRJTime;
	cudaEvent_t start_shared, stop_shared;
	cudaEventCreate(&start_shared);
	cudaEventCreate(&stop_shared);
	cudaEventRecord(start_shared, 0);	
	solutionJacobiShared = jacobiSharedSRJShifted(initX, rhs, nGrids, srjSchemes, indexPointer, numSchemes, numSchemeParams, threadsPerBlock, overlap, numCycles, levelSRJ, Mcopies);
	cudaEventRecord(stop_shared, 0);	
	cudaEventSynchronize(stop_shared);
	cudaEventElapsedTime(&sharedSRJTime, start_shared, stop_shared);
	sharedJacobiResidual = residual1DPoisson(solutionJacobiShared, rhs, nGrids);
	printf("Residual of the Jacobi Shared solution is %.15f\n", sharedJacobiResidual);
	printf("Time needed for SRJ Shared: %f ms\n", sharedSRJTime);
/*	for (int i = 0; i < Mcopies * nGrids; i++) {
		printf("solutionJacobiShared[%d] = %f\n", i, solutionJacobiShared[i]);	
	}
*/
#endif 
   
    // FREE MEMORY
    delete[] initX;
    delete[] rhs;
	delete[] srjSchemes;
#ifdef RUN_CPU_FLAG
    delete[] solutionJacobiCpu;
#endif 
#ifdef RUN_GPU_FLAG 
    delete[] solutionJacobiGpu;
#endif
#ifdef RUN_SHARED_FLAG
    delete[] solutionJacobiShared;
#endif

    return 0;
}
