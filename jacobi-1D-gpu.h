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

/* Perform step of relaxed Jacobi on the GPU */
__global__
void _jacobiGpuSRJIteration(double * x1, const double * x0, const double * rhs, const int nGrids, const double dx, const double relaxation_value)
{
    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (iGrid > 0 && iGrid  < (nGrids - 1)) {
        double leftX = x0[iGrid - 1];
        double rightX = x0[iGrid + 1];
        double centerX = x0[iGrid];
        x1[iGrid] = jacobi1DPoissonRelaxed(leftX, centerX, rightX, rhs[iGrid], dx, relaxation_value);
    }
    __syncthreads();
}

/* Perform SRJ on GPU with global memory with a specific level */
double * jacobiGpuSRJ(const double * initX, const double * rhs, const int nGrids, const double * srjSchemes, const int * indexPointer, const int numSchemes, const int threadsPerBlock, const int numCycles, const int levelSRJ)
{
	/* Define/Initialize key parameters */
    double dx = 1.0 / (nGrids - 1);
	const int nTotalGrids = nGrids;
    int nBlocks = (int)ceil(nTotalGrids / (double)threadsPerBlock);
	double residual_before, residual_after;
	int level;

    /* Allocate memory on the device and copy over variables */
    double * x0Gpu, * x1Gpu, * rhsGpu, * residualGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nTotalGrids);
    cudaMalloc(&x1Gpu, sizeof(double) * nTotalGrids);
    cudaMalloc(&rhsGpu, sizeof(double) * nTotalGrids);
    cudaMalloc(&residualGpu, sizeof(double) * nTotalGrids);
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nTotalGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nTotalGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nTotalGrids, cudaMemcpyHostToDevice);

	/* Perform SRJ cycles */
	for (int cycle = 0; cycle < numCycles; cycle++) {
		/* Select which level to use */
		levelSelect(level, cycle, residual_before, residual_after, numSchemes);
		if (cycle == 0) {
			level = 0;
		}
		else {
			level = levelSRJ;
		}
    	/* Obtain residual before performing cycles */
		residual_before = residualFastGpu(residualGpu, x0Gpu, rhsGpu, nGrids, threadsPerBlock, nBlocks);
		/* Perform all iterations associated with a given SRJ cycle */
    	for (int relaxationParameterID = indexPointer[level]; relaxationParameterID < indexPointer[level + 1]; relaxationParameterID++) {
			// Jacobi iteration on the GPU
        	_jacobiGpuSRJIteration<<<nBlocks, threadsPerBlock>>>(x1Gpu, x0Gpu, rhsGpu, nGrids, dx, srjSchemes[relaxationParameterID]);
        	{ 
        		double * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
			}
    	}
    	/* Obtain residual after performing cycles */
		residual_after = residualFastGpu(residualGpu, x0Gpu, rhsGpu, nGrids, threadsPerBlock, nBlocks);
		/* Print information */
		printf("Cycle %d of Level %d complete: The residual is %f\n", cycle, level, residual_after);
	}

    /* Write solution from GPU to CPU variable */
	double * solution = new double[nTotalGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(double) * nTotalGrids, cudaMemcpyDeviceToHost);

    /* Free all memory */
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);

    return solution;
}

/* Perform SRJ on GPU with global memory and heuristic for selecting the next level scheme */
double * jacobiGpuSRJHeuristic(const double * initX, const double * rhs, const int nGrids, const double * srjSchemes, const int * indexPointer, const int numSchemes, const int threadsPerBlock, const int numCycles)
{
	/* Define/Initialize key parameters */
    double dx = 1.0 / (nGrids - 1);
	const int nTotalGrids = nGrids;
    int nBlocks = (int)ceil(nTotalGrids / (double)threadsPerBlock);
	double residual_before, residual_after;
	int level;

    /* Allocate memory on the device and copy over variables */
    double * x0Gpu, * x1Gpu, * rhsGpu, * residualGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nTotalGrids);
    cudaMalloc(&x1Gpu, sizeof(double) * nTotalGrids);
    cudaMalloc(&rhsGpu, sizeof(double) * nTotalGrids);
    cudaMalloc(&residualGpu, sizeof(double) * nTotalGrids);
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nTotalGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nTotalGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nTotalGrids, cudaMemcpyHostToDevice);

	/* Perform SRJ cycles */
	for (int cycle = 0; cycle < numCycles; cycle++) {
		/* Select which level to use */
		levelSelect(level, cycle, residual_before, residual_after, numSchemes);
    	/* Obtain residual before performing cycles */
		residual_before = residualFastGpu(residualGpu, x0Gpu, rhsGpu, nGrids, threadsPerBlock, nBlocks);
		/* Perform all iterations associated with a given SRJ cycle */
    	for (int relaxationParameterID = indexPointer[level]; relaxationParameterID < indexPointer[level + 1]; relaxationParameterID++) {
			// Jacobi iteration on the GPU
        	_jacobiGpuSRJIteration<<<nBlocks, threadsPerBlock>>>(x1Gpu, x0Gpu, rhsGpu, nGrids, dx, srjSchemes[relaxationParameterID]);
        	{ 
        		double * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
			}
    	}
    	/* Obtain residual after performing cycles */
		residual_after = residualFastGpu(residualGpu, x0Gpu, rhsGpu, nGrids, threadsPerBlock, nBlocks);
		/* Print information */
		printf("Cycle %d of Level %d complete: The residual is %f\n", cycle, level, residual_after);
	}

    /* Write solution from GPU to CPU variable */
	double * solution = new double[nTotalGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(double) * nTotalGrids, cudaMemcpyDeviceToHost);

    /* Free all memory */
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);

    return solution;
}

__global__
void _jacobiGpuClassicIteration(double * x1, const double * x0, const double * rhs, const int nGrids, const double dx, const double relaxation_value, const int Mcopies)
{
    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;
	const int nTotalGrids = nGrids * Mcopies;
    if ((iGrid % nGrids) > 0 && (iGrid % nGrids) < (nGrids - 1) && iGrid < nTotalGrids-1) {
        double leftX = x0[iGrid - 1];
        double rightX = x0[iGrid + 1];
        x1[iGrid] = jacobi1DPoisson(leftX, rightX, rhs[iGrid], dx);
    }
    __syncthreads();
}

double * jacobiGpu(const double * initX, const double * rhs, const int nGrids, const int nIters,
                  const int threadsPerBlock, const double relaxation_value, const int Mcopies)
{
    // Compute dx for use in jacobi1DPoisson
    double dx = 1.0 / (nGrids - 1);

	// Total number of grid points
	const int nTotalGrids = nGrids * Mcopies;
    
    // Allocate memory in the CPU for all inputs and solutions
    double * x0Gpu, * x1Gpu, * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nTotalGrids);
    cudaMalloc(&x1Gpu, sizeof(double) * nTotalGrids);
    cudaMalloc(&rhsGpu, sizeof(double) * nTotalGrids);
    
    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nTotalGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nTotalGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nTotalGrids, cudaMemcpyHostToDevice);

    // Run the classic iteration for prescribed number of iterations
    int nBlocks = (int)ceil(nTotalGrids / (double)threadsPerBlock);

    for (int iIter = 0; iIter < nIters; ++iIter) {
		// Jacobi iteration on the GPU
        _jacobiGpuClassicIteration<<<nBlocks, threadsPerBlock>>>(x1Gpu, x0Gpu, rhsGpu, nGrids, dx, relaxation_value, Mcopies);
        { 
        	double * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
		}
    }

    // Write solution from GPU to CPU variable
    double * solution = new double[nTotalGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(double) * nTotalGrids,
            cudaMemcpyDeviceToHost);

    // Free all memory
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);

    return solution;
}

int jacobiGpuIterationCountResidual(const double * initX, const double * rhs, const int nGrids, const double TOL, const int threadsPerBlock, const double relaxation_value)
{
    // Compute dx for use in jacobi1DPoisson
    double dx = 1.0 / (nGrids - 1);
 
    // Initial residual
    double residual = 1000000000000.0;
    
    // Allocate memory in the CPU for all inputs and solutions
    double * x0Gpu, * x1Gpu, * rhsGpu, * residualGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nGrids);
    cudaMalloc(&x1Gpu, sizeof(double) * nGrids);
    cudaMalloc(&rhsGpu, sizeof(double) * nGrids);
    cudaMalloc(&residualGpu, sizeof(double) * nGrids);

    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(residualGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);

    // Container to hold CPU solution if one wants to compute residual purely on the CPU 
    
    // Run the classic iteration for prescribed number of iterations
    int nBlocks = (int)ceil(nGrids / (double)threadsPerBlock);
    int iIter = 0;
    double * residualCpu = new double[nGrids];
    while (residual > TOL) {
        // Jacobi iteration on the GPU
        _jacobiGpuClassicIteration<<<nBlocks, threadsPerBlock>>>(
                x1Gpu, x0Gpu, rhsGpu, nGrids, dx, relaxation_value, 1); 
        cudaError_t errSync  = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess)
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        if (errAsync != cudaSuccess)
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
        {
            double * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
        }
        iIter++;
		residual = residualFastGpu(residualGpu, x0Gpu, rhsGpu, nGrids, threadsPerBlock, nBlocks);
        // Print out the residual
        if (iIter % 1000 == 0) {
			printf("GPU: The residual at step %d is %f\n", iIter, residual);
        }
	}

    // Free all memory
    // CPU
    delete[] residualCpu;
    // GPU
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
    cudaFree(residualGpu);

    int nIters = iIter;
    return nIters;
}
