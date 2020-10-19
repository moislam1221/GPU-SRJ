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

#define GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST 1

/* Perform step of relaxed Jacobi on the GPU */
__global__
void _jacobiGpuSRJIteration(double * x1, const double * x0, const double * rhs, const int nxGrids, const int nyGrids,  const double dx, const double dy, const double relaxation_value)
{
	/* Define local ID of dof */
    int ixGrid = blockIdx.x * blockDim.x + threadIdx.x;
    int iyGrid = blockIdx.y * blockDim.y + threadIdx.y;
    int dof = iyGrid * nxGrids + ixGrid;
    int nDofs = nxGrids * nyGrids;
    double leftX, rightX, topX, bottomX, centerX; 

	/* Perform relaxed Jacobi updates on all interior points in domain */
    if (dof < nDofs) {
		if ((ixGrid > 0) && (ixGrid < nxGrids - 1) && (iyGrid > 0) && (iyGrid < nyGrids - 1)) {
			leftX = x0[dof - 1];
			rightX = x0[dof + 1];
			topX = x0[dof + nxGrids];
			bottomX = x0[dof - nxGrids];
			centerX = x0[dof];
			x1[dof] = jacobi2DPoissonRelaxed(leftX, centerX, rightX, topX, bottomX, rhs[dof], dx, dy, relaxation_value);
		}
    }
    __syncthreads();
}

/* Perform SRJ on GPU with global memory with a specific level */
double * jacobiGpuSRJ(const double * initX, const double * rhs, const int nxGrids, const int nyGrids, const double * srjSchemes, const int * indexPointer, const int numSchemes, const int numCycles, const int levelSRJ, const int threadsPerBlock_x, const int threadsPerBlock_y)
{
	/* Define/Initialize key parameters */
    double dx = 1.0 / (nxGrids - 1);
    double dy = 1.0 / (nyGrids - 1);
	int nDofs = nxGrids * nyGrids;
	double residual_after;
	int level;

    /* Allocate memory on the device and copy over variables */
    double * x0Gpu, * x1Gpu, * rhsGpu, * residualGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nDofs);
    cudaMalloc(&x1Gpu, sizeof(double) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(double) * nDofs);
    cudaMalloc(&residualGpu, sizeof(double) * nDofs);
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);

	/* Establish 2D grid and block structures */
	dim3 block(threadsPerBlock_x, threadsPerBlock_y);
    int nxBlocks = (int)ceil(nxGrids / (double)threadsPerBlock_x);
    int nyBlocks = (int)ceil(nyGrids / (double)threadsPerBlock_y);
	dim3 grid(nxBlocks, nyBlocks);

	/* Perform SRJ cycles */
	for (int cycle = 0; cycle < numCycles; cycle++) {
		/* Select which level to use */
		if (cycle == 0) {
			level = 0;
		}
		else {
			level = levelSRJ;
		}
		/* Perform all iterations associated with a given SRJ cycle */
    	for (int relaxationParameterID = indexPointer[level]; relaxationParameterID < indexPointer[level + 1]; relaxationParameterID++) {
			/* Jacobi iteration on the GPU */
        	_jacobiGpuSRJIteration<<<grid, block>>>(x1Gpu, x0Gpu, rhsGpu, nxGrids, nyGrids, dx, dy, srjSchemes[relaxationParameterID]);
			{
				double * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
			}
		}
    	/* Compute the residual afterwards */
		residual_after = residualFastGpu2D(residualGpu, x0Gpu, rhsGpu, nxGrids, nyGrids, threadsPerBlock_x, threadsPerBlock_y, nxBlocks, nyBlocks);
		/* Print information */
		printf("Cycle %d of Level %d complete: The residual is %f\n", cycle, level, residual_after);
	}
		
	/* Write solution from GPU to CPU variable */
	double * solution = new double[nDofs];
	cudaMemcpy(solution, x0Gpu, sizeof(double) * nDofs, cudaMemcpyDeviceToHost);

	/* Free all memory */
	cudaFree(x0Gpu);
	cudaFree(x1Gpu);
	cudaFree(rhsGpu);

	return solution;
}

__global__
void _jacobiGpuClassicIteration(double * x1, const double * x0, const double * rhs, const int nxGrids, const int nyGrids, const double dx, const double dy)
{
    int ixGrid = blockIdx.x * blockDim.x + threadIdx.x;
    int iyGrid = blockIdx.y * blockDim.y + threadIdx.y;
    int dof = iyGrid * nxGrids + ixGrid;
    int nDofs = nxGrids * nyGrids;
    if (dof < nDofs) {
		if ((ixGrid > 0) && (ixGrid < nxGrids - 1) && (iyGrid > 0) && (iyGrid < nyGrids - 1)) {
			double leftX = x0[dof - 1];
			double rightX = x0[dof + 1];
			double topX = x0[dof + nxGrids];
			double bottomX = x0[dof - nxGrids];
			x1[dof] = jacobi2DPoisson(leftX, rightX, topX, bottomX, rhs[dof], dx, dy);
			// x1[dof] = jacobi1DPoisson(leftX, rightX, rhs[dof], dx);
			// x1[dof] = (leftX + rightX + topX + bottomX) / 4;
		}
    }
    __syncthreads();
}

double * jacobiGpu(const double * initX, const double * rhs, const int nxGrids, const int nyGrids,
                  const int nIters, const int threadsPerBlock_x, const int threadsPerBlock_y)
{
    // Compute dx, dy for use in jacobi2DPoisson
    double dx = 1.0 / (nxGrids - 1);
    double dy = 1.0 / (nyGrids - 1);
    int nDofs = nxGrids * nyGrids;
    
    // Allocate memory in the CPU for all inputs and solutions
    double * x0Gpu, * x1Gpu, * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nDofs);
    cudaMalloc(&x1Gpu, sizeof(double) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(double) * nDofs);
    
    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);

    // Establish 2D grid and block structures
    dim3 block(threadsPerBlock_x, threadsPerBlock_y);
    const int nxBlocks = (int)ceil(nxGrids / (double)threadsPerBlock_x);
    const int nyBlocks = (int)ceil(nyGrids / (double)threadsPerBlock_y);
    dim3 grid(nxBlocks, nyBlocks);

    double * solution = new double[nDofs];
    for (int iIter = 0; iIter < nIters; ++iIter) {
	// Jacobi iteration on the GPU
        _jacobiGpuClassicIteration<<<grid, block>>>(
                x1Gpu, x0Gpu, rhsGpu, nxGrids, nyGrids, dx, dy); 
        {
            double * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
        }
    }

    // Write solution from GPU to CPU variable
    // double * solution = new double[nDofs];
    cudaMemcpy(solution, x0Gpu, sizeof(double) * nDofs,
            cudaMemcpyDeviceToHost);

    // Free all memory
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);

    return solution;
}

int jacobiGpuIterationCountResidual(const double * initX, const double * rhs, const int nxGrids, const int nyGrids, 
                            		const double TOL, const int threadsPerBlock_x, const int threadsPerBlock_y)
{
    // Compute dx, dy for use in jacobi2DPoisson
    double dx = 1.0 / (nxGrids - 1);
    double dy = 1.0 / (nyGrids - 1);
    int nDofs = nxGrids * nyGrids;
    
    // Allocate memory in the CPU for all inputs and solutions
    double * x0Gpu, * x1Gpu, * rhsGpu, * residualGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nDofs);
    cudaMalloc(&x1Gpu, sizeof(double) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(double) * nDofs);
    cudaMalloc(&residualGpu, sizeof(double) * nDofs);
    
    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(residualGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);

    // Establish 2D grid and block structures
    dim3 block(threadsPerBlock_x, threadsPerBlock_y);
    const int nxBlocks = (int)ceil(nxGrids / (double)threadsPerBlock_x);
    const int nyBlocks = (int)ceil(nyGrids / (double)threadsPerBlock_y);
    dim3 grid(nxBlocks, nyBlocks);

    // Run the classic iteration for prescribed number of iterations
    double residual = 1000000000000.0;
    int iIter = 0;
    double * residualCpu = new double[nDofs];
    while (residual > TOL) {
		// Jacobi iteration on the GPU
        _jacobiGpuClassicIteration<<<grid, block>>>(
                x1Gpu, x0Gpu, rhsGpu, nxGrids, nyGrids, dx, dy); 
        {
        	double * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
        }
		iIter++;
		if (iIter % 1 == 0) {
#ifdef GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST
			residual2DPoissonGPU <<<grid, block>>> (residualGpu, x0Gpu, rhsGpu, nxGrids, nyGrids);
			// residual1DPoissonGPU <<<grid, block>>> (residualGpu, x0Gpu, rhsGpu, nxGrids, nyGrids);
			cudaDeviceSynchronize();
			cudaMemcpy(residualCpu, residualGpu, sizeof(double) * nDofs, cudaMemcpyDeviceToHost);
			residual = 0.0;
			for (int j = 0; j < nDofs; j++) {
			// for (int j = nxGrids; j < 2 * nxGrids; j++) {
				residual = residual + residualCpu[j];
			}
			residual = sqrt(residual);
#else
			// Write solution from GPU to CPU variable
			cudaMemcpy(solution, x0Gpu, sizeof(double) * nDofs, cudaMemcpyDeviceToHost);
			residual = residual2DPoisson(solution, rhs, nxGrids, nyGrids);
#endif        
			if (iIter % 1000 == 0) {
				// Print out the residual
				printf("GPU: The residual at step %d is %f\n", iIter, residual);
			}
		}
    }

    // Free all memory
    delete[] residualCpu;
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
    cudaFree(residualGpu);

    int nIters = iIter;
    return nIters;
}


int jacobiGpuIterationCountSolutionError(const double * initX, const double * rhs, const int nxGrids, const int nyGrids, const double TOL, const int threadsPerBlock_x, const int threadsPerBlock_y, const double * solution_exact)
{
    // Compute dx, dy for use in jacobi2DPoisson
    double dx = 1.0 / (nxGrids - 1);
    double dy = 1.0 / (nyGrids - 1);
    int nDofs = nxGrids * nyGrids;
   
    // Initial solution error
    double solution_error = 1000000000000.0;
 
    // Allocate memory in the CPU for all inputs and solutions
    double * x0Gpu, * x1Gpu, * rhsGpu, * solutionErrorGpu, * solution_exactGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nDofs);
    cudaMalloc(&x1Gpu, sizeof(double) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(double) * nDofs);
    cudaMalloc(&solutionErrorGpu, sizeof(double) * nDofs);
    cudaMalloc(&solution_exactGpu, sizeof(double) * nDofs);
    
    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(solutionErrorGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(solution_exactGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);

    // Establish 2D grid and block structures
    dim3 block(threadsPerBlock_x, threadsPerBlock_y);
    const int nxBlocks = (int)ceil(nxGrids / (double)threadsPerBlock_x);
    const int nyBlocks = (int)ceil(nyGrids / (double)threadsPerBlock_y);
    dim3 grid(nxBlocks, nyBlocks);

    // Run the classic iteration for prescribed number of iterations
    int iIter = 0;
    double * solutionErrorCpu = new double[nDofs];
    while (solution_error > TOL) {
		// Jacobi iteration on the GPU
        _jacobiGpuClassicIteration<<<grid, block>>>(
                x1Gpu, x0Gpu, rhsGpu, nxGrids, nyGrids, dx, dy); 
        {
        	double * tmp = x0Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
        }
		iIter++;
		if (iIter % 1000 == 0) {
#ifdef GPU_RESIDUAL_SOLUTION_ERROR_CALCULATION_FAST
			solutionError2DPoissonGPU <<<grid, block>>> (solutionErrorGpu, x0Gpu, solution_exactGpu, nDofs);
			cudaDeviceSynchronize();
			cudaMemcpy(solutionErrorCpu, solutionErrorGpu, sizeof(double) * nDofs, cudaMemcpyDeviceToHost);
			solution_error = 0.0;
			for (int j = 0; j < nDofs; j++) {
				solution_error = solution_error + solutionErrorCpu[j];
			}
			solution_error = sqrt(solution_error);
#else
			// Write solution from GPU to CPU variable
			cudaMemcpy(solution, x0Gpu, sizeof(double) * nDofs, cudaMemcpyDeviceToHost);
			solution_error = solutionError2DPoisson(solution, solution_exact, nDofs);
#endif        
			// Print out the residual
            printf("GPU: The solution error at step %d is %f\n", iIter, solution_error);
        }
    }

    // Free all memory
    delete[] solutionErrorCpu;
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
    cudaFree(solutionErrorGpu);
    cudaFree(solution_exactGpu);

    int nIters = iIter;
    return nIters;
}
