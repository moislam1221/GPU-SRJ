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

// HEADER FILES
#include "Helper/jacobi.h"
#include "Helper/residual.h"
#include "Helper/setGPU.h"
#include "Helper/solution_error.h"
#include "Helper/srjSchemes.h"

#define RUN_CPU_FLAG 1
#define RUN_GPU_FLAG 1
#define RUN_SHARED_FLAG 1

// Determine which header files to include based on which directives are active
#ifdef RUN_CPU_FLAG
#include "jacobi-2D-cpu.h"
#endif
#ifdef RUN_GPU_FLAG
#include "jacobi-2D-gpu.h"
#endif
#ifdef RUN_SHARED_FLAG
#include "jacobi-2D-shared-srj.h"
#endif

int main(int argc, char *argv[])
{
    /* Inputs and Settings */
    const int nxDim = atoi(argv[1]);
    const int nyDim = atoi(argv[1]); 
    const int threadsPerBlock_x = atoi(argv[2]); 
    const int threadsPerBlock_y = atoi(argv[2]); 
	const int numCycles = atoi(argv[3]);
	const int levelSRJ = atoi(argv[4]);
   
	/* Set the correct GPU to use (Endeavour GPUs: "TITAN V" OR "GeForce GTX 1080 Ti") */ 
	std::string gpuToUse = "TITAN V"; // "TITAN V";
    setGPU(gpuToUse);

    /* Initialize initial condition and rhs */
    int dof;
    int nxGrids = nxDim + 2;
    int nyGrids = nyDim + 2;
    int nDofs = nxGrids * nyGrids;
    double * initX = new double[nDofs];
    double * rhs = new double[nDofs];
    for (int jGrid = 0; jGrid < nyGrids; ++jGrid) {
        for (int iGrid = 0; iGrid < nxGrids; ++iGrid) {
            dof = iGrid + jGrid * nxGrids;
			if (iGrid == 0 || iGrid == nxGrids-1 || jGrid == 0 || jGrid == nyGrids-1) {
				initX[dof] = 0.0f;
			}
			else {
				initX[dof] = 1.0f; 
			}
			rhs[dof] = 1.0f;
        }
    }
	
	/* Load SRJ schemes from Python txt files */
	int numSchemes = 25;
	int numSchemeParams = 9710;
	double * srjSchemes = new double[numSchemeParams];
	int indexPointer[numSchemes] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	loadSRJSchemes(srjSchemes, numSchemeParams);
	loadIndexPointer(indexPointer, numSchemes);
    
    /* CPU SRJ Jacobi */
#ifdef RUN_CPU_FLAG
    double cpuJacobiResidual;
    double * solutionJacobiCpu;
	solutionJacobiCpu = jacobiCpuSRJ(initX, rhs, nxGrids, nyGrids, srjSchemes, indexPointer, numSchemes, numCycles, levelSRJ);
	cpuJacobiResidual = residual2DPoisson(solutionJacobiCpu, rhs, nxGrids, nyGrids); 
	printf("Residual of the Jacobi CPU solution is %f\n", cpuJacobiResidual);
#endif

    /* GPU SRJ Jacobi */
#ifdef RUN_GPU_FLAG
    double gpuJacobiResidual;
	double * solutionJacobiGpu;
	solutionJacobiGpu = jacobiGpuSRJ(initX, rhs, nxGrids, nyGrids, srjSchemes, indexPointer, numSchemes, numCycles, levelSRJ, threadsPerBlock_x, threadsPerBlock_y);
	gpuJacobiResidual = residual2DPoisson(solutionJacobiGpu, rhs, nxGrids, nyGrids); 
	printf("Residual of the Jacobi GPU solution is %f\n", gpuJacobiResidual);
#endif

    /* Shared SRJ Jacobi */
#ifdef RUN_SHARED_FLAG
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    double sharedJacobiResidual;
    double * solutionJacobiShared;  
    solutionJacobiShared = jacobiSharedSRJ(initX, rhs, nxGrids, nyGrids, srjSchemes, indexPointer, numSchemes, numSchemeParams, numCycles, levelSRJ, threadsPerBlock_x, threadsPerBlock_y, 0, 0);
    sharedJacobiResidual = residual2DPoisson(solutionJacobiShared, rhs, nxGrids, nyGrids);
	printf("Residual of the Jacobi Shared solution is %f\n", sharedJacobiResidual);
#endif
   
	/* Free Memory */
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
