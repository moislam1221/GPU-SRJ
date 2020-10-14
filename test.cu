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
/* Helper Files */
#include "Helper/fillThreadsPerBlock.h"
#include "Helper/level.h"
#include "Helper/jacobi.h"
#include "Helper/residual.h"
#include "Helper/setGPU.h"
#include "Helper/srjSchemes.h"
/* Load SRJ implementation with Shared Memory */
#include "jacobi-1D-shared-srj-shifted.h"

int main(int argc, char *argv[])
{
    /* Inputs and Settings*/
    const int nDim = atoi(argv[1]); 
    const int threadsPerBlock_shared = atoi(argv[2]);
	const int numCycles = atoi(argv[3]); 
	const int levelSRJ = atoi(argv[4]);
	const int printSolutionFlag = 0;   
 
    /* Set the correct GPU to use (Endeavour GPUs: "TITAN V" OR "GeForce GTX 1080 Ti") */
	std::string gpuToUse = "TITAN V"; // "GeForce GTX 1080 Ti"; 
    setGPU(gpuToUse);

    /* Initialize initial condition and rhs */
    int nGrids = nDim + 2;
    double * initX = new double[nGrids];
    double * rhs = new double[nGrids];
	for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
		if (iGrid == 0 || iGrid == nGrids-1) {
			initX[iGrid] = 0.0f;
		}
		else {
			initX[iGrid] = 1.0f;
		}
		rhs[iGrid] = 1.0f;
	}
    
	/* Load SRJ schemes from Python txt files */
	int numSchemes = 25;
	int numSchemeParams = 9710;
	double * srjSchemes = new double[numSchemeParams];
	int indexPointer[numSchemes] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	loadSRJSchemes(srjSchemes, numSchemeParams);
	loadIndexPointer(indexPointer, numSchemes);

	/* Run Shared SRJ Algorithm */	
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    double sharedJacobiResidual;
    double * solutionJacobiShared = new double[nGrids];
	int overlap = 0;
	solutionJacobiShared = jacobiSharedSRJShifted(initX, rhs, nGrids, srjSchemes, indexPointer, numSchemes, numSchemeParams, threadsPerBlock_shared, overlap, numCycles, levelSRJ);
	sharedJacobiResidual = residual1DPoisson(solutionJacobiShared, rhs, nGrids);
	printf("Residual of the Jacobi Shared solution is %f\n", sharedJacobiResidual);
	
	/* Print final solution */
	if (printSolutionFlag == 1) {
		for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
			printf("solution[%d] = %f\n", iGrid, solutionJacobiShared[iGrid]); 
		}
	}
    
	/* Free Memory */
    delete[] initX;
    delete[] rhs;
	delete[] srjSchemes;
    delete[] solutionJacobiShared;

    return 0;
}
