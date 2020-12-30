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

#define PI 3.14159265358979323

/* Perform one Jacobi update step on the CPU */
void jacobiCpuSRJIteration(double * x1, const double * x0, const double * rhs, const int nGrids, const double dx, const double relaxation_value)
{
	double leftX, rightX, centerX;
	for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
		/* Incorporate BCs for edgemost DOFs */
    	leftX = (iGrid > 0) ? x0[iGrid - 1] : 0.0f; 
        rightX = (iGrid < nGrids - 1) ? x0[iGrid + 1] : 0.0f;
        centerX = x0[iGrid];
        x1[iGrid] = jacobi1DPoissonRelaxed(leftX, centerX, rightX, rhs[iGrid], dx, relaxation_value);
	}
}

/* Perform SRJ with a specific level for all cycles */
double * jacobiCpuSRJ(const double * initX, const double * rhs, const int nGrids, const double * srjSchemes, const int * indexPointer, const int numSchemes, const int numCycles, const int levelSRJ)
{
	/* Instantiate variables */
    double dx = 1.0 / (nGrids + 1);
    double * x0 = new double[nGrids];
    double * x1 = new double[nGrids];
    memcpy(x0, initX, sizeof(double) * nGrids);
    memcpy(x1, initX, sizeof(double) * nGrids);
    double residual_after; 
	int level = levelSRJ;

	/* Perform SRJ Cycles */
	for (int cycle = 0; cycle < numCycles; cycle++) {
		/* Perform all iterations associated with SRJ cycle on all DOFs */
    	for (int relaxationParameterID = indexPointer[level]; relaxationParameterID < indexPointer[level+1]; relaxationParameterID++) {
			jacobiCpuSRJIteration(x1, x0, rhs, nGrids, dx, srjSchemes[relaxationParameterID]);
			{
        		double * tmp = x0; x0 = x1; x1 = tmp;
			}
		}
		/* Compute the residual afterwards */
		residual_after = residual1DPoisson(x0, rhs, nGrids);
		/* Print information at the end of the cycle */
		printf("Cycle %d of Level %d complete: The residual is %f\n", cycle, level, residual_after);
    }

    delete[] x1;
    return x0;
}

/* Perform SRJ with the heuristic for selecting the next level */
double * jacobiCpuSRJHeuristic(const double * initX, const double * rhs, const int nGrids, const double * srjSchemes, const int * indexPointer, const int numSchemes, const int numCycles)
{
	/* Instantiate variables */
    double dx = 1.0 / (nGrids + 1);
    double * x0 = new double[nGrids];
    double * x1 = new double[nGrids];
    memcpy(x0, initX, sizeof(double) * nGrids);
    memcpy(x1, initX, sizeof(double) * nGrids);
    double residual_before, residual_after; 
	int level;

	/* Perform SRJ Cycles */
	for (int cycle = 0; cycle < numCycles; cycle++) {
		/* Select the next level scheme to use */
		levelSelect(level, cycle, residual_before, residual_after, numSchemes);
		/* Compute the residual prior to the cycle */		
		residual_before = residual1DPoisson(x0, rhs, nGrids);
		/* Perform all iterations associated with SRJ cycle on all DOFs */
    	for (int relaxationParameterID = indexPointer[level]; relaxationParameterID < indexPointer[level+1]; relaxationParameterID++) {
			jacobiCpuSRJIteration(x1, x0, rhs, nGrids, dx, srjSchemes[relaxationParameterID]);
        	{
				double * tmp = x0; x0 = x1; x1 = tmp;
			}
		}
		/* Compute the residual afterwards */
		residual_after = residual1DPoisson(x0, rhs, nGrids);
		/* Print information at the end of the cycle */
		printf("Cycle %d of Level %d complete: The residual is %f\n", cycle, level, residual_after);
    }

    delete[] x1;
    return x0;
}

/* Perform Jacobi iterations for prescribed number of iterations */
/*
double * jacobiCpu(const double * initX, const double * rhs, const int nGrids, const int nIters)
{
    double dx = 1.0 / (nGrids - 1);
    double * x0 = new double[nGrids];
    double * x1 = new double[nGrids];
    memcpy(x0, initX, sizeof(double) * nGrids);
    memcpy(x1, initX, sizeof(double) * nGrids);

    for (int iIter = 0; iIter < nIters; ++ iIter) {
        for (int iGrid = 1; iGrid < nGrids-1; ++iGrid) {
            double leftX = x0[iGrid - 1];
            double rightX = x0[iGrid + 1];
            x1[iGrid] = jacobi1DPoisson(leftX, rightX, rhs[iGrid], dx);
        }
        double * tmp = x0; x0 = x1; x1 = tmp;
    }

    delete[] x1;
    return x0;
}
*/

/* Perform Jacobi iterations until a prescribed tolerance is reached */
/*
int jacobiCpuIterationCountResidual(const double * initX, const double * rhs, int nGrids, double TOL)
{
    double dx = 1.0 / (nGrids - 1);
    double * x0 = new double[nGrids];
    double * x1 = new double[nGrids];
    memcpy(x0, initX, sizeof(double) * nGrids);
    memcpy(x1, initX, sizeof(double) * nGrids);

    double residual = 1000000000000.0;
    int iIter = 0;
    while (residual > TOL) {
        for (int iGrid = 1; iGrid < nGrids-1; ++iGrid) {
            double leftX = x0[iGrid - 1];
            double rightX = x0[iGrid + 1];
            x1[iGrid] = jacobi1DPoisson(leftX, rightX, rhs[iGrid], dx);
        }
        double * tmp = x0; x0 = x1; x1 = tmp;
        iIter++;
		residual = residual1DPoisson(x0, rhs, nGrids);
        if (iIter % 10 == 0) {
			printf("CPU: The residual at step %d is %f\n", iIter, residual);
		}
    }

    int nIters = iIter;
    delete[] x0;
    delete[] x1;
    return nIters;
}
*/
