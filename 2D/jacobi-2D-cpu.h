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

/* Perform SRJ with a specific level for all cycles */
double * jacobiCpuSRJ(const double * initX, const double * rhs, const int nxGrids, const int nyGrids,  const double * srjSchemes, const int * indexPointer, const int numSchemes, const int numCycles, const int levelSRJ)
{
	/* Instantiate variables */
    double dx = 1.0 / (nxGrids - 1);
    double dy = 1.0 / (nyGrids - 1);
    int nDofs = nxGrids * nyGrids;
    double * x0 = new double[nDofs];
    double * x1 = new double[nDofs];
    memcpy(x0, initX, sizeof(double) * nDofs);
    memcpy(x1, initX, sizeof(double) * nDofs);
	double leftX, rightX, topX, bottomX, centerX;
    double residual_after; 
	int level, dof;

	/* Perform SRJ Cycles */
	for (int cycle = 0; cycle < numCycles; cycle++) {
		/* Select the next level scheme to use */
		if (cycle == 0) {
			level = 0;
		}
		else {
			level = levelSRJ;
		}
		/* Perform all iterations associated with SRJ cycle on all DOFs */
    	for (int relaxationParameterID = indexPointer[level]; relaxationParameterID < indexPointer[level+1]; relaxationParameterID++) {
        	for (int jGrid = 1; jGrid < nyGrids-1; ++jGrid) {
				for (int iGrid = 1; iGrid < nxGrids-1; ++iGrid) {
		        	dof = jGrid * nxGrids + iGrid;
                	leftX = x0[dof - 1];
					rightX = x0[dof + 1];
                	topX = x0[dof + nxGrids];
                	bottomX = x0[dof - nxGrids];
					centerX = x0[dof];
					x1[dof] = jacobi2DPoissonRelaxed(leftX, centerX, rightX, topX, bottomX, rhs[dof], dx, dy, srjSchemes[relaxationParameterID]);
				}
        	}
        	double * tmp = x0; x0 = x1; x1 = tmp;
    	}
		/* Compute the residual afterwards */
		residual_after = residual2DPoisson(x0, rhs, nxGrids, nyGrids);
		/* Print information at the end of the cycle */
		printf("Cycle %d of Level %d complete: The residual is %f\n", cycle, level, residual_after);
    }

    delete[] x1;
    return x0;
}

double * jacobiCpu(const double * initX, const double * rhs, int nxGrids, int nyGrids,  int nIters)
{
    double dx = 1.0 / (nxGrids - 1);
    double dy = 1.0 / (nyGrids - 1);
    int nDofs = nxGrids * nyGrids;
    double * x0 = new double[nDofs];
    double * x1 = new double[nDofs];
    memcpy(x0, initX, sizeof(double) * nDofs);
    memcpy(x1, initX, sizeof(double) * nDofs);

    int dof;
    for (int iIter = 0; iIter < nIters; ++ iIter) {
        for (int jGrid = 1; jGrid < nyGrids-1; ++jGrid) {
			for (int iGrid = 1; iGrid < nxGrids-1; ++iGrid) {
		        dof = jGrid * nxGrids + iGrid;
                double leftX = x0[dof - 1];
				double rightX = x0[dof + 1];
                double topX = x0[dof + nxGrids];
                double bottomX = x0[dof - nxGrids];
				x1[dof] = jacobi2DPoisson(leftX, rightX, topX, bottomX, rhs[dof], dx, dy);
			}
        }
        double * tmp = x0; x0 = x1; x1 = tmp;
    }

    delete[] x1;
    return x0;
}

int jacobiCpuIterationCountResidual(const double * initX, const double * rhs, int nxGrids, int nyGrids, double TOL)
{
    double dx = 1.0 / (nxGrids - 1);
    double dy = 1.0 / (nyGrids - 1);
    int nDofs = nxGrids * nyGrids;
    double * x0 = new double[nDofs];
    double * x1 = new double[nDofs];
    memcpy(x0, initX, sizeof(double) * nDofs);
    memcpy(x1, initX, sizeof(double) * nDofs);

    double residual = 1000000000000.0;
    int iIter = 0;
    int dof;
    while (residual > TOL) {
        for (int jGrid = 1; jGrid < nyGrids-1; ++jGrid) {
			for (int iGrid = 1; iGrid < nxGrids-1; ++iGrid) {
		        dof = jGrid * nxGrids + iGrid;
        		double leftX = x0[dof - 1];
				double rightX = x0[dof + 1];
                double topX = x0[dof + nxGrids];
                double bottomX = x0[dof - nxGrids];
				x1[dof] = jacobi2DPoisson(leftX, rightX, topX, bottomX, rhs[dof], dx, dy);
			}
        }
		double * tmp = x0; x0 = x1; x1 = tmp;
		iIter++;
		if (iIter % 100 == 0) {
			residual = residual2DPoisson(x0, rhs, nxGrids, nyGrids);
			if (iIter % 1000 == 0) {
				printf("CPU: The residual at step %d is %f\n", iIter, residual);
			}
		}
    }
    int nIters = iIter;
    delete[] x0;
    delete[] x1;
    return nIters;
}

int jacobiCpuIterationCountSolutionError(const double * initX, const double * rhs, int nxGrids, int nyGrids, double TOL, const double * solution_exact)
{
    double dx = 1.0 / (nxGrids - 1);
    double dy = 1.0 / (nyGrids - 1);
    int nDofs = nxGrids * nyGrids;
    double * x0 = new double[nDofs];
    double * x1 = new double[nDofs];
    memcpy(x0, initX, sizeof(double) * nDofs);
    memcpy(x1, initX, sizeof(double) * nDofs);

    double solution_error = 1000000000000.0;
    int iIter = 0;
    int dof;
    while (solution_error > TOL) {
        for (int jGrid = 1; jGrid < nyGrids-1; ++jGrid) {
			for (int iGrid = 1; iGrid < nxGrids-1; ++iGrid) {
		        dof = jGrid * nxGrids + iGrid;
        		double leftX = x0[dof - 1];
				double rightX = x0[dof + 1];
                double topX = x0[dof + nxGrids];
                double bottomX = x0[dof - nxGrids];
				x1[dof] = jacobi2DPoisson(leftX, rightX, topX, bottomX, rhs[dof], dx, dy);
			}
        }
		double * tmp = x0; x0 = x1; x1 = tmp;
		iIter++;
		solution_error = solutionError2DPoisson(x0, solution_exact, nDofs);
		if (iIter % 1000 == 0) {
			printf("CPU: The solution error at step %d is %f\n", iIter, solution_error);
		}
    }
    
    int nIters = iIter;
    delete[] x0;
    delete[] x1;
    return nIters;
}
