/* 1 - Global to Shared Transfer */
__device__
void __jacobiGlobalToSharedSRJ(const double * x0Gpu, const double * rhsGpu, const int subdomainLength_x, const int subdomainLength_y, const int nxGrids, const int nyGrids, const int OVERLAP_X, const int OVERLAP_Y)
{
	/* Define shared memory */
	extern __shared__ double sharedMemory[];
	
	/* Compute global ID of bottomleft corner point handled by specific blockIdx.x, blockIdx.y (serves as useful reference ID point) */
    int blockShift = blockIdx.x * (blockDim.x - OVERLAP_X) + blockIdx.y * (blockDim.y - OVERLAP_Y) * nyGrids;
    int Idx, Idy, I;
    int sharedID = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
	int nPerSubdomain = subdomainLength_x * subdomainLength_y;
	int nDofs = nxGrids * nyGrids;

	/* While we haven't moved all points in global subdomain over to shared */
    for (int i = sharedID; i < nPerSubdomain; i += stride) {
        /* Compute the global ID of point in the grid */
		Idx = (i % subdomainLength_x); 
		Idy = i/subdomainLength_x; 
		I = blockShift + Idx + Idy * nyGrids; // global ID 
        /* If the global ID is less than number of points, or local ID is less than number of points in subdomain */
        if (I < nDofs && i < nPerSubdomain) {
            sharedMemory[i] = x0Gpu[I]; 
            sharedMemory[i + nPerSubdomain] = x0Gpu[I];
            sharedMemory[i + 2 * nPerSubdomain] = rhsGpu[I];
        }
    }
	__syncthreads();
}

/* 2 - Update within shared memory */
__device__
void __jacobiUpdateKernelSRJ(const int nxGrids, const int nyGrids, const int subdomainLength_x, const int subdomainLength_y, const int OVERLAP_X, const int OVERLAP_Y, const int level, const int * indexPointerGpu, const double * srjSchemesGpu)
{
    /* Create shared memory pointers */
	extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory, * x1 = sharedMemory + subdomainLength_x * subdomainLength_y, * x2 = sharedMemory + 2 * subdomainLength_x * subdomainLength_y;
    const double dx = 1.0 / (nxGrids - 1);
    const double dy = 1.0 / (nyGrids - 1);
	double leftX, rightX, topX, bottomX, centerX, rhs;

	/* Define local and global ID in x and y */
	int ix = threadIdx.x + 1;
	int iy = threadIdx.y + 1;
	int i = ix + iy * subdomainLength_x;
    int Ix = blockIdx.x * (blockDim.x - OVERLAP_X) + ix;
    int Iy = blockIdx.y * (blockDim.y - OVERLAP_Y) + iy;

	/* Perform one SRJ cycle based on the current level */
	for (int relaxationParameterID = indexPointerGpu[level]; relaxationParameterID < indexPointerGpu[level+1]; relaxationParameterID++) {
 		if (Ix < nxGrids-1 && Iy < nyGrids-1) {
			leftX = x0[i-1];
			rightX = x0[i+1];
			topX = x0[i+subdomainLength_x];
			bottomX = x0[i-subdomainLength_x];
			rhs = x2[i];
			centerX = x0[i];
			x1[i] = jacobi2DPoissonRelaxed(leftX, centerX, rightX, topX, bottomX, rhs, dx, dy, srjSchemesGpu[relaxationParameterID]);
        }
	    __syncthreads();
        double * tmp = x0; x0 = x1; x1 = tmp;
    }
}

/* 3 - Shared to Global Transfer */
__device__
void __jacobiSharedToGlobalSRJ(double * x1Gpu, const int subdomainLength_x, const int subdomainLength_y, const int nxGrids, const int nyGrids, const int * indexPointerGpu, const int level, const int OVERLAP_X, const int OVERLAP_Y)
{
	/* Define shared memory */
	extern __shared__ double sharedMemory[];

    /* Define booleans indicating whether grid point should be handled by this block */
    bool inxRange = 0;
    bool inyRange = 0;

    /* Check if x point should be handled by this particular block */
    int Ix = blockIdx.x * (blockDim.x - OVERLAP_X) + (threadIdx.x + 1);
    if (Ix < nxGrids-1){
		if (blockIdx.x == 0) {
			if (threadIdx.x <= blockDim.x - 1 - OVERLAP_X/2) {
				inxRange = 1;
			}
		}
		else if (blockIdx.x == gridDim.x - 1) {
			if (threadIdx.x >= OVERLAP_X/2) {
				inxRange = 1;
			}
		}
		else {
			if (threadIdx.x >= OVERLAP_X/2 && threadIdx.x <= blockDim.x - 1 - OVERLAP_X/2) {
				inxRange = 1;
			}
		}
    }

	/*  Check if y point should be handled by this particular block */
    int Iy = blockIdx.y * (blockDim.y - OVERLAP_Y) + (threadIdx.y + 1);
    if (Iy < nyGrids-1) {
		if (blockIdx.y == 0) {
			if (threadIdx.y <= blockDim.y - 1 - OVERLAP_Y/2) {
				inyRange = 1;
			}
		}
		else if (blockIdx.y == gridDim.y - 1) {
			if (threadIdx.y >= OVERLAP_Y/2) {
				inyRange = 1;
			}
		}
		else {
			if (threadIdx.y >= OVERLAP_Y/2 && threadIdx.y <= blockDim.y - 1 - OVERLAP_Y/2) {
				inyRange = 1;
			}
		}
    }

	/* Define the number of iterations which were performed */
	int numIters = indexPointerGpu[level+1] - indexPointerGpu[level];

	/* Define blockShift */
    int blockShift = blockIdx.x * (blockDim.x - OVERLAP_X) + blockIdx.y * (blockDim.y - OVERLAP_Y) * nyGrids;

	/* Define nPerSubdomain */
	int nPerSubdomain = subdomainLength_x * subdomainLength_y;

    /* If point is within bound of points to be handled by particular blockIdx.x, blockIdx.y, then move value over to global memory */   
	if (inxRange == 1 && inyRange == 1) {
		const int i_inner = (threadIdx.x + 1) + (threadIdx.y + 1) * subdomainLength_x;
		const int Idx_inner = (i_inner % subdomainLength_x); // local ID
		const int Idy_inner = i_inner/subdomainLength_x; // local ID
		const int I_inner = blockShift + Idx_inner + Idy_inner * nyGrids; // global ID
        if ((numIters % 2) == 0) { 
		    x1Gpu[I_inner] = sharedMemory[i_inner];
		}
        else { 
		    x1Gpu[I_inner] = sharedMemory[i_inner + nPerSubdomain];
        }
    }

    __syncthreads();

}

/* Perform one cycle of hierarchical SRJ */
__global__
void _jacobiUpdateSRJ(double * x1Gpu, double * x0Gpu, const double * rhsGpu, const int nxGrids, const int nyGrids, const int OVERLAP_X, const int OVERLAP_Y, const int level, const int * indexPointerGpu, const double * srjSchemesGpu)
{
    /* Move to shared memory */
    extern __shared__ double sharedMemory[];
   
    /* Define useful constants regarding subdomain edge length and number of points within a 2D subdomain */
    const int subdomainLength_x = blockDim.x + 2;
    const int subdomainLength_y = blockDim.y + 2;

    /* STEP 1 - MOVE ALL VALUES TO SHARED MEMORY: Assume no overlap */
	__jacobiGlobalToSharedSRJ(x0Gpu, rhsGpu, subdomainLength_x, subdomainLength_y, nxGrids, nyGrids, OVERLAP_X, OVERLAP_Y);
    
    /* STEP 2 - UPDATE ALL INNER POINTS IN EACH BLOCK */
	__jacobiUpdateKernelSRJ(nxGrids, nyGrids, subdomainLength_x, subdomainLength_y, OVERLAP_X, OVERLAP_Y, level, indexPointerGpu, srjSchemesGpu);
    
	/* STEP 3 - MOVE ALL VALUES FROM SHARED MEMORY BACK TO GLOBAL MEMORY */
	__jacobiSharedToGlobalSRJ(x1Gpu, subdomainLength_x, subdomainLength_y, nxGrids, nyGrids, indexPointerGpu, level, OVERLAP_X, OVERLAP_Y);

}

/************************ Diagonal *****************************/

/* 1 - Global to Shared Transfer Shifted */
__device__
void __jacobiGlobalToSharedSRJShiftedDiagonal(const double * x0Gpu, const double * rhsGpu, const int subdomainLength_x, const int subdomainLength_y, const int nxGrids, const int nyGrids, const int OVERLAP_X, const int OVERLAP_Y)
{
	/* Define shared memory */
	extern __shared__ double sharedMemory[];
	
	/* Compute global ID of bottomleft corner point handled by specific blockIdx.x, blockIdx.y (serves as useful reference ID point) */
    int xShift = (blockDim.x/2); 
	int yShift = (blockDim.y/2);
	int IxBlockShift = blockIdx.x * (blockDim.x - OVERLAP_X);
	int IyBlockShift = (blockIdx.y * (blockDim.y - OVERLAP_Y));
    int idx, idy, I, Ixglobal, Iyglobal;
    int sharedID = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
	int nPerSubdomain = subdomainLength_x * subdomainLength_y;
	int nDofs = nxGrids * nyGrids;

	for (int i = sharedID; i < nPerSubdomain; i += stride) {
		/* Compute the global ID of point in the grid */
		idx = (i % subdomainLength_x); 
		idy = i/subdomainLength_x; 
		Ixglobal = xShift + IxBlockShift + idx;
		Iyglobal = yShift + IyBlockShift + idy;
		/* Check the x coord of point and use mod if it falls outside of range */
		if (blockIdx.x == gridDim.x - 1) {
			if (Ixglobal > nxGrids - 1) {
				Ixglobal = Ixglobal % nxGrids;
			}
		}
		/* Check the y coord of point and use mod if it falls outside of range */
		if (blockIdx.y == gridDim.y - 1) {
			if (Iyglobal > nyGrids - 1) {
				Iyglobal = Iyglobal % nyGrids;
			}
		}
		I = Ixglobal + Iyglobal * nxGrids; // global ID
		/* If the global ID is less than number of points, or local ID is less than number of points in subdomain */
		if (I < nDofs && i < nPerSubdomain) {
			sharedMemory[i] = x0Gpu[I]; 
			sharedMemory[i + nPerSubdomain] = x0Gpu[I];
			sharedMemory[i + 2 * nPerSubdomain] = rhsGpu[I];
/*			if (blockIdx.x == 1 && blockIdx.y == 1) {
				printf("In block (%d,%d): sharedMemory[%d] = %f, I = %d, Ixglobal = %d, Iyglobal = %d\n", blockIdx.x, blockIdx.y, i, sharedMemory[i], I, Ixglobal, Iyglobal);
			}
*/		}
	}
	__syncthreads();
}

/* 2 - Update within shared memory shifted */
__device__
void __jacobiUpdateKernelSRJShiftedDiagonal(const int nxGrids, const int nyGrids, const int subdomainLength_x, const int subdomainLength_y, const int OVERLAP_X, const int OVERLAP_Y, const int level, const int * indexPointerGpu, const double * srjSchemesGpu)
{
    /* Create shared memory pointers */
	extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory, * x1 = sharedMemory + subdomainLength_x * subdomainLength_y, * x2 = sharedMemory + 2 * subdomainLength_x * subdomainLength_y;
    const double dx = 1.0 / (nxGrids - 1);
    const double dy = 1.0 / (nyGrids - 1);
	double leftX, rightX, topX, bottomX, centerX, rhs;

	/* Define local and global ID in x and y */
	int ix = threadIdx.x + 1;
	int iy = threadIdx.y + 1;
	int i = ix + iy * subdomainLength_x;
    int xShift = (blockDim.x/2); 
	int yShift = (blockDim.y/2);
    int Ix = blockIdx.x * (blockDim.x - OVERLAP_X) + xShift + ix;
    int Iy = blockIdx.y * (blockDim.y - OVERLAP_Y) + yShift + iy;
		
	/* Make adjustments to updated local/global IDs based on the block */
	if (blockIdx.x == gridDim.x - 1) {	
		if (ix > blockDim.x / 2) {
			ix = ix + 2;
			i = ix + iy * subdomainLength_x;
			Ix = (blockIdx.x * (blockDim.x - OVERLAP_X) + xShift + ix) % nxGrids;
		}
	}	
	if (blockIdx.y == gridDim.y - 1) {	
		if (iy > blockDim.y / 2) {
			iy = iy + 2;
			i = ix + iy * subdomainLength_x;
			Iy = (blockIdx.y * (blockDim.y - OVERLAP_Y) + yShift + iy) % nyGrids;
		}
	}

	/* Perform one SRJ cycle based on the current level */
	for (int relaxationParameterID = indexPointerGpu[level]; relaxationParameterID < indexPointerGpu[level+1]; relaxationParameterID++) {
		/* Update interior points */	
		if (Ix < nxGrids-1 && Iy < nyGrids-1) {
			leftX = x0[i-1];
			rightX = x0[i+1];
			topX = x0[i+subdomainLength_x];
			bottomX = x0[i-subdomainLength_x];
			rhs = x2[i];
			centerX = x0[i];
			x1[i] = jacobi2DPoissonRelaxed(leftX, centerX, rightX, topX, bottomX, rhs, dx, dy, srjSchemesGpu[relaxationParameterID]);
		}
		__syncthreads();
		double * tmp = x0; x0 = x1; x1 = tmp;
	}
}

/* 3 - Shared to Global Transfer shifted */
__device__
void __jacobiSharedToGlobalSRJShiftedDiagonal(double * x1Gpu, const int subdomainLength_x, const int subdomainLength_y, const int nxGrids, const int nyGrids, const int * indexPointerGpu, const int level, const int OVERLAP_X, const int OVERLAP_Y)
{
	/* Define shared memory */
	extern __shared__ double sharedMemory[];

	/* Define the amount of shift */
    int xShift = (blockDim.x/2); 
    int yShift = (blockDim.y/2); 

    /* Define local and global x coordinate of point to be updated */
    int ixlocal = threadIdx.x + 1;
	int Ixglobal = blockIdx.x * (blockDim.x - OVERLAP_X) + xShift + ixlocal;
 
    /* Define local and global y coordinate of point to be updated */
    int iylocal = threadIdx.y + 1;
    int Iyglobal = blockIdx.y * (blockDim.y - OVERLAP_Y) + yShift + iylocal;

	/* Define the number of iterations which were performed */
	int numIters = indexPointerGpu[level+1] - indexPointerGpu[level];

	/* Define nPerSubdomain */
	int nPerSubdomain = subdomainLength_x * subdomainLength_y;

    /* If point is within bound of points to be handled by particular blockIdx.x, blockIdx.y, then move value over to global memory */   
	if (blockIdx.x == gridDim.x - 1) {
		if (ixlocal > blockDim.x / 2) {
			ixlocal = ixlocal + 2;
			Ixglobal = (blockIdx.x * (blockDim.x - OVERLAP_X) + xShift + ixlocal) % nxGrids;
		}
	}	
	if (blockIdx.y == gridDim.y - 1) {
		if (iylocal > blockDim.y / 2) { 
			iylocal = iylocal + 2;
			Iyglobal = (blockIdx.y * (blockDim.y - OVERLAP_Y) + yShift + iylocal) % nyGrids;
		}
	}	
	int ilocal = ixlocal + iylocal * subdomainLength_x;
	int Iglobal = Ixglobal + Iyglobal * nxGrids;

	if ((numIters % 2) == 0) { 
		x1Gpu[Iglobal] = sharedMemory[ilocal];
	}
	else {
		x1Gpu[Iglobal] = sharedMemory[ilocal + nPerSubdomain];
	}

    __syncthreads();
}

/* Perform one cycle of hierarchichal SRJ with a shift */
__global__
void _jacobiUpdateSRJShiftedDiagonal(double * x1Gpu, double * x0Gpu, const double * rhsGpu, const int nxGrids, const int nyGrids, const int OVERLAP_X, const int OVERLAP_Y, const int level, const int * indexPointerGpu, const double * srjSchemesGpu)
{
    /* Move to shared memory */
    extern __shared__ double sharedMemory[];
   
    /* Define useful constants regarding subdomain edge length and number of points within a 2D subdomain */
    int subdomainLength_x, subdomainLength_y;
	if (blockIdx.x < gridDim.x - 1) {
		subdomainLength_x = blockDim.x + 2;
	}
	else {
		subdomainLength_x = blockDim.x + 4;
	}
	if (blockIdx.y < gridDim.y - 1) {
		subdomainLength_y = blockDim.y + 2;
	}
	else {
		subdomainLength_y = blockDim.y + 4;
	}

    /* STEP 1 - MOVE ALL VALUES TO SHARED MEMORY: Assume no overlap */
	__jacobiGlobalToSharedSRJShiftedDiagonal(x0Gpu, rhsGpu, subdomainLength_x, subdomainLength_y, nxGrids, nyGrids, OVERLAP_X, OVERLAP_Y);
    
    /* STEP 2 - UPDATE ALL INNER POINTS IN EACH BLOCK */
	__jacobiUpdateKernelSRJShiftedDiagonal(nxGrids, nyGrids, subdomainLength_x, subdomainLength_y, OVERLAP_X, OVERLAP_Y, level, indexPointerGpu, srjSchemesGpu);
    
	/* STEP 3 - MOVE ALL VALUES FROM SHARED MEMORY BACK TO GLOBAL MEMORY */
	__jacobiSharedToGlobalSRJShiftedDiagonal(x1Gpu, subdomainLength_x, subdomainLength_y, nxGrids, nyGrids, indexPointerGpu, level, OVERLAP_X, OVERLAP_Y);

}

/********************************* Horizontal *********************************/

/* 1 - Global to Shared Transfer Shifted */
__device__
void __jacobiGlobalToSharedSRJShiftedHorizontal(const double * x0Gpu, const double * rhsGpu, const int subdomainLength_x, const int subdomainLength_y, const int nxGrids, const int nyGrids, const int OVERLAP_X, const int OVERLAP_Y)
{
	/* Define shared memory */
	extern __shared__ double sharedMemory[];
	
	/* Compute global ID of bottomleft corner point handled by specific blockIdx.x, blockIdx.y (serves as useful reference ID point) */
    int xShift = (blockDim.x/2); 
	int yShift = 0;
	int IxBlockShift = blockIdx.x * (blockDim.x - OVERLAP_X);
	int IyBlockShift = (blockIdx.y * (blockDim.y - OVERLAP_Y));
    int idx, idy, I, Ixglobal, Iyglobal;
    int sharedID = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
	int nPerSubdomain = subdomainLength_x * subdomainLength_y;
	int nDofs = nxGrids * nyGrids;

	for (int i = sharedID; i < nPerSubdomain; i += stride) {
		/* Compute the global ID of point in the grid */
		idx = (i % subdomainLength_x); 
		idy = i/subdomainLength_x; 
		Ixglobal = xShift + IxBlockShift + idx;
		Iyglobal = yShift + IyBlockShift + idy;
		/* Check the x coord of point and use mod if it falls outside of range */
		if (blockIdx.x == gridDim.x - 1) {
			if (Ixglobal > nxGrids - 1) {
				Ixglobal = Ixglobal % nxGrids;
			}
		}
		I = Ixglobal + Iyglobal * nxGrids; // global ID
		/* If the global ID is less than number of points, or local ID is less than number of points in subdomain */
		if (I < nDofs && i < nPerSubdomain) {
			sharedMemory[i] = x0Gpu[I]; 
			sharedMemory[i + nPerSubdomain] = x0Gpu[I];
			sharedMemory[i + 2 * nPerSubdomain] = rhsGpu[I];
/*			if (blockIdx.x == 1 && blockIdx.y == 1) {
				printf("In block (%d,%d): sharedMemory[%d] = %f, I = %d, Ixglobal = %d, Iyglobal = %d\n", blockIdx.x, blockIdx.y, i, sharedMemory[i], I, Ixglobal, Iyglobal);
			}
*/		}
	}
	__syncthreads();

}

/* 2 - Update within shared memory shifted */
__device__
void __jacobiUpdateKernelSRJShiftedHorizontal(const int nxGrids, const int nyGrids, const int subdomainLength_x, const int subdomainLength_y, const int OVERLAP_X, const int OVERLAP_Y, const int level, const int * indexPointerGpu, const double * srjSchemesGpu)
{
    /* Create shared memory pointers */
	extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory, * x1 = sharedMemory + subdomainLength_x * subdomainLength_y, * x2 = sharedMemory + 2 * subdomainLength_x * subdomainLength_y;
    const double dx = 1.0 / (nxGrids - 1);
    const double dy = 1.0 / (nyGrids - 1);
	double leftX, rightX, topX, bottomX, centerX, rhs;

	/* Define local and global ID in x and y */
	int ix = threadIdx.x + 1;
	int iy = threadIdx.y + 1;
	int i = ix + iy * subdomainLength_x;
    int xShift = (blockDim.x/2); 
	int yShift = 0;
    int Ix = blockIdx.x * (blockDim.x - OVERLAP_X) + xShift + ix;
    int Iy = blockIdx.y * (blockDim.y - OVERLAP_Y) + yShift + iy;
		
	/* Make adjustments to updated local/global IDs based on the block */
	if (blockIdx.x == gridDim.x - 1) {	
		if (ix > blockDim.x / 2) {
			ix = ix + 2;
			i = ix + iy * subdomainLength_x;
			Ix = (blockIdx.x * (blockDim.x - OVERLAP_X) + xShift + ix) % nxGrids;
		}
	}	

	/* Perform one SRJ cycle based on the current level */
	for (int relaxationParameterID = indexPointerGpu[level]; relaxationParameterID < indexPointerGpu[level+1]; relaxationParameterID++) {
		/* Update interior points */	
		if (Ix < nxGrids-1 && Iy < nyGrids-1) {
			leftX = x0[i-1];
			rightX = x0[i+1];
			topX = x0[i+subdomainLength_x];
			bottomX = x0[i-subdomainLength_x];
			rhs = x2[i];
			centerX = x0[i];
			x1[i] = jacobi2DPoissonRelaxed(leftX, centerX, rightX, topX, bottomX, rhs, dx, dy, srjSchemesGpu[relaxationParameterID]);
		}
		__syncthreads();
		double * tmp = x0; x0 = x1; x1 = tmp;
		__syncthreads();
/*		if (blockIdx.x == 0 && blockIdx.y == 1) {
			printf("In block (%d,%d): sharedMemory[%d] = %f, Ix = %d, Iy = %d\n", blockIdx.x, blockIdx.y, i, x0[i], Ix, Iy);
		}
*/	}
		
}


/* 3 - Shared to Global Transfer shifted */
__device__
void __jacobiSharedToGlobalSRJShiftedHorizontal(double * x1Gpu, const int subdomainLength_x, const int subdomainLength_y, const int nxGrids, const int nyGrids, const int * indexPointerGpu, const int level, const int OVERLAP_X, const int OVERLAP_Y)
{
	/* Define shared memory */
	extern __shared__ double sharedMemory[];

	/* Define the amount of shift */
    int xShift = (blockDim.x/2); 
    int yShift = 0; 

    /* Define local and global x coordinate of point to be updated */
    int ixlocal = threadIdx.x + 1;
	int Ixglobal = blockIdx.x * (blockDim.x - OVERLAP_X) + xShift + ixlocal;
 
    /* Define local and global y coordinate of point to be updated */
    int iylocal = threadIdx.y + 1;
    int Iyglobal = blockIdx.y * (blockDim.y - OVERLAP_Y) + yShift + iylocal;

	/* Define the number of iterations which were performed */
	int numIters = indexPointerGpu[level+1] - indexPointerGpu[level];

	/* Define nPerSubdomain */
	int nPerSubdomain = subdomainLength_x * subdomainLength_y;

    /* If point is within bound of points to be handled by particular blockIdx.x, blockIdx.y, then move value over to global memory */   
	if (blockIdx.x == gridDim.x - 1) {
		if (ixlocal > blockDim.x / 2) {
			ixlocal = ixlocal + 2;
			Ixglobal = (blockIdx.x * (blockDim.x - OVERLAP_X) + xShift + ixlocal) % nxGrids;
		}
	}	
	int ilocal = ixlocal + iylocal * subdomainLength_x;
	int Iglobal = Ixglobal + Iyglobal * nxGrids;

	if ((numIters % 2) == 0) { 
		x1Gpu[Iglobal] = sharedMemory[ilocal];
	}
	else {
		x1Gpu[Iglobal] = sharedMemory[ilocal + nPerSubdomain];
	}

    __syncthreads();
}

/* Perform one cycle of hierarchichal SRJ with a shift */
__global__
void _jacobiUpdateSRJShiftedHorizontal(double * x1Gpu, double * x0Gpu, const double * rhsGpu, const int nxGrids, const int nyGrids, const int OVERLAP_X, const int OVERLAP_Y, const int level, const int * indexPointerGpu, const double * srjSchemesGpu)
{
    /* Move to shared memory */
    extern __shared__ double sharedMemory[];
   
    /* Define useful constants regarding subdomain edge length and number of points within a 2D subdomain */
    int subdomainLength_x, subdomainLength_y;
	if (blockIdx.x < gridDim.x - 1) {
		subdomainLength_x = blockDim.x + 2;
	}
	else {
		subdomainLength_x = blockDim.x + 4;
	}
	subdomainLength_y = blockDim.y + 2;

    /* STEP 1 - MOVE ALL VALUES TO SHARED MEMORY: Assume no overlap */
	__jacobiGlobalToSharedSRJShiftedHorizontal(x0Gpu, rhsGpu, subdomainLength_x, subdomainLength_y, nxGrids, nyGrids, OVERLAP_X, OVERLAP_Y);
    
    /* STEP 2 - UPDATE ALL INNER POINTS IN EACH BLOCK */
	__jacobiUpdateKernelSRJShiftedHorizontal(nxGrids, nyGrids, subdomainLength_x, subdomainLength_y, OVERLAP_X, OVERLAP_Y, level, indexPointerGpu, srjSchemesGpu);
    
	/* STEP 3 - MOVE ALL VALUES FROM SHARED MEMORY BACK TO GLOBAL MEMORY */
	__jacobiSharedToGlobalSRJShiftedHorizontal(x1Gpu, subdomainLength_x, subdomainLength_y, nxGrids, nyGrids, indexPointerGpu, level, OVERLAP_X, OVERLAP_Y);
}

/********************************* Vertical *********************************/

/* 1 - Global to Shared Transfer Shifted */
__device__
void __jacobiGlobalToSharedSRJShiftedVertical(const double * x0Gpu, const double * rhsGpu, const int subdomainLength_x, const int subdomainLength_y, const int nxGrids, const int nyGrids, const int OVERLAP_X, const int OVERLAP_Y)
{
	/* Define shared memory */
	extern __shared__ double sharedMemory[];
	
	/* Compute global ID of bottomleft corner point handled by specific blockIdx.x, blockIdx.y (serves as useful reference ID point) */
    int xShift = 0; 
	int yShift = (blockDim.y/2);
	int IxBlockShift = blockIdx.x * (blockDim.x - OVERLAP_X);
	int IyBlockShift = (blockIdx.y * (blockDim.y - OVERLAP_Y));
    int idx, idy, I, Ixglobal, Iyglobal;
    int sharedID = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
	int nPerSubdomain = subdomainLength_x * subdomainLength_y;
	int nDofs = nxGrids * nyGrids;

	for (int i = sharedID; i < nPerSubdomain; i += stride) {
		/* Compute the global ID of point in the grid */
		idx = (i % subdomainLength_x); 
		idy = i/subdomainLength_x; 
		Ixglobal = xShift + IxBlockShift + idx;
		Iyglobal = yShift + IyBlockShift + idy;
		/* Check the y coord of point and use mod if it falls outside of range */
		if (blockIdx.y == gridDim.y - 1) {
			if (Iyglobal > nyGrids - 1) {
				Iyglobal = Iyglobal % nyGrids;
			}
		}
		I = Ixglobal + Iyglobal * nxGrids; // global ID
		/* If the global ID is less than number of points, or local ID is less than number of points in subdomain */
		if (I < nDofs && i < nPerSubdomain) {
			sharedMemory[i] = x0Gpu[I]; 
			sharedMemory[i + nPerSubdomain] = x0Gpu[I];
			sharedMemory[i + 2 * nPerSubdomain] = rhsGpu[I];
/*			if (blockIdx.x == 1 && blockIdx.y == 1) {
				printf("In block (%d,%d): sharedMemory[%d] = %f, I = %d, Ixglobal = %d, Iyglobal = %d\n", blockIdx.x, blockIdx.y, i, sharedMemory[i], I, Ixglobal, Iyglobal);
			}
*/		}
	}
	__syncthreads();

}

/* 2 - Update within shared memory shifted */
__device__
void __jacobiUpdateKernelSRJShiftedVertical(const int nxGrids, const int nyGrids, const int subdomainLength_x, const int subdomainLength_y, const int OVERLAP_X, const int OVERLAP_Y, const int level, const int * indexPointerGpu, const double * srjSchemesGpu)
{
    /* Create shared memory pointers */
	extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory, * x1 = sharedMemory + subdomainLength_x * subdomainLength_y, * x2 = sharedMemory + 2 * subdomainLength_x * subdomainLength_y;
    const double dx = 1.0 / (nxGrids - 1);
    const double dy = 1.0 / (nyGrids - 1);
	double leftX, rightX, topX, bottomX, centerX, rhs;

	/* Define local and global ID in x and y */
	int ix = threadIdx.x + 1;
	int iy = threadIdx.y + 1;
	int i = ix + iy * subdomainLength_x;
    int xShift = 0; 
	int yShift = (blockDim.y/2);
    int Ix = blockIdx.x * (blockDim.x - OVERLAP_X) + xShift + ix;
    int Iy = blockIdx.y * (blockDim.y - OVERLAP_Y) + yShift + iy;
		
	/* Make adjustments to updated local/global IDs based on the block */
	if (blockIdx.y == gridDim.y - 1) {	
		if (iy > blockDim.y / 2) {
			iy = iy + 2;
			i = ix + iy * subdomainLength_x;
			Iy = (blockIdx.y * (blockDim.y - OVERLAP_Y) + yShift + iy) % nyGrids;
		}
	}

	/* Perform one SRJ cycle based on the current level */
	for (int relaxationParameterID = indexPointerGpu[level]; relaxationParameterID < indexPointerGpu[level+1]; relaxationParameterID++) {
		/* Update interior points */	
		if (Ix < nxGrids-1 && Iy < nyGrids-1) {
			leftX = x0[i-1];
			rightX = x0[i+1];
			topX = x0[i+subdomainLength_x];
			bottomX = x0[i-subdomainLength_x];
			rhs = x2[i];
			centerX = x0[i];
			x1[i] = jacobi2DPoissonRelaxed(leftX, centerX, rightX, topX, bottomX, rhs, dx, dy, srjSchemesGpu[relaxationParameterID]);
		}
		__syncthreads();
		double * tmp = x0; x0 = x1; x1 = tmp;
	}
}

/* 3 - Shared to Global Transfer shifted */
__device__
void __jacobiSharedToGlobalSRJShiftedVertical(double * x1Gpu, const int subdomainLength_x, const int subdomainLength_y, const int nxGrids, const int nyGrids, const int * indexPointerGpu, const int level, const int OVERLAP_X, const int OVERLAP_Y)
{
	/* Define shared memory */
	extern __shared__ double sharedMemory[];

	/* Define the amount of shift */
    int xShift = 0; 
    int yShift = (blockDim.y/2); 

    /* Define local and global x coordinate of point to be updated */
    int ixlocal = threadIdx.x + 1;
	int Ixglobal = blockIdx.x * (blockDim.x - OVERLAP_X) + xShift + ixlocal;
 
    /* Define local and global y coordinate of point to be updated */
    int iylocal = threadIdx.y + 1;
    int Iyglobal = blockIdx.y * (blockDim.y - OVERLAP_Y) + yShift + iylocal;

	/* Define the number of iterations which were performed */
	int numIters = indexPointerGpu[level+1] - indexPointerGpu[level];

	/* Define nPerSubdomain */
	int nPerSubdomain = subdomainLength_x * subdomainLength_y;

    /* If point is within bound of points to be handled by particular blockIdx.x, blockIdx.y, then move value over to global memory */   
	if (blockIdx.y == gridDim.y - 1) {
		if (iylocal > blockDim.y / 2) { 
			iylocal = iylocal + 2;
			Iyglobal = (blockIdx.y * (blockDim.y - OVERLAP_Y) + yShift + iylocal) % nyGrids;
		}
	}	
	int ilocal = ixlocal + iylocal * subdomainLength_x;
	int Iglobal = Ixglobal + Iyglobal * nxGrids;

	if ((numIters % 2) == 0) { 
		x1Gpu[Iglobal] = sharedMemory[ilocal];
	}
	else {
		x1Gpu[Iglobal] = sharedMemory[ilocal + nPerSubdomain];
	}

    __syncthreads();
}

/* Perform one cycle of hierarchichal SRJ with a shift */
__global__
void _jacobiUpdateSRJShiftedVertical(double * x1Gpu, double * x0Gpu, const double * rhsGpu, const int nxGrids, const int nyGrids, const int OVERLAP_X, const int OVERLAP_Y, const int level, const int * indexPointerGpu, const double * srjSchemesGpu)
{
    /* Move to shared memory */
    extern __shared__ double sharedMemory[];
   
    /* Define useful constants regarding subdomain edge length and number of points within a 2D subdomain */
    int subdomainLength_x, subdomainLength_y;
	subdomainLength_x = blockDim.x + 2;
	if (blockIdx.y < gridDim.y - 1) {
		subdomainLength_y = blockDim.y + 2;
	}
	else {
		subdomainLength_y = blockDim.y + 4;
	}

    /* STEP 1 - MOVE ALL VALUES TO SHARED MEMORY: Assume no overlap */
	__jacobiGlobalToSharedSRJShiftedVertical(x0Gpu, rhsGpu, subdomainLength_x, subdomainLength_y, nxGrids, nyGrids, OVERLAP_X, OVERLAP_Y);
    
    /* STEP 2 - UPDATE ALL INNER POINTS IN EACH BLOCK */
	__jacobiUpdateKernelSRJShiftedVertical(nxGrids, nyGrids, subdomainLength_x, subdomainLength_y, OVERLAP_X, OVERLAP_Y, level, indexPointerGpu, srjSchemesGpu);
    
	/* STEP 3 - MOVE ALL VALUES FROM SHARED MEMORY BACK TO GLOBAL MEMORY */
	__jacobiSharedToGlobalSRJShiftedVertical(x1Gpu, subdomainLength_x, subdomainLength_y, nxGrids, nyGrids, indexPointerGpu, level, OVERLAP_X, OVERLAP_Y);
}

/********************* SRJ with Shared Memory Implementation *************************/
double * jacobiSharedSRJ(const double * initX, const double * rhs, const int nxGrids, const int nyGrids, const double * srjSchemes, const int * indexPointer, const int numSchemes, const int numSchemeParams, const int numCycles, const int levelSRJ, const int threadsPerBlock_x, const int threadsPerBlock_y, const int OVERLAP_X, const int OVERLAP_Y)
{
    /* Number of grid points handled by a subdomain in each direction */
    const int subdomainLength_x = threadsPerBlock_x + 2;
    const int subdomainLength_y = threadsPerBlock_y + 2;

    /* Number of blocks necessary in each direction */
    const int nxBlocks = ceil(((double)nxGrids-2.0-(double)OVERLAP_X) / ((double)threadsPerBlock_x-(double)OVERLAP_X));
    const int nyBlocks = ceil(((double)nyGrids-2.0-(double)OVERLAP_Y) / ((double)threadsPerBlock_y-(double)OVERLAP_Y));

    /* Define the grid and block parameters */
    dim3 grid(nxBlocks, nyBlocks);
    dim3 block(threadsPerBlock_x, threadsPerBlock_y);

    /* Define total number of degrees of freedom */
    int nDofs = nxGrids * nyGrids;
    
    /* Allocate GPU memory via cudaMalloc */
    double * x0Gpu, * x1Gpu, * rhsGpu, * residualGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nDofs);
    cudaMalloc(&x1Gpu, sizeof(double) * nDofs);
    cudaMalloc(&rhsGpu, sizeof(double) * nDofs);
    cudaMalloc(&residualGpu, sizeof(double) * nDofs);
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nDofs, cudaMemcpyHostToDevice);
	
	/* Allocate additional variables related to SRJ schemes */
    double * srjSchemesGpu;
    int * indexPointerGpu;
	cudaMalloc(&srjSchemesGpu, sizeof(double) * numSchemeParams);
    cudaMalloc(&indexPointerGpu, sizeof(int) * numSchemes);
    cudaMemcpy(srjSchemesGpu, srjSchemes, sizeof(double) * numSchemeParams, cudaMemcpyHostToDevice);
    cudaMemcpy(indexPointerGpu, indexPointer, sizeof(int) * numSchemes, cudaMemcpyHostToDevice);

    /* Define amount of shared memory needed */
    const int sharedBytes = 3 * (subdomainLength_x + 2) * (subdomainLength_y + 2) * sizeof(double);

	/* Initialize level */
	int level = 0;

	/* Initialize residual variables */
	// double residual_after;

	/* Perform cycles of hierarchical SRJ */
	for (int cycle = 0; cycle < numCycles; cycle++) {
		/* Perform cycles */
		if (cycle % 4 == 0) {
    		_jacobiUpdateSRJ <<<grid, block, sharedBytes>>> (x1Gpu, x0Gpu, rhsGpu, nxGrids, nyGrids, OVERLAP_X, OVERLAP_Y, level, indexPointerGpu, srjSchemesGpu);
		} 
       	else if (cycle % 4 == 1) {
        	_jacobiUpdateSRJShiftedHorizontal <<<grid, block, sharedBytes>>> (x1Gpu, x0Gpu, rhsGpu, nxGrids, nyGrids, OVERLAP_X, OVERLAP_Y, level, indexPointerGpu, srjSchemesGpu);
		}
		else if (cycle % 4 == 2) {
			_jacobiUpdateSRJShiftedDiagonal <<<grid, block, sharedBytes>>> (x1Gpu, x0Gpu, rhsGpu, nxGrids, nyGrids, OVERLAP_X, OVERLAP_Y, level, indexPointerGpu, srjSchemesGpu);
		}
		else {
			_jacobiUpdateSRJShiftedVertical <<<grid, block, sharedBytes>>> (x1Gpu, x0Gpu, rhsGpu, nxGrids, nyGrids, OVERLAP_X, OVERLAP_Y, level, indexPointerGpu, srjSchemesGpu);
		}
		/* Swap pointers */
		{
            double * tmp = x1Gpu; x1Gpu = x0Gpu; x0Gpu = tmp;
        }
		/* Obtain residual after performing cycles of SRJ */
		// residual_after = residualFastGpu2D(residualGpu, x0Gpu, rhsGpu, nxGrids, nyGrids, threadsPerBlock_x, threadsPerBlock_y, nxBlocks, nyBlocks);
		/* Set the subsequent levesl to levelSRJ */
		level = levelSRJ;
		/* Print info */
		// printf("The residual after cycle %d where we applied SRJ level %d is %f\n", cycle, level, residual_after);
    }

	/* Copy solution to the CPU */
    double * solution = new double[nDofs];
    cudaMemcpy(solution, x0Gpu, sizeof(double) * nDofs, cudaMemcpyDeviceToHost);
/*	printf("PRINTING SOLUTION\n");
	for (int i = 0; i < nDofs; i++) {
		printf("solution(%d) = %f\n", i, solution[i]);
	}
*/
    /* Clean up */
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
    cudaFree(residualGpu);
    cudaFree(srjSchemesGpu);
    cudaFree(indexPointerGpu);

    return solution;
}
