/* 1 - Global to Shared Transfer */
__device__
void __jacobiGlobalToShared(const double * x0Gpu, const double * rhsGpu, const int nPerSubdomain, const int nGrids, const int OVERLAP, const int blocksPerDomain, const int Mcopies)
{
    /* Define shared memory */
	extern __shared__ double sharedMemory[];

	/* Total number of grid points */
	const int nTotalGrids = nGrids * Mcopies;
    
	/* Define local and global ID */
	const int I = threadIdx.x + (blockDim.x - OVERLAP) * (blockIdx.x % blocksPerDomain) + nGrids * (blockIdx.x / blocksPerDomain);
    const int i = threadIdx.x;

	// Move first tpb points in subdomain */
    if (I < nTotalGrids && I < nGrids * (1 + blockIdx.x/blocksPerDomain)) {
        sharedMemory[i] = x0Gpu[I];
        sharedMemory[i + nPerSubdomain] = sharedMemory[i];
        sharedMemory[i + 2 * nPerSubdomain] = rhsGpu[I]; 
    }
   
	/* Move remaining 2 points */
	const int I2 = I + blockDim.x;
    const int i2 = i + blockDim.x;
    if (i2 < nPerSubdomain && I2 < nTotalGrids && I2 < nGrids * (1 + blockIdx.x/blocksPerDomain)) {
        sharedMemory[i2] = x0Gpu[I2];
        sharedMemory[i2 + nPerSubdomain] = sharedMemory[i2];
        sharedMemory[i2 + 2 * nPerSubdomain] = rhsGpu[I2]; 
    }
}

/* 2 - Update within shared memory */
__device__
void __jacobiUpdateKernelSRJ(const int nGrids, const int nSub, const int OVERLAP, const int blocksPerDomain, const int level, const int * indexPointerGpu, const double * srjSchemesGpu, const int Mcopies)
{
	/* Create shared memory pointers */
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory, * x1 = sharedMemory + nSub, * x2 = sharedMemory + 2 * nSub;
    const double dx = 1.0 / (nGrids - 1);
    
	/* Perform one SRJ cycle based on the current level */
	for (int relaxationParameterID = indexPointerGpu[level]; relaxationParameterID < indexPointerGpu[level+1]; relaxationParameterID++) {
        int i = threadIdx.x + 1;
    	int I = i + (blockDim.x - OVERLAP) * (blockIdx.x % blocksPerDomain) + nGrids * (blockIdx.x / blocksPerDomain);
        if (I < Mcopies * nGrids-1 && I < nGrids * (1 + blockIdx.x/blocksPerDomain) - 1) {
            double leftX = x0[i-1];
            double rightX = x0[i+1];
            double rhs = x2[i];
            double centerX = x0[i];
            x1[i] = jacobi1DPoissonRelaxed(leftX, centerX, rightX, rhs, dx, srjSchemesGpu[relaxationParameterID]);
		}
        __syncthreads();
        double * tmp = x1; x1 = x0; x0 = tmp;
    }
}

/* 3 - Shared to Global Transfer */
__device__
void __jacobiSharedToGlobal(double * x1Gpu, const int nPerSubdomain, const int nGrids, const int * indexPointerGpu, const int level, const int OVERLAP, const int blocksPerDomain, const int Mcopies)
{
    /* Define shared memory */
	extern __shared__ double sharedMemory[];
	
	/* Define local and global ID */
	const int I = threadIdx.x + (blockDim.x - OVERLAP) * (blockIdx.x % blocksPerDomain) + nGrids * (blockIdx.x / blocksPerDomain);
    const int i = threadIdx.x;
	
	/* Define the numnber of iterations which were performed */
	int numIters = indexPointerGpu[level+1] - indexPointerGpu[level];

	/* Transfer values from shared memory to global memory */
    if ((I+1) < Mcopies * nGrids - 1 && (I+1) < nGrids * (1 + blockIdx.x/blocksPerDomain) - 1) {
		if ((numIters % 2) == 0) {
			x1Gpu[I+1] = sharedMemory[i+1];
		}
		else {
			x1Gpu[I+1] = sharedMemory[i+1 + nPerSubdomain];
		}
    }
}

/* Function to perform one hierarchichal cycle */
__global__
void _jacobiUpdateSRJ(double * x1Gpu, const double * x0Gpu, const double * rhsGpu, const int nGrids, const int OVERLAP, const int blocksPerDomain, const int level, const int * indexPointerGpu, const double * srjSchemesGpu, const int Mcopies)
{
    // Move to shared memory
    extern __shared__ double sharedMemory[];

	// Define number of points in subdomain
    const int nPerSubdomain = blockDim.x + 2;
	
	// STEP 1 - MOVE ALL VALUES TO SHARED MEMORY
	__jacobiGlobalToShared(x0Gpu, rhsGpu, nPerSubdomain, nGrids, OVERLAP, blocksPerDomain, Mcopies);
    
	// STEP 2 - UPDATE ALL INNER POINTS IN EACH BLOCK
    __jacobiUpdateKernelSRJ(nGrids, nPerSubdomain, OVERLAP, blocksPerDomain, level, indexPointerGpu, srjSchemesGpu, Mcopies);

    // STEP 3 - MOVE ALL VALUES FROM SHARED MEMORY BACK TO GLOBAL MEMORY
	__jacobiSharedToGlobal(x1Gpu, nPerSubdomain, nGrids, indexPointerGpu, level, OVERLAP, blocksPerDomain, Mcopies);

}

/* */

/* 1 - Global to Shared Transfer Shifted */
__device__
void __jacobiGlobalToSharedShifted(const double * x0Gpu, const double * rhsGpu, const int nPerSubdomain, const int nGrids, const int OVERLAP, const int blocksPerDomain, const int Mcopies)
{
    /* Define shared memory */
	extern __shared__ double sharedMemory[];

	/* Define total grids */
	const int nTotalGrids = nGrids * Mcopies;
    
	/* Define shift and local and global ID */
	const int shift = blockDim.x / 2;
	const int I = threadIdx.x + (blockDim.x - OVERLAP) * (blockIdx.x % blocksPerDomain) + nGrids * (blockIdx.x / blocksPerDomain) + shift;
    const int i = threadIdx.x;

	/* If not the last GPU block, do everything as normal with the shift */
	if ((blockIdx.x % blocksPerDomain) != blocksPerDomain - 1) {
		// Move first tpb points
		if (I < nTotalGrids && I < nGrids * (1 + blockIdx.x/blocksPerDomain)) {
			sharedMemory[i] = x0Gpu[I];
			sharedMemory[i + nPerSubdomain] = sharedMemory[i];
			sharedMemory[i + 2 * nPerSubdomain] = rhsGpu[I]; 
		}
		// Move remaining 2 points 
		const int I2 = I + blockDim.x;
		const int i2 = i + blockDim.x;
		if (i2 < nPerSubdomain && I2 < nTotalGrids && I2 < nGrids * (1 + blockIdx.x/blocksPerDomain)) {
			sharedMemory[i2] = x0Gpu[I2];
			sharedMemory[i2 + nPerSubdomain] = sharedMemory[i2];
			sharedMemory[i2 + 2 * nPerSubdomain] = rhsGpu[I2]; 
		}
	}
	/* For the last block of each domain */
	else {
		/* Move the last grid points to first half of shared memory */
		if (i < nPerSubdomain / 2) {
			sharedMemory[i] = x0Gpu[I];
			sharedMemory[i + nPerSubdomain] = sharedMemory[i];
			sharedMemory[i + 2 * nPerSubdomain] = rhsGpu[I]; 
		}
		/* Move the first grid points to the second half of shared memory for last block */
		else { 
			sharedMemory[i] = x0Gpu[i-nPerSubdomain/2];
			sharedMemory[i + nPerSubdomain] = sharedMemory[i];
			sharedMemory[i + 2 * nPerSubdomain] = rhsGpu[i-nPerSubdomain/2]; 
		}
		const int i2 = i + blockDim.x;
		if (i2 < nPerSubdomain) {
			sharedMemory[i2] = x0Gpu[i2-nPerSubdomain/2];
			sharedMemory[i2 + nPerSubdomain] = sharedMemory[i2];
			sharedMemory[i2 + 2 * nPerSubdomain] = rhsGpu[i2-nPerSubdomain/2]; 
		}
	}
	__syncthreads();

	/* Print values if necessary */

}

/* 2 - Update within shared memory shifted */
__device__
void __jacobiUpdateKernelSRJShifted(const int nGrids, const int nSub, const int OVERLAP, const int blocksPerDomain, const int level, const int * indexPointerGpu, const double * srjSchemesGpu, const int Mcopies)
{
	/* Define shared memory and pointers */
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory, * x1 = sharedMemory + nSub, * x2 = sharedMemory + 2 * nSub;
    const double dx = 1.0 / (nGrids - 1);
    
	/* Perform one cycle of the SRJ method */
	for (int relaxationParameterID = indexPointerGpu[level]; relaxationParameterID < indexPointerGpu[level+1]; relaxationParameterID++) {
		/* If not the last block, perform same steps as before */
		if ((blockIdx.x % blocksPerDomain) != blocksPerDomain - 1) {
        	int i = threadIdx.x + 1;
    		int I = i + (blockDim.x - OVERLAP) * (blockIdx.x % blocksPerDomain) + nGrids * (blockIdx.x / blocksPerDomain);
        	// int I = blockIdx.x * (blockDim.x - OVERLAP)+ i;
        	if (I < Mcopies * nGrids-1 && I < nGrids * (1 + blockIdx.x/blocksPerDomain) - 1) {
            	double leftX = x0[i-1];
            	double rightX = x0[i+1];
            	double rhs = x2[i];
            	double centerX = x0[i];
            	x1[i] = jacobi1DPoissonRelaxed(leftX, centerX, rightX, rhs, dx, srjSchemesGpu[relaxationParameterID]);
			}
		}
		/* For last block, be sure to update only the interior DOFs */
		else {
			int i = threadIdx.x + 1;
			if (i <= (nSub/2-2)) {
            	double leftX = x0[i-1];
            	double rightX = x0[i+1];
            	double rhs = x2[i];
            	double centerX = x0[i];
            	x1[i] = jacobi1DPoissonRelaxed(leftX, centerX, rightX, rhs, dx, srjSchemesGpu[relaxationParameterID]);
				// printf("ThreadIdx.x %d, shared DOF %d: leftX %f, centerX %f, rightX %f gives %f, rhs %f\n", threadIdx.x, i, x0[i-1], x0[i], x0[i+1], x1[i], x2[i]);
			}
			else { 
				int i_adjust = i + 2;
            	double leftX = x0[i_adjust-1];
            	double rightX = x0[i_adjust+1];
            	double rhs = x2[i_adjust];
            	double centerX = x0[i_adjust];
            	x1[i_adjust] = jacobi1DPoissonRelaxed(leftX, centerX, rightX, rhs, dx, srjSchemesGpu[relaxationParameterID]);
				// printf("SECOND HALF - ThreadIdx.x %d, shared DOF %d: leftX %f, centerX %f, rightX %f gives %f, rhs %f\n", threadIdx.x, i_adjust, x0[i_adjust-1], x0[i_adjust], x0[i_adjust+1], x1[i_adjust], x2[i_adjust]);
			}
		}
        __syncthreads();
		/* Swap pointers */
        double * tmp = x1; x1 = x0; x0 = tmp;
    }

	/* Print out values if necessary */
/*	
	if (threadIdx.x == 0) {
		printf("STEP 2\n");
		for (int i = 0; i < nSub; i++) {
			printf("BlockID %d: sharedMemory[%d] = %f, rhs[%d] = %f\n", blockIdx.x, i, x0[i], i, x2[i]);
		}
	}	
*/
}

/* 3 - Shared to Global Transfer shifted */
__device__
void __jacobiSharedToGlobalShifted(double * x1Gpu, const int nPerSubdomain, const int nGrids, const int * indexPointerGpu, const int level, const int OVERLAP, const int blocksPerDomain, const int Mcopies)
{
    /* Define shared memory */
	extern __shared__ double sharedMemory[];
	
	/* Define local and global ID */
	const int shift = blockDim.x / 2;
	const int I = threadIdx.x + (blockDim.x - OVERLAP) * (blockIdx.x % blocksPerDomain) + nGrids * (blockIdx.x / blocksPerDomain) + shift;
    const int i = threadIdx.x;
	
	/* Define the numnber of iterations which were performed */
	int numIters = indexPointerGpu[level+1] - indexPointerGpu[level];

	/* Move back from shared to global memory */
	if ((blockIdx.x % blocksPerDomain) != blocksPerDomain - 1) {
		if ((I+1) < Mcopies * nGrids && (I+1) < nGrids * (1 + blockIdx.x/blocksPerDomain) - 1) {
			if ((numIters % 2) == 0) {
				x1Gpu[I+1] = sharedMemory[i+1];
			}
			else {
				x1Gpu[I+1] = sharedMemory[i+1 + nPerSubdomain];
			}
		}
	}
	/* For the last block, allocate points appropriately (second half of shared goes back to beginning of grid) */
	else {
		if ((i+1) <= (nPerSubdomain-4) / 2) {
			if ((numIters % 2) == 0) {
				x1Gpu[I+1] = sharedMemory[i+1];
			}
			else {
				x1Gpu[I+1] = sharedMemory[i+1 + nPerSubdomain];
			}
		}
		else {
			int i_adjust = i + 2;
			int I_adjust = (i_adjust + (blockDim.x - OVERLAP) * (blockIdx.x % blocksPerDomain) + nGrids * (blockIdx.x / blocksPerDomain) + shift) % (nGrids * Mcopies);
			if ((numIters % 2) == 0) {
				x1Gpu[I_adjust+1] = sharedMemory[i_adjust+1];
			}
			else {
				x1Gpu[I_adjust+1] = sharedMemory[i_adjust +1 + nPerSubdomain];
			}
		}
	}
}

/* Perform one cycle of hierarchical SRJ with a shift */
__global__
void _jacobiUpdateSRJShifted(double * x1Gpu, const double * x0Gpu, const double * rhsGpu, const int nGrids, const int OVERLAP, const int blocksPerDomain, const int level, const int * indexPointerGpu, const double * srjSchemesGpu, const int Mcopies)
{
    /* Move to shared memory */
    extern __shared__ double sharedMemory[];

	/* The last block has 2 extra grid points */
	int nPerSubdomain;
	if ((blockIdx.x % blocksPerDomain) != blocksPerDomain - 1) {
    	nPerSubdomain = blockDim.x + 2;
	}
	else {
    	nPerSubdomain = blockDim.x + 4;
	}
	
	/* Total number of grid points */
	const int nTotalGrids = nGrids;
    
	/* STEP 1 - MOVE ALL VALUES TO SHARED MEMORY */
	__jacobiGlobalToSharedShifted(x0Gpu, rhsGpu, nPerSubdomain, nTotalGrids, OVERLAP, blocksPerDomain, Mcopies);
     
	/* STEP 2 - UPDATE ALL INNER POINTS IN EACH BLOCK */
    __jacobiUpdateKernelSRJShifted(nGrids, nPerSubdomain, OVERLAP, blocksPerDomain, level, indexPointerGpu, srjSchemesGpu, Mcopies);

    /* STEP 3 - MOVE ALL VALUES FROM SHARED MEMORY BACK TO GLOBAL MEMORY */
	__jacobiSharedToGlobalShifted(x1Gpu, nPerSubdomain, nGrids, indexPointerGpu, level, OVERLAP, blocksPerDomain, Mcopies);
}

/* SRJ with Shared Memory Implementation */
double * jacobiSharedSRJShifted(const double * initX, const double * rhs, const int nGrids, const double * srjSchemes, const int * indexPointer, const int numSchemes, const int numSchemeParams, const int threadsPerBlock, int OVERLAP, const int numCycles, const int levelSRJ, const int Mcopies)
{
    /* Number of grid points handled by a subdomain (except the last one)*/
    const int nSub = threadsPerBlock + 2;

	/* Total number of grid points */
	const int nTotalGrids = nGrids * Mcopies;

    /* Allocate GPU memory and copy arrays */
    double * x0Gpu, * x1Gpu, * rhsGpu; // * residualGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nTotalGrids);
    cudaMalloc(&x1Gpu, sizeof(double) * nTotalGrids);
    cudaMalloc(&rhsGpu, sizeof(double) * nTotalGrids);
    // cudaMalloc(&residualGpu, sizeof(double) * nGrids);
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nTotalGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(x1Gpu, initX, sizeof(double) * nTotalGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nTotalGrids, cudaMemcpyHostToDevice);
    
	/* Allocate additional variables related to SRJ schemes */
    double * srjSchemesGpu;
    int * indexPointerGpu;
	cudaMalloc(&srjSchemesGpu, sizeof(double) * numSchemeParams);
    cudaMalloc(&indexPointerGpu, sizeof(int) * numSchemes);
    cudaMemcpy(srjSchemesGpu, srjSchemes, sizeof(double) * numSchemeParams, cudaMemcpyHostToDevice);
    cudaMemcpy(indexPointerGpu, indexPointer, sizeof(int) * numSchemes, cudaMemcpyHostToDevice);
    
	/* Number of blocks necessary */
    const int blocksPerDomain = ceil(((double)nGrids-2.0-(double)OVERLAP) / ((double)threadsPerBlock-(double)OVERLAP));
	const int numBlocks = blocksPerDomain * Mcopies;

    /* Define amount of shared memory needed (give each subdomain tpb + 4 * 3 entries of memory) */
    const int sharedBytes = 3 * (nSub + 2) * sizeof(double);
    
	/* Initialize level */
	int level = levelSRJ;

	/* Initialize residual variables */
	// double residual_after;

	/* Perform cycles of hierarchical SRJ */ 
    for (int cycle = 0; cycle < numCycles; cycle++) {
		/* Perform cycles (alternating between unshifted and shifted iterations) */
        if (cycle % 2 == 0) {
			_jacobiUpdateSRJ <<<numBlocks, threadsPerBlock, sharedBytes>>> (x1Gpu, x0Gpu, rhsGpu, nGrids, OVERLAP, blocksPerDomain, level, indexPointerGpu, srjSchemesGpu, Mcopies);
		}
		else {
			_jacobiUpdateSRJShifted <<<numBlocks, threadsPerBlock, sharedBytes>>> (x1Gpu, x0Gpu, rhsGpu, nGrids, OVERLAP, blocksPerDomain, level, indexPointerGpu, srjSchemesGpu, Mcopies);
		}
		/* Swap pointers */
		double * tmp = x1Gpu; x1Gpu = x0Gpu; x0Gpu = tmp;
		/* Obtain residual after performing cycles of SRJ */
		// residual_after = residualFastGpu(residualGpu, x0Gpu, rhsGpu, nGrids, threadsPerBlock, blocksPerDomain);
		/* Set the subsequent levels to levelSRJ */
		// level = levelSRJ;
		/* Print info */
		// printf("The residual after cycle %d where we applied SRJ level %d is %.15f\n", cycle, level, residual_after);
    }

	/* Copy solution to the CPU */
    double * solution = new double[nTotalGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(double) * nTotalGrids, cudaMemcpyDeviceToHost);

    /* Clean up */
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
	// cudaFree(residualGpu);
	cudaFree(srjSchemesGpu);
	cudaFree(indexPointerGpu);

    return solution;
}
