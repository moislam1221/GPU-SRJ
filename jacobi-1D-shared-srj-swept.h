__constant__ double constantsrjSchemesGpu[8000];

//double normFromRow(double leftMatrix, double centerMatrix, double rightMatrix, double leftX, double centerX, double rightX,  double centerRhs) 
//{
//    return centerRhs - (leftMatrix*leftX + centerMatrix*centerX + rightMatrix*rightX);
//}

__host__ __device__
double normFromRow2(const double leftX, const double centerX, const double rightX, const double centerRhs, const double dx)
{
    return centerRhs + (leftX - 2.0*centerX + rightX) / (dx*dx);
}

double Residual(const double * solution, const double * rhs, int nGrids)
{
    int nDofs = nGrids;
    double dx = 1.0f / (nGrids + 1);
    double residual = 0.0;
    for (int iGrid = 0; iGrid < nDofs; iGrid++) {
        double leftX = (iGrid > 0) ? solution[iGrid - 1] : 0.0f; 
        double centerX = solution[iGrid];
        double rightX = (iGrid < nGrids - 1) ? solution[iGrid + 1] : 0.0f;
        double residualContributionFromRow = normFromRow2(leftX, centerX, rightX, rhs[iGrid], dx);
	residual = residual + residualContributionFromRow * residualContributionFromRow;
	// printf("For gridpoint %d, residual contribution is %f\n", iGrid, residualContributionFromRow);
    }
    residual = sqrt(residual);
    return residual;
}

__device__ 
void __jacobiBlockUpperTriangleFromShared(double * xLeftBlock, double *xRightBlock, const double *rhsBlock, int nGrids, int iGrid, const double * srjSchemesGpu, const int lowerIndex, const int upperIndex)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x; 

    double dx = 1.0f / (nGrids + 1);
	// printf("UPPER\n");

	#pragma unroll
    for (int k = 1; k < blockDim.x/2; ++k) {
        if (threadIdx.x >= k && threadIdx.x <= blockDim.x-k-1) {
            double leftX = x0[threadIdx.x - 1];
            double centerX = x0[threadIdx.x];
            double rightX = x0[threadIdx.x + 1];
            double rhs = rhsBlock[threadIdx.x];
			double omega = 0.0;
			if (lowerIndex + (k-1) < upperIndex) {
				// CMemory
				// omega = srjSchemesGpu[lowerIndex + (k-1)];
				omega = constantsrjSchemesGpu[lowerIndex + (k-1)];
			}
			else {
				omega = 0.0;				
			}
			// printf("k = %d, omega = %f\n", k, omega);
			if (iGrid == 0) {
				leftX = 0.0f;
			}
			if (iGrid == nGrids-1) {
				rightX = 0.0f;
			}
			// x1[threadIdx.x] = jacobiGrid(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs);
			x1[threadIdx.x] = jacobi1DPoissonRelaxed(leftX, centerX, rightX, rhs, dx, omega);
			// printf("blockIdx.x = %d, x1[%d] = %f, leftX = %f, centerX = %f, rightX = %f, omega = %f\n", blockIdx.x, threadIdx.x, x1[threadIdx.x], leftX, centerX, rightX, omega);
        }
        __syncthreads();	
		double * tmp = x1; x1 = x0; x0 = tmp;
    }
/*
	for (int i = 0; i < blockDim.x; i++) {
		printf("blockIdx = %d, x0[%d] = %f\n", blockIdx.x, threadIdx.x, x0[threadIdx.x]); 
	} 
*/  
    double * tmp = x1; x1 = x0; x0 = tmp;

    int remainder = threadIdx.x % 4;
    xLeftBlock[threadIdx.x] = x0[(threadIdx.x+1)/2 + blockDim.x*(remainder > 1)];
    xRightBlock[threadIdx.x] = x0[blockDim.x-1-(threadIdx.x+1)/2 + blockDim.x*(remainder > 1)];

}

__global__
void _jacobiGpuUpperTriangle(double * xLeftGpu, double *xRightGpu, const double * x0Gpu, const double *rhsGpu, int nGrids, const double * srjSchemesGpu, const int lowerIndex, const int upperIndex)
{
    int blockShift = blockDim.x * blockIdx.x;
    double * xLeftBlock = xLeftGpu + blockShift;
    double * xRightBlock = xRightGpu + blockShift;
    const double * x0Block = x0Gpu + blockShift;
    const double * rhsBlock = rhsGpu + blockShift;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double sharedMemory[];
    sharedMemory[threadIdx.x] = x0Block[threadIdx.x];
    sharedMemory[threadIdx.x + blockDim.x] = x0Block[threadIdx.x];

    __jacobiBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock, nGrids, iGrid, srjSchemesGpu, lowerIndex, upperIndex);

}

__device__ 
void __jacobiBlockLowerTriangleFromShared(const double * xLeftBlock, const double *xRightBlock, const double *rhsBlock, int nGrids, int iGrid, const double * srjSchemesGpu, const int lowerIndex, const int upperIndex)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x;

    int remainder = threadIdx.x % 4;

    if (threadIdx.x != blockDim.x-1) {
        x0[blockDim.x-1-((blockDim.x+threadIdx.x+1)/2) + blockDim.x*(remainder>1)] = xLeftBlock[threadIdx.x];
		x0[(blockDim.x+threadIdx.x+1)/2 + blockDim.x*(remainder>1)] = xRightBlock[threadIdx.x];
    }

    double dx = 1.0f / (nGrids + 1);
	// printf("LOWER\n");

    # pragma unroll
    for (int k = blockDim.x/2; k > 0; --k) {
	if (k < blockDim.x/2) {
	    if (threadIdx.x >= k && threadIdx.x <= blockDim.x-k-1) {
        	double leftX = x0[threadIdx.x - 1];
            double centerX = x0[threadIdx.x];
            double rightX = x0[threadIdx.x + 1];
			double rhs = rhsBlock[threadIdx.x]; 
			double omega;
			if (lowerIndex + blockDim.x/2 - 1 - k < upperIndex) {
				// double omega = srjSchemesGpu[lowerIndex + blockDim.x/2 -1 + (blockDim.x/2 - 1 - k)]
				// CMemory
				// omega = srjSchemesGpu[lowerIndex + blockDim.x/2 - 1 - k];
				omega = constantsrjSchemesGpu[lowerIndex + blockDim.x/2 - 1 - k];
			}
			else {
				omega = 0.0;
			}
			// printf("k = %d, omega = %f\n", k, omega);
			if (iGrid == 0) {
		    	leftX = 0.0f;
			}
			if (iGrid == nGrids-1) {
		    	rightX = 0.0f;
			}
	        // x1[threadIdx.x] = jacobiGrid(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs);
			x1[threadIdx.x] = jacobi1DPoissonRelaxed(leftX, centerX, rightX, rhs, dx, omega);
	    }
 	    double * tmp = x1; x1 = x0; x0 = tmp;
    }
	__syncthreads();
    }

    double leftX = (threadIdx.x == 0) ? xLeftBlock[blockDim.x - 1] : x0[threadIdx.x - 1];
    double centerX = x0[threadIdx.x];
    double rightX = (threadIdx.x == blockDim.x-1) ? xRightBlock[blockDim.x - 1] : x0[threadIdx.x + 1];
    double rhs = rhsBlock[threadIdx.x];
	double omega;
	if ((lowerIndex + blockDim.x/2 - 1) < upperIndex) {
		// CMemory
		// omega = srjSchemesGpu[lowerIndex + blockDim.x/2 - 1];
		omega = constantsrjSchemesGpu[lowerIndex + blockDim.x/2 - 1];
	}
	else {
		omega = 0.0;
	}
	// printf("k = 0, omega = %f\n", omega);
    if (iGrid == 0) {
       leftX = 0.0;    
    }
    if (iGrid == nGrids-1) {
        rightX = 0.0;
    }
    // x1[threadIdx.x] = jacobiGrid(leftMatrixBlock[threadIdx.x], centerMatrixBlock[threadIdx.x], rightMatrixBlock[threadIdx.x], leftX, centerX, rightX, rhsBlock[threadIdx.x]);
	x1[threadIdx.x] = jacobi1DPoissonRelaxed(leftX, centerX, rightX, rhs, dx, omega);
    
	double * tmp = x1; x1 = x0; x0 = tmp; 
}

__global__
void _jacobiGpuLowerTriangle(double * x0Gpu, double *xLeftGpu, double * xRightGpu, double *rhsGpu, int nGrids, const double * srjSchemesGpu, int lowerIndex, const int upperIndex)
{
    int blockShift = blockDim.x * blockIdx.x;
    double * xLeftBlock = xLeftGpu + blockShift;
    double * xRightBlock = xRightGpu + blockShift;
    double * x0Block = x0Gpu + blockShift;
    double * rhsBlock = rhsGpu + blockShift;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ double sharedMemory[];
    
	lowerIndex = lowerIndex + blockDim.x/2; 
    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock, nGrids, iGrid, srjSchemesGpu, lowerIndex, upperIndex);

    x0Block[threadIdx.x] = sharedMemory[threadIdx.x];

}

__global__       
void _jacobiGpuShiftedDiamond(double * xLeftGpu, double * xRightGpu, double * rhsGpu, int nGrids, const double * srjSchemes, int lowerIndex, const int upperIndex)
{
    int blockShift = blockDim.x * blockIdx.x;
    double * xLeftBlock = xRightGpu + blockShift;
    double * xRightBlock = (blockIdx.x == (gridDim.x-1)) ?
                          xLeftGpu : 
                          xLeftGpu + blockShift + blockDim.x;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x/2;
    iGrid = (iGrid < nGrids) ? iGrid : threadIdx.x - blockDim.x/2;

    int indexShift = blockDim.x/2;
    double * rhsBlock = rhsGpu + blockShift + indexShift;
    
    extern __shared__ double sharedMemory[];
    
    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock, nGrids, iGrid, srjSchemes, lowerIndex, upperIndex);  
	lowerIndex = lowerIndex + blockDim.x/2; 
    __jacobiBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock, nGrids, iGrid, srjSchemes, lowerIndex, upperIndex);

}

__global__
void _jacobiGpuDiamond(double * xLeftGpu, double * xRightGpu, const double * rhsGpu, int nGrids, const double * srjSchemes, int lowerIndex, const int upperIndex)
{
    int blockShift = blockDim.x * blockIdx.x;
    double * xLeftBlock = xLeftGpu + blockShift;
    double * xRightBlock = xRightGpu + blockShift;
    const double * rhsBlock = rhsGpu + blockShift;

    int iGrid = blockDim.x * blockIdx.x + threadIdx.x;
    
    extern __shared__ double sharedMemory[];

    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock, nGrids, iGrid, srjSchemes, lowerIndex, upperIndex);
	lowerIndex = lowerIndex + blockDim.x/2; 
    __jacobiBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock, nGrids, iGrid, srjSchemes, lowerIndex, upperIndex);
}
double * jacobiGpuSwept(const double * initX, const double * rhs, int nGrids, int nIters, const int threadsPerBlock, const double * srjSchemes, const int * indexPointer, const int numSchemes, const int numSchemeParams, const int level) { 
    
	// Determine number of threads and blocks 
    const int nBlocks = (int)ceil(nGrids / (double)threadsPerBlock);

    // Allocate memory for solution and inputs
    double *xLeftGpu, *xRightGpu;
    cudaMalloc(&xLeftGpu, sizeof(double) * threadsPerBlock * nBlocks);
    cudaMalloc(&xRightGpu, sizeof(double) * threadsPerBlock * nBlocks);
    double * x0Gpu, * rhsGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&rhsGpu, sizeof(double) * (nGrids + threadsPerBlock/2));
    double * srjSchemesGpu;
	int * indexPointerGpu;
    cudaMalloc(&srjSchemesGpu, sizeof(double) * numSchemeParams);
    cudaMalloc(&indexPointerGpu, sizeof(int) * numSchemes);
    // cudaMemcpy(srjSchemesGpu, srjSchemes, sizeof(double) * numSchemeParams, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constantsrjSchemesGpu, srjSchemes, sizeof(double) * 8000);
    cudaMemcpy(indexPointerGpu, indexPointer, sizeof(int) * numSchemes, cudaMemcpyHostToDevice);

    // Allocate memory in the GPU
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);

    // Allocate a bit more memory to avoid memcpy within shifted kernels
    cudaMemcpy(x0Gpu + nGrids, initX, sizeof(double) * threadsPerBlock/2, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu + nGrids, rhs, sizeof(double) * threadsPerBlock/2, cudaMemcpyHostToDevice);

    int sharedFloatsPerBlock = threadsPerBlock * 2;
/*
	for (int sweptCycle = 0; sweptCycle < nIters; sweptCycle++) {
		int lowerIndex = indexPointer[level];
		int upperIndex = indexPointer[level+1];
		int subLevelCycles = ceil(((float)upperIndex - (float)lowerIndex) / (float)threadsPerBlock);
		for (int subLevelCycle = 0; subLevelCycle < subLevelCycles; subLevelCycle++) {
			_jacobiGpuUpperTriangle <<<nBlocks, threadsPerBlock, sizeof(double) * sharedFloatsPerBlock>>>(xLeftGpu, xRightGpu, x0Gpu, rhsGpu, nGrids, srjSchemesGpu, lowerIndex, upperIndex);
			_jacobiGpuShiftedDiamond <<<nBlocks, threadsPerBlock, sizeof(double) * sharedFloatsPerBlock>>>(xLeftGpu, xRightGpu, rhsGpu, nGrids, srjSchemesGpu, lowerIndex, upperIndex);
			_jacobiGpuLowerTriangle <<<nBlocks, threadsPerBlock, sizeof(double) * sharedFloatsPerBlock>>>(x0Gpu, xLeftGpu, xRightGpu, rhsGpu, nGrids, srjSchemesGpu, lowerIndex, upperIndex);
			lowerIndex = lowerIndex + threadsPerBlock;
		}
	}
*/

	for (int sweptCycle = 0; sweptCycle < nIters; sweptCycle++) {
		int lowerIndex = indexPointer[level];
		int upperIndex = indexPointer[level+1];
		int numIters = upperIndex - lowerIndex;
		int diamondCycles = floor((((float)numIters - 33.0f) / (float)threadsPerBlock) + 1);
		_jacobiGpuUpperTriangle <<<nBlocks, threadsPerBlock, sizeof(double) * sharedFloatsPerBlock>>>(xLeftGpu, xRightGpu, x0Gpu, rhsGpu, nGrids, srjSchemesGpu, lowerIndex, upperIndex);
		_jacobiGpuShiftedDiamond <<<nBlocks, threadsPerBlock, sizeof(double) * sharedFloatsPerBlock>>>(xLeftGpu, xRightGpu, rhsGpu, nGrids, srjSchemesGpu, lowerIndex, upperIndex);
		for (int diamondCycle = 0; diamondCycle < diamondCycles; diamondCycle++) {
			lowerIndex = lowerIndex + threadsPerBlock / 2;
			_jacobiGpuDiamond <<<nBlocks, threadsPerBlock, sizeof(double) * sharedFloatsPerBlock>>>(xLeftGpu, xRightGpu, rhsGpu, nGrids, srjSchemesGpu, lowerIndex, upperIndex);
			lowerIndex = lowerIndex + threadsPerBlock / 2;
			_jacobiGpuShiftedDiamond <<<nBlocks, threadsPerBlock, sizeof(double) * sharedFloatsPerBlock>>>(xLeftGpu, xRightGpu, rhsGpu, nGrids, srjSchemesGpu, lowerIndex, upperIndex);
		}
		_jacobiGpuLowerTriangle <<<nBlocks, threadsPerBlock, sizeof(double) * sharedFloatsPerBlock>>>(x0Gpu, xLeftGpu, xRightGpu, rhsGpu, nGrids, srjSchemesGpu, lowerIndex, upperIndex);
	}
 
    double * solution = new double[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(double) * nGrids,
            cudaMemcpyDeviceToHost);

    cudaFree(x0Gpu);
    cudaFree(xLeftGpu);
    cudaFree(xRightGpu);
    cudaFree(rhsGpu);

    return solution;
}
