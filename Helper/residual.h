__host__ __device__
double normFromRow(const double leftX, const double centerX, const double rightX, const double centerRhs, const double dx)
{
    return centerRhs + (leftX - 2.0*centerX + rightX) / (dx*dx);
}

double residual1DPoisson(const double * solution, const double * rhs, int nGrids)
{
    double residual = 0.0;
    double dx = 1.0 / (nGrids - 1);
    double leftX, centerX, rightX, residualContributionFromRow;

    for (int iGrid = 0; iGrid < nGrids; iGrid++) {
        leftX = (iGrid > 0) ? solution[iGrid - 1] : 0.0f;
        centerX = solution[iGrid];
        rightX = (iGrid < nGrids) ? solution[iGrid + 1] : 0.0f;
        residualContributionFromRow = normFromRow(leftX, centerX, rightX, rhs[iGrid], dx);
        residual = residual + residualContributionFromRow * residualContributionFromRow;
	}

    residual = sqrt(residual);
    return residual;
}

__global__
void residual1DPoissonGPU(double * residualGpu, const double * solution, const double * rhs, int nGrids)
{
    double dx = 1.0 / (nGrids - 1);
    double leftX, centerX, rightX, residualContributionFromRow;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < nGrids; i += stride) {
        leftX = (i > 0) ? solution[i - 1] : 0.0f;
        centerX = solution[i];
        rightX = (i < nGrids) ? solution[i + 1] : 0.0f;
        residualContributionFromRow = normFromRow(leftX, centerX, rightX, rhs[i], dx);
        residualGpu[i] = residualContributionFromRow * residualContributionFromRow;
    }
    __syncthreads();
}

double residualFastGpu(double * residualGpu, const double * x0Gpu, const double * rhsGpu, const int nGrids, const int threadsPerBlock, const int numBlocks)
{
	/* Obtain the final L2 norm residual of a solution existing on the device */
	residual1DPoissonGPU <<<numBlocks, threadsPerBlock>>> (residualGpu, x0Gpu, rhsGpu, nGrids);
    cudaDeviceSynchronize();
	double * residualCpu = new double[nGrids];
    cudaMemcpy(residualCpu, residualGpu, sizeof(double) * nGrids, cudaMemcpyDeviceToHost);
    double residual = 0.0;
    for (int j = 0; j < nGrids; j++) {
         residual = residual + residualCpu[j];
    }
    residual = sqrt(residual);

	return residual;
}

