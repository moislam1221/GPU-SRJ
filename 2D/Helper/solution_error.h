void loadSolutionExact(double * solution_exact, std::string SOLUTIONEXACT_FILENAME, const int nDofs) 
{
    std::ifstream solution_exact_file(SOLUTIONEXACT_FILENAME);
    for (int i = 0; i < nDofs; i++) {
        solution_exact_file >> solution_exact[i];
    }
}

double solutionError2DPoisson(const double * solution, const double * solution_exact, int nDofs)
{
    double solution_error = 0.0;
    for (int i = 0; i < nDofs; i++) {
        solution_error = solution_error + (solution_exact[i] - solution[i]) * (solution_exact[i] - solution[i]);
    }
    solution_error = sqrt(solution_error);
    
    return solution_error;
}

__global__
void solutionError2DPoissonGPU(double * solutionErrorGpu, const double * solution, const double * solution_exact, int nDofs)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < nDofs; i += stride) {
	    solutionErrorGpu[i] = (solution_exact[i] - solution[i]) * (solution_exact[i] - solution[i]);
    }
    __syncthreads();
}

/*
__global__
void solutionError1DPoissonGPUNorm(double * solutionErrorGpu, const double * solution, const double * solution_exact, int nGrids, double * solution_error)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < nGrids; i += stride) {
	    solutionErrorGpu[i] = (solution_exact[i] - solution[i]) * (solution_exact[i] - solution[i]);
    }
    __syncthreads();

    *solution_error = 100.0;
    
    extern __shared__ double sharedMemory[];

    // Placing error components in shared memory
    if (index < nGrids) {
        sharedMemory[threadIdx.x] = solutionErrorGpu[index]
    }
    __syncthreads;
 
    // Reduction within shared memory
    for (int s = blockDim.x/2; s > 0; s = s/2) {
        if threadIdx.x < s {
            sharedMemory[threadIdx.x] = sharedMemory[threadIdx.x] + sharedMemory[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        solutionErrorGpu[blockIdx.x] = sharedMemory[0];
    } 

    __syncthreads();
      

//    solution_error = atomicAdd(solutionErrorGpu, nGrids);
}
*/
