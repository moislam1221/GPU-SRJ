# GPU-SRJ

We solve a 1D Poisson equation with Dirichlet boundary conditions using a Scheduled Relaxation Jacobi (SRJ) method.

Three implementations are developed
* CPU Implementation
* Equivalent GPU Implementation using global memory
* GPU Implementation using shared memory

The goal is to explore the speedup achievable by utilizing quick to access shared memory in conjunction with SRJ.

One can run the main file using the following commands:

nvcc main-1D-poisson-automated.cu

./a.out N tpb numCycles levelSRJ
  
The four inputs to the function are:
* N - the number of equally spaced interior grid points (not counting boundary points)
* tpb - number of threads per block for the GPU implementation
* numCycles - the number of SRJ cycles to perform
* levelSRJ - the level of the SRJ scheme you wish to use (higher level schemes contain more steps - we have schemes up to level 24)

The code requires a cuda installation (we used cuda-10.1 in development). This can be obtained from NVIDIA's website. An NVIDIA GPU is also required.

Enjoy :smile:
