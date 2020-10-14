void setGPU(std::string gpuToUse) {
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        // printf("%s\n",deviceProperties.name);
        // printf("%d\n", deviceProperties.name == gpuToUse);
        if (deviceProperties.name == gpuToUse) {
            cudaSetDevice(deviceIndex);
            printf("GPU in use: %s\n", deviceProperties.name);
        }
    }
}
