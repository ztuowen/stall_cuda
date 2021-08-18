#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

#define BDIM 32

CUdevice   device;
CUcontext  context;
CUmodule   module;
CUfunction function;

#define module_file "kernel.cubin"
#define kernel_name "arr_kernel"

void initCUDA()
{
    int deviceCount = 0;
    CUresult err = cuInit(0);
    if (err == CUDA_SUCCESS)
        cuDeviceGetCount(&deviceCount);

    if (deviceCount == 0) {
        fprintf(stderr, "Error: no devices supporting CUDA\n");
        exit(-1);
    }

    // get first CUDA device
    cuDeviceGet(&device, 0);
    char name[100];
    cuDeviceGetName(name, 100, device);
    printf("> Using device 0: %s\n", name);

    err = cuCtxCreate(&context, 0, device);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error initializing the CUDA context.\n");
        cuCtxDestroy(context);
        exit(-1);
    }

    err = cuModuleLoad(&module, module_file);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error loading the module %s\n", module_file);
        const char * str;
        cuGetErrorString(err, &str);
        fprintf(stderr, "%s\n", str);
        cuCtxDestroy(context);
        exit(-1);
    }

    err = cuModuleGetFunction(&function, module, kernel_name);

    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
        const char * str;
        cuGetErrorString(err, &str);
        fprintf(stderr, "%s\n", str);
        cuCtxDestroy(context);
        exit(-1);
    }
}

int main() {
    int size = BDIM * 16 * sizeof(int);
    int *in = (int *)malloc(size);
    int *out = (int *)malloc(size);
    int *in_dev, *out_dev;
    initCUDA();
    cudaMalloc(&in_dev, size);
    cudaMalloc(&out_dev, size);
    for (int i = 0; i < BDIM; ++i)
        in[i] = i;
    cudaMemcpy(in_dev, in, size, cudaMemcpyHostToDevice);
    void * args[2] = {&in_dev, &out_dev};
    cuLaunchKernel(function, 
            1, 1, 1,
            BDIM, 1, 1,
            0, 0, args, 0);
    // Test
    cudaMemcpy(out, out_dev, size, cudaMemcpyDeviceToHost);
    printf("%d\n",out[0]);
    return 0;
}
