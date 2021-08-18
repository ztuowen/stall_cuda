#include <stdint.h>
#define BDIM 32
#define ITERTOT 1024

extern "C" __global__ void 
__launch_bounds__(32, 1)
arr_kernel(int *inptr, int *outptr) {
    int x = threadIdx.x;
    int a = inptr[x];
    uint32_t start = 0;
    uint32_t stop = 0;
    int b;
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
#pragma unroll 8
    for (uint32_t i = 0; i < ITERTOT; ++i) {
        asm volatile ("{\n"
                ".reg .u32 t1;\n"
                "shfl.sync.down.b32 	t1, %1, 4, 31, -1;\n"
                "add.s32 	%0, t1, 1;\n"
                "}\n"
            : "=r"(b) : "r"(a) : "memory");
        a = b;
        // outptr[x] = __shfl_down_sync(0xFFFFFFFF, a, 4, 32) + 1;
    }
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
    outptr[x] = b;
    if (threadIdx.x == 0)
        outptr[x] = stop - start;
}

