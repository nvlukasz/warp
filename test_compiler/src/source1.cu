#include <cuda/std/array>

extern "C" __global__ void float_value_kernel(float x)
{
    cuda::std::array<float, 3> a;
    a[0] = x;
    printf("%f\n", a[0]);
}
