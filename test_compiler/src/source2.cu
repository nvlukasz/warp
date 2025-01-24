#include <array>
#include <cmath>

namespace std
{
using namespace cuda::std;  // e.g., std::array
}

extern "C" __global__ void float_value_kernel(float x)
{
    std::array<float, 3> a;  // voila
    a[0] = x;
    a[1] = std::sqrt(x);
    a[2] = std::pow(x, 2.0f);
    printf("%g, %g, %g\n", a[0], a[1], a[2]);
}
