#include "wim.h"

#include <cuda_runtime.h>

#include <unordered_map>

#if defined(_WIN32)
    #define WIM_API __declspec(dllexport)
#else
    #define WIM_API __attribute__ ((visibility ("default")))
#endif

#define check_cuda(code) (wim::check_cuda_result(code, __FILE__, __LINE__))

// internal stuff
namespace wim
{

bool check_cuda_result(cudaError_t code, const char* file, int line)
{
    if (code == cudaSuccess)
        return true;

    fprintf(stderr, "WIM CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file, line);
    return false;
}

}

// API for Python bindings
extern "C"
{

WIM_API wim::Image* create_image_cpu(int width, int height)
{
    wim::Color* data = new wim::Color[width * height];
    wim::Image* img = new wim::Image(width, height, data);
    return img;
}

WIM_API void destroy_image_cpu(wim::Image* img)
{
    if (img)
    {
        delete [] img->getData();
        delete img;
    }
}

WIM_API wim::Image* create_image_cuda(int device, int width, int height)
{
    if (!check_cuda(cudaSetDevice(device)))
        return nullptr;

    wim::Color* data = nullptr;
    if (!check_cuda(cudaMalloc(&data, width * height * sizeof(wim::Color))))
        return nullptr;
    if (!check_cuda(cudaMemset(data, 0, width * height * sizeof(wim::Color))))
        return nullptr;

    wim::Image* img = new wim::Image(width, height, data);
    return img;
}

WIM_API void destroy_image_cuda(int device, wim::Image* img)
{
    if (!img)
        return;

    if (!check_cuda(cudaSetDevice(device)))
        return;

    check_cuda(cudaFree(img->getData()));

    delete img;
}

}
